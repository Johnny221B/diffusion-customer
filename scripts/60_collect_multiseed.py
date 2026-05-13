"""Collect multi-seed images per word for noise-aware ground truth.

For each word in [word_start, word_end), generate `n_seeds` images with seeds
[0, 1, ..., n_seeds-1], compute DreamSim against the reference, save all PNGs
plus a partial dreams matrix.

This script processes ONE GPU's slice. Launch 4 instances in parallel with
disjoint [word_start, word_end) ranges, then run the merge step.

Usage (single GPU):
  python scripts/60_collect_multiseed.py \
      --model_path .../stable-diffusion-3.5-large \
      --device cuda:0 \
      --pool_npz outputs/strict_pool_s228_0429_0119/embeddings.npz \
      --ref_png  outputs/strict_pool_s228_0429_0119/reference.png \
      --out_dir  outputs/multiseed_s228_M40_<stamp>/ \
      --word_start 0 --word_end 57 \
      --n_seeds 40 --gen_batch 8 \
      --partial_id 0
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from PIL import Image

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pool_npz", required=True,
                        help="existing strict_pool embeddings.npz with embs + words")
    parser.add_argument("--ref_png", required=True,
                        help="existing reference.png to score against")
    parser.add_argument("--out_dir", required=True,
                        help="shared output dir (PNGs + partial npz here)")
    parser.add_argument("--word_start", type=int, required=True)
    parser.add_argument("--word_end",   type=int, required=True)
    parser.add_argument("--n_seeds", type=int, default=40)
    parser.add_argument("--gen_batch", type=int, default=8,
                        help="how many seeds per pipe call (avoid OOM)")
    parser.add_argument("--partial_id", type=int, required=True,
                        help="ID for partial output filename (typically GPU id)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    # Save config alongside partial
    with open(os.path.join(args.out_dir, f"config_partial_{args.partial_id}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # === Load pool ===
    data = np.load(args.pool_npz, allow_pickle=True)
    embs_all = data["embs"]              # (228, 4096)
    words_all = list(data["words"])       # (228,)
    print(f"[{args.device}] Loaded pool: {len(words_all)} words", flush=True)

    subset = list(range(args.word_start, args.word_end))
    print(f"[{args.device}] Processing words [{args.word_start}, {args.word_end}) = {len(subset)} words", flush=True)

    # === Load models ===
    print(f"[{args.device}] Loading SD3.5 ...", flush=True)
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=args.device)
    print(f"[{args.device}] Loading DreamSim ...", flush=True)
    scorer = DreamSimScorer(device=args.device)

    # === Reference image: load from disk and preprocess ===
    print(f"[{args.device}] Loading reference image: {args.ref_png}", flush=True)
    ref_img = Image.open(args.ref_png).convert("RGB")
    ref_tensor = scorer.preprocess(ref_img)

    # === Iterate words ===
    seeds = list(range(args.n_seeds))
    n_subset = len(subset)
    dreams_partial = np.full((n_subset, args.n_seeds), np.nan, dtype=np.float32)
    word_index = []      # global indices, in order of subset

    t0 = time.time()
    for local_i, global_i in enumerate(subset):
        w = str(words_all[global_i])
        emb = embs_all[global_i]
        word_index.append(global_i)

        try:
            emb_t = torch.from_numpy(emb.astype(np.float32)).to(
                args.device, dtype=torch.float16).unsqueeze(0)
            embeds = gen.encode_batch_insert(prompt="", z_vectors_batch=emb_t)

            # Generate in chunks to avoid OOM
            imgs_for_word = []
            for chunk_start in range(0, args.n_seeds, args.gen_batch):
                chunk = seeds[chunk_start:chunk_start + args.gen_batch]
                # encode_batch_insert was called for batch=1; expand for batch=len(chunk)
                # Easier: re-call encode_batch_insert with batch size = len(chunk)
                emb_chunk = emb_t.expand(len(chunk), -1).contiguous()
                embeds_chunk = gen.encode_batch_insert(prompt="", z_vectors_batch=emb_chunk)
                imgs = gen.generate_batch(embeds_chunk, chunk)
                imgs_for_word.extend(imgs)

            for s, img in zip(seeds, imgs_for_word):
                # Save PNG: imgs/<gid:03d>_<word>_seed<ss:02d>.png
                fname = f"{global_i:03d}_{w}_seed{s:02d}.png"
                img.save(os.path.join(img_dir, fname))
                d = scorer.model(ref_tensor, scorer.preprocess(img)).item()
                dreams_partial[local_i, s] = d
                img.close()

            elapsed = time.time() - t0
            rate = (local_i + 1) / elapsed
            eta = (n_subset - local_i - 1) / rate if rate > 0 else 0
            print(f"[{args.device}] [{local_i + 1}/{n_subset}] '{w}' "
                  f"d_mean={dreams_partial[local_i].mean():.4f} "
                  f"d_std={dreams_partial[local_i].std():.4f}  "
                  f"rate={rate*60:.1f}/min  eta={eta/60:.1f}min",
                  flush=True)

        except Exception as e:
            print(f"[{args.device}] [err] '{w}': {e}", flush=True)
            continue

        # Save partial periodically (every 5 words)
        if (local_i + 1) % 5 == 0 or (local_i + 1) == n_subset:
            np.savez(os.path.join(args.out_dir, f"dreams_partial_{args.partial_id}.npz"),
                     dreams=dreams_partial,
                     word_index=np.array(word_index, dtype=np.int32),
                     word_start=args.word_start,
                     word_end=args.word_end,
                     n_seeds=args.n_seeds)

    # Final save
    np.savez(os.path.join(args.out_dir, f"dreams_partial_{args.partial_id}.npz"),
             dreams=dreams_partial,
             word_index=np.array(word_index, dtype=np.int32),
             word_start=args.word_start,
             word_end=args.word_end,
             n_seeds=args.n_seeds)
    print(f"[{args.device}] DONE. Saved dreams_partial_{args.partial_id}.npz", flush=True)


if __name__ == "__main__":
    main()
