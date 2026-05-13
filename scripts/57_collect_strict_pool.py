"""Collect SD3.5 + DreamSim for the strict single-token shoe-relevant pool.

Reads pool from strict_pool_v3.csv (228 words, all single-token under T5).
Reuses (emb, dreamsim) for the 129 seed_kept words from existing exp 47 data.
Generates SD3.5 + DreamSim only for new words.

Output: outputs/strict_pool_<tag>_<timestamp>/
  raw_data.csv  -- columns: word, dreamsim, source, valid
  embeddings.npz -- embs (N,4096), dreams (N,), words list, ref_emb, comp_emb,
                    d_B, b_idx, valid, sources

Usage:
  python scripts/57_collect_strict_pool.py --model_path <path>
"""

import os
import argparse
import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


PROMPT = ("Product photo of a single shoe, full shoe visible, side profile, "
          "centered on a plain white background")


def get_word_emb(pipe, word):
    """Same logic as exp 47: token embedding via diff with empty prompt."""
    with torch.no_grad():
        out = pipe.encode_prompt(prompt=word, prompt_2=word, prompt_3=word, negative_prompt="")
        out_empty = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", negative_prompt="")
    pe, ee = out[0], out_empty[0]
    L_w, L_e = pe.shape[1], ee.shape[1]
    if L_w > L_e:
        n = L_w - L_e
        return pe[0, :n, :].mean(dim=0).detach()
    ml = min(L_w, L_e)
    diffs = (pe[0, :ml] - ee[0, :ml]).norm(dim=1)
    return pe[0, diffs.argmax().item(), :].detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--ref_word", type=str, default="red")
    parser.add_argument("--pool_csv", type=str,
                        default="outputs/strict_pool_v3.csv")
    parser.add_argument("--existing_npz", type=str,
                        default="outputs/scaling_red_vs_green_d128_0416_1723/data.npz")
    parser.add_argument("--existing_csv", type=str,
                        default="outputs/scaling_red_vs_green_d128_0416_1723/raw_data.csv")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="strict")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(args.out_root, f"strict_pool_{args.tag}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # === Load pool ===
    df_pool = pd.read_csv(args.pool_csv)
    pool_words = df_pool["word"].astype(str).tolist()
    pool_sources = df_pool["source"].astype(str).tolist()
    print(f"Pool: {len(pool_words)} words")
    print(df_pool["source"].value_counts())

    # === Load existing seed data ===
    existing = np.load(args.existing_npz)
    df_existing = pd.read_csv(args.existing_csv)
    existing_words = df_existing["word"].astype(str).tolist()
    word_to_idx = {w: i for i, w in enumerate(existing_words)}
    existing_embs = existing["embs"]      # (172, 4096)
    existing_dreams = existing["dreams"]  # (172,)

    # Map seed_kept words to existing indices
    seed_kept_words = [w for w, s in zip(pool_words, pool_sources) if s == "seed_kept"]
    new_words = [(i, w) for i, (w, s) in enumerate(zip(pool_words, pool_sources))
                 if s != "seed_kept"]
    print(f"Seed reused: {len(seed_kept_words)}, new to generate: {len(new_words)}")

    # === Allocate full arrays ===
    N = len(pool_words)
    embs_full = np.zeros((N, 4096), dtype=np.float32)
    dreams_full = np.full(N, np.nan, dtype=np.float32)

    for i, w in enumerate(pool_words):
        if pool_sources[i] == "seed_kept" and w in word_to_idx:
            j = word_to_idx[w]
            embs_full[i] = existing_embs[j]
            dreams_full[i] = existing_dreams[j]

    print(f"Pre-loaded {(~np.isnan(dreams_full)).sum()} entries from existing data")

    # === Load models ===
    print("Loading SD3.5 ...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=args.device)
    print("Loading DreamSim ...")
    scorer = DreamSimScorer(device=args.device)

    def gen_image(emb_4096):
        emb_t = torch.from_numpy(emb_4096.astype(np.float32)).to(
            args.device, dtype=torch.float16).unsqueeze(0)
        embeds = gen.encode_batch_insert(PROMPT, emb_t)
        return gen.generate_batch(embeds, [args.seed])[0]

    # === Reference ===
    print(f"\nGenerating reference '{args.ref_word}' ...")
    ref_emb = get_word_emb(gen.pipe, args.ref_word).float().cpu().numpy()
    ref_img = gen_image(ref_emb)
    ref_img.save(os.path.join(out_dir, "reference.png"))
    ref_tensor = scorer.preprocess(ref_img)

    # === Iterate over new words ===
    t0 = time.time()
    for idx_count, (idx, w) in enumerate(new_words):
        try:
            emb = get_word_emb(gen.pipe, w).float().cpu().numpy()
            img = gen_image(emb)
            d = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            img.close()
            embs_full[idx] = emb
            dreams_full[idx] = d
        except Exception as e:
            print(f"[err] '{w}': {e}")
            continue

        if (idx_count + 1) % args.save_every == 0 or (idx_count + 1) == len(new_words):
            elapsed = time.time() - t0
            rate = (idx_count + 1) / elapsed
            eta = (len(new_words) - idx_count - 1) / rate if rate > 0 else 0
            print(f"[{idx_count + 1}/{len(new_words)}] '{w}' d={d:.4f}  "
                  f"rate={rate:.2f}/s  eta={eta/60:.1f}min")
            df = pd.DataFrame({
                "word": pool_words,
                "dreamsim": dreams_full,
                "source": pool_sources,
            })
            df.to_csv(os.path.join(out_dir, "raw_data.csv"), index=False)
            np.savez(os.path.join(out_dir, "embeddings.npz"),
                     embs=embs_full, dreams=dreams_full,
                     words=np.array(pool_words), sources=np.array(pool_sources),
                     ref_emb=ref_emb)

    # === Final + competitor selection ===
    valid = ~np.isnan(dreams_full)
    print(f"\nValid entries: {valid.sum()}/{N}")

    valid_dreams = dreams_full[valid]
    median_d = float(np.median(valid_dreams))
    valid_idx = np.where(valid)[0]
    rel_b = int(np.argmin(np.abs(valid_dreams - median_d)))
    b_idx = int(valid_idx[rel_b])
    d_B = float(dreams_full[b_idx])
    print(f"B='{pool_words[b_idx]}' d_B={d_B:.4f}  (median over {valid.sum()})")

    comp_img = gen_image(embs_full[b_idx])
    comp_img.save(os.path.join(out_dir, "competitor.png"))
    comp_img.close()

    np.savez(os.path.join(out_dir, "embeddings.npz"),
             embs=embs_full, dreams=dreams_full,
             words=np.array(pool_words), sources=np.array(pool_sources),
             ref_emb=ref_emb, comp_emb=embs_full[b_idx],
             d_B=d_B, b_idx=b_idx, valid=valid)

    df = pd.DataFrame({
        "word": pool_words,
        "dreamsim": dreams_full,
        "source": pool_sources,
        "valid": valid,
    })
    df.to_csv(os.path.join(out_dir, "raw_data.csv"), index=False)
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
