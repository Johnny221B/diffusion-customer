"""Collect a large word pool (>=1500 words) for GOF analysis.

Reuses existing 172-word data when possible (loads embs/dreams from exp 47 npz).
Extends with random samples from WordNet adjectives + concrete nouns.

For each new word:
  - Get token embedding via get_word_emb (same logic as exp 47)
  - Generate SD3.5 image with fixed seed and prompt
  - Compute DreamSim distance to reference image
  - Save (word, embedding, dreamsim) to checkpoint after every 50 words

Output: outputs/large_pool_<timestamp>/
  raw_data.csv  -- columns: word, dreamsim, source
  embeddings.npz -- embs (N,4096), dreams (N,), words list, ref_emb, comp_emb,
                    d_B (competitor distance), source list
  reference.png, competitor.png
"""

import os
import argparse
import json
import time
import re
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


PROMPT = ("Product photo of a single shoe, full shoe visible, side profile, "
          "centered on a plain white background")


# -------- helpers --------

def get_word_emb(pipe, word):
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


def build_word_pool(seed_words, target_n, rng_seed=0):
    """Combine seed words + random WordNet samples to reach target_n unique words."""
    from nltk.corpus import wordnet as wn

    # Existing seeds (the 172-word list) -- always include
    pool = list(dict.fromkeys(seed_words))  # dedupe preserving order
    pool_set = set(pool)
    sources = ["seed"] * len(pool)

    # Random WordNet adjectives + concrete nouns
    word_re = re.compile(r"^[a-z]+$")
    adjs = [l.name() for s in wn.all_synsets("a") for l in s.lemmas()]
    nouns = [l.name() for s in wn.all_synsets("n") for l in s.lemmas()]

    def filt(words):
        return sorted({
            w for w in words
            if word_re.match(w) and 3 <= len(w) <= 10
        })

    adjs = filt(adjs)
    nouns = filt(nouns)

    rng = np.random.RandomState(rng_seed)
    rng.shuffle(adjs); rng.shuffle(nouns)

    # Add adjectives first (more visually descriptive), then nouns
    for batch, tag in [(adjs, "wn_adj"), (nouns, "wn_noun")]:
        for w in batch:
            if w in pool_set:
                continue
            pool.append(w); sources.append(tag); pool_set.add(w)
            if len(pool) >= target_n:
                break
        if len(pool) >= target_n:
            break
    return pool, sources


# -------- main --------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--ref_word", type=str, default="red")
    parser.add_argument("--target_n", type=int, default=1500)
    parser.add_argument("--existing_npz", type=str,
                        default="outputs/scaling_red_vs_green_d128_0416_1723/data.npz",
                        help="existing 172-word data to seed and reuse")
    parser.add_argument("--existing_csv", type=str,
                        default="outputs/scaling_red_vs_green_d128_0416_1723/raw_data.csv")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(args.out_root,
                           f"large_pool{('_' + args.tag) if args.tag else ''}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # === Load existing 172-word data (reuse embs + dreams to save GPU time) ===
    existing = np.load(args.existing_npz)
    df_existing = pd.read_csv(args.existing_csv)
    existing_words = df_existing["word"].tolist()
    existing_embs = existing["embs"]      # (172, 4096)
    existing_dreams = existing["dreams"]  # (172,)
    print(f"Reusing {len(existing_words)} existing words (no SD3.5 needed for these)")

    # === Build pool: existing + WordNet ===
    pool, sources = build_word_pool(existing_words, args.target_n, rng_seed=args.seed)
    new_words = [(i, w) for i, w in enumerate(pool) if i >= len(existing_words)]
    print(f"Total pool: {len(pool)}, new words to generate: {len(new_words)}")

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

    # Sanity: existing dreams already use this same ref via prior run, but we
    # cannot guarantee bit-exact equality of the new SD3.5 ref image. Recompute
    # ref dreamsim of the seed words using the new ref to be consistent.
    # ---> Strategy: re-use existing dreams (consistent within their own run),
    # and run new words against the freshly generated ref. We separate them by
    # "source" column so downstream can choose to use only new or pooled.

    # === Allocate full arrays ===
    N = len(pool)
    embs_full = np.zeros((N, 4096), dtype=np.float32)
    dreams_full = np.full(N, np.nan, dtype=np.float32)

    embs_full[: len(existing_words)] = existing_embs
    dreams_full[: len(existing_words)] = existing_dreams

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
            eta = (len(new_words) - idx_count - 1) / rate
            done = (idx_count + 1) + len(existing_words)
            print(f"[{done}/{N}] '{w}' d={d:.4f}  rate={rate:.2f}/s  eta={eta/60:.1f}min")
            # Checkpoint save
            df = pd.DataFrame({
                "word": pool,
                "dreamsim": dreams_full,
                "source": sources,
            })
            df.to_csv(os.path.join(out_dir, "raw_data.csv"), index=False)
            np.savez(os.path.join(out_dir, "embeddings.npz"),
                     embs=embs_full, dreams=dreams_full,
                     words=np.array(pool), sources=np.array(sources),
                     ref_emb=ref_emb)

    # === Final save + competitor selection ===
    valid = ~np.isnan(dreams_full)
    print(f"\nValid entries: {valid.sum()}/{N}")

    # Pick competitor at median dreamsim (over valid entries only)
    valid_dreams = dreams_full[valid]
    median_d = float(np.median(valid_dreams))
    valid_idx = np.where(valid)[0]
    rel_b = int(np.argmin(np.abs(valid_dreams - median_d)))
    b_idx = int(valid_idx[rel_b])
    d_B = float(dreams_full[b_idx])
    print(f"B='{pool[b_idx]}' d_B={d_B:.4f}  (median over {valid.sum()} valid)")

    # Generate competitor image (for reproducibility / paper)
    comp_img = gen_image(embs_full[b_idx])
    comp_img.save(os.path.join(out_dir, "competitor.png"))
    comp_img.close()

    np.savez(os.path.join(out_dir, "embeddings.npz"),
             embs=embs_full, dreams=dreams_full,
             words=np.array(pool), sources=np.array(sources),
             ref_emb=ref_emb, comp_emb=embs_full[b_idx],
             d_B=d_B, b_idx=b_idx, valid=valid)

    df = pd.DataFrame({
        "word": pool,
        "dreamsim": dreams_full,
        "source": sources,
        "valid": valid,
    })
    df.to_csv(os.path.join(out_dir, "raw_data.csv"), index=False)

    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
