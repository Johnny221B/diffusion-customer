"""Merge dreams_partial_{0..3}.npz into a unified dreams_matrix.npz.

Each partial covers a contiguous slice [word_start, word_end) and stores:
  - dreams: (slice_size, n_seeds)
  - word_index: global word indices in the order they appear in `dreams`
  - word_start / word_end / n_seeds

Output: dreams_matrix.npz with:
  - dreams: (N_words_total, n_seeds)  -- placed in canonical word order
  - n_seeds, n_words
"""
import os
import json
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="dir containing dreams_partial_{0..K}.npz")
    ap.add_argument("--pool_npz", required=True,
                    help="canonical pool to inherit total word count")
    args = ap.parse_args()

    pool = np.load(args.pool_npz, allow_pickle=True)
    n_words_total = len(pool["embs"])
    pool_words = list(pool["words"])

    partials = sorted([f for f in os.listdir(args.out_dir)
                       if f.startswith("dreams_partial_") and f.endswith(".npz")])
    print(f"Found {len(partials)} partial files: {partials}")

    # Determine n_seeds from first partial
    first = np.load(os.path.join(args.out_dir, partials[0]))
    n_seeds = int(first["n_seeds"])
    print(f"n_seeds = {n_seeds}")

    full = np.full((n_words_total, n_seeds), np.nan, dtype=np.float32)
    coverage = np.zeros(n_words_total, dtype=bool)

    for fname in partials:
        d = np.load(os.path.join(args.out_dir, fname))
        rows = d["dreams"]                # (slice_size, n_seeds)
        idx  = d["word_index"]            # (slice_size,) global indices
        if rows.shape[1] != n_seeds:
            raise ValueError(f"{fname} n_seeds mismatch")
        for local_i, gid in enumerate(idx):
            full[gid] = rows[local_i]
            coverage[gid] = True
        print(f"  {fname}: covered {len(idx)} words "
              f"[{int(d['word_start'])}, {int(d['word_end'])})")

    miss = (~coverage).sum()
    print(f"\nCoverage: {coverage.sum()}/{n_words_total}  (missing {miss})")
    if miss > 0:
        missing_words = [pool_words[i] for i in np.where(~coverage)[0]]
        print(f"Missing words: {missing_words[:20]}")

    np.savez(os.path.join(args.out_dir, "dreams_matrix.npz"),
             dreams=full,
             n_seeds=n_seeds,
             n_words=n_words_total,
             words=np.array(pool_words),
             coverage=coverage)
    print(f"\nSaved -> {os.path.join(args.out_dir, 'dreams_matrix.npz')}")
    print(f"shape = ({n_words_total}, {n_seeds})")

    # Quick stats
    valid = ~np.any(np.isnan(full), axis=1)
    if valid.sum() > 0:
        per_word_mean = np.nanmean(full[valid], axis=1)
        per_word_std  = np.nanstd(full[valid], axis=1)
        print(f"\nPer-word dreamsim stats (over {valid.sum()} valid words):")
        print(f"  mean range: [{per_word_mean.min():.4f}, {per_word_mean.max():.4f}]")
        print(f"  std  range: [{per_word_std.min():.4f}, {per_word_std.max():.4f}]")
        print(f"  mean of std: {per_word_std.mean():.4f}")
        print(f"  median std:  {np.median(per_word_std):.4f}")


if __name__ == "__main__":
    main()
