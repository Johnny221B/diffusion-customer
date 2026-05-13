"""Learning curve over a (d, N) grid with multi-seed gt + balanced B.

Wraps 61's core (run_one_seed_v2 + helpers) to sweep
    d in d_list   x   N in N_list
using identical sampling/labels/eval. PCA is fit once at max(d_list) and sliced.

Usage:
  python scripts/62_learncurve_dimsweep.py \
      --pool_npz outputs/strict_pool_s228_0429_0119/embeddings.npz \
      --raw_csv  outputs/strict_pool_s228_0429_0119/raw_data.csv \
      --dreams_npz outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz \
      --alpha 30 \
      --d_list 8,16,32,64,128 \
      --N_list 100,500,1000,2000,5000,10000 \
      --n_seeds 100 \
      --gp_max_N 1000 \
      --B_word canvas --B_seed 34 \
      --tag dimsweep
"""

import os
import sys
import json
import argparse
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "mod61", os.path.join(_HERE, "61_learncurve_multiseed.py")
)
_M61 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_M61)

run_one_seed_v2 = _M61.run_one_seed_v2


def plot_dimsweep(agg, p_max, p_mean, out_dir, args, d_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ("top1_p",      "top1 winning rate",  True),
        ("topk_mean_p", "top5 winning rate",  True),
        ("regret_top1", "regret (top-1)",     False),
    ]
    colors = {
        "logistic":       "tab:blue",
        "logistic_l2":    "tab:cyan",
        "poly2_logistic": "tab:purple",
        "gp_rbf":         "tab:green",
        "random_forest":  "tab:orange",
        "random":         "gray",
    }
    nrows, ncols = len(d_list), len(metrics_to_plot)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
        sharex=True, squeeze=False,
    )
    for r, d in enumerate(d_list):
        for c, (metric, ylabel, draw_ceiling) in enumerate(metrics_to_plot):
            ax = axes[r, c]
            sub_d = agg[agg["d"] == d]
            for model in sorted(sub_d["model"].unique()):
                sub = sub_d[sub_d["model"] == model].sort_values("N")
                mean = sub[f"{metric}_mean"].values
                std  = sub[f"{metric}_std"].values
                n    = sub[f"{metric}_count"].values
                se = std / np.sqrt(np.maximum(n, 1))
                mask = ~np.isnan(mean)
                ax.errorbar(
                    sub["N"].values[mask], mean[mask], yerr=se[mask],
                    marker="o", capsize=3, label=model,
                    color=colors.get(model, "black"),
                )
            if draw_ceiling:
                ax.axhline(p_max,  color="k", linestyle=":",  alpha=0.5)
                ax.axhline(p_mean, color="k", linestyle="--", alpha=0.4)
            else:
                ax.axhline(0.0, color="k", linestyle=":", alpha=0.5)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(ylabel)
            if r == nrows - 1:
                ax.set_xlabel("N (samples)")
            if c == 0:
                ax.set_ylabel(f"d={d}")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"Dim sweep learning curve  α={args.alpha}, M={args.M_used}, "
        f"{args.n_seeds} seeds, B={args.B_word_used} s_B={args.B_seed}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curve_dimsweep.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_npz",   required=True)
    parser.add_argument("--raw_csv",    required=True)
    parser.add_argument("--dreams_npz", required=True)
    parser.add_argument("--alpha", type=float, default=30.0)
    parser.add_argument("--d_list", type=str, default="8,16,32,64,128")
    parser.add_argument("--N_list", type=str,
                        default="100,500,1000,2000,5000,10000")
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--gp_max_N", type=int, default=1000)
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="dimsweep")
    parser.add_argument("--B_word", type=str, default="",
                        help="explicit competitor word; empty -> moon or median")
    parser.add_argument("--B_seed", type=int, default=0)
    args = parser.parse_args()

    d_list = [int(x) for x in args.d_list.split(",")]
    N_list = [int(x) for x in args.N_list.split(",")]
    d_max  = max(d_list)

    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(
        args.out_root,
        f"learncurve_ms_{args.tag}_{stamp}" if args.tag else f"learncurve_ms_{stamp}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # Load pool
    pool = np.load(args.pool_npz, allow_pickle=True)
    embs = pool["embs"]
    df_raw = pd.read_csv(args.raw_csv)
    words = df_raw["word"].astype(str).tolist()

    dm = np.load(args.dreams_npz, allow_pickle=True)
    dreams_matrix = dm["dreams"]
    M = int(dreams_matrix.shape[1])
    args.M_used = M

    valid = ~np.any(np.isnan(dreams_matrix), axis=1)
    embs   = embs[valid]
    dreams_matrix = dreams_matrix[valid]
    words  = [w for w, v in zip(words, valid) if v]
    print(f"Loaded {len(embs)} valid words, M={M} seeds each.")

    # PCA once at d_max, then slice
    mean_emb = embs.mean(0)
    pca = PCA(n_components=d_max, random_state=0).fit(embs - mean_emb)
    Z_full_max = (embs - mean_emb) @ pca.components_.T
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"PCA cumvar at d in {d_list}: " +
          ", ".join(f"d={d}:{cum[d-1]:.3f}" for d in d_list))

    # Pick B
    if args.B_word:
        if args.B_word not in words:
            raise ValueError(f"--B_word {args.B_word!r} not in pool")
        B_idx = words.index(args.B_word)
    elif "moon" in words:
        B_idx = words.index("moon")
    else:
        per_word_mean = dreams_matrix.mean(axis=1)
        median_d = float(np.median(per_word_mean))
        B_idx = int(np.argmin(np.abs(per_word_mean - median_d)))
    args.B_idx = B_idx
    if args.B_seed < 0 or args.B_seed >= M:
        raise ValueError(f"--B_seed {args.B_seed} out of [0, {M})")
    D_B = float(dreams_matrix[B_idx, args.B_seed])
    args.B_word_used = words[B_idx]
    print(f"B='{words[B_idx]}' (idx={B_idx})  s_B={args.B_seed}  D_B = {D_B:.4f}")

    # Drop B from candidate pool
    keep = np.arange(len(words)) != B_idx
    Z_full_max_kept = Z_full_max[keep]
    dreams_matrix_words = dreams_matrix[keep]
    words_kept = [w for w, k in zip(words, keep) if k]

    p_oracle = (dreams_matrix_words < D_B).mean(axis=1).astype(np.float64)
    p_max  = float(p_oracle.max())
    p_mean = float(p_oracle.mean())
    print(f"Oracle: min={p_oracle.min():.3f} median={np.median(p_oracle):.3f} "
          f"max={p_max:.3f} mean={p_mean:.3f}")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    np.savez(os.path.join(out_dir, "oracle.npz"),
             words=np.array(words_kept), p_oracle=p_oracle,
             D_B=D_B, B_word=words[B_idx], M=M)

    print(f"\nd_list = {d_list}  N_list = {N_list}  n_seeds = {args.n_seeds}")
    print(f"GP skipped when N > {args.gp_max_N}  (poly2 skipped when d > 32)\n")

    rows = []
    for d in d_list:
        Z_full_d = Z_full_max_kept[:, :d]
        print(f"=== d={d}  (PCA cumvar = {cum[d-1]:.3f}) ===")
        for N in N_list:
            for seed in range(args.n_seeds):
                recs = run_one_seed_v2(
                    Z_full_d, dreams_matrix_words, D_B, p_oracle,
                    N, d, args.alpha, seed, None, args.gp_max_N,
                )
                rows.extend(recs)
            sub = pd.DataFrame([r for r in rows if r["N"] == N and r["d"] == d])
            per_model = sub.groupby("model")["top1_p"].mean().round(3)
            print(f"  [d={d:3d}  N={N:5d}] top1_p:")
            for m, v in per_model.items():
                print(f"      {m:<16s} {v:.3f}")
        # Snapshot CSV after each d so partial results survive crashes
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "learncurve_per_seed.csv"), index=False
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "learncurve_per_seed.csv"), index=False)

    metrics = ["top1_p", "topk_mean_p", "regret_top1", "regret_topk"]
    agg = (
        df.groupby(["d", "N", "model"])[metrics]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    agg.columns = [c if isinstance(c, str) else "_".join([x for x in c if x])
                   for c in agg.columns]
    agg["p_max"] = p_max
    agg["p_mean"] = p_mean
    agg.to_csv(os.path.join(out_dir, "learncurve_summary.csv"), index=False)

    plot_dimsweep(agg, p_max, p_mean, out_dir, args, d_list)
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
