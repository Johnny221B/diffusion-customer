"""Recompute the SOFT oracle for a finished script-63 run and draw the canonical
per-model BO curve (picked-at-t + best-so-far running max), matching
bo_thompson_curve_per_model_soft.png.

The soft oracle uses the SAME alpha as the run's labels (read from config.json):
  p_soft(word) = mean_k sigmoid(alpha * (D_B - D_{word,k}))
picked_idx in bo_per_step.csv indexes the candidate pool (competitor B dropped),
exactly as run_one_sim_seed produced it.

Usage:
  python scripts/69_soft_oracle_curve.py --run_dir outputs/bo_thompson_cmp_d16a12_0521_2255
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

MODELS = ["logistic_bayesian", "logistic_l2", "poly2_logistic", "gp_rbf", "random_forest"]
COLORS = {
    "logistic_bayesian": "tab:blue", "logistic_l2": "tab:cyan",
    "poly2_logistic": "tab:purple", "gp_rbf": "tab:green", "random_forest": "tab:orange",
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pool_npz", default="outputs/strict_pool_s228_0429_0119/embeddings.npz")
    ap.add_argument("--raw_csv",  default="outputs/strict_pool_s228_0429_0119/raw_data.csv")
    ap.add_argument("--dreams_npz", default="outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    alpha = float(cfg["alpha"]); d = cfg["d"]
    B_word = cfg.get("B_word_used", cfg.get("B_word", "canvas"))
    B_seed = int(cfg["B_seed"])

    # rebuild the candidate pool exactly like script 63
    dreams = np.load(args.dreams_npz, allow_pickle=True)["dreams"].astype(np.float64)
    words = pd.read_csv(args.raw_csv)["word"].astype(str).tolist()
    valid = ~np.any(np.isnan(dreams), axis=1)
    dreams = dreams[valid]; words = [w for w, v in zip(words, valid) if v]
    B_idx = words.index(B_word); D_B = float(dreams[B_idx, B_seed])
    keep = np.arange(len(words)) != B_idx
    dreams_kept = dreams[keep]                      # (n_cand, M)

    # per-candidate soft expected success (the gold)
    p_soft_word = sigmoid(alpha * (D_B - dreams_kept)).mean(axis=1)   # (n_cand,)
    oracle_max = float(p_soft_word.max())
    uniform_mean = float(p_soft_word.mean())

    # attach soft oracle to every step's pick, then running max within (sim,model)
    df = pd.read_csv(os.path.join(args.run_dir, "bo_per_step.csv"))
    df["p_soft"] = p_soft_word[df["picked_idx"].values]
    df = df.sort_values(["sim_seed", "model", "step"])
    df["p_soft_run_max"] = df.groupby(["sim_seed", "model"])["p_soft"].cummax()
    df.to_csv(os.path.join(args.run_dir, "bo_per_step_softoracle.csv"), index=False)
    n_sim = df["sim_seed"].nunique()

    # aggregate per (step, model)
    def agg(col):
        g = df.groupby(["step", "model"])[col].agg(["mean", "std", "count"]).reset_index()
        return g

    a_pick = agg("p_soft")
    a_best = agg("p_soft_run_max")

    # ---- plot: 5 panels, picked-at-t (thin) + best-so-far (thick) ----
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.2 * len(MODELS), 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        c = COLORS[model]
        for a, lw, alp, lab in [(a_pick, 1.0, 0.9, "picked at t"),
                                (a_best, 2.6, 1.0, "best-so-far (running max)")]:
            s = a[a["model"] == model].sort_values("step")
            if s.empty:
                continue
            se = s["std"].values / np.sqrt(np.maximum(s["count"].values, 1))
            ax.plot(s["step"], s["mean"], color=c, lw=lw, alpha=alp, label=lab)
            ax.fill_between(s["step"], s["mean"] - se, s["mean"] + se, color=c, alpha=0.12)
        ax.axhline(oracle_max, color="k", ls=":", alpha=0.6, lw=1, label=f"oracle max ({oracle_max:.2f})")
        ax.axhline(uniform_mean, color="k", ls="--", alpha=0.5, lw=1, label=f"uniform mean ({uniform_mean:.2f})")
        ax.set_title(model)
        ax.set_xlabel("BO step t")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("expected success rate (soft)")
    fig.suptitle(
        f"Per-model BO curves (soft oracle, α={alpha})  d={d}, N0={cfg['N0']}, "
        f"B={cfg['B']}, T={cfg['T']}, sims={n_sim}, B_word={B_word} s_B={B_seed}",
        fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.run_dir, "bo_thompson_curve_per_model_soft.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"alpha={alpha}, d={d}, sims={n_sim}, oracle_max={oracle_max:.3f}, uniform={uniform_mean:.3f}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
