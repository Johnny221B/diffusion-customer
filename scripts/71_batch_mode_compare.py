"""Compare two batch-acquisition regimes of script 63 on the SOFT oracle:

  seed_batch  : 1 arm (one posterior sample) x B random seeds   -- depth
  theta_batch : B posterior samples x 1 fixed seed              -- breadth (parallel TS)

Both spend the same per-step image budget B; they differ only in how the budget
is split between exploiting one word's seeds vs sampling B words once each.

For each run we recompute the soft expected success per candidate word
  p_soft(word) = mean_k sigmoid(alpha * (D_B - D_{word,k}))
and, per (sim, model, step), reduce the (possibly B) picks of that step to:
  picked_at_t  = MEAN p_soft over the step's picks   (quality of images made now)
  step_best    = MAX  p_soft over the step's picks
then best_so_far = cummax(step_best) within (sim, model).
For seed_batch (1 row/step) mean==max==the single pick, so this is identical to
the canonical curve; for theta_batch it correctly treats the B picks as one batch.

Overlays the two regimes per surrogate (5 panels), best-so-far thick + picked-at-t
thin, with the shared soft oracle ceiling / uniform mean.

Usage:
  python scripts/71_batch_mode_compare.py \
      --seed_run  outputs/bo_thompson_cmp_d16a12_0521_2255 \
      --theta_run outputs/bo_thompson_thetabatch_d16a12_<stamp> \
      --out outputs/batch_mode_compare
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
# regime -> linestyle for the overlay
STYLE = {"seed_batch": "-", "theta_batch": "--"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_run(run_dir, raw_csv, dreams_npz):
    """Return (label, alpha, d, agg_pick, agg_best, oracle_max, uniform_mean).

    agg_pick / agg_best are per-(step,model) frames with mean/std/count.
    """
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    alpha = float(cfg["alpha"]); d = int(cfg["d"])
    acq = cfg.get("acq_mode", "seed_batch")
    B_word = cfg.get("B_word_used", cfg.get("B_word", "canvas"))
    B_seed = int(cfg["B_seed"])

    dreams = np.load(dreams_npz, allow_pickle=True)["dreams"].astype(np.float64)
    words = pd.read_csv(raw_csv)["word"].astype(str).tolist()
    valid = ~np.any(np.isnan(dreams), axis=1)
    dreams = dreams[valid]; words = [w for w, v in zip(words, valid) if v]
    B_idx = words.index(B_word); D_B = float(dreams[B_idx, B_seed])
    keep = np.arange(len(words)) != B_idx
    dreams_kept = dreams[keep]

    p_soft_word = sigmoid(alpha * (D_B - dreams_kept)).mean(axis=1)
    oracle_max = float(p_soft_word.max()); uniform_mean = float(p_soft_word.mean())

    df = pd.read_csv(os.path.join(run_dir, "bo_per_step.csv"))
    df["p_soft"] = p_soft_word[df["picked_idx"].values]

    # collapse the (possibly B) picks of a step into one mean and one max
    step = (df.groupby(["sim_seed", "model", "step"])["p_soft"]
              .agg(picked=("mean"), step_best=("max")).reset_index())
    step = step.sort_values(["sim_seed", "model", "step"])
    step["best_so_far"] = step.groupby(["sim_seed", "model"])["step_best"].cummax()

    def agg(col):
        return (step.groupby(["step", "model"])[col]
                    .agg(["mean", "std", "count"]).reset_index())

    label = f"{acq} (B={cfg['B']}, fs={cfg.get('fixed_seed', '-')})"
    return dict(label=label, acq=acq, alpha=alpha, d=d, B=cfg["B"],
                B_word=B_word, B_seed=B_seed,
                agg_pick=agg("picked"), agg_best=agg("best_so_far"),
                oracle_max=oracle_max, uniform_mean=uniform_mean,
                n_sim=step["sim_seed"].nunique())


def line(ax, frame, model, color, ls, lw, alpha, label):
    s = frame[frame["model"] == model].sort_values("step")
    if s.empty:
        return
    se = s["std"].values / np.sqrt(np.maximum(s["count"].values, 1))
    ax.plot(s["step"], s["mean"], color=color, ls=ls, lw=lw, alpha=alpha, label=label)
    ax.fill_between(s["step"], s["mean"] - se, s["mean"] + se, color=color, alpha=0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_run", required=True)
    ap.add_argument("--theta_run", required=True)
    ap.add_argument("--out", default="outputs/batch_mode_compare")
    ap.add_argument("--raw_csv", default="outputs/strict_pool_s228_0429_0119/raw_data.csv")
    ap.add_argument("--dreams_npz", default="outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = [load_run(args.seed_run, args.raw_csv, args.dreams_npz),
            load_run(args.theta_run, args.raw_csv, args.dreams_npz)]
    oracle_max = runs[0]["oracle_max"]; uniform_mean = runs[0]["uniform_mean"]

    # ---- figure: 5 panels, overlay both regimes ----
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.3 * len(MODELS), 4.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        c = COLORS[model]
        for r in runs:
            ls = STYLE.get(r["acq"], "-")
            line(ax, r["agg_pick"], model, c, ls, 1.0, 0.5, f"{r['acq']}: picked@t")
            line(ax, r["agg_best"], model, c, ls, 2.6, 1.0, f"{r['acq']}: best-so-far")
        ax.axhline(oracle_max, color="k", ls=":", alpha=0.6, lw=1, label=f"oracle max ({oracle_max:.2f})")
        ax.axhline(uniform_mean, color="k", ls="-.", alpha=0.4, lw=1, label=f"uniform ({uniform_mean:.2f})")
        ax.set_title(model)
        ax.set_xlabel("BO step t")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("expected success rate (soft)")
    r0, r1 = runs
    fig.suptitle(
        f"seed_batch (solid) vs theta_batch (dashed)  "
        f"d={r0['d']}, α={r0['alpha']:g}, B={r0['B']}, "
        f"B_word={r0['B_word']} s_B={r0['B_seed']}, sims={r0['n_sim']}/{r1['n_sim']}",
        fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(args.out, "batch_mode_compare_soft.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    # ---- numeric summary: final best-so-far + steady-state picked@t (last 50) ----
    rows = []
    for r in runs:
        for model in MODELS:
            b = r["agg_best"][r["agg_best"]["model"] == model].sort_values("step")
            p = r["agg_pick"][r["agg_pick"]["model"] == model].sort_values("step")
            if b.empty:
                continue
            tmax = int(b["step"].max())
            rows.append({
                "model": model, "regime": r["acq"],
                "final_best_so_far": float(b["mean"].iloc[-1]),
                "picked_last50": float(p[p["step"] >= tmax - 50]["mean"].mean()),
                "auc_best": float(np.trapz(b["mean"], b["step"]) / tmax),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, "batch_mode_summary.csv"), index=False)

    print(f"oracle_max={oracle_max:.3f}  uniform={uniform_mean:.3f}")
    for metric in ["final_best_so_far", "picked_last50", "auc_best"]:
        print(f"\n=== {metric} ===")
        print(tab.pivot(index="model", columns="regime", values=metric).round(3).to_string())
    print(f"\nSaved -> {out_png}")
    print(f"Saved -> {os.path.join(args.out, 'batch_mode_summary.csv')}")


if __name__ == "__main__":
    main()
