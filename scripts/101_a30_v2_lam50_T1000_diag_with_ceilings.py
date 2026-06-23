#!/usr/bin/env python3
"""Four-panel T1000 diagnostic with hard/soft oracle ceilings.

Run dir note: the directory name still says T200, but the same sims have been
extended in place to T300/T500/T1000.

Ceilings:
  - Mathematical ceiling for both hard win-rate and true_p is 1.0.
  - The more informative dashed lines are the discrete oracle ceilings computed
    from the precomputed 228-word × 40-seed dreams matrix:
      hard: max_w mean_seed 1[D[w,seed] < D_B]
      soft: max_w mean_seed sigmoid(alpha * (D_B - D[w,seed]))
    These are not the exact continuous-run estimand, because continuous renders
    arbitrary PCA-manifold points with one fixed render seed, but they are a
    useful task-level reference.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("outputs/cmts_a30_v2_lam50_bbright_d16_B8_T200_0621_0437")
OUT = Path("outputs/cmts_top4_summary_0621_0437")
OUT.mkdir(parents=True, exist_ok=True)

DREAMS_NPZ = Path("outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz")
D_B = 0.4704614281654358
ALPHA = 30.0
PREV_BEST_DS = 0.2803


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def oracle_ceilings():
    z = np.load(DREAMS_NPZ, allow_pickle=True)
    dreams = z["dreams"].astype(float)
    words = z["words"].astype(str)

    hard_by_word = (dreams < D_B).mean(axis=1)
    soft_by_word = sigmoid(ALPHA * (D_B - dreams)).mean(axis=1)

    ih = int(np.argmax(hard_by_word))
    is_ = int(np.argmax(soft_by_word))
    return {
        "hard_oracle": float(hard_by_word[ih]),
        "hard_word": str(words[ih]),
        "soft_oracle": float(soft_by_word[is_]),
        "soft_word": str(words[is_]),
        "hard_top5_mean": float(np.sort(hard_by_word)[-5:].mean()),
        "soft_top5_mean": float(np.sort(soft_by_word)[-5:].mean()),
    }


def load_rounds():
    parts = []
    for sim_dir in sorted(ROOT.glob("sim[0-9][0-9][0-9]")):
        path = sim_dir / "trajectory.csv"
        if not path.exists():
            continue
        m = pd.read_csv(path)
        m = m.query("phase == 'main'").copy()
        if m.empty:
            continue
        x = (
            m.groupby("t")
            .agg(
                hard=("y_hard", "mean"),
                true_p=("true_p_soft", "mean"),
                pred_p=("predicted_p", "mean"),
                mean_ds=("ds_to_R", "mean"),
                round_min_ds=("ds_to_R", "min"),
                beta_norm=("beta_norm", "first"),
            )
            .reset_index()
        )
        x["best_ds"] = x["round_min_ds"].cummin()
        x["sim"] = sim_dir.name
        parts.append(x)
    if not parts:
        raise SystemExit(f"No trajectory.csv data found under {ROOT}")
    return pd.concat(parts, ignore_index=True)


def stats(r, col):
    g = r.groupby("t")[col]
    return g.mean(), (g.std(ddof=1) / np.sqrt(g.count())).fillna(0), g.count()


def draw(r, ax, col, color, label, ls="-", band=True, roll=15):
    mean, err, _ = stats(r, col)
    smooth = mean.rolling(roll, center=True, min_periods=1).mean()
    ax.plot(
        mean.index,
        smooth,
        color=color,
        lw=2.0 if ls == "-" else 1.6,
        ls=ls,
        label=label,
    )
    if band:
        ax.fill_between(mean.index, mean - err, mean + err, color=color, alpha=0.14)


def add_hline(ax, y, label, color="0.2", ls=":", lw=1.2):
    ax.axhline(y, color=color, ls=ls, lw=lw, alpha=0.85, label=label)


def plot(r, ceil):
    max_t_by_sim = r.groupby("sim")["t"].max().to_dict()
    complete = all(v >= 999 for v in max_t_by_sim.values()) and len(max_t_by_sim) == 5
    suffix = "T1000" if complete else "T1000_partial"

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.8), sharex=True)
    (ax_hard, ax_belief), (ax_ds, ax_best) = axes

    draw(r, ax_hard, "hard", "tab:red", "hard win-rate")
    add_hline(ax_hard, 1.0, "math ceiling=1.0", color="0.25", ls=":")
    add_hline(
        ax_hard,
        ceil["hard_oracle"],
        f"discrete oracle={ceil['hard_oracle']:.3f} ({ceil['hard_word']})",
        color="tab:red",
        ls="--",
    )
    add_hline(
        ax_hard,
        ceil["hard_top5_mean"],
        f"top-5 mean={ceil['hard_top5_mean']:.3f}",
        color="tab:red",
        ls="-.",
        lw=1.0,
    )
    ax_hard.axhline(0.5, color="0.4", ls=":", lw=1.0)

    draw(r, ax_belief, "true_p", "tab:green", "true p")
    draw(r, ax_belief, "pred_p", "tab:red", "predicted p", ls="--", band=False)
    add_hline(ax_belief, 1.0, "math ceiling=1.0", color="0.25", ls=":")
    add_hline(
        ax_belief,
        ceil["soft_oracle"],
        f"discrete soft oracle={ceil['soft_oracle']:.3f} ({ceil['soft_word']})",
        color="tab:green",
        ls="--",
    )
    add_hline(
        ax_belief,
        ceil["soft_top5_mean"],
        f"top-5 mean={ceil['soft_top5_mean']:.3f}",
        color="tab:green",
        ls="-.",
        lw=1.0,
    )
    ax_belief.axhline(0.5, color="0.4", ls=":", lw=1.0)

    draw(r, ax_ds, "mean_ds", "tab:purple", "mean ds-to-R")
    add_hline(ax_ds, D_B, f"$D_B$={D_B:.4f}", color="0.4", ls=":")

    draw(r, ax_best, "best_ds", "tab:blue", "best-so-far ds-to-R")
    add_hline(ax_best, PREV_BEST_DS, f"previous best {PREV_BEST_DS:.4f}", color="k", ls=":")

    titles = [
        "(a) hard win-rate (roll15; mean ± SE over available sims)",
        "(b) belief vs truth (solid=true, dashed=predicted)",
        "(c) mean ds-to-R per round (lower is better)",
        "(d) best-so-far ds-to-R (lower is better)",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.axvline(200, color="0.35", ls="--", lw=1.1, alpha=0.7)
        ax.axvline(300, color="0.35", ls="-.", lw=1.1, alpha=0.7)
        ax.axvline(500, color="0.35", ls=":", lw=1.1, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("round t")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    # Show how many sims support the late part of the curve.
    _, _, counts = stats(r, "hard")
    ax_count = ax_hard.twinx()
    ax_count.plot(counts.index, counts.values, color="tab:blue", lw=1.0, alpha=0.65)
    ax_count.set_ylabel("available sims", color="tab:blue")
    ax_count.tick_params(axis="y", labelcolor="tab:blue")
    ax_count.set_ylim(0, max(5.5, counts.max() + 0.5))

    ax_hard.set_ylabel("win-rate")
    ax_belief.set_ylabel("probability")
    ax_ds.set_ylabel("mean ds-to-R")
    ax_best.set_ylabel("best ds-to-R")

    sim_status = ", ".join(f"{k}={v}" for k, v in max_t_by_sim.items())
    fig.suptitle(
        r"Continuous CM-TS: $\alpha$=30, $v$=2, $\lambda$=50, bright/18, B=8"
        + f"\n{suffix}; max t by sim: {sim_status}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png = OUT / f"a30_v2_lam50_{suffix}_diagnostics_with_ceilings.png"
    fig.savefig(png, dpi=170)
    plt.close(fig)
    return png, suffix


def save_windows(r, suffix):
    specs = [
        (0, 19, "first20"),
        (180, 199, "T200_last20"),
        (280, 299, "T300_last20"),
        (480, 499, "T500_last20"),
        (900, 949, "T1000_900_949"),
        (950, 999, "T1000_950_999"),
        (980, 999, "T1000_last20_available"),
        (990, 999, "T1000_last10_available"),
    ]
    windows = []
    for lo, hi, label in specs:
        sub = r[r.t.between(lo, hi)]
        if sub.empty:
            continue
        per_sim = sub.groupby("sim")[["hard", "true_p", "pred_p", "mean_ds", "best_ds"]].mean()
        row = {"window": label, "t_lo": lo, "t_hi": hi, "n_sims": int(len(per_sim))}
        for col in per_sim:
            row[col] = float(per_sim[col].mean())
            row[f"{col}_se"] = float(per_sim[col].std(ddof=1) / np.sqrt(len(per_sim))) if len(per_sim) > 1 else 0.0
        windows.append(row)
    out = OUT / f"a30_v2_lam50_{suffix}_windows.csv"
    pd.DataFrame(windows).to_csv(out, index=False)
    return out, pd.DataFrame(windows)


def main():
    r = load_rounds()
    ceil = oracle_ceilings()
    png, suffix = plot(r, ceil)
    csv, windows = save_windows(r, suffix)

    print(f"saved: {png}")
    print(f"saved: {csv}")
    print("max t by sim:")
    print(r.groupby("sim")["t"].max().to_string())
    print("\nceilings:")
    for k, v in ceil.items():
        print(f"  {k}: {v}")
    print("\nwindows:")
    print(windows.to_string(index=False))


if __name__ == "__main__":
    main()
