"""Overlay BO learning curves across (d, alpha) configs produced by script 63.

Reads every outputs/bo_thompson_<tag_glob>_* run, pulls the per-(step,model)
p_oracle_running_max from bo_summary.csv, and overlays the configs per surrogate.
p_oracle is the HARD oracle (alpha-independent), so all configs share one y-axis
and one ceiling -- a fair common ruler for "which (d,alpha) picks better words faster".

Usage:
  python scripts/68_bo_config_compare.py --glob "outputs/bo_thompson_cmp_*"
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/bo_thompson_cmp_*")
    ap.add_argument("--out", default="outputs/bo_config_compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dirs = sorted(glob.glob(args.glob))
    runs = []
    for d in dirs:
        summ = os.path.join(d, "bo_summary.csv")
        cfg = os.path.join(d, "config.json")
        if not (os.path.exists(summ) and os.path.exists(cfg)):
            continue
        c = json.load(open(cfg))
        runs.append({"dir": d, "d": c["d"], "alpha": c["alpha"],
                     "label": f"d={c['d']}, α={c['alpha']:g}",
                     "summary": pd.read_csv(summ)})
    if not runs:
        raise SystemExit(f"no runs matched {args.glob}")
    # stable ordering: by (alpha desc, d) so baseline d8a30 reads first
    runs.sort(key=lambda r: (-r["alpha"], r["d"]))
    print(f"Found {len(runs)} configs: " + ", ".join(r["label"] for r in runs))

    # ceiling/mean from any oracle.npz (hard oracle is config-independent).
    # best-so-far running-max saturates at p_max=1.0 (one outlier word has all-40
    # seeds beating B), so it barely separates configs. The top-5% mean is a robust
    # ceiling, and the RAW per-step picked p_oracle (no running-max) is the
    # discriminating "steady-state exploit quality" view.
    orc = np.load(os.path.join(runs[0]["dir"], "oracle.npz"), allow_pickle=True)["p_oracle"]
    p_max = float(orc.max())
    p_top5 = float(np.sort(orc)[::-1][:max(1, len(orc) // 20)].mean())
    p_mean = float(orc.mean())

    models = sorted(runs[0]["summary"]["model"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {r["label"]: cmap(i) for i, r in enumerate(runs)}

    def smooth(y, w=11):
        if len(y) < w:
            return y
        k = np.ones(w) / w
        return np.convolve(y, k, mode="same")

    def plot_metric(metric, ylabel, title, fname, do_smooth=False):
        ncol = 3
        nrow = int(np.ceil(len(models) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.2 * nrow), squeeze=False)
        axes = axes.reshape(-1)
        for mi, model in enumerate(models):
            ax = axes[mi]
            for r in runs:
                sub = r["summary"]
                sub = sub[(sub["model"] == model) & (sub["metric"] == metric)].sort_values("step")
                if sub.empty:
                    continue
                y = smooth(sub["mean"].values) if do_smooth else sub["mean"].values
                ax.plot(sub["step"], y, color=colors[r["label"]], label=r["label"], lw=1.5)
            ax.axhline(p_top5, color="k", ls="-", alpha=0.5, label=f"top-5% ceiling {p_top5:.3f}")
            ax.axhline(p_mean, color="k", ls="--", alpha=0.4, label=f"uniform {p_mean:.3f}")
            ax.set_title(model)
            ax.set_xlabel("BO step")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="lower right")
        for k in range(len(models), len(axes)):
            axes[k].axis("off")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, fname), dpi=140)
        plt.close(fig)

    plot_metric("p_oracle_running_max", "best-so-far hard p_oracle",
                f"BO best-so-far by (d, α) — saturates near max={p_max:.2f}, low separation",
                "compare_running_max.png")
    plot_metric("p_oracle", "per-step picked p_oracle (smoothed)",
                "BO per-step pick quality by (d, α) — discriminating exploit view",
                "compare_per_step.png", do_smooth=True)

    # ---- summary table: best-so-far final, AUC, and steady-state per-step mean ----
    final_rows = []
    for model in models:
        for r in runs:
            s = r["summary"]
            rm = s[(s["model"] == model) & (s["metric"] == "p_oracle_running_max")].sort_values("step")
            ps = s[(s["model"] == model) & (s["metric"] == "p_oracle")].sort_values("step")
            if rm.empty:
                continue
            tmax = rm["step"].max()
            ss = ps[ps["step"] >= tmax - 50]["mean"].mean()  # steady-state pick quality
            final_rows.append({"model": model, "config": r["label"],
                               "final_runmax": float(rm["mean"].iloc[-1]),
                               "auc_runmax": float(np.trapz(rm["mean"], rm["step"]) / tmax),
                               "perstep_last50": float(ss)})
    ft = pd.DataFrame(final_rows)
    ft.to_csv(os.path.join(args.out, "final_table.csv"), index=False)

    print(f"\nceilings: max={p_max:.3f}  top5%mean={p_top5:.3f}  uniform={p_mean:.3f}")
    print("\n=== best-so-far final-step (saturated, low separation) ===")
    print(ft.pivot(index="model", columns="config", values="final_runmax").round(3).to_string())
    print("\n=== per-step pick quality, mean of last 50 steps (DISCRIMINATING) ===")
    print(ft.pivot(index="model", columns="config", values="perstep_last50").round(3).to_string())
    print("\n=== AUC of best-so-far (learning speed) ===")
    print(ft.pivot(index="model", columns="config", values="auc_runmax").round(3).to_string())
    print(f"\nSaved -> {args.out}/  (compare_running_max.png, compare_per_step.png, final_table.csv)")


if __name__ == "__main__":
    main()
