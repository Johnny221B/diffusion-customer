"""Generate a markdown report from a GOF run directory.

Reads gof_per_fold.csv, gof_summary.csv, paired_tests_vs_logistic.csv,
n_scaling_summary.csv (if present) and produces REPORT.md with tables,
key findings, and pointers to the figures.

Usage:
  python scripts/55_gof_report.py --run_dir outputs/gof_pilot172_0428_1658
"""

import os
import argparse
import numpy as np
import pandas as pd


def fmt_mean_se(mean, std, n):
    """mean ± SE, with SE = std / sqrt(n)."""
    if pd.isna(mean):
        return "—"
    se = std / np.sqrt(max(n, 1)) if not pd.isna(std) else 0.0
    return f"{mean:.3f} ± {se:.3f}"


def make_metric_table(agg, metric):
    """Pivot to (rows=d, cols=model) with mean ± SE."""
    pivot_mean = agg.pivot_table(index="d", columns="model", values=f"{metric}_mean")
    pivot_std  = agg.pivot_table(index="d", columns="model", values=f"{metric}_std")
    pivot_n    = agg.pivot_table(index="d", columns="model", values=f"{metric}_count")

    rows = []
    for d in pivot_mean.index:
        row = {"d": d}
        for model in pivot_mean.columns:
            m = pivot_mean.loc[d, model]
            s = pivot_std.loc[d, model]
            n = pivot_n.loc[d, model]
            row[model] = fmt_mean_se(m, s, n)
        rows.append(row)
    return pd.DataFrame(rows).set_index("d")


def find_best_per_dim(agg, metric, lower_is_better):
    """Return dict d -> best_model name."""
    pivot = agg.pivot_table(index="d", columns="model", values=f"{metric}_mean")
    out = {}
    for d in pivot.index:
        row = pivot.loc[d].dropna()
        if len(row) == 0:
            continue
        out[d] = row.idxmin() if lower_is_better else row.idxmax()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    rd = args.run_dir
    agg = pd.read_csv(os.path.join(rd, "gof_summary.csv"))

    paired_path = os.path.join(rd, "paired_tests_vs_logistic.csv")
    paired = pd.read_csv(paired_path) if os.path.exists(paired_path) else None

    sc_path = os.path.join(rd, "n_scaling_summary.csv")
    has_sc = os.path.exists(sc_path)
    sc = pd.read_csv(sc_path) if has_sc else None

    # ------- generate report -------
    out_md = os.path.join(rd, "REPORT.md")
    L = []

    L.append(f"# GOF Report — `{os.path.basename(rd)}`\n")
    L.append("## 1. Aggregate metrics (5-fold CV, mean ± SE)\n")

    for metric, lower_is_better in [
        ("auc",      False),
        ("logloss",  True),
        ("accuracy", False),
        ("brier",    True),
        ("ece",      True),
    ]:
        tbl = make_metric_table(agg, metric)
        best = find_best_per_dim(agg, metric, lower_is_better)
        L.append(f"### {metric.upper()}  ({'lower better' if lower_is_better else 'higher better'})\n")
        L.append(tbl.to_markdown())
        L.append("\n\nBest model per d: " +
                 ", ".join(f"d={d} → **{m}**" for d, m in best.items()))
        L.append("\n")

    # ------- significance summary -------
    if paired is not None:
        L.append("## 2. Paired Wilcoxon vs `logistic` (p-values)\n")
        L.append("p < 0.05 means model differs significantly from plain logistic.\n")
        for metric in ["auc", "brier", "ece"]:
            sub = paired[paired["metric"] == metric]
            if len(sub) == 0:
                continue
            piv = sub.pivot_table(index="d", columns="model", values="p_value")
            L.append(f"\n### {metric.upper()} p-values\n")
            L.append(piv.round(3).to_markdown())
            L.append("\n")

    # ------- key takeaways -------
    L.append("## 3. Key takeaways\n")
    auc = agg.pivot_table(index="d", columns="model", values="auc_mean")
    brier = agg.pivot_table(index="d", columns="model", values="brier_mean")

    # find sweet spot
    best_auc_per_d = auc.max(axis=1)
    sweet_d = int(best_auc_per_d.idxmax())
    sweet_val = float(best_auc_per_d.max())
    sweet_model = auc.loc[sweet_d].idxmax()
    L.append(f"- **Sweet spot**: d={sweet_d}, best model `{sweet_model}`, AUC={sweet_val:.3f}\n")

    # logistic adequacy
    if "logistic" in auc.columns:
        log_auc = auc["logistic"]
        adequate_dims = log_auc[log_auc >= 0.65].index.tolist()
        if adequate_dims:
            L.append(f"- **Plain logistic adequate (AUC ≥ 0.65) at**: d ∈ {adequate_dims}\n")
        else:
            L.append("- **Plain logistic never reaches AUC 0.65** — linear model insufficient at all tested d\n")

    # GP collapse
    if "gp_rbf" in auc.columns:
        gp_auc = auc["gp_rbf"]
        collapse = gp_auc[gp_auc < 0.55].index.tolist()
        if collapse:
            L.append(f"- **GP-RBF collapses (AUC < 0.55) at**: d ∈ {collapse}\n")

    # L2 helps?
    if "logistic" in auc.columns and "logistic_l2" in auc.columns:
        diff = auc["logistic_l2"] - auc["logistic"]
        big_help = diff[diff > 0.03].index.tolist()
        if big_help:
            L.append(f"- **L2 regularization helps logistic substantially (Δ AUC > 0.03) at**: d ∈ {big_help}\n")

    # ------- N scaling -------
    if has_sc:
        L.append("\n## 4. N-scaling (per-N mean across repeats)\n")
        for d in sorted(sc["d"].unique()):
            sub = sc[sc["d"] == d]
            piv = sub.pivot_table(index="N", columns="model", values="auc_mean")
            L.append(f"\n### d = {d} — AUC vs N\n")
            L.append(piv.round(3).to_markdown())
            L.append("\n")

    # ------- pointers -------
    L.append("\n## 5. Files\n")
    for fn in [
        "gof_per_fold.csv",
        "gof_summary.csv",
        "paired_tests_vs_logistic.csv",
        "n_scaling_per_rep.csv",
        "n_scaling_summary.csv",
        "metrics_vs_dim.png",
        "reliability_diagrams.png",
        "config.json",
    ]:
        p = os.path.join(rd, fn)
        if os.path.exists(p):
            L.append(f"- `{fn}`\n")

    with open(out_md, "w") as f:
        f.write("\n".join(L))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
