"""Goodness-of-Fit (GOF) analysis on existing word data.

Tests whether P(y=1|z) = sigmoid(beta'z) is a reasonable model on PCA features.

Procedure:
  - Load 172-word data (embs, dreamsim) from existing exp 47 outputs
  - Recompute B at dreamsim median, drop B, generate binary labels
  - For each PCA dim d in {8,16,32,64,128}:
      For each model in {logistic, logistic_l2, poly2_logistic, gp_rbf, random_forest}:
          5-fold stratified CV
          Record AUC / LogLoss / Accuracy / Brier / ECE per fold
  - Report mean ± SE across folds, paired Wilcoxon comparing models
  - Reliability diagrams + scaling curves

Usage:
  python scripts/53_gof_offline.py \\
      --data_npz outputs/scaling_red_vs_green_d128_0416_1723/data.npz \\
      --raw_csv  outputs/scaling_red_vs_green_d128_0416_1723/raw_data.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score, brier_score_loss
)
from scipy import stats


# ---------- metric helpers ----------

def expected_calibration_error(y_true, p_pred, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (p_pred >= bins[i]) & (p_pred < bins[i + 1] if i < n_bins - 1
                                       else p_pred <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = p_pred[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def all_metrics(y_true, p_pred):
    y_pred = (p_pred >= 0.5).astype(int)
    out = {
        "auc":      np.nan,
        "logloss":  np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "brier":    brier_score_loss(y_true, p_pred),
        "ece":      expected_calibration_error(y_true, p_pred),
    }
    if len(set(y_true)) == 2:
        out["auc"]     = roc_auc_score(y_true, p_pred)
        out["logloss"] = log_loss(y_true, np.clip(p_pred, 1e-6, 1 - 1e-6))
    return out


# ---------- model factories ----------

def make_models(d):
    """Return dict of name -> sklearn-compatible estimator (with predict_proba)."""
    models = {}

    # Plain logistic, near-zero regularisation -> behaves like MLE
    models["logistic"] = LogisticRegression(
        C=1e6, max_iter=5000, solver="lbfgs"
    )

    # L2-regularised logistic, C tuned by inner CV (uses lbfgs; up to ~150 samples)
    models["logistic_l2"] = LogisticRegressionCV(
        Cs=[1e-3, 1e-2, 1e-1, 1, 10, 100], cv=3,
        max_iter=5000, solver="lbfgs",
    )

    # Polynomial degree-2 logistic — only meaningful for low d
    if d <= 32:
        models["poly2_logistic"] = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
            ("clf", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")),
        ])

    # GP with RBF kernel (auto-fit hyperparameters by marginal likelihood)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    models["gp_rbf"] = GaussianProcessClassifier(
        kernel=kernel, n_restarts_optimizer=2, random_state=0,
    )

    # Random forest baseline
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200, random_state=0, n_jobs=-1
    )

    return models


# ---------- one fold ----------

def fit_predict(model, Z_tr, y_tr, Z_te):
    """Fit model on (Z_tr, y_tr), return P(y=1) on Z_te.
    Standardize features; GP may use them at unit scale."""
    mu = Z_tr.mean(0)
    sd = Z_tr.std(0) + 1e-9
    Ztr = (Z_tr - mu) / sd
    Zte = (Z_te - mu) / sd
    if len(set(y_tr)) < 2:
        # constant class -> return constant prediction
        return np.full(len(Zte), float(y_tr[0]))
    model.fit(Ztr, y_tr)
    return model.predict_proba(Zte)[:, 1]


# ---------- main GOF loop ----------

def run_gof(Z_full, y, dims, model_specs, n_splits=5, seed=42):
    """Cross-validation across (d, model). Returns long-format DataFrame."""
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for d in dims:
        Z = Z_full[:, :d]
        for fold, (tr, te) in enumerate(skf.split(Z, y)):
            Z_tr, Z_te = Z[tr], Z[te]
            y_tr, y_te = y[tr], y[te]
            models = make_models(d)
            for name, model in models.items():
                try:
                    p = fit_predict(model, Z_tr, y_tr, Z_te)
                    m = all_metrics(y_te, p)
                except Exception as e:
                    print(f"[skip] d={d} model={name} fold={fold}: {e}")
                    m = {k: np.nan for k in ["auc", "logloss", "accuracy", "brier", "ece"]}
                rows.append({"d": d, "model": name, "fold": fold, **m})
            print(f"  d={d:3d} fold={fold} done")
    return pd.DataFrame(rows)


# ---------- aggregation ----------

def aggregate(df_long):
    metrics = ["auc", "logloss", "accuracy", "brier", "ece"]
    agg = (
        df_long
        .groupby(["d", "model"])[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # flatten columns
    agg.columns = [
        c if isinstance(c, str) else "_".join([x for x in c if x])
        for c in agg.columns
    ]
    return agg


def paired_tests(df_long, baseline="logistic"):
    """Compare each non-baseline model to baseline on each (d, metric) using
    paired Wilcoxon signed-rank test on per-fold values."""
    rows = []
    metrics = ["auc", "logloss", "accuracy", "brier", "ece"]
    for d in df_long["d"].unique():
        sub = df_long[df_long["d"] == d]
        base = sub[sub["model"] == baseline].sort_values("fold")
        if len(base) == 0:
            continue
        for model in sub["model"].unique():
            if model == baseline:
                continue
            other = sub[sub["model"] == model].sort_values("fold")
            if len(other) != len(base):
                continue
            for metric in metrics:
                a = base[metric].values
                b = other[metric].values
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.sum() < 3:
                    continue
                try:
                    stat, pval = stats.wilcoxon(a[mask], b[mask], zero_method="wilcox")
                except ValueError:
                    stat, pval = np.nan, 1.0
                rows.append({
                    "d": d, "metric": metric, "model": model, "baseline": baseline,
                    "n_pairs": int(mask.sum()),
                    "diff_mean": float(np.mean(b[mask] - a[mask])),
                    "wilcoxon_stat": stat, "p_value": pval,
                })
    return pd.DataFrame(rows)


# ---------- plots ----------

def plot_metrics_vs_dim(agg, out_path, metrics=("auc", "logloss", "brier", "ece")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5))
    colors = {
        "logistic":       "tab:blue",
        "logistic_l2":    "tab:cyan",
        "poly2_logistic": "tab:purple",
        "gp_rbf":         "tab:green",
        "random_forest":  "tab:orange",
    }
    for ax, metric in zip(axes, metrics):
        for model in agg["model"].unique():
            sub = agg[agg["model"] == model].sort_values("d")
            mean = sub[f"{metric}_mean"].values
            std  = sub[f"{metric}_std"].values
            n    = sub[f"{metric}_count"].values
            se = std / np.sqrt(np.maximum(n, 1))
            ax.errorbar(sub["d"], mean, yerr=se, marker="o", capsize=3,
                        label=model, color=colors.get(model, "black"))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("PCA dim d")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.grid(True, alpha=0.3)
        if metric == "auc":
            ax.axhline(0.5, color="k", alpha=0.3, linestyle="--", label="random")
        ax.legend(fontsize=8)

    fig.suptitle("GOF: metric vs PCA dimension (5-fold CV, mean ± SE)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagrams(Z_full, y, dims, out_path, n_bins=10, seed=42):
    """For each d and each model, plot empirical y rate vs predicted prob bin.
    Uses single 5-fold CV concatenated predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_dims = len(dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(4.2 * n_dims, 4.5), sharey=True)
    if n_dims == 1:
        axes = [axes]

    for ax, d in zip(axes, dims):
        Z = Z_full[:, :d]
        models = make_models(d)
        for name, model in models.items():
            preds = np.full(len(y), np.nan)
            try:
                for tr, te in skf.split(Z, y):
                    p = fit_predict(model, Z[tr], y[tr], Z[te])
                    preds[te] = p
            except Exception:
                continue
            mask = ~np.isnan(preds)
            if mask.sum() == 0:
                continue
            bins = np.linspace(0, 1, n_bins + 1)
            xs, ys = [], []
            for i in range(n_bins):
                m = mask & (preds >= bins[i]) & (preds < bins[i + 1] + (1e-9 if i == n_bins - 1 else 0))
                if m.sum() < 3:
                    continue
                xs.append(preds[m].mean())
                ys.append(y[m].mean())
            if xs:
                ax.plot(xs, ys, marker="o", label=name, alpha=0.85)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted P(y=1)")
        if d == dims[0]:
            ax.set_ylabel("empirical y rate")
        ax.set_title(f"d = {d}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Reliability diagrams (CV-pooled predictions)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------- N-scaling ----------

def n_scaling(Z_full, y, dims, model_specs, n_list, n_test=40, n_repeats=10, seed=42):
    """For each (d, model, N), repeatedly subsample N training points + n_test
    held-out, record metrics. Used to see at what N each metric saturates."""
    rng = np.random.RandomState(seed)
    rows = []
    N_total = len(y)
    for d in dims:
        Z = Z_full[:, :d]
        for N in n_list:
            if N + n_test > N_total:
                continue
            for rep in range(n_repeats):
                idx = rng.permutation(N_total)
                tr = idx[:N]; te = idx[N:N + n_test]
                if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                    continue
                models = make_models(d)
                for name, model in models.items():
                    try:
                        p = fit_predict(model, Z[tr], y[tr], Z[te])
                        m = all_metrics(y[te], p)
                    except Exception:
                        continue
                    rows.append({"d": d, "model": name, "N": N, "rep": rep, **m})
            print(f"  d={d} N={N} done")
    return pd.DataFrame(rows)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", type=str, required=True,
                        help="data.npz with embs (N, 4096), dreams (N,)")
    parser.add_argument("--raw_csv", type=str, required=True,
                        help="raw_data.csv with 'word' column matching embs order")
    parser.add_argument("--dims", type=str, default="8,16,32,64,128",
                        help="PCA dims to test, comma-separated")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_scaling", action="store_true",
                        help="Also run N-scaling sub-experiment")
    parser.add_argument("--n_list", type=str, default="30,50,80,120",
                        help="Train sizes for scaling experiment")
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(args.out_root, f"gof{('_' + args.tag) if args.tag else ''}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load
    data = np.load(args.data_npz)
    embs = data["embs"]                 # (N, 4096)
    dreams = data["dreams"]             # (N,)
    df_raw = pd.read_csv(args.raw_csv)
    words = df_raw["word"].tolist()

    print(f"Loaded {len(embs)} words, dim_emb={embs.shape[1]}")

    # === Recompute PCA on full embs (single global fit, 4096 -> max(dims)) ===
    dims = [int(x) for x in args.dims.split(",")]
    d_max = max(dims)
    from sklearn.decomposition import PCA
    mean_emb = embs.mean(0)
    pca = PCA(n_components=d_max, random_state=0).fit(embs - mean_emb)
    Z_full = (embs - mean_emb) @ pca.components_.T   # (N, d_max)
    print(f"PCA cumvar at d={d_max}: {pca.explained_variance_ratio_.sum():.3f}")

    # === Define B at dreamsim median, generate binary labels ===
    median_d = float(np.median(dreams))
    b_idx = int(np.argmin(np.abs(dreams - median_d)))
    d_B = float(dreams[b_idx])
    print(f"B='{words[b_idx]}' d_B={d_B:.4f} (median)")

    keep = np.arange(len(embs)) != b_idx
    Z_full = Z_full[keep]
    dreams = dreams[keep]
    y = (dreams < d_B).astype(int)
    words = [words[i] for i in range(len(words)) if keep[i]]
    print(f"After dropping B: N={len(y)}, positives={y.sum()} ({y.mean():.2f})")

    # === Main GOF: model x dim x fold ===
    print(f"\n=== GOF: dims={dims}, n_splits={args.n_splits} ===")
    df_long = run_gof(Z_full, y, dims, None, n_splits=args.n_splits, seed=args.seed)
    df_long.to_csv(os.path.join(out_dir, "gof_per_fold.csv"), index=False)

    agg = aggregate(df_long)
    agg.to_csv(os.path.join(out_dir, "gof_summary.csv"), index=False)
    print("\n=== Aggregate (mean over folds) ===")
    pivot = agg.pivot_table(index=["d", "model"], values=["auc_mean", "brier_mean", "ece_mean"])
    print(pivot.round(4))

    paired = paired_tests(df_long, baseline="logistic")
    paired.to_csv(os.path.join(out_dir, "paired_tests_vs_logistic.csv"), index=False)

    # === Plots ===
    plot_metrics_vs_dim(agg, os.path.join(out_dir, "metrics_vs_dim.png"))
    plot_reliability_diagrams(Z_full, y, dims, os.path.join(out_dir, "reliability_diagrams.png"))

    # === Optional: N-scaling ===
    if args.n_scaling:
        n_list = [int(x) for x in args.n_list.split(",")]
        scaling_dims = [d for d in dims if d in (16, 64)]
        print(f"\n=== N-scaling: dims={scaling_dims}, N={n_list} ===")
        df_sc = n_scaling(Z_full, y, scaling_dims, None, n_list,
                          n_test=min(40, max(20, len(y) - max(n_list))),
                          n_repeats=10, seed=args.seed)
        df_sc.to_csv(os.path.join(out_dir, "n_scaling_per_rep.csv"), index=False)

        sc_agg = (df_sc.groupby(["d", "model", "N"])
                       [["auc", "logloss", "accuracy", "brier", "ece"]]
                       .agg(["mean", "std", "count"])
                       .reset_index())
        sc_agg.columns = [
            c if isinstance(c, str) else "_".join([x for x in c if x])
            for c in sc_agg.columns
        ]
        sc_agg.to_csv(os.path.join(out_dir, "n_scaling_summary.csv"), index=False)

    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
