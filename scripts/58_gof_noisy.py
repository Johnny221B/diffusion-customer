"""GOF analysis with noisy Bernoulli labels per word.

Per-word probability:    p_i = sigmoid(alpha * (d_B - dreams_i))
Per-word labels:         y_i^(j) ~ Bernoulli(p_i),  j = 1..k

CV uses GroupKFold by word index, so all k samples of a word are in the same
split (train OR test, never both -> no leakage).

Effective sample size:
  - rows: N_words * k
  - independent groups for AUC: N_words

Usage:
  python scripts/58_gof_noisy.py \\
      --data_npz outputs/strict_pool_s228_<stamp>/embeddings.npz \\
      --raw_csv  outputs/strict_pool_s228_<stamp>/raw_data.csv \\
      --alpha 5 --k_per_word 20
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
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, brier_score_loss
from scipy import stats


# ---------- helpers ----------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(y_true, p_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (p_pred >= bins[i]) & (p_pred < bins[i + 1])
        else:
            mask = (p_pred >= bins[i]) & (p_pred <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = p_pred[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def all_metrics(y_true, p_pred, p_oracle=None):
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
    # If oracle p available, also report calibration vs oracle directly
    if p_oracle is not None:
        out["mse_vs_oracle"] = float(np.mean((p_pred - p_oracle) ** 2))
        out["mae_vs_oracle"] = float(np.mean(np.abs(p_pred - p_oracle)))
    return out


# ---------- model factories ----------

def make_models(d, skip=()):
    models = {}
    models["logistic"] = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs")
    models["logistic_l2"] = LogisticRegressionCV(
        Cs=[1e-3, 1e-2, 1e-1, 1, 10, 100], cv=3,
        max_iter=5000, solver="lbfgs",
    )
    if d <= 32:
        models["poly2_logistic"] = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("clf", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")),
        ])
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    models["gp_rbf"] = GaussianProcessClassifier(
        kernel=kernel, n_restarts_optimizer=2, random_state=0,
    )
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200, random_state=0, n_jobs=-1
    )
    for s in skip:
        models.pop(s, None)
    return models


def fit_predict(model, Z_tr, y_tr, Z_te):
    mu = Z_tr.mean(0)
    sd = Z_tr.std(0) + 1e-9
    Ztr = (Z_tr - mu) / sd
    Zte = (Z_te - mu) / sd
    if len(set(y_tr)) < 2:
        return np.full(len(Zte), float(y_tr[0]))
    model.fit(Ztr, y_tr)
    return model.predict_proba(Zte)[:, 1]


# ---------- noisy label generation ----------

def make_noisy_dataset(Z_word, p_word, k_per_word, label_seed):
    """Expand per-word data to k Bernoulli samples per word.
    Returns (Z_rows, y_rows, p_oracle_rows, group_rows)."""
    rng = np.random.RandomState(label_seed)
    N_words = len(Z_word)
    Z_rows = np.repeat(Z_word, k_per_word, axis=0)
    p_rows = np.repeat(p_word, k_per_word)
    group_rows = np.repeat(np.arange(N_words), k_per_word)
    y_rows = (rng.uniform(size=len(p_rows)) < p_rows).astype(int)
    return Z_rows, y_rows, p_rows, group_rows


# ---------- main GOF loop ----------

def run_gof_noisy(Z_full_word, p_word, dims, n_splits, k_per_word,
                  label_seed, cv_seed, skip_models=()):
    rows = []
    for d in dims:
        Z_word = Z_full_word[:, :d]
        # NOTE: same noisy labels across all (d, fold) -> apples-to-apples
        Z_rows, y_rows, p_oracle_rows, group_rows = make_noisy_dataset(
            Z_word, p_word, k_per_word, label_seed
        )

        gkf = GroupKFold(n_splits=n_splits)
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(Z_rows, y_rows, groups=group_rows)):
            Z_tr, Z_te = Z_rows[tr_idx], Z_rows[te_idx]
            y_tr, y_te = y_rows[tr_idx], y_rows[te_idx]
            p_te_oracle = p_oracle_rows[te_idx]

            models = make_models(d, skip=skip_models)
            for name, model in models.items():
                try:
                    p = fit_predict(model, Z_tr, y_tr, Z_te)
                    m = all_metrics(y_te, p, p_oracle=p_te_oracle)
                except Exception as e:
                    print(f"[skip] d={d} model={name} fold={fold}: {e}")
                    m = {k: np.nan for k in
                         ["auc", "logloss", "accuracy", "brier", "ece",
                          "mse_vs_oracle", "mae_vs_oracle"]}
                rows.append({"d": d, "model": name, "fold": fold, **m})
            print(f"  d={d:3d} fold={fold} done  "
                  f"(train={len(tr_idx)}, test={len(te_idx)})")
    return pd.DataFrame(rows)


# ---------- aggregation ----------

def aggregate(df_long):
    metrics = ["auc", "logloss", "accuracy", "brier", "ece",
               "mse_vs_oracle", "mae_vs_oracle"]
    agg = (
        df_long
        .groupby(["d", "model"])[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        c if isinstance(c, str) else "_".join([x for x in c if x])
        for c in agg.columns
    ]
    return agg


def paired_tests(df_long, baseline="logistic"):
    rows = []
    metrics = ["auc", "logloss", "accuracy", "brier", "ece",
               "mse_vs_oracle", "mae_vs_oracle"]
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

def plot_metrics_vs_dim(agg, out_path,
                        metrics=("auc", "logloss", "brier", "ece", "mse_vs_oracle")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.6 * n_metrics, 4.5))
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
        ax.legend(fontsize=7)
    fig.suptitle("GOF (noisy labels): metric vs PCA dimension", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration_vs_oracle(Z_full_word, p_word, dims, k_per_word,
                                label_seed, cv_seed, out_path, skip_models=()):
    """For each (d, model), pool CV-predicted P̂ and compare to oracle p directly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_dims = len(dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(4.2 * n_dims, 4.5), sharey=True)
    if n_dims == 1:
        axes = [axes]

    for ax, d in zip(axes, dims):
        Z_word = Z_full_word[:, :d]
        Z_rows, y_rows, p_oracle_rows, group_rows = make_noisy_dataset(
            Z_word, p_word, k_per_word, label_seed
        )
        gkf = GroupKFold(n_splits=5)
        models = make_models(d, skip=skip_models)
        for name, model in models.items():
            preds = np.full(len(y_rows), np.nan)
            try:
                for tr, te in gkf.split(Z_rows, y_rows, groups=group_rows):
                    p = fit_predict(model, Z_rows[tr], y_rows[tr], Z_rows[te])
                    preds[te] = p
            except Exception:
                continue
            mask = ~np.isnan(preds)
            if mask.sum() == 0:
                continue
            # scatter oracle p vs predicted p
            ax.scatter(p_oracle_rows[mask], preds[mask], alpha=0.05, s=4, label=name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("oracle P(y=1)")
        if d == dims[0]:
            ax.set_ylabel("predicted P̂(y=1)")
        ax.set_title(f"d = {d}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Calibration vs oracle (CV-pooled)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", type=str, required=True)
    parser.add_argument("--raw_csv", type=str, required=True)
    parser.add_argument("--dims", type=str, default="8,16,32,64,128")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="Sigmoid sensitivity for label generation")
    parser.add_argument("--k_per_word", type=int, default=20,
                        help="Bernoulli samples per word")
    parser.add_argument("--label_seed", type=int, default=42)
    parser.add_argument("--cv_seed", type=int, default=42)
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--skip_models", type=str, default="",
                        help="Comma-separated model names to skip (e.g. gp_rbf)")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(args.out_root,
                           f"gof_noisy{('_' + args.tag) if args.tag else ''}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # === Load + filter to valid words ===
    data = np.load(args.data_npz, allow_pickle=True)
    embs = data["embs"]
    dreams = data["dreams"]
    df_raw = pd.read_csv(args.raw_csv)
    words = df_raw["word"].tolist()

    if "valid" in data.files:
        valid = data["valid"]
    else:
        valid = ~np.isnan(dreams)
    print(f"Loaded {len(embs)} entries, valid={valid.sum()}")

    embs = embs[valid]
    dreams = dreams[valid]
    words = [w for w, v in zip(words, valid) if v]

    # === PCA ===
    dims = [int(x) for x in args.dims.split(",")]
    d_max = max(dims)
    mean_emb = embs.mean(0)
    pca = PCA(n_components=d_max, random_state=0).fit(embs - mean_emb)
    Z_full_word = (embs - mean_emb) @ pca.components_.T  # (N_words, d_max)
    print(f"PCA cumvar at d={d_max}: {pca.explained_variance_ratio_.sum():.3f}")

    # === Pick competitor B at dreamsim median ===
    median_d = float(np.median(dreams))
    b_idx = int(np.argmin(np.abs(dreams - median_d)))
    d_B = float(dreams[b_idx])
    print(f"B='{words[b_idx]}' d_B={d_B:.4f}")

    keep = np.arange(len(embs)) != b_idx
    Z_full_word = Z_full_word[keep]
    dreams = dreams[keep]
    words = [w for w, k_ in zip(words, keep) if k_]

    # === Generate per-word probabilities ===
    p_word = sigmoid(args.alpha * (d_B - dreams))
    print(f"\nLabel-prob stats:  min={p_word.min():.3f}  median={np.median(p_word):.3f}  "
          f"max={p_word.max():.3f}")
    print(f"Frac near 0/1 (|p-0.5|>0.4): {((np.abs(p_word - 0.5) > 0.4)).mean():.3f}")
    print(f"Effective N: {len(p_word)} words x {args.k_per_word} = "
          f"{len(p_word) * args.k_per_word} rows")

    # === Run GOF ===
    skip = tuple(s.strip() for s in args.skip_models.split(",") if s.strip())
    df_long = run_gof_noisy(Z_full_word, p_word, dims, args.n_splits,
                            args.k_per_word, args.label_seed, args.cv_seed,
                            skip_models=skip)
    df_long.to_csv(os.path.join(out_dir, "gof_per_fold.csv"), index=False)

    agg = aggregate(df_long)
    agg.to_csv(os.path.join(out_dir, "gof_summary.csv"), index=False)
    print("\n=== Aggregate ===")
    print(agg.pivot_table(index=["d", "model"],
                          values=["auc_mean", "brier_mean", "ece_mean",
                                  "mse_vs_oracle_mean"]).round(4))

    paired = paired_tests(df_long, baseline="logistic")
    paired.to_csv(os.path.join(out_dir, "paired_tests_vs_logistic.csv"), index=False)

    # === Plots ===
    plot_metrics_vs_dim(agg, os.path.join(out_dir, "metrics_vs_dim.png"))
    plot_calibration_vs_oracle(
        Z_full_word, p_word, dims, args.k_per_word,
        args.label_seed, args.cv_seed,
        os.path.join(out_dir, "calibration_vs_oracle.png"),
        skip_models=skip,
    )

    # Save oracle p for inspection
    np.savez(os.path.join(out_dir, "label_probs.npz"),
             words=np.array(words), dreams=dreams, p_word=p_word, d_B=d_B,
             alpha=args.alpha)

    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
