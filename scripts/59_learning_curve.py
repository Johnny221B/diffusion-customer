"""Learning curve: expected winning rate vs sample size N.

Setup
-----
- Pool: 227 strict single-token shoe-relevant words (B excluded)
- True winning prob:  p_i = sigmoid(alpha * (d_B - dreams_i))
- iid sampling WITH REPLACEMENT from the pool, one Bernoulli label per draw
- For each N: fit each model, score all 227 candidate words (PCA d), pick
  argmax_i p_hat_i, record TRUE p of that pick = "expected winning rate".
- Repeat across many seeds; plot learning curve.

Usage
-----
  python scripts/59_learning_curve.py \
      --data_npz outputs/strict_pool_s228_<stamp>/embeddings.npz \
      --raw_csv  outputs/strict_pool_s228_<stamp>/raw_data.csv \
      --alpha 30 --d 16 \
      --N_list 100,200,500,1000,2000,5000,10000 \
      --n_seeds 100 \
      --gp_max_N 2000 \
      --tag a30d16
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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
        n_estimators=200, random_state=0, n_jobs=-1,
    )
    for s in skip:
        models.pop(s, None)
    return models


def fit_predict_full(model, Z_tr, y_tr, Z_full):
    mu = Z_tr.mean(0)
    sd = Z_tr.std(0) + 1e-9
    Ztr = (Z_tr - mu) / sd
    Zfull = (Z_full - mu) / sd
    if len(set(y_tr)) < 2:
        # Degenerate: all labels same. Predict that constant for everyone.
        return np.full(len(Zfull), float(y_tr[0]))
    model.fit(Ztr, y_tr)
    return model.predict_proba(Zfull)[:, 1]


def evaluate_pick(p_hat, p_word, top_k=5):
    """Given predicted scores over 227 words, return metrics on the picked word."""
    order = np.argsort(-p_hat)  # descending
    top1 = int(order[0])
    topk = order[:top_k]
    return {
        "top1_idx":      top1,
        "top1_p":        float(p_word[top1]),
        "topk_mean_p":   float(p_word[topk].mean()),
        "regret_top1":   float(p_word.max() - p_word[top1]),
        "regret_topk":   float(p_word.max() - p_word[topk].mean()),
    }


def run_one_seed(Z_full_word, p_word, N, d, seed, models_to_run, gp_max_N):
    """One iid sample of size N (with replacement), fit each model, evaluate."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(Z_full_word), size=N)
    Z_tr = Z_full_word[idx][:, :d]
    p_tr = p_word[idx]
    y_tr = (rng.uniform(size=N) < p_tr).astype(int)
    Z_full = Z_full_word[:, :d]

    skip = ("gp_rbf",) if N > gp_max_N else ()
    models = make_models(d, skip=skip)
    if models_to_run is not None:
        models = {k: v for k, v in models.items() if k in models_to_run}

    out = []
    for name, model in models.items():
        try:
            p_hat = fit_predict_full(model, Z_tr, y_tr, Z_full)
        except Exception as e:
            out.append({
                "model": name, "N": N, "d": d, "seed": seed,
                "top1_idx": -1, "top1_p": np.nan, "topk_mean_p": np.nan,
                "regret_top1": np.nan, "regret_topk": np.nan,
                "error": str(e)[:200],
            })
            continue
        m = evaluate_pick(p_hat, p_word)
        out.append({"model": name, "N": N, "d": d, "seed": seed, **m})

    # === baselines (no model) ===
    # Random pick: one uniform draw from 227
    r_pick = int(rng.randint(0, len(p_word)))
    out.append({
        "model": "random", "N": N, "d": d, "seed": seed,
        "top1_idx": r_pick, "top1_p": float(p_word[r_pick]),
        "topk_mean_p": float(p_word.mean()),  # ev over uniform
        "regret_top1": float(p_word.max() - p_word[r_pick]),
        "regret_topk": float(p_word.max() - p_word.mean()),
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", type=str, required=True)
    parser.add_argument("--raw_csv",  type=str, required=True)
    parser.add_argument("--alpha", type=float, default=30.0)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--N_list", type=str, default="100,200,500,1000,2000,5000,10000")
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--gp_max_N", type=int, default=2000,
                        help="Skip gp_rbf when N > this (O(N^3) too slow)")
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(
        args.out_root, f"learncurve{('_' + args.tag) if args.tag else ''}_{stamp}"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # === Load + filter to valid words ===
    data = np.load(args.data_npz, allow_pickle=True)
    embs = data["embs"]
    dreams = data["dreams"]
    df_raw = pd.read_csv(args.raw_csv)
    words = df_raw["word"].tolist()
    valid = data["valid"] if "valid" in data.files else ~np.isnan(dreams)
    embs = embs[valid]
    dreams = dreams[valid]
    words = [w for w, v in zip(words, valid) if v]

    # === PCA up to args.d ===
    mean_emb = embs.mean(0)
    pca = PCA(n_components=args.d, random_state=0).fit(embs - mean_emb)
    Z_full_word = (embs - mean_emb) @ pca.components_.T  # (N_words, d)
    print(f"Loaded {len(embs)} words. PCA cumvar at d={args.d}: "
          f"{pca.explained_variance_ratio_.sum():.3f}")

    # === Pick competitor B at dreamsim median, drop B from pool ===
    median_d = float(np.median(dreams))
    b_idx = int(np.argmin(np.abs(dreams - median_d)))
    d_B = float(dreams[b_idx])
    print(f"B='{words[b_idx]}' d_B={d_B:.4f}")
    keep = np.arange(len(embs)) != b_idx
    Z_full_word = Z_full_word[keep]
    dreams = dreams[keep]
    words = [w for w, k_ in zip(words, keep) if k_]

    # === True winning probabilities ===
    p_word = sigmoid(args.alpha * (d_B - dreams))
    p_max = float(p_word.max())
    p_mean = float(p_word.mean())
    print(f"\nLabel-prob stats:  min={p_word.min():.3f}  median={np.median(p_word):.3f}  "
          f"max={p_max:.3f}  mean={p_mean:.3f}")
    print(f"Frac extreme (|p-0.5|>0.4): {((np.abs(p_word - 0.5) > 0.4)).mean():.3f}")
    print(f"Oracle ceiling (max_p) = {p_max:.4f}")
    print(f"Random baseline (mean_p) = {p_mean:.4f}\n")

    N_list = [int(x) for x in args.N_list.split(",")]
    print(f"N_list = {N_list},  n_seeds = {args.n_seeds},  d = {args.d},  alpha = {args.alpha}")
    print(f"GP skipped when N > {args.gp_max_N}\n")

    # === Main loop ===
    rows = []
    for N in N_list:
        for seed in range(args.n_seeds):
            recs = run_one_seed(
                Z_full_word, p_word, N, args.d, seed,
                models_to_run=None, gp_max_N=args.gp_max_N,
            )
            rows.extend(recs)
        # progress
        completed = [r for r in rows if r["N"] == N]
        per_model_top1 = (
            pd.DataFrame(completed).groupby("model")["top1_p"].mean().round(3)
        )
        print(f"[N={N:5d}] done  -> mean top1_p per model:")
        for m, v in per_model_top1.items():
            print(f"    {m:<16s} {v:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "learncurve_per_seed.csv"), index=False)

    # === Aggregate ===
    metrics = ["top1_p", "topk_mean_p", "regret_top1", "regret_topk"]
    agg = (
        df.groupby(["N", "model"])[metrics]
        .agg(["mean", "std", "count"]).reset_index()
    )
    agg.columns = [c if isinstance(c, str) else "_".join([x for x in c if x])
                   for c in agg.columns]
    agg["p_max"] = p_max
    agg["p_mean"] = p_mean
    agg.to_csv(os.path.join(out_dir, "learncurve_summary.csv"), index=False)

    # === Plot ===
    plot_learning_curves(agg, p_max, p_mean, out_dir, args)

    # === Save oracle ===
    np.savez(os.path.join(out_dir, "label_probs.npz"),
             words=np.array(words), dreams=dreams, p_word=p_word,
             d_B=d_B, alpha=args.alpha)
    print(f"\nSaved -> {out_dir}")


def plot_learning_curves(agg, p_max, p_mean, out_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ("top1_p",        "Expected winning rate (top-1)",        True),
        ("topk_mean_p",   "Expected winning rate (top-5 mean)",   True),
        ("regret_top1",   "Regret = max_p - p_picked (top-1)",    False),
    ]
    colors = {
        "logistic":       "tab:blue",
        "logistic_l2":    "tab:cyan",
        "poly2_logistic": "tab:purple",
        "gp_rbf":         "tab:green",
        "random_forest":  "tab:orange",
        "random":         "gray",
    }

    fig, axes = plt.subplots(1, len(metrics_to_plot),
                             figsize=(6 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    for ax, (metric, ylabel, draw_ceiling) in zip(axes, metrics_to_plot):
        for model in agg["model"].unique():
            sub = agg[agg["model"] == model].sort_values("N")
            mean = sub[f"{metric}_mean"].values
            std  = sub[f"{metric}_std"].values
            n    = sub[f"{metric}_count"].values
            se = std / np.sqrt(np.maximum(n, 1))
            mask = ~np.isnan(mean)
            ax.errorbar(sub["N"].values[mask], mean[mask], yerr=se[mask],
                        marker="o", capsize=3, label=model,
                        color=colors.get(model, "black"))
        if draw_ceiling:
            ax.axhline(p_max, color="k", linestyle=":",
                       alpha=0.5, label=f"oracle ceiling ({p_max:.3f})")
            ax.axhline(p_mean, color="k", linestyle="--",
                       alpha=0.4, label=f"uniform mean ({p_mean:.3f})")
        else:
            ax.axhline(0.0, color="k", linestyle=":", alpha=0.5,
                       label="zero regret")
        ax.set_xscale("log")
        ax.set_xlabel("N (iid samples)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Learning curve  (α={args.alpha}, d={args.d}, "
                 f"{args.n_seeds} seeds, with-replacement)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
