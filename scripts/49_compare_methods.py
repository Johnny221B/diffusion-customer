"""Compare three models on identical (Z, dreamsim, label) data:

(a) Logistic Thompson on binary labels  (current BO setup)
(b) Ridge regression on continuous -dreamsim
(c) Gaussian Process with RBF kernel on continuous -dreamsim

All use the same train/test splits and random seeds. Metric: Top-K precision,
Pearson(pred, -dreamsim), cos(linear_dir, z_dir_true) where applicable.

Usage:
  python scripts/49_compare_methods.py \
      --data_npz outputs/scaling_red_vs_green_d32_0416_1723/data.npz \
      --n_trials 10
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from src.thompson_optimizer import LogisticThompsonOptimizer


def topk_prec(pred_high_is_good, truth_low_is_good, k):
    pred_top = set(np.argsort(pred_high_is_good)[::-1][:k])
    true_top = set(np.argsort(truth_low_is_good)[:k])
    return len(pred_top & true_top) / k


def pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return stats.pearsonr(a, b)[0]


def eval_logistic(Z_tr, y_tr_bin, Z_te, d_te, true_dir, k):
    opt = LogisticThompsonOptimizer(dim_latent=Z_tr.shape[1], prior_var=3.0)
    for i in range(len(Z_tr)):
        opt.add_comparison_data(Z_tr[i], [int(y_tr_bin[i])])
    opt.update_posterior()
    learned = opt.mu_map[1:]
    intercept = opt.mu_map[0]
    pred = Z_te @ learned + intercept
    cos = np.dot(learned, true_dir) / (np.linalg.norm(learned) * np.linalg.norm(true_dir) + 1e-9)
    return {
        "cos": float(cos),
        "r": pearson(pred, -d_te),
        "topk": topk_prec(pred, d_te, k),
    }


def eval_ridge(Z_tr, d_tr, Z_te, d_te, true_dir, k, alpha=1.0):
    reg = Ridge(alpha=alpha).fit(Z_tr, -d_tr)
    pred = reg.predict(Z_te)
    learned = reg.coef_
    cos = np.dot(learned, true_dir) / (np.linalg.norm(learned) * np.linalg.norm(true_dir) + 1e-9)
    return {
        "cos": float(cos),
        "r": pearson(pred, -d_te),
        "topk": topk_prec(pred, d_te, k),
    }


def eval_gp(Z_tr, d_tr, Z_te, d_te, k):
    # Normalize inputs for stable kernel scale
    mu = Z_tr.mean(0)
    sd = Z_tr.std(0) + 1e-9
    Ztr = (Z_tr - mu) / sd
    Zte = (Z_te - mu) / sd
    ytr = -d_tr  # high = good
    ytr_mean = ytr.mean()
    ytr_c = ytr - ytr_mean

    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=2, random_state=0)
    gp.fit(Ztr, ytr_c)
    pred = gp.predict(Zte) + ytr_mean
    return {
        "r": pearson(pred, -d_te),
        "topk": topk_prec(pred, d_te, k),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--N_list", type=str, default="30,50,80,120,150")
    args = parser.parse_args()

    data = np.load(args.data_npz)
    Z = data["Z"]
    dreams = data["dreams"]
    labels = data["labels"]
    true_dir = data["z_dir_true"]

    N_total = len(Z)
    print(f"N_total={N_total}, dim_z={Z.shape[1]}, positives={labels.sum()}")

    N_list = [int(x) for x in args.N_list.split(",")]
    N_list = [n for n in N_list if 5 <= n < N_total]

    rng = np.random.RandomState(42)
    rows = []
    for N in N_list:
        buf = {"log": [], "rid": [], "gp_": []}
        for trial in range(args.n_trials):
            idx = rng.permutation(N_total)
            tr, te = idx[:N], idx[N:]
            Z_tr, Z_te = Z[tr], Z[te]
            d_tr, d_te = dreams[tr], dreams[te]
            y_tr = labels[tr]

            buf["log"].append(eval_logistic(Z_tr, y_tr, Z_te, d_te, true_dir, args.top_k))
            buf["rid"].append(eval_ridge(Z_tr, d_tr, Z_te, d_te, true_dir, args.top_k))
            buf["gp_"].append(eval_gp(Z_tr, d_tr, Z_te, d_te, args.top_k))

        for method, records in buf.items():
            rec = {
                "method": {"log": "logistic", "rid": "ridge", "gp_": "gp_rbf"}[method],
                "N": N,
                "topk_mean": np.mean([r["topk"] for r in records]),
                "topk_std":  np.std ([r["topk"] for r in records]),
                "r_mean":    np.mean([r["r"]    for r in records]),
                "r_std":     np.std ([r["r"]    for r in records]),
            }
            if "cos" in records[0]:
                rec["cos_mean"] = np.mean([r["cos"] for r in records])
                rec["cos_std"]  = np.std ([r["cos"] for r in records])
            rows.append(rec)
        print(f"N={N}: "
              f"log_top={rows[-3]['topk_mean']:.3f}  "
              f"ridge_top={rows[-2]['topk_mean']:.3f}  "
              f"gp_top={rows[-1]['topk_mean']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.data_npz)
    df.to_csv(os.path.join(out_dir, "compare_methods.csv"), index=False)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, color in [("logistic", "tab:blue"), ("ridge", "tab:orange"), ("gp_rbf", "tab:green")]:
        sub = df[df["method"] == method]
        axes[0].errorbar(sub["N"], sub["topk_mean"], yerr=sub["topk_std"],
                         marker="o", capsize=3, label=method, color=color)
        axes[1].errorbar(sub["N"], sub["r_mean"], yerr=sub["r_std"],
                         marker="o", capsize=3, label=method, color=color)

    axes[0].set_xlabel("N (iid training samples)")
    axes[0].set_ylabel(f"Top-{args.top_k} Precision")
    axes[0].set_title("Top-K precision on held-out")
    axes[0].axhline(args.top_k / N_total, color="gray", linestyle="--", alpha=0.5, label="random")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("N (iid training samples)")
    axes[1].set_ylabel("Pearson(pred, -dreamsim)")
    axes[1].set_title("Reward prediction correlation")
    axes[1].axhline(0, color="k", alpha=0.3)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    dim_tag = os.path.basename(out_dir)
    fig.suptitle(f"logistic vs ridge vs GP-RBF  ({dim_tag})", fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "compare_methods.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
