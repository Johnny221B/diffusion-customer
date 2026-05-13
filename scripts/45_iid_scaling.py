"""
Learning curve: how does Bayesian update performance scale with number of iid samples?

Loads pre-collected data from script 44's output (data.npz), then for each N in
[10, 20, 30, 50, 80, 120, 150, all]:
- Randomly subsample N words
- Train fresh LogisticThompsonOptimizer on these
- Evaluate on held-out words (or all words)
- Repeat M trials per N to get error bars

Metrics:
- Cosine similarity between learned direction and true direction
- Pearson correlation between predicted reward and -dreamsim
- Top-K precision (fraction of top-K predictions that are actually in true top-K)
"""

import os
import argparse
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split

from src.thompson_optimizer import LogisticThompsonOptimizer


def evaluate(opt, Z_eval, dreams_eval, probs_eval, true_dir, top_k=10):
    learned_dir = opt.mu_map[1:]
    intercept = opt.mu_map[0]
    pred_reward = Z_eval @ learned_dir + intercept

    # Cosine with true direction
    cos = np.dot(learned_dir, true_dir) / (
        np.linalg.norm(learned_dir) * np.linalg.norm(true_dir) + 1e-9)

    # Pearson correlation with -dreamsim (higher reward = lower dist)
    if np.std(pred_reward) > 0 and np.std(dreams_eval) > 0:
        r_pred, _ = stats.pearsonr(pred_reward, -dreams_eval)
    else:
        r_pred = 0.0

    # Top-K precision
    pred_topk = set(np.argsort(pred_reward)[::-1][:top_k])
    true_topk = set(np.argsort(dreams_eval)[:top_k])  # smallest dreamsim
    topk_prec = len(pred_topk & true_topk) / top_k

    return {"cos": cos, "r_pred": r_pred, "topk_prec": topk_prec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", type=str, required=True, help="Path to data.npz from script 44")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    data = np.load(args.data_npz)
    Z = data["Z"]
    dreams = data["dreams"]
    labels = data["labels"]
    probs = data["probs"]
    z_dir_true = data["z_dir_true"]

    N_total = len(Z)
    dim_z = Z.shape[1]
    print(f"Loaded {N_total} samples, dim_z={dim_z}")
    print(f"Positives: {labels.sum()}/{N_total}")
    print(f"True direction norm: {np.linalg.norm(z_dir_true):.4f}")

    out_dir = os.path.dirname(args.data_npz)

    # N values to try
    N_list = [5, 10, 20, 30, 50, 80, 120, min(150, N_total - 10), N_total - 10]
    N_list = sorted(set([n for n in N_list if 5 <= n < N_total]))
    print(f"Testing N values: {N_list}")

    rng = np.random.RandomState(42)
    results = []

    for N in N_list:
        cos_list, rpred_list, topk_list = [], [], []
        for trial in range(args.n_trials):
            indices = rng.permutation(N_total)
            train_idx = indices[:N]
            test_idx = indices[N:]

            opt = LogisticThompsonOptimizer(dim_latent=dim_z, prior_var=3.0)
            for i in train_idx:
                opt.add_comparison_data(Z[i], [int(labels[i])])
            opt.update_posterior()

            metrics = evaluate(opt, Z[test_idx], dreams[test_idx], probs[test_idx],
                              z_dir_true, top_k=args.top_k)
            cos_list.append(metrics["cos"])
            rpred_list.append(metrics["r_pred"])
            topk_list.append(metrics["topk_prec"])

        results.append({
            "N": N,
            "cos_mean": np.mean(cos_list), "cos_std": np.std(cos_list),
            "rpred_mean": np.mean(rpred_list), "rpred_std": np.std(rpred_list),
            "topk_mean": np.mean(topk_list), "topk_std": np.std(topk_list),
        })
        print(f"N={N:4d}: cos={np.mean(cos_list):.4f}±{np.std(cos_list):.4f}, "
              f"r_pred={np.mean(rpred_list):.4f}±{np.std(rpred_list):.4f}, "
              f"top{args.top_k}={np.mean(topk_list):.4f}±{np.std(topk_list):.4f}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "iid_scaling.csv"), index=False)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.errorbar(df["N"], df["cos_mean"], yerr=df["cos_std"], marker='o', capsize=3)
    ax.set_xlabel("Number of iid samples")
    ax.set_ylabel("Cos(learned_dir, true_dir)")
    ax.set_title("Direction Alignment")
    ax.set_xscale("log")
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(df["N"], df["rpred_mean"], yerr=df["rpred_std"], marker='o', capsize=3, color='green')
    ax.set_xlabel("Number of iid samples")
    ax.set_ylabel("Pearson(pred_reward, -dreamsim)")
    ax.set_title("Reward Prediction Correlation")
    ax.set_xscale("log")
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.errorbar(df["N"], df["topk_mean"], yerr=df["topk_std"], marker='o', capsize=3, color='red')
    ax.set_xlabel("Number of iid samples")
    ax.set_ylabel(f"Top-{args.top_k} Precision")
    ax.set_title(f"Top-{args.top_k} Precision (held-out)")
    ax.set_xscale("log")
    ax.axhline(y=args.top_k / N_total, color='gray', linestyle='--', alpha=0.5, label='random baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Bayesian Update: Learning Curve (dim_z={dim_z}, {args.n_trials} trials)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "iid_scaling.png"), dpi=150)
    print(f"\nResults saved to {out_dir}/iid_scaling.png")


if __name__ == "__main__":
    main()
