"""Under the real-pipeline constraint (binary label, fixed competitor B):
pick B so that d_B is the MEDIAN of word-dreamsim-to-ref, then compare
logistic classifier vs GP classifier on the same binary labels.

Uses existing data.npz (no regeneration).

Usage:
  python scripts/50_classifier_binary.py \
      --data_npz outputs/scaling_red_vs_green_d32_0416_1723/data.npz \
      --raw_csv  outputs/scaling_red_vs_green_d32_0416_1723/raw_data.csv \
      --n_trials 10
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import roc_auc_score


def topk_prec(pred_score, truth_low_is_good, k):
    pred_top = set(np.argsort(pred_score)[::-1][:k])
    true_top = set(np.argsort(truth_low_is_good)[:k])
    return len(pred_top & true_top) / k


def eval_logistic(Z_tr, y_tr, Z_te, d_te, k):
    if len(set(y_tr)) < 2:
        p = np.full(len(Z_te), y_tr[0])
        auc = 0.5
    else:
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Z_tr, y_tr)
        p = clf.predict_proba(Z_te)[:, 1]
    y_te = (d_te < np.median(d_te)).astype(int)  # used only for local auc check
    return {"auc_local": 0.5, "topk": topk_prec(p, d_te, k), "pred": p}


def eval_gp_clf(Z_tr, y_tr, Z_te, k):
    if len(set(y_tr)) < 2:
        p = np.full(len(Z_te), y_tr[0])
    else:
        mu = Z_tr.mean(0); sd = Z_tr.std(0) + 1e-9
        Ztr = (Z_tr - mu) / sd; Zte = (Z_te - mu) / sd
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=2, random_state=0)
        gp.fit(Ztr, y_tr)
        p = gp.predict_proba(Zte)[:, 1]
    return {"pred": p}


def run_one(Z, dreams, labels_binary, k, n_trials, N_list, rng):
    rows = []
    for N in N_list:
        log_auc = []; log_top = []
        gp_auc = []; gp_top = []
        for t in range(n_trials):
            idx = rng.permutation(len(Z))
            tr, te = idx[:N], idx[N:]
            Z_tr, Z_te = Z[tr], Z[te]
            y_tr, y_te = labels_binary[tr], labels_binary[te]
            d_te = dreams[te]

            r1 = eval_logistic(Z_tr, y_tr, Z_te, d_te, k)
            r2 = eval_gp_clf(Z_tr, y_tr, Z_te, k)

            if len(set(y_te)) == 2:
                log_auc.append(roc_auc_score(y_te, r1["pred"]))
                gp_auc.append(roc_auc_score(y_te, r2["pred"]))
            log_top.append(topk_prec(r1["pred"], d_te, k))
            gp_top.append(topk_prec(r2["pred"], d_te, k))

        rows.append({
            "N": N,
            "log_auc_mean": np.mean(log_auc) if log_auc else np.nan,
            "log_auc_std":  np.std (log_auc) if log_auc else np.nan,
            "gp_auc_mean":  np.mean(gp_auc)  if gp_auc else np.nan,
            "gp_auc_std":   np.std (gp_auc)  if gp_auc else np.nan,
            "log_topk_mean": np.mean(log_top), "log_topk_std": np.std(log_top),
            "gp_topk_mean":  np.mean(gp_top),  "gp_topk_std":  np.std(gp_top),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", type=str, required=True)
    parser.add_argument("--raw_csv", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--N_list", type=str, default="30,50,80,120,150")
    args = parser.parse_args()

    data = np.load(args.data_npz)
    Z = data["Z"]
    dreams = data["dreams"]
    df_raw = pd.read_csv(args.raw_csv)
    words = df_raw["word"].tolist()

    # Pick B = word whose dreamsim is closest to median
    median_d = np.median(dreams)
    b_idx = int(np.argmin(np.abs(dreams - median_d)))
    b_word = words[b_idx]
    d_B = dreams[b_idx]
    print(f"Median dreamsim = {median_d:.4f}")
    print(f"B_competitor = '{b_word}' (idx={b_idx}, d_B={d_B:.4f})")

    # Binary labels: y = 1 if word beats B (closer to ref)
    y = (dreams < d_B).astype(int)
    # Drop B itself from dataset (it IS the competitor)
    keep = np.arange(len(Z)) != b_idx
    Z, dreams, y = Z[keep], dreams[keep], y[keep]
    print(f"After dropping B: {len(Z)} words, positives={y.sum()}/{len(y)} ({y.mean():.2f})")

    N_list = [int(x) for x in args.N_list.split(",")]
    N_list = [n for n in N_list if 5 <= n < len(Z)]

    rng = np.random.RandomState(42)
    df = run_one(Z, dreams, y, args.top_k, args.n_trials, N_list, rng)

    out_dir = os.path.dirname(args.data_npz)
    df.to_csv(os.path.join(out_dir, "classifier_binary.csv"), index=False)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(df["N"], df["log_auc_mean"], yerr=df["log_auc_std"], marker="o", capsize=3, label="logistic", color="tab:blue")
    axes[0].errorbar(df["N"], df["gp_auc_mean"], yerr=df["gp_auc_std"], marker="o", capsize=3, label="GP classifier", color="tab:green")
    axes[0].axhline(0.5, color="k", alpha=0.3, label="random")
    axes[0].set_xlabel("N (train size)"); axes[0].set_ylabel("AUC")
    axes[0].set_title(f"Held-out AUC  (B='{b_word}', pos={y.mean():.2f})")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(df["N"], df["log_topk_mean"], yerr=df["log_topk_std"], marker="o", capsize=3, label="logistic", color="tab:blue")
    axes[1].errorbar(df["N"], df["gp_topk_mean"], yerr=df["gp_topk_std"], marker="o", capsize=3, label="GP classifier", color="tab:green")
    axes[1].axhline(args.top_k / len(Z), color="gray", linestyle="--", alpha=0.5, label="random")
    axes[1].set_xlabel("N (train size)"); axes[1].set_ylabel(f"Top-{args.top_k} precision")
    axes[1].set_title(f"Top-{args.top_k} Precision (held-out)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    dim_tag = os.path.basename(out_dir)
    fig.suptitle(f"Binary-label classifiers  ({dim_tag}, B median)", fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "classifier_binary.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
