"""Sequential active BO vs iid baseline.

Same 172-word data, d=16 PCA, B=median word, binary labels.
At each step t, agent picks the next word to query from the un-queried pool.

Three strategies:
  (a) random      — iid baseline
  (b) ucb_log     — logistic + UCB on p(y=1)
  (c) ucb_gp      — GP classifier + UCB

Metrics tracked along t:
  - AUC on still-held-out pool (how well model predicts labels)
  - Top-10 precision on held-out
  - Best-so-far dreamsim among queried (regret)
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


def fit_logistic(Z, y):
    if len(set(y)) < 2:
        return None
    return LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)


def fit_gp(Z, y):
    if len(set(y)) < 2:
        return None
    mu = Z.mean(0); sd = Z.std(0) + 1e-9
    Zn = (Z - mu) / sd
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=2, random_state=0)
    gp.fit(Zn, y)
    return (gp, mu, sd)


def pred_logistic(model, Z):
    if model is None:
        return np.full(len(Z), 0.5)
    return model.predict_proba(Z)[:, 1]


def pred_gp(model_tup, Z):
    if model_tup is None:
        return np.full(len(Z), 0.5)
    gp, mu, sd = model_tup
    Zn = (Z - mu) / sd
    return gp.predict_proba(Zn)[:, 1]


def run_trial(Z, y, dreams, strategy, T, warmup, rng, k=10):
    """Run one sequential trial. Returns per-step metrics."""
    N = len(Z)
    queried = np.array([False] * N)
    order = []

    # Warmup: random
    warmup_ix = rng.permutation(N)[:warmup]
    for i in warmup_ix:
        queried[i] = True
        order.append(i)

    rows = []
    for t in range(warmup, T + 1):
        q_idx = np.where(queried)[0]
        h_idx = np.where(~queried)[0]

        # Fit model on queried so far
        if strategy == "random":
            model = None
        elif strategy == "ucb_log":
            model = fit_logistic(Z[q_idx], y[q_idx])
        elif strategy == "ucb_gp":
            model = fit_gp(Z[q_idx], y[q_idx])
        else:
            raise ValueError(strategy)

        # Predict on held-out
        if strategy == "ucb_log":
            p_held = pred_logistic(model, Z[h_idx])
        elif strategy == "ucb_gp":
            p_held = pred_gp(model, Z[h_idx])
        else:
            p_held = None

        # Metrics on held-out
        if p_held is not None and len(set(y[h_idx])) == 2:
            auc = roc_auc_score(y[h_idx], p_held)
            topk = topk_prec(p_held, dreams[h_idx], k)
        else:
            auc = np.nan
            topk = np.nan

        best_d = dreams[q_idx].min()
        pos_rate = y[q_idx].mean()

        rows.append({
            "t": t, "auc_held": auc, "topk_held": topk,
            "best_d": best_d, "pos_rate": pos_rate,
        })

        # Decide next query
        if t == T:
            break
        if strategy == "random":
            next_i = h_idx[rng.randint(len(h_idx))]
        elif strategy == "ucb_log":
            # UCB: logistic prob + exploration via uncertainty (use distance-to-nearest-queried as proxy)
            p = p_held if p_held is not None else np.full(len(h_idx), 0.5)
            dist_to_queried = np.min(
                np.linalg.norm(Z[h_idx][:, None] - Z[q_idx][None, :], axis=-1), axis=1)
            score = p + 0.3 * (dist_to_queried / dist_to_queried.max())
            next_i = h_idx[np.argmax(score)]
        elif strategy == "ucb_gp":
            # GP inherent uncertainty via prob near 0.5
            p = p_held
            uncert = 1 - np.abs(p - 0.5) * 2  # high near 0.5
            score = p + 0.3 * uncert
            next_i = h_idx[np.argmax(score)]
        queried[next_i] = True
        order.append(next_i)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", type=str, required=True)
    parser.add_argument("--raw_csv", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=8)
    parser.add_argument("--T", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    data = np.load(args.data_npz)
    Z = data["Z"]
    dreams = data["dreams"]
    df_raw = pd.read_csv(args.raw_csv)
    words = df_raw["word"].tolist()

    median_d = np.median(dreams)
    b_idx = int(np.argmin(np.abs(dreams - median_d)))
    b_word = words[b_idx]
    d_B = dreams[b_idx]
    print(f"B='{b_word}' d_B={d_B:.4f} (median)")

    y_all = (dreams < d_B).astype(int)
    keep = np.arange(len(Z)) != b_idx
    Z, dreams, y_all = Z[keep], dreams[keep], y_all[keep]
    print(f"N={len(Z)}, positives={y_all.sum()}")

    strategies = ["random", "ucb_log", "ucb_gp"]
    all_rows = []
    for s in strategies:
        print(f"\n=== {s} ===")
        for trial in range(args.n_trials):
            rng = np.random.RandomState(100 + trial)
            df = run_trial(Z, y_all, dreams, s, args.T, args.warmup, rng, k=args.top_k)
            df["strategy"] = s
            df["trial"] = trial
            all_rows.append(df)
        print(f"  done {args.n_trials} trials")

    df_all = pd.concat(all_rows, ignore_index=True)
    out_dir = os.path.dirname(args.data_npz)
    df_all.to_csv(os.path.join(out_dir, "active_vs_iid_raw.csv"), index=False)

    # Aggregate: mean ± std over trials at each t
    agg = df_all.groupby(["strategy", "t"]).agg(
        auc_mean=("auc_held", "mean"), auc_std=("auc_held", "std"),
        topk_mean=("topk_held", "mean"), topk_std=("topk_held", "std"),
        bestd_mean=("best_d", "mean"), bestd_std=("best_d", "std"),
        pos_mean=("pos_rate", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(out_dir, "active_vs_iid_agg.csv"), index=False)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"random": "gray", "ucb_log": "tab:blue", "ucb_gp": "tab:green"}
    for s in strategies:
        sub = agg[agg["strategy"] == s]
        axes[0].plot(sub["t"], sub["auc_mean"], label=s, color=colors[s])
        axes[0].fill_between(sub["t"], sub["auc_mean"] - sub["auc_std"], sub["auc_mean"] + sub["auc_std"], alpha=0.2, color=colors[s])
        axes[1].plot(sub["t"], sub["topk_mean"], label=s, color=colors[s])
        axes[1].fill_between(sub["t"], sub["topk_mean"] - sub["topk_std"], sub["topk_mean"] + sub["topk_std"], alpha=0.2, color=colors[s])
        axes[2].plot(sub["t"], sub["bestd_mean"], label=s, color=colors[s])
        axes[2].fill_between(sub["t"], sub["bestd_mean"] - sub["bestd_std"], sub["bestd_mean"] + sub["bestd_std"], alpha=0.2, color=colors[s])

    axes[0].set_xlabel("t (queries)"); axes[0].set_ylabel("AUC held-out")
    axes[0].axhline(0.5, color="k", alpha=0.3); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Held-out AUC over time")

    axes[1].set_xlabel("t (queries)"); axes[1].set_ylabel(f"Top-{args.top_k} Precision")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f"Held-out Top-{args.top_k}")

    axes[2].set_xlabel("t (queries)"); axes[2].set_ylabel("Best dreamsim so far (lower=better)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Best word found (regret-style)")

    dim_tag = os.path.basename(out_dir)
    fig.suptitle(f"Active vs iid  ({dim_tag}, B='{b_word}')", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "active_vs_iid.png"), dpi=150)
    print(f"\nSaved {out_dir}/active_vs_iid.png")


if __name__ == "__main__":
    main()
