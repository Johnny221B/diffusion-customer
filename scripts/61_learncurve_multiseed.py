"""Learning curve with multi-seed image-gen ground truth.

Two-layer label model
---------------------
- Oracle (used for evaluation):
    p_i = (1/M) sum_{s_w=0..M-1} 1{ D_i^(s_w) < D_B }
  where D_B = dreams_matrix[B_idx, 0] is the FIXED B image (s_B = 0).

- Training samples (with image-gen noise + sigmoid user noise):
    For each iid draw:
        i  ~ Uniform{0..N_words-1}
        sw ~ Uniform{0..M-1}
        D_w = dreams_matrix[i, sw]
        y ~ Bernoulli(sigmoid(alpha * (D_B - D_w)))

Surrogate input is z_i (PCA of word embedding) -- deterministic per word.
Predicted argmax over the 227 candidate words; recorded "true winning rate"
is the oracle p_{i*}.

Usage:
  python scripts/61_learncurve_multiseed.py \
      --pool_npz outputs/strict_pool_s228_0429_0119/embeddings.npz \
      --raw_csv  outputs/strict_pool_s228_0429_0119/raw_data.csv \
      --dreams_npz outputs/multiseed_s228_M40_<stamp>/dreams_matrix.npz \
      --alpha 30 --d 16 \
      --N_list 100,200,500,1000,2000,5000,10000 \
      --n_seeds 100 \
      --gp_max_N 1000 \
      --tag multiseed_a30d16
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
        return np.full(len(Zfull), float(y_tr[0]))
    model.fit(Ztr, y_tr)
    return model.predict_proba(Zfull)[:, 1]


def evaluate_pick(p_hat, p_oracle, top_k=5):
    order = np.argsort(-p_hat)
    top1 = int(order[0])
    topk = order[:top_k]
    return {
        "top1_idx":       top1,
        "top1_p":         float(p_oracle[top1]),
        "topk_mean_p":    float(p_oracle[topk].mean()),
        "regret_top1":    float(p_oracle.max() - p_oracle[top1]),
        "regret_topk":    float(p_oracle.max() - p_oracle[topk].mean()),
    }


def run_one_seed(Z_full_word, dreams_matrix, B_idx, p_oracle,
                 N, d, alpha, seed, models_to_run, gp_max_N):
    """One full BO simulation:
       sample N (word, image_seed) iid, train each model, evaluate."""
    rng = np.random.RandomState(seed)
    N_words, M = dreams_matrix.shape

    idx = rng.randint(0, N_words, size=N)        # which word
    sw  = rng.randint(0, M,       size=N)        # which image seed for the candidate
    D_w = dreams_matrix[idx, sw]                 # candidate image dreamsim
    D_B = dreams_matrix[B_idx, 0]                # B fixed at s_B = 0

    p_train = sigmoid(alpha * (D_B - D_w))       # soft prob (with image-gen noise)
    y_tr = (rng.uniform(size=N) < p_train).astype(int)

    Z_tr   = Z_full_word[idx][:, :d]
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
        m = evaluate_pick(p_hat, p_oracle)
        out.append({"model": name, "N": N, "d": d, "seed": seed, **m})

    # Random baseline
    r_pick = int(rng.randint(0, N_words))
    out.append({
        "model": "random", "N": N, "d": d, "seed": seed,
        "top1_idx": r_pick, "top1_p": float(p_oracle[r_pick]),
        "topk_mean_p": float(p_oracle.mean()),
        "regret_top1": float(p_oracle.max() - p_oracle[r_pick]),
        "regret_topk": float(p_oracle.max() - p_oracle.mean()),
    })
    return out


def plot_learning_curves(agg, p_max, p_mean, out_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ("top1_p",      "Expected winning rate (top-1)",      True),
        ("topk_mean_p", "Expected winning rate (top-5 mean)", True),
        ("regret_top1", "Regret = max_p - p_picked (top-1)",  False),
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
        ax.set_xlabel("N (iid (word, image_seed) samples)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Learning curve (multi-seed gt)  α={args.alpha}, d={args.d}, "
                 f"M={args.M_used}, {args.n_seeds} seeds, B={args.B_word_used} s_B={args.B_seed}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_npz",  required=True)
    parser.add_argument("--raw_csv",   required=True)
    parser.add_argument("--dreams_npz", required=True,
                        help="merged dreams_matrix.npz produced after step 60")
    parser.add_argument("--alpha", type=float, default=30.0)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--N_list", type=str, default="100,200,500,1000,2000,5000,10000")
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--gp_max_N", type=int, default=1000)
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--B_word", type=str, default="",
                        help="explicit competitor word; if empty, auto-pick moon or median")
    parser.add_argument("--B_seed", type=int, default=0,
                        help="which image seed of B to use as the fixed competitor image")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(args.out_root,
                           f"learncurve_ms{('_' + args.tag) if args.tag else ''}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    # === Load pool (embeddings) ===
    pool = np.load(args.pool_npz, allow_pickle=True)
    embs = pool["embs"]
    df_raw = pd.read_csv(args.raw_csv)
    words = df_raw["word"].astype(str).tolist()

    # === Load multi-seed dreams matrix ===
    dm = np.load(args.dreams_npz, allow_pickle=True)
    dreams_matrix = dm["dreams"]    # (228, M)
    M = int(dreams_matrix.shape[1])
    args.M_used = M

    # Filter valid words
    valid = ~np.any(np.isnan(dreams_matrix), axis=1)
    embs    = embs[valid]
    dreams_matrix = dreams_matrix[valid]
    words   = [w for w, v in zip(words, valid) if v]
    print(f"Loaded {len(embs)} valid words, M={M} seeds each.")

    # === PCA ===
    mean_emb = embs.mean(0)
    pca = PCA(n_components=args.d, random_state=0).fit(embs - mean_emb)
    Z_full_word = (embs - mean_emb) @ pca.components_.T
    print(f"PCA cumvar at d={args.d}: {pca.explained_variance_ratio_.sum():.3f}")

    # === Pick B ===
    if args.B_word:
        if args.B_word not in words:
            raise ValueError(f"--B_word {args.B_word!r} not in pool")
        B_idx = words.index(args.B_word)
    elif "moon" in words:
        B_idx = words.index("moon")
    else:
        per_word_mean = dreams_matrix.mean(axis=1)
        median_d = float(np.median(per_word_mean))
        B_idx = int(np.argmin(np.abs(per_word_mean - median_d)))
    args.B_idx = B_idx
    if args.B_seed < 0 or args.B_seed >= M:
        raise ValueError(f"--B_seed {args.B_seed} out of [0, {M})")
    D_B = float(dreams_matrix[B_idx, args.B_seed])
    args.B_word_used = words[B_idx]
    print(f"B='{words[B_idx]}' (idx={B_idx})  s_B={args.B_seed}  D_B = {D_B:.4f}")

    # === Drop B from candidate pool (we never want to recommend B against itself) ===
    keep = np.arange(len(words)) != B_idx
    Z_full_word = Z_full_word[keep]
    dreams_matrix_words = dreams_matrix[keep]   # (227, M)
    words_kept = [w for w, k in zip(words, keep) if k]
    # Re-derive per-row map: B_idx is now -1 (not in pool). We still need D_B as scalar.

    # === ORACLE p_i (hard fraction) ===
    # p_i = (1/M) sum_{s_w} 1{ D_i^(s_w) < D_B }
    p_oracle = (dreams_matrix_words < D_B).mean(axis=1).astype(np.float64)
    p_max  = float(p_oracle.max())
    p_mean = float(p_oracle.mean())
    top5_words = [(words_kept[i], float(p_oracle[i])) for i in np.argsort(-p_oracle)[:5]]
    print(f"\nOracle p stats: min={p_oracle.min():.3f}  median={np.median(p_oracle):.3f}  "
          f"max={p_max:.3f}  mean={p_mean:.3f}")
    print(f"Frac extreme (|p-0.5|>0.4): {((np.abs(p_oracle - 0.5) > 0.4)).mean():.3f}")
    print(f"Top-5 words by oracle p: {top5_words}")

    # Save config + oracle
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    np.savez(os.path.join(out_dir, "oracle.npz"),
             words=np.array(words_kept), p_oracle=p_oracle, D_B=D_B,
             B_word=words[B_idx], M=M)

    N_list = [int(x) for x in args.N_list.split(",")]
    print(f"\nN_list = {N_list},  n_seeds = {args.n_seeds},  d = {args.d},  alpha = {args.alpha}")
    print(f"GP skipped when N > {args.gp_max_N}\n")

    # === Build per-row dreams matrix, but B_idx is in original index space ===
    # Easier: store full dreams_matrix and map idx via keep.
    # In run_one_seed we sample idx in [0..len(words_kept)-1] and look up dreams_matrix_words[idx, sw].
    # B is fixed scalar D_B; no row indexing needed for B in the sampler.
    rows = []
    for N in N_list:
        for seed in range(args.n_seeds):
            recs = run_one_seed_v2(
                Z_full_word, dreams_matrix_words, D_B, p_oracle,
                N, args.d, args.alpha, seed, None, args.gp_max_N,
            )
            rows.extend(recs)
        sub = pd.DataFrame([r for r in rows if r["N"] == N])
        per_model = sub.groupby("model")["top1_p"].mean().round(3)
        print(f"[N={N:5d}] done  -> mean top1_p per model:")
        for m, v in per_model.items():
            print(f"    {m:<16s} {v:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "learncurve_per_seed.csv"), index=False)

    metrics = ["top1_p", "topk_mean_p", "regret_top1", "regret_topk"]
    agg = (
        df.groupby(["N", "model"])[metrics]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    agg.columns = [c if isinstance(c, str) else "_".join([x for x in c if x])
                   for c in agg.columns]
    agg["p_max"] = p_max
    agg["p_mean"] = p_mean
    agg.to_csv(os.path.join(out_dir, "learncurve_summary.csv"), index=False)

    plot_learning_curves(agg, p_max, p_mean, out_dir, args)
    print(f"\nSaved -> {out_dir}")


def run_one_seed_v2(Z_full_word, dreams_matrix_words, D_B, p_oracle,
                    N, d, alpha, seed, models_to_run, gp_max_N):
    rng = np.random.RandomState(seed)
    N_words, M = dreams_matrix_words.shape

    idx = rng.randint(0, N_words, size=N)
    sw  = rng.randint(0, M,       size=N)
    D_w = dreams_matrix_words[idx, sw]

    p_train = sigmoid(alpha * (D_B - D_w))
    y_tr = (rng.uniform(size=N) < p_train).astype(int)

    Z_tr   = Z_full_word[idx][:, :d]
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
        m = evaluate_pick(p_hat, p_oracle)
        out.append({"model": name, "N": N, "d": d, "seed": seed, **m})

    r_pick = int(rng.randint(0, N_words))
    out.append({
        "model": "random", "N": N, "d": d, "seed": seed,
        "top1_idx": r_pick, "top1_p": float(p_oracle[r_pick]),
        "topk_mean_p": float(p_oracle.mean()),
        "regret_top1": float(p_oracle.max() - p_oracle[r_pick]),
        "regret_topk": float(p_oracle.max() - p_oracle.mean()),
    })
    return out


if __name__ == "__main__":
    main()
