"""Exp: choose (d, alpha) for the discrete-bandit setting.

Two coupled knobs:
  - alpha  (label sharpness): y_{w,k} ~ Bern(sigmoid(alpha*(D_B - D_{w,k}))).
            Too large -> per-word soft oracle p_w saturates to {0,1} (bimodal cliff,
            BO can't rank the middle). Too small -> p_w all pile at 0.5 (labels are
            coin flips, weak signal). We want p_w spread evenly over [0,1].
  - d      (PCA dim of z): only enters the SURROGATE, not the oracle. Question is
            whether logit(p_w) is linear in z_d, i.e. whether sigma(beta^T z) holds.

Headline metric for alpha: histogram-entropy of the per-word soft oracle p_w.
  Peaks when p_w ~ Uniform[0,1]; low for both the "all 0.5" and the "all 0/1" regimes.
Headline metric for d: cross-validated R^2 of  logit(p_w) ~ linear(z_d).
  High R^2 == the logistic-linear surrogate assumption is faithful at that (d, alpha).

Pure CPU, no SD3.5, no regeneration. Reuses the precomputed dreams_matrix.

Usage:
  python scripts/66_sensitivity_dim_diagnostic.py \
      --pool_npz   outputs/strict_pool_s228_0429_0119/embeddings.npz \
      --raw_csv    outputs/strict_pool_s228_0429_0119/raw_data.csv \
      --dreams_npz outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz \
      --B_word canvas --B_seed 34
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def soft_oracle(dreams, D_B, alpha):
    """p_w = mean_k sigmoid(alpha*(D_B - D_{w,k})), shape (n_words,)."""
    return sigmoid(alpha * (D_B - dreams)).mean(axis=1)


def hist_entropy(p, bins=10):
    """Normalized Shannon entropy of p's histogram on [0,1]. 1.0 == uniform."""
    h, _ = np.histogram(p, bins=bins, range=(0.0, 1.0))
    q = h / h.sum()
    q = q[q > 0]
    return float(-(q * np.log(q)).sum() / np.log(bins))


def ks_to_uniform(p):
    """Kolmogorov-Smirnov distance of p to Uniform[0,1] (smaller == more uniform)."""
    x = np.sort(p)
    n = len(x)
    cdf_emp = np.arange(1, n + 1) / n
    return float(np.max(np.abs(cdf_emp - x)))


def cv_linear_r2(z, y, n_splits=5, seed=0):
    """5-fold CV R^2 of OLS  y ~ linear(z).  Also returns in-sample R^2."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(z):
        reg = LinearRegression().fit(z[tr], y[tr])
        pred = reg.predict(z[te])
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[tr].mean()) ** 2)  # baseline = train mean
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
    reg_full = LinearRegression().fit(z, y)
    r2_in = reg_full.score(z, y)
    return float(np.mean(r2s)), float(r2_in)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_npz",   default="outputs/strict_pool_s228_0429_0119/embeddings.npz")
    ap.add_argument("--raw_csv",    default="outputs/strict_pool_s228_0429_0119/raw_data.csv")
    ap.add_argument("--dreams_npz", default="outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz")
    ap.add_argument("--B_word", default="canvas")
    ap.add_argument("--B_seed", type=int, default=34)
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--tag", default="bcanvas")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(args.out_root, f"sens_dim_diag_{args.tag}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load ----
    pool = np.load(args.pool_npz, allow_pickle=True)
    embs = pool["embs"].astype(np.float64)
    dm = np.load(args.dreams_npz, allow_pickle=True)
    dreams = dm["dreams"].astype(np.float64)
    words = pd.read_csv(args.raw_csv)["word"].astype(str).tolist()

    valid = ~np.any(np.isnan(dreams), axis=1)
    embs, dreams = embs[valid], dreams[valid]
    words = [w for w, v in zip(words, valid) if v]
    M = dreams.shape[1]

    B_idx = words.index(args.B_word)
    D_B = float(dreams[B_idx, args.B_seed])

    # candidate pool: drop the competitor word (matches script 63)
    keep = np.arange(len(words)) != B_idx
    embs_k, dreams_k = embs[keep], dreams[keep]
    n = len(embs_k)
    print(f"n_candidates={n}, M={M}, D_B={D_B:.4f}")
    print(f"D: median={np.median(dreams_k):.4f} std={dreams_k.std():.4f} "
          f"across-word std={dreams_k.mean(1).std():.4f} within-word std={dreams_k.std(1).mean():.4f}")

    # ---- PCA once at max d, slice for smaller d (components are ordered) ----
    d_grid = [2, 4, 8, 12, 16, 24, 32, 48, 64, 128]
    d_grid = [d for d in d_grid if d <= n - 2]
    max_d = max(d_grid)
    mean_emb = embs_k.mean(0)
    pca = PCA(n_components=max_d, random_state=0).fit(embs_k - mean_emb)
    Z_full = (embs_k - mean_emb) @ pca.components_.T  # (n, max_d)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # ---- alpha sweep: p_w distribution ----
    alpha_grid = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 50]
    rows = []
    pw_by_alpha = {}
    for a in alpha_grid:
        pw = soft_oracle(dreams_k, D_B, a)
        pw_by_alpha[a] = pw
        rows.append({
            "alpha": a,
            "entropy": hist_entropy(pw),
            "ks_to_unif": ks_to_uniform(pw),
            "std": float(pw.std()),
            "mean": float(pw.mean()),
            "frac_in_band_0.2_0.8": float(((pw > 0.2) & (pw < 0.8)).mean()),
            "frac_saturated": float(((pw < 0.05) | (pw > 0.95)).mean()),
        })
    df_a = pd.DataFrame(rows)
    df_a.to_csv(os.path.join(out_dir, "alpha_sweep.csv"), index=False)

    # ---- principled alpha: match the label sigmoid's slope to the distance scale ----
    # We want a "1-std-better image" (Delta D = std_D) to be preferred with prob p_target,
    # i.e. sigmoid(alpha * std_D) = p_target  ->  alpha = logit(p_target) / std_D.
    # This makes the annotator graded across the distances that actually occur,
    # instead of a near-deterministic step (which is the alpha=30 symptom).
    std_D = float(dreams_k.std())
    p_target = 0.75
    alpha_matched = float(np.log(p_target / (1 - p_target)) / std_D)
    best_alpha = min(alpha_grid, key=lambda a: abs(a - alpha_matched))
    print("\n=== alpha sweep ===")
    print(df_a.to_string(index=False))
    print(f"  distance std = {std_D:.4f}")
    print(f"  alpha matched to scale (1-std-better preferred at p={p_target}) = {alpha_matched:.1f}")
    print(f"  -> nearest grid alpha = {best_alpha}")
    print(f"  (NOTE: hist-entropy keeps rising with alpha because p_w drifts toward the")
    print(f"   hard-oracle limit, never bimodal -> entropy is NOT the right objective here.)")

    # ---- (d, alpha) grid: linearity of logit(p_w) in z_d ----
    eps = 1e-3
    r2_cv = np.full((len(d_grid), len(alpha_grid)), np.nan)
    r2_in = np.full((len(d_grid), len(alpha_grid)), np.nan)
    for j, a in enumerate(alpha_grid):
        pw = np.clip(pw_by_alpha[a], eps, 1 - eps)
        y = np.log(pw / (1 - pw))  # logit
        for i, d in enumerate(d_grid):
            Zd = Z_full[:, :d]
            r2_cv[i, j], r2_in[i, j] = cv_linear_r2(Zd, y)
    np.savez(os.path.join(out_dir, "dim_alpha_r2.npz"),
             d_grid=np.array(d_grid), alpha_grid=np.array(alpha_grid),
             r2_cv=r2_cv, r2_in=r2_in, cumvar=cumvar)

    # ============================ PLOTS ============================
    # 0) THE label sigmoid over the real distance distribution (the user's concern)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.linspace(dreams_k.min(), dreams_k.max(), 400)
    ax2 = ax.twinx()
    ax2.hist(dreams_k.ravel(), bins=60, color="lightgray", alpha=0.7, density=True)
    ax2.set_ylabel("distance density (gray)", color="gray")
    for a in [5, 10, 12, 20, 30, 50]:
        ys = sigmoid(a * (D_B - xs))
        lw = 3 if a == best_alpha else 1.5
        ax.plot(xs, ys, lw=lw,
                label=f"alpha={a}" + ("  <- matched" if a == best_alpha else ""))
    ax.axvline(D_B, color="k", ls="--", alpha=0.7, label=f"D_B={D_B:.3f}")
    ax.set_xlabel("DreamSim distance D (lower = closer to reference R)")
    ax.set_ylabel("P(prefer this image over B) = sigmoid(alpha*(D_B - D))")
    ax.set_title("Label sigmoid vs distance: alpha=30 is a near-step within the distance bulk")
    ax.legend(fontsize=9, loc="center left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "label_sigmoid_vs_distance.png"), dpi=140)
    plt.close(fig)

    # 1) p_w histograms across alpha (small multiples)
    ncol = 4
    nrow = int(np.ceil(len(alpha_grid) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.array(axes).reshape(-1)
    for k, a in enumerate(alpha_grid):
        ax = axes[k]
        ax.hist(pw_by_alpha[a], bins=20, range=(0, 1), color="tab:blue",
                edgecolor="white", alpha=0.85)
        ax.set_title(f"alpha={a}  H={hist_entropy(pw_by_alpha[a]):.2f}"
                     + ("  <-- best" if a == best_alpha else ""), fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("soft oracle p_w")
    for k in range(len(alpha_grid), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"Per-word soft oracle distribution vs alpha (B={args.B_word}, D_B={D_B:.3f})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pw_distribution_by_alpha.png"), dpi=140)
    plt.close(fig)

    # 2) alpha tradeoff curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_a["alpha"], df_a["entropy"], "o-", color="tab:red", label="hist entropy (uniformity)")
    ax.plot(df_a["alpha"], df_a["frac_in_band_0.2_0.8"], "s-", color="tab:green",
            label="frac p_w in [0.2,0.8]")
    ax.plot(df_a["alpha"], df_a["frac_saturated"], "^-", color="tab:gray",
            label="frac saturated (<.05 or >.95)")
    ax.axvline(best_alpha, color="k", ls=":", alpha=0.6, label=f"best alpha={best_alpha}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("metric")
    ax.set_title("alpha tradeoff: graded oracle vs saturation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alpha_tradeoff.png"), dpi=140)
    plt.close(fig)

    # 3) PCA cumvar
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, "o-", color="tab:blue")
    for d in d_grid:
        ax.axvline(d, color="gray", ls=":", alpha=0.3)
    ax.axvline(8, color="tab:red", ls="--", alpha=0.7, label="current d=8")
    ax.set_xlabel("PCA dim d")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"PCA cumvar (d=8 -> {cumvar[7]:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_cumvar.png"), dpi=140)
    plt.close(fig)

    # 4) R^2(d, alpha) heatmap (CV)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, mat, title in [(axes[0], r2_cv, "CV R^2  logit(p_w) ~ linear(z_d)"),
                           (axes[1], r2_in, "in-sample R^2 (overfit reference)")]:
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                       vmin=min(0.0, np.nanmin(r2_cv)), vmax=1.0)
        ax.set_xticks(range(len(alpha_grid)))
        ax.set_xticklabels(alpha_grid)
        ax.set_yticks(range(len(d_grid)))
        ax.set_yticklabels(d_grid)
        ax.set_xlabel("alpha")
        ax.set_ylabel("d")
        ax.set_title(title, fontsize=11)
        for i in range(len(d_grid)):
            for j in range(len(alpha_grid)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.6 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Logistic-linear faithfulness across (d, alpha) — higher = sigma(beta^T z) holds better",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "logit_linearfit_r2.png"), dpi=140)
    plt.close(fig)

    # ---- summary ----
    # best d at the recommended alpha: largest CV R^2 with diminishing returns
    j_best = alpha_grid.index(best_alpha)
    col = r2_cv[:, j_best]
    summary = {
        "D_B": D_B, "n_candidates": n, "M": M,
        "across_word_std": float(dreams_k.mean(1).std()),
        "within_word_std": float(dreams_k.std(1).mean()),
        "cumvar_d8": float(cumvar[7]),
        "alpha_matched_to_scale": alpha_matched,
        "recommended_alpha": best_alpha,
        "best_d_cv_r2": int(d_grid[int(np.argmax(r2_cv[:, alpha_grid.index(best_alpha)]))]),
        "alpha_grid": alpha_grid, "d_grid": d_grid,
        "cv_r2_at_best_alpha": {int(d): float(col[i]) for i, d in enumerate(d_grid)},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CV R^2 of logit(p_w)~linear(z_d) at best alpha={} ===".format(best_alpha))
    for i, d in enumerate(d_grid):
        print(f"  d={d:3d}: CV R2={col[i]:.3f}  (in-sample {r2_in[i, j_best]:.3f})")
    print(f"\nSaved -> {out_dir}/")
    for f in ["pw_distribution_by_alpha.png", "alpha_tradeoff.png",
              "pca_cumvar.png", "logit_linearfit_r2.png",
              "alpha_sweep.csv", "summary.json"]:
        print("  -", f)


if __name__ == "__main__":
    main()
