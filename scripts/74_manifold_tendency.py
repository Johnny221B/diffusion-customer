"""74 — Manifold-scaling tendency: how kNN-distance / tau / coverage move as the
vocabulary (anchor count N) grows toward the full 227.

Same construction as scripts/73_cmts_dreamsim.py: drop competitor 'black', fit
PCA(d) on the 227 kept words -> canonical latent Z, manifold M = {z : d_k(z) <= tau},
tau = q-quantile of leave-one-out k-NN distances (calibrate_tau, q=0.95).

The PCA space is FIXED (fit once on all 227); we only SUBSAMPLE N anchors from it,
so the curves isolate "anchor density", not a shifting embedding geometry.

For each N (reps random subsamples):
  knn_dist : mean over anchors of the within-subsample leave-one-out k-th NN distance
  tau      : calibrate_tau(subsample) = 95th pct of those LOO kNN distances
  coverage : HELD-OUT real-word coverage = fraction of the (227-N) words NOT in the
             subsample whose k-th NN distance to the subsample is <= tau.
             (In-sample coverage is pinned to q=0.95 by construction, so it carries
              no information; held-out coverage is the real generalization test.)

CPU only — does NOT touch the GPU / the running v-sweep.

  conda run -n diverse --no-capture-output python scripts/74_manifold_tendency.py --dim 16
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cmts_sim import calibrate_tau   # reuse the exact tau definition


def loo_knn_dists(A, k):
    """leave-one-out k-th nearest-neighbour distance for each row of A."""
    D = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def kth_dist_to_set(P, A, k):
    """k-th nearest distance from each row of P to the anchor set A (P not in A)."""
    D = np.linalg.norm(P[:, None, :] - A[None, :, :], axis=2)
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def kth_dist_all_probes(Z, anchor_mask, k):
    """k-th NN distance from EVERY word in Z to the anchor subset, excluding self.
    Defined for all N including N=len(Z) (then it reduces to leave-one-out)."""
    A = Z[anchor_mask]
    D = np.linalg.norm(Z[:, None, :] - A[None, :, :], axis=2)   # (M, N)
    # for words that ARE anchors, blank out the self-distance (0) so we don't count self
    anchor_rows = np.where(anchor_mask)[0]
    D[anchor_rows, np.arange(len(anchor_rows))] = np.inf
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", default="outputs/strict_pool_s228_0429_0119")
    ap.add_argument("--B_word", default="black", help="competitor word dropped from anchors")
    ap.add_argument("--dim", type=int, default=16, help="PCA dim (canonical run uses 16)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("--reps", type=int, default=30, help="random subsamples per N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or f"outputs/manifold_tendency_d{args.dim}"
    os.makedirs(out_dir, exist_ok=True)

    # --- load + canonical PCA (identical recipe to script 73) ---
    pool = np.load(os.path.join(args.pool_dir, "embeddings.npz"), allow_pickle=True)
    embs = pool["embs"].astype(np.float32)
    words = [str(w) for w in pool["words"]]
    b_idx = words.index(args.B_word)
    keep = np.ones(len(words), dtype=bool); keep[b_idx] = False
    embs_kept = embs[keep]
    pca = PCA(n_components=args.dim, random_state=0)
    Z = pca.fit_transform(embs_kept).astype(np.float64)        # (227, d) FIXED space
    M = len(Z)
    print(f"PCA({args.dim}) on {M} words, expl_var_ratio_sum={pca.explained_variance_ratio_.sum():.4f}")

    tau_full = calibrate_tau(Z, args.k, q=args.q)
    knn_full = loo_knn_dists(Z, args.k).mean()
    print(f"FULL N={M}: mean LOO kNN={knn_full:.4f}, tau(q={args.q})={tau_full:.4f}  "
          f"(script-73 reports tau_d=20.93 @ d=16)\n")

    # --- N grid (must exceed k; densest near the small end where change is fastest) ---
    Ns = sorted(set(int(round(x)) for x in
                    np.geomspace(max(args.k + 2, 12), M, 16)))
    Ns = [n for n in Ns if n >= args.k + 2]
    if Ns[-1] != M:
        Ns.append(M)

    rng = np.random.default_rng(args.seed)
    rows = []
    hdr = (f"{'N':>5} {'knn_dist':>16} {'tau':>16} {'held_cov':>16} {'n_held':>7}")
    print(hdr); print("-" * len(hdr))
    for N in Ns:
        knn_r, tau_r, cov_r, full_r = [], [], [], []
        n_held_last = 0
        for _ in range(args.reps):
            idx = rng.choice(M, size=N, replace=False)
            mask = np.zeros(M, bool); mask[idx] = True
            A = Z[mask]
            dd = loo_knn_dists(A, args.k)
            knn_r.append(dd.mean())
            tau = np.quantile(dd, args.q)
            tau_r.append(tau)
            # full-population coverage (defined for ALL N up to M; ->q at N=M)
            dk_all = kth_dist_all_probes(Z, mask, args.k)
            full_r.append(float(np.mean(dk_all <= tau)))
            # held-out-only coverage (undefined at N=M: no held-out words)
            held = Z[~mask]
            if len(held) >= 5:
                dk = kth_dist_to_set(held, A, args.k)
                cov_r.append(float(np.mean(dk <= tau)))
                n_held_last = len(held)
        def ms(a): return (float(np.mean(a)), float(np.std(a) / np.sqrt(len(a)))) if a else (np.nan, np.nan)
        knn_m, knn_se = ms(knn_r); tau_m, tau_se = ms(tau_r)
        cov_m, cov_se = ms(cov_r); full_m, full_se = ms(full_r)
        rows.append(dict(N=N, knn_dist=knn_m, knn_se=knn_se, tau=tau_m, tau_se=tau_se,
                         held_cov=cov_m, held_cov_se=cov_se,
                         full_cov=full_m, full_cov_se=full_se, n_held=n_held_last, reps=args.reps))
        cov_s = f"{cov_m:.4f}" if not np.isnan(cov_m) else " n/a "
        print(f"{N:>5} {knn_m:>8.4f}±{knn_se:<6.4f} {tau_m:>8.4f}±{tau_se:<6.4f} "
              f"full={full_m:.4f}±{full_se:<6.4f} held={cov_s:>7} {n_held_last:>7}")

    df = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "tendency.csv"); df.to_csv(csv, index=False)

    # intrinsic-dim estimate: knn_dist ~ N^{-1/d_int}  =>  slope of log-knn vs log-N
    lo = df[df.N <= M * 0.8]
    slope = np.polyfit(np.log(lo.N), np.log(lo.knn_dist), 1)[0]
    d_int = -1.0 / slope if slope < 0 else np.nan
    print(f"\nlog-log slope d(knn)/d(N) = {slope:.3f}  =>  intrinsic dim ~ {d_int:.1f} "
          f"(PCA dim was {args.dim})")

    # --- figure ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    cov_df = df[~df.held_cov.isna()]
    for ax, (col, se, ylab, c, d_) in zip(
            axes[:2],
            [("knn_dist", "knn_se", "mean LOO $k$-NN distance", "tab:blue", df),
             ("tau", "tau_se", rf"$\tau$ (95th pct LOO $k$-NN)", "tab:green", df)]):
        ax.errorbar(d_.N, d_[col], yerr=1.96 * d_[se], marker="o", ms=4,
                    color=c, capsize=2, lw=1.4)
        ax.set_xlabel("number of words $N$"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    # coverage panel: full-population (reaches N=227) + held-out-only (stops early)
    ax = axes[2]
    ax.errorbar(df.N, df.full_cov, yerr=1.96 * df.full_cov_se, marker="o", ms=4,
                color="tab:red", capsize=2, lw=1.4, label="full-population (self-excl.)")
    ax.errorbar(cov_df.N, cov_df.held_cov, yerr=1.96 * cov_df.held_cov_se, marker="s",
                ms=3, color="tab:orange", capsize=2, lw=1.0, ls="--", alpha=0.7,
                label="held-out only")
    ax.set_xlabel("number of words $N$"); ax.set_ylabel("word coverage"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    axes[0].axhline(knn_full, color="gray", ls=":", lw=1)
    axes[1].axhline(tau_full, color="gray", ls=":", lw=1)
    axes[1].annotate(f"full $\\tau$={tau_full:.2f}", (Ns[0], tau_full),
                     fontsize=8, color="gray", va="bottom")
    axes[2].axhline(args.q, color="gray", ls=":", lw=1)
    axes[2].annotate(f"calibration $q$={args.q}", (df.N.iloc[0], args.q),
                     fontsize=8, color="gray", va="bottom")
    fig.suptitle(f"Manifold scaling vs vocabulary size  (PCA $d$={args.dim}, $k$={args.k}, "
                 f"{args.reps} subsamples/point)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = os.path.join(out_dir, "tendency.png"); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"\nsaved: {csv}\n       {png}")


if __name__ == "__main__":
    main()
