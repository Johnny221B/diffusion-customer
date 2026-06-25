"""Cheap discrete proxy for tuning CM-TS before rendering continuous designs.

This intentionally differs from the historical discrete experiment (script 63):

* the model feature is ``phi = z - z_comp`` with no intercept, as in script 73;
* each round uses B independent Thompson draws, hence B (possibly repeated) words;
* by default every selected word gets an independently sampled pre-rendered seed;
  with --fixed_image_seed, each word maps to exactly one pre-rendered image,
  matching the fixed-render-seed continuous setting;
* Laplace MAP, ridge precision ``lam``, norm clip ``S`` and covariance inflation
  ``v`` are exactly the implementations used by continuous CM-TS;
* training alpha and evaluation alpha are separate.  This is required for a
  meaningful alpha sweep because changing alpha also changes the definition of
  true probability.

No SD3.5 or GPU is used.  The 228 x 40 DreamSim matrix is the image oracle.

Example smoke test::

  conda run -n diverse python scripts/92_discrete_cmts_sweep.py \
    --alphas 15 --vs 0.5,1 --lams 1,10 --n_sim 2 --T 10 --tag smoke

Full first-pass screen::

  conda run -n diverse python scripts/92_discrete_cmts_sweep.py \
    --alphas 10,15,20,30 --vs 0.25,0.5,1,2,4 \
    --lams 0.5,1,5,10,50 --n_sim 20 --T 200 --tag grid1

Fixed-image-seed screen, matching the current continuous setting more closely::

  conda run -n diverse python scripts/92_discrete_cmts_sweep.py \
    --fixed_image_seed 18 --alphas 3,5,7.5,10,15,20,30 \
    --vs 0.5,1,2,4 --lams 10,20,50,100 \
    --eval_alpha 30 --n_sim 20 --T 200 --tag fixedseed_alpha
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from src.cmts_sim import laplace_map, project_norm, sigma  # noqa: E402


def parse_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def config_name(alpha, v, lam):
    return f"a{alpha:g}_v{v:g}_lam{lam:g}"


def load_problem(args):
    pool = np.load(args.pool_npz, allow_pickle=True)
    embs = np.asarray(pool["embs"], dtype=np.float64)
    words = pd.read_csv(args.raw_csv)["word"].astype(str).tolist()
    dreams = np.asarray(np.load(args.dreams_npz, allow_pickle=True)["dreams"],
                        dtype=np.float64)

    valid = ~np.any(np.isnan(dreams), axis=1)
    embs, dreams = embs[valid], dreams[valid]
    words = [w for w, ok in zip(words, valid) if ok]
    if args.B_word not in words:
        raise ValueError(f"competitor word {args.B_word!r} is not in the pool")
    if not 0 <= args.B_seed < dreams.shape[1]:
        raise ValueError(f"B_seed must be in [0, {dreams.shape[1]})")

    centered = embs - embs.mean(axis=0)
    pca = PCA(n_components=args.d, random_state=0).fit(centered)
    Z = pca.transform(centered).astype(np.float64)

    comp_idx = words.index(args.B_word)
    z_comp = Z[comp_idx].copy()
    D_B = float(dreams[comp_idx, args.B_seed])
    keep = np.arange(len(words)) != comp_idx
    return (Z[keep], dreams[keep], [w for w, ok in zip(words, keep) if ok],
            z_comp, D_B, float(pca.explained_variance_ratio_.sum()))


def run_one(alpha, v, lam, sim_seed, Z, dreams, z_comp, D_B,
            p_eval_word, mean_ds_word, args):
    """Run one discrete CM-TS simulation and return one aggregate row per round."""
    n_words, n_seeds = dreams.shape
    fixed_seed = None
    if args.fixed_image_seed >= 0:
        fixed_seed = int(args.fixed_image_seed) % n_seeds
    # Identical initial random stream across configurations for a given sim.
    rng = np.random.default_rng(sim_seed * 1000 + 7)
    Phi_all = Z - z_comp
    p_train_word = (sigma(alpha * (D_B - dreams[:, fixed_seed]))
                    if fixed_seed is not None
                    else sigma(alpha * (D_B - dreams)).mean(axis=1))

    warm_idx = rng.integers(n_words, size=args.N0)
    warm_seed = (np.full(args.N0, fixed_seed, dtype=int)
                 if fixed_seed is not None
                 else rng.integers(n_seeds, size=args.N0))
    warm_ds = dreams[warm_idx, warm_seed]
    warm_prob = sigma(alpha * (D_B - warm_ds))
    warm_y = rng.binomial(1, warm_prob).astype(float)
    Phi = Phi_all[warm_idx].copy()
    y = warm_y.copy()
    beta_hat, H = laplace_map(Phi, y, lam, args.d)
    beta_hat = project_norm(beta_hat, args.S)

    rows = []
    cfg = config_name(alpha, v, lam)
    for t in range(args.T):
        center = beta_hat.copy()
        Hinv = np.linalg.pinv(H)
        cov = v * v * 0.5 * (Hinv + Hinv.T)
        cov_eigs = np.linalg.eigvalsh(cov)
        betas = rng.multivariate_normal(beta_hat, cov, size=args.B)

        # Subtracting z_comp is constant over arms, so either Z or Phi_all gives
        # the same argmax.  Phi_all makes the connection to the model explicit.
        arms = np.argmax(betas @ Phi_all.T, axis=1)
        image_seeds = (np.full(args.B, fixed_seed, dtype=int)
                       if fixed_seed is not None
                       else rng.integers(n_seeds, size=args.B))
        ds = dreams[arms, image_seeds]
        p_train = sigma(alpha * (D_B - ds))
        p_eval_sample = sigma(args.eval_alpha * (D_B - ds))
        y_batch = rng.binomial(1, p_train).astype(float)
        Phi_batch = Phi_all[arms]
        Phi = np.vstack((Phi, Phi_batch))
        y = np.append(y, y_batch)
        beta_hat, H = laplace_map(Phi, y, lam, args.d, beta0=beta_hat)
        beta_hat = project_norm(beta_hat, args.S)
        predicted = sigma(Phi_batch @ beta_hat)

        rows.append({
            "config": cfg,
            "alpha_train": alpha,
            "alpha_eval": args.eval_alpha,
            "v": v,
            "lam": lam,
            "sim_seed": sim_seed,
            "t": t,
            "true_p_soft_sample": float(np.mean(p_eval_sample)),
            "true_p_soft_oracle": float(np.mean(p_eval_word[arms])),
            "true_p_train_oracle": float(np.mean(p_train_word[arms])),
            "predicted_p": float(np.mean(predicted)),
            "mean_ds_sample": float(np.mean(ds)),
            "mean_ds_oracle": float(np.mean(mean_ds_word[arms])),
            "round_min_ds_sample": float(np.min(ds)),
            "round_min_ds_oracle": float(np.min(mean_ds_word[arms])),
            "hard_winrate": float(np.mean(ds < D_B)),
            "fixed_image_seed": -1 if fixed_seed is None else int(fixed_seed),
            "label_rate": float(np.mean(y_batch)),
            "beta_norm": float(np.linalg.norm(beta_hat)),
            "cov_eig_max": float(cov_eigs[-1]),
            "cov_eig_min": float(cov_eigs[0]),
            "n_unique_arms": int(np.unique(arms).size),
        })
    return rows


def make_summary(df, args):
    window = min(20, max(1, args.T // 5))
    early = df[df["t"] < window]
    late = df[df["t"] >= args.T - window]
    metrics = [
        "true_p_soft_sample", "true_p_soft_oracle", "true_p_train_oracle", "predicted_p",
        "mean_ds_sample", "mean_ds_oracle", "hard_winrate", "beta_norm",
    ]
    keys = ["config", "alpha_train", "alpha_eval", "v", "lam"]
    e = early.groupby(keys)[metrics].mean().add_suffix("_early")
    l = late.groupby(keys)[metrics].mean().add_suffix("_late")
    out = e.join(l).reset_index()
    for metric in metrics:
        out[f"{metric}_delta"] = out[f"{metric}_late"] - out[f"{metric}_early"]
    out["belief_gap_late"] = (
        out["predicted_p_late"] - out["true_p_soft_oracle_late"]
    )
    return out.sort_values(
        ["true_p_soft_oracle_late", "belief_gap_late"],
        ascending=[False, True],
    )


def rolling_mean(frame, column, window):
    return frame[column].rolling(window, min_periods=1, center=True).mean()


def plot_top(df, summary, out_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = summary.head(args.plot_top)["config"].tolist()
    mean = (df[df["config"].isin(top)]
            .groupby(["config", "t"], as_index=False)
            .mean(numeric_only=True))
    ncols = min(3, max(1, len(top)))
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, cfg in zip(axes.flat, top):
        sub = mean[mean["config"] == cfg].sort_values("t")
        x = sub["t"].to_numpy()
        # Exact 40-seed expectation is the primary truth.  The pale line is the
        # finite B-image quantity corresponding exactly to continuous panel (b).
        ax.plot(x, rolling_mean(sub, "true_p_soft_oracle", args.roll),
                color="tab:green", lw=2.2, label="true p (40-seed oracle)")
        ax.plot(x, rolling_mean(sub, "true_p_soft_sample", args.roll),
                color="tab:green", lw=1.0, alpha=0.30,
                label="true p (sampled images)")
        ax.plot(x, rolling_mean(sub, "predicted_p", args.roll),
                color="tab:red", lw=1.6, ls="--", label="predicted p")
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_title(cfg)
        ax.grid(alpha=0.25)
        ax.set_xlabel("round t")
        ax.set_ylabel("probability")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(top):]:
        ax.axis("off")
    fig.suptitle(
        f"Discrete CM-TS belief vs truth; B={args.B_word}/{args.B_seed}, "
        f"eval alpha={args.eval_alpha:g}, d={args.d}, sims={args.n_sim}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "belief_vs_truth_top.png"), dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_npz", default="outputs/strict_pool_s228_0429_0119/embeddings.npz")
    ap.add_argument("--raw_csv", default="outputs/strict_pool_s228_0429_0119/raw_data.csv")
    ap.add_argument("--dreams_npz", default="outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz")
    ap.add_argument("--alphas", default="10,15,20,30", help="comma-separated training alphas")
    ap.add_argument("--eval_alpha", type=float, default=15.0,
                    help="fixed alpha used for cross-configuration true-p evaluation")
    ap.add_argument("--vs", default="0.25,0.5,1,2,4")
    ap.add_argument("--lams", default="0.5,1,5,10,50")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--N0", type=int, default=24)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--n_sim", type=int, default=20)
    ap.add_argument("--S", type=float, default=8.0)
    ap.add_argument("--B_word", default="bright")
    ap.add_argument("--B_seed", type=int, default=18)
    ap.add_argument("--fixed_image_seed", type=int, default=-1,
                    help="if >=0, every word is evaluated only at this image seed; "
                         "default -1 samples a pre-rendered seed independently")
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--tag", default="grid1")
    ap.add_argument("--plot_top", type=int, default=12)
    ap.add_argument("--roll", type=int, default=15)
    args = ap.parse_args()

    alphas, vs, lams = map(parse_floats, (args.alphas, args.vs, args.lams))
    if not alphas or not vs or not lams:
        raise ValueError("alphas, vs and lams must each contain at least one value")
    n_cfg = len(alphas) * len(vs) * len(lams)
    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(args.out_root, f"discrete_cmts_{args.tag}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    Z, dreams, words, z_comp, D_B, pca_var = load_problem(args)
    if args.fixed_image_seed >= 0:
        fixed_seed = int(args.fixed_image_seed) % dreams.shape[1]
        p_eval_word = sigma(args.eval_alpha * (D_B - dreams[:, fixed_seed]))
        mean_ds_word = dreams[:, fixed_seed]
    else:
        fixed_seed = None
        p_eval_word = sigma(args.eval_alpha * (D_B - dreams)).mean(axis=1)
        mean_ds_word = dreams.mean(axis=1)
    config = vars(args).copy()
    config.update({"alphas_parsed": alphas, "vs_parsed": vs,
                   "lams_parsed": lams, "D_B": D_B,
                   "n_candidates": len(words), "n_image_seeds": dreams.shape[1],
                   "fixed_image_seed_resolved": fixed_seed,
                   "pca_explained_variance": pca_var})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Loaded {len(words)} candidates x {dreams.shape[1]} seeds; "
          f"B={args.B_word}/{args.B_seed}, D_B={D_B:.6f}, PCA var={pca_var:.3f}")
    print(f"Running {n_cfg} configs x {args.n_sim} sims x {args.T} rounds")
    rows = []
    start = time.time()
    done = 0
    for alpha in alphas:
        for v in vs:
            for lam in lams:
                cfg = config_name(alpha, v, lam)
                for sim_seed in range(args.n_sim):
                    rows.extend(run_one(alpha, v, lam, sim_seed, Z, dreams, z_comp,
                                        D_B, p_eval_word, mean_ds_word, args))
                done += 1
                elapsed = time.time() - start
                eta = elapsed * (n_cfg - done) / done
                print(f"[{done:3d}/{n_cfg}] {cfg}  elapsed={elapsed/60:.1f}m "
                      f"eta={eta/60:.1f}m", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(out_dir, "round_metrics.csv"),
                                          index=False)

    df = pd.DataFrame(rows)
    summary = make_summary(df, args)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    plot_top(df, summary, out_dir, args)
    print("\nTop configurations by late exact true p:")
    cols = ["config", "true_p_soft_oracle_early", "true_p_soft_oracle_late",
            "true_p_soft_oracle_delta", "true_p_train_oracle_delta",
            "predicted_p_late", "belief_gap_late",
            "mean_ds_oracle_late", "beta_norm_late"]
    print(summary[cols].head(min(12, len(summary))).to_string(index=False))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
