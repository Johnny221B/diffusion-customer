"""Exp2: BO learning curve via Thompson sampling.

Five surrogates compete on the same task:
  warm-start with N0 iid (word, seed) labels, then run T BO steps.
  At each step:
    1. surrogate picks z* (its acquisition; Thompson except logistic_l2 which is eps-greedy)
    2. B images of word(z*) are "generated" by looking up dreams_matrix[i*, s] for
       B distinct seeds s (no SD3.5 call — we precomputed the matrix)
    3. B noisy labels y_b ~ Bern(sigmoid(alpha * (D_B - D_{w*, s_b}))) are added
    4. surrogate updates posterior (refit on accumulated data;
       GP refit only every gp_refit_every steps to amortize O(N^3))
  Record per-step true expected success p_oracle[i*] (the gold; available
  because dreams_matrix gives us all 40 seeds per word) AND the empirical
  success rate among the B images this step.

Usage:
  python scripts/63_bo_thompson_curve.py \
      --pool_npz   outputs/strict_pool_s228_0429_0119/embeddings.npz \
      --raw_csv    outputs/strict_pool_s228_0429_0119/raw_data.csv \
      --dreams_npz outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz \
      --d 8 --N0 50 --T 200 --B 8 --n_sim 100 \
      --alpha 30 --prior_var 3.0 --epsilon 0.1 \
      --B_word canvas --B_seed 34 \
      --tag bcanvas
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# silence sklearn convergence warnings to keep the log readable
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

# import LogisticThompsonOptimizer from the project src
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
from src.thompson_optimizer import LogisticThompsonOptimizer  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
#  Surrogate wrappers: uniform interface
#     .acquire(Z, rng)  -> int index in [0, n_words)
#     .add(z, y)        -> log (z, y) to history
#     .step_update(t)   -> refit (subject to internal refit policy)
# =============================================================================

class BayesianLogisticSurrogate:
    """Laplace + Thompson on sigma(beta^T z)."""
    name = "logistic_bayesian"
    def __init__(self, d, prior_var=3.0, exploration_a=1.0):
        self.d = d
        self.opt = LogisticThompsonOptimizer(
            dim_latent=d, prior_var=prior_var, exploration_a=exploration_a,
        )
        self._has_data = False
    def acquire(self, Z, rng):
        if not self._has_data:
            return int(rng.randint(0, len(Z)))
        b0, beta = self.opt.sample_theta()
        scores = b0 + Z @ beta
        return int(np.argmax(scores))
    def add(self, z, y):
        self.opt.add_comparison_data(z, int(y))
        self._has_data = True
    def step_update(self, t):
        if self._has_data:
            self.opt.update_posterior()


class L2LogisticSurrogate:
    """MAP frequentist sigma(beta^T z) + epsilon-greedy.

    C=prior_var: makes the loss surface identical to BayesianLogisticSurrogate
    under a N(0, prior_var I) Gaussian prior. Only acquisition differs.
    """
    name = "logistic_l2"
    def __init__(self, d, C=3.0, epsilon=0.1):
        self.d = d
        self.C = C
        self.epsilon = epsilon
        self.X = []
        self.y = []
        self.clf = None
    def acquire(self, Z, rng):
        if self.clf is None or rng.uniform() < self.epsilon:
            return int(rng.randint(0, len(Z)))
        scores = self.clf.predict_proba(Z)[:, 1]
        return int(np.argmax(scores))
    def add(self, z, y):
        self.X.append(np.asarray(z, dtype=np.float64))
        self.y.append(int(y))
    def step_update(self, t):
        if len(set(self.y)) < 2:
            return
        self.clf = LogisticRegression(
            C=self.C, penalty="l2", max_iter=5000, solver="lbfgs",
        ).fit(np.array(self.X), np.array(self.y))


class Poly2Surrogate:
    """Laplace + Thompson on sigma(beta^T phi(z)) with phi = degree-2 expansion."""
    name = "poly2_logistic"
    def __init__(self, d, prior_var=3.0, exploration_a=1.0):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        dummy = self.poly.fit_transform(np.zeros((1, d), dtype=np.float64))
        self.poly_d = int(dummy.shape[1])
        self.opt = LogisticThompsonOptimizer(
            dim_latent=self.poly_d, prior_var=prior_var,
            exploration_a=exploration_a,
        )
        self._has_data = False
    def _expand_one(self, z):
        return self.poly.transform(z.reshape(1, -1))[0]
    def _expand_many(self, Z):
        return self.poly.transform(Z)
    def acquire(self, Z, rng):
        if not self._has_data:
            return int(rng.randint(0, len(Z)))
        b0, beta = self.opt.sample_theta()
        scores = b0 + self._expand_many(Z) @ beta
        return int(np.argmax(scores))
    def add(self, z, y):
        self.opt.add_comparison_data(self._expand_one(z), int(y))
        self._has_data = True
    def step_update(self, t):
        if self._has_data:
            self.opt.update_posterior()


class GPRBFSurrogate:
    """GPR (binary y as continuous) + Thompson sampling f ~ N(mu, sigma^2)."""
    name = "gp_rbf"
    def __init__(self, d, refit_every=5):
        self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.refit_every = refit_every
        self.X = []
        self.y = []
        self.gpr = None
    def acquire(self, Z, rng):
        if self.gpr is None:
            return int(rng.randint(0, len(Z)))
        mu, std = self.gpr.predict(Z, return_std=True)
        f_sample = mu + std * rng.standard_normal(len(Z))
        return int(np.argmax(f_sample))
    def add(self, z, y):
        self.X.append(np.asarray(z, dtype=np.float64))
        self.y.append(float(y))
    def step_update(self, t):
        if len(self.y) < 2:
            return
        if t % self.refit_every != 0 and self.gpr is not None:
            return
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel, alpha=1e-3, normalize_y=True,
            n_restarts_optimizer=1, random_state=0,
        ).fit(np.array(self.X), np.array(self.y))


class RFBootstrapSurrogate:
    """RF + bootstrap-style Thompson: sample one tree from the forest, argmax."""
    name = "random_forest"
    def __init__(self, d, n_estimators=200):
        self.n_estimators = n_estimators
        self.X = []
        self.y = []
        self.rf = None
    def acquire(self, Z, rng):
        if self.rf is None or len(set(self.y)) < 2:
            return int(rng.randint(0, len(Z)))
        tree_idx = int(rng.randint(0, len(self.rf.estimators_)))
        tree = self.rf.estimators_[tree_idx]
        proba = tree.predict_proba(Z)
        if proba.shape[1] == 2:
            scores = proba[:, 1]
        else:
            # tree saw only one class; predict_proba is degenerate
            scores = np.full(len(Z), float(tree.classes_[0]))
        return int(np.argmax(scores + 1e-6 * rng.standard_normal(len(Z))))
    def add(self, z, y):
        self.X.append(np.asarray(z, dtype=np.float64))
        self.y.append(int(y))
    def step_update(self, t):
        if len(set(self.y)) < 2:
            return
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, n_jobs=-1, random_state=None,
        ).fit(np.array(self.X), np.array(self.y))


def make_surrogates(d, args):
    surr = [
        BayesianLogisticSurrogate(d, prior_var=args.prior_var),
        L2LogisticSurrogate(d, C=args.prior_var, epsilon=args.epsilon),
        Poly2Surrogate(d, prior_var=args.prior_var),
        GPRBFSurrogate(d, refit_every=args.gp_refit_every),
        RFBootstrapSurrogate(d),
    ]
    # poly2's degree-2 expansion is d*(d+3)/2 features (324 at d=24): the
    # Laplace Hessian inversion becomes prohibitive at high d. Allow skipping it.
    if getattr(args, "skip_poly2", False):
        surr = [s for s in surr if s.name != "poly2_logistic"]
    return surr


def run_one_sim_seed(seed, Z_full, dreams, D_B, p_oracle, args):
    n_words, M = dreams.shape
    d = Z_full.shape[1]

    # Seed BOTH np.random (used by LogisticThompsonOptimizer.sample_theta
    # via multivariate_normal) and a local RandomState for everything else.
    np.random.seed(seed * 97 + 3)
    rng = np.random.RandomState(seed * 1000 + 7)

    surrogates = make_surrogates(d, args)

    theta_batch = (args.acq_mode == "theta_batch")
    s0 = int(args.fixed_seed) % M   # the single fixed seed in theta_batch mode

    # ---- Warm-start: N0 iid word samples shared across all surrogates ----
    # seed_batch: each warm sample uses a random seed.
    # theta_batch: every image uses the single fixed seed s0 (the new regime
    #   keeps ONE seed throughout, so warm-start must too, else the two runs
    #   would differ in their warm data and confound the comparison).
    warm_idx = rng.randint(0, n_words, size=args.N0)
    warm_s   = (np.full(args.N0, s0, dtype=int) if theta_batch
                else rng.randint(0, M, size=args.N0))
    warm_D   = dreams[warm_idx, warm_s]
    warm_p   = sigmoid(args.alpha * (D_B - warm_D))
    warm_y   = (rng.uniform(size=args.N0) < warm_p).astype(int)

    for sur in surrogates:
        for k in range(args.N0):
            sur.add(Z_full[warm_idx[k]], warm_y[k])
        sur.step_update(t=0)

    # ---- BO loop ----
    records = []
    for t in range(1, args.T + 1):
        for sur in surrogates:
            if theta_batch:
                # Parallel/batch Thompson: draw B independent acquisitions from
                # the SAME current posterior (B theta samples for TS surrogates;
                # B eps-greedy draws for logistic_l2), each proposing one word.
                # Each proposed word is "generated" at the single fixed seed s0,
                # so the batch's diversity comes from posterior sampling, not
                # from seed replication. Observe B labels, then ONE update.
                arms = np.array([sur.acquire(Z_full, rng) for _ in range(args.B)],
                                dtype=int)
                D_arms = dreams[arms, s0]
                p_arms = sigmoid(args.alpha * (D_B - D_arms))
                y_arms = (rng.uniform(size=args.B) < p_arms).astype(int)
                for b in range(args.B):
                    sur.add(Z_full[arms[b]], y_arms[b])
                sur.step_update(t=t)
                for b in range(args.B):
                    records.append({
                        "sim_seed": seed,
                        "model":   sur.name,
                        "step":    t,
                        "picked_idx":  int(arms[b]),
                        "p_oracle":    float(p_oracle[arms[b]]),
                        "empirical_B": float(D_arms[b] < D_B),
                        "arm_in_batch": b,
                    })
            else:
                i_star = sur.acquire(Z_full, rng)
                # B distinct seeds for word i_star
                s_batch = rng.choice(M, size=args.B, replace=False)
                D_batch = dreams[i_star, s_batch]
                p_batch = sigmoid(args.alpha * (D_B - D_batch))
                y_batch = (rng.uniform(size=args.B) < p_batch).astype(int)
                empirical = float((D_batch < D_B).mean())
                for b in range(args.B):
                    sur.add(Z_full[i_star], y_batch[b])
                sur.step_update(t=t)
                records.append({
                    "sim_seed": seed,
                    "model":   sur.name,
                    "step":    t,
                    "picked_idx":  i_star,
                    "p_oracle":    float(p_oracle[i_star]),
                    "empirical_B": empirical,
                })
    return records


def plot_bo_curve(agg, p_max, p_mean, out_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "logistic_bayesian": "tab:blue",
        "logistic_l2":       "tab:cyan",
        "poly2_logistic":    "tab:purple",
        "gp_rbf":            "tab:green",
        "random_forest":     "tab:orange",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    # (a) cumulative best-so-far  -- monotonic, reads as "running max" curve
    # (b) per-step picked         -- jagged, shows surrogate's confidence each step
    for ax, ycol, title in [
        (axes[0], "p_oracle_running_max",
         "Running max of expected success (best-so-far recommendation)"),
        (axes[1], "p_oracle",
         "Expected success of step-t pick (raw)"),
    ]:
        for model in sorted(agg["model"].unique()):
            sub = agg[(agg["model"] == model) & (agg["metric"] == ycol)].sort_values("step")
            mean = sub["mean"].values
            std  = sub["std"].values
            n    = sub["count"].values
            se = std / np.sqrt(np.maximum(n, 1))
            ax.plot(sub["step"].values, mean, label=model,
                    color=colors.get(model, "black"))
            ax.fill_between(sub["step"].values, mean - se, mean + se,
                            alpha=0.18, color=colors.get(model, "black"))
        ax.axhline(p_max,  color="k", linestyle=":",  alpha=0.5,
                   label=f"oracle ceiling ({p_max:.3f})")
        ax.axhline(p_mean, color="k", linestyle="--", alpha=0.4,
                   label=f"uniform mean ({p_mean:.3f})")
        ax.set_xlabel("BO step t")
        ax.set_ylabel("Expected success rate")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"BO Thompson curve  α={args.alpha}, d={args.d}, "
        f"N₀={args.N0}, B={args.B}, T={args.T}, sims={args.n_sim_done}, "
        f"B_word={args.B_word_used} s_B={args.B_seed}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bo_thompson_curve.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_npz",   required=True)
    parser.add_argument("--raw_csv",    required=True)
    parser.add_argument("--dreams_npz", required=True)
    parser.add_argument("--alpha",       type=float, default=30.0)
    parser.add_argument("--d",           type=int,   default=8)
    parser.add_argument("--N0",          type=int,   default=50)
    parser.add_argument("--T",           type=int,   default=200)
    parser.add_argument("--B",           type=int,   default=8)
    parser.add_argument("--n_sim",       type=int,   default=100)
    parser.add_argument("--prior_var",   type=float, default=3.0)
    parser.add_argument("--epsilon",     type=float, default=0.1)
    parser.add_argument("--gp_refit_every", type=int, default=5)
    parser.add_argument("--B_word",      type=str, default="canvas")
    parser.add_argument("--B_seed",      type=int, default=34)
    parser.add_argument("--out_root",    type=str, default="outputs")
    parser.add_argument("--tag",         type=str, default="bcanvas")
    parser.add_argument("--skip_poly2",  action="store_true",
                        help="drop poly2_logistic (prohibitive Hessian at high d)")
    parser.add_argument("--acq_mode", type=str, default="seed_batch",
                        choices=["seed_batch", "theta_batch"],
                        help="seed_batch (canonical): 1 arm x B random seeds. "
                             "theta_batch: B posterior samples x 1 fixed seed "
                             "(parallel Thompson sampling).")
    parser.add_argument("--fixed_seed", type=int, default=0,
                        help="the single seed used per image in theta_batch mode")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(args.out_root, f"bo_thompson_{args.tag}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load embeddings ----
    pool = np.load(args.pool_npz, allow_pickle=True)
    embs = pool["embs"]
    df_raw = pd.read_csv(args.raw_csv)
    words = df_raw["word"].astype(str).tolist()

    dm = np.load(args.dreams_npz, allow_pickle=True)
    dreams = dm["dreams"]
    M = int(dreams.shape[1])
    args.M_used = M

    valid = ~np.any(np.isnan(dreams), axis=1)
    embs   = embs[valid]
    dreams = dreams[valid]
    words  = [w for w, v in zip(words, valid) if v]
    print(f"Loaded {len(embs)} valid words, M={M} seeds each.")

    # ---- PCA at d ----
    mean_emb = embs.mean(0)
    pca = PCA(n_components=args.d, random_state=0).fit(embs - mean_emb)
    Z_full = ((embs - mean_emb) @ pca.components_.T).astype(np.float64)
    cum = float(np.cumsum(pca.explained_variance_ratio_)[args.d - 1])
    print(f"PCA cumvar at d={args.d}: {cum:.3f}")

    # ---- Pick B, drop from candidate pool, compute oracle ----
    if args.B_word not in words:
        raise ValueError(f"B_word {args.B_word!r} not in pool")
    B_idx = words.index(args.B_word)
    D_B = float(dreams[B_idx, args.B_seed])
    args.B_word_used = args.B_word

    keep = np.arange(len(words)) != B_idx
    Z_full_kept = Z_full[keep]
    dreams_kept = dreams[keep]
    words_kept  = [w for w, k in zip(words, keep) if k]
    p_oracle = (dreams_kept < D_B).mean(axis=1).astype(np.float64)
    p_max  = float(p_oracle.max())
    p_mean = float(p_oracle.mean())
    print(f"B='{args.B_word}' s_B={args.B_seed}  D_B={D_B:.4f}")
    print(f"p_oracle: min={p_oracle.min():.3f} mean={p_mean:.3f} max={p_max:.3f}")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    np.savez(
        os.path.join(out_dir, "oracle.npz"),
        words=np.array(words_kept), p_oracle=p_oracle,
        D_B=D_B, B_word=args.B_word, M=M,
    )

    print(f"\nd={args.d}, N0={args.N0}, T={args.T}, B={args.B}, "
          f"n_sim={args.n_sim}, prior_var={args.prior_var}\n")

    # ---- Main loop over simulation seeds ----
    all_records = []
    t0 = time.time()
    for sim in range(args.n_sim):
        recs = run_one_sim_seed(
            sim, Z_full_kept, dreams_kept, D_B, p_oracle, args,
        )
        all_records.extend(recs)
        if (sim + 1) % 5 == 0 or sim == 0:
            elapsed = time.time() - t0
            eta = elapsed * (args.n_sim - sim - 1) / (sim + 1)
            df_so_far = pd.DataFrame(all_records)
            final = df_so_far[df_so_far["step"] == args.T]
            tail = final.groupby("model")["p_oracle"].mean().round(3).to_dict()
            print(f"[sim {sim+1:3d}/{args.n_sim}] "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  "
                  f"final-step p_oracle: {tail}")
            df_so_far.to_csv(
                os.path.join(out_dir, "bo_per_step.csv"), index=False,
            )

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(out_dir, "bo_per_step.csv"), index=False)
    args.n_sim_done = int(df["sim_seed"].nunique())

    # ---- Aggregate: per (step, model) for both raw p_oracle and running max ----
    df = df.sort_values(["sim_seed", "model", "step"])
    df["p_oracle_running_max"] = df.groupby(["sim_seed", "model"])["p_oracle"].cummax()

    long = []
    for metric in ["p_oracle", "p_oracle_running_max", "empirical_B"]:
        g = (
            df.groupby(["step", "model"])[metric]
              .agg(["mean", "std", "count"]).reset_index()
        )
        g["metric"] = metric
        long.append(g)
    agg = pd.concat(long, axis=0, ignore_index=True)
    agg.to_csv(os.path.join(out_dir, "bo_summary.csv"), index=False)

    plot_bo_curve(agg, p_max, p_mean, out_dir, args)
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
