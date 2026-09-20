#!/usr/bin/env python3
"""Synthetic discrete validation for the five paper surrogates.

The data-generating model is logistic-linear, matching the stated validation
assumption.  Every method sees the same initial observations; subsequent data
are collected by its own acquisition rule.  Function RMSE is evaluated on the
entire finite action set, so it is comparable for parametric and nonparametric
methods alike.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import PolynomialFeatures


NAMES = ["Bayesian logistic + TS", "L2 logistic + epsilon-greedy",
         "Quadratic logistic + TS", "GP + randomized acquisition",
         "Random forest + tree sampling"]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def laplace_fit(X, y, lam, beta=None, n_iter=30):
    q = X.shape[1]
    beta = np.zeros(q) if beta is None else beta.copy()
    eye = np.eye(q)
    for _ in range(n_iter):
        p = sigmoid(X @ beta)
        grad = X.T @ (p - y) + lam * beta
        w = np.clip(p * (1 - p), 1e-6, None)
        H = X.T @ (X * w[:, None]) + lam * eye
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.linalg.norm(step) < 1e-7:
            break
    p = sigmoid(X @ beta)
    w = np.clip(p * (1 - p), 1e-6, None)
    H = X.T @ (X * w[:, None]) + lam * eye
    return beta, H


class LaplacePolicy:
    def __init__(self, features, lam, rng, quadratic=False, posterior_scale=1.0):
        self.rng, self.lam, self.posterior_scale = rng, lam, posterior_scale
        self.poly = PolynomialFeatures(2, include_bias=False) if quadratic else None
        self.F = self.poly.fit_transform(features) if quadratic else features
        self.X, self.y, self.beta, self.H = [], [], None, None

    def add(self, i, y): self.X.append(self.F[i]); self.y.append(y)
    def fit(self):
        self.beta, self.H = laplace_fit(np.asarray(self.X), np.asarray(self.y), self.lam, self.beta)
    def acquire(self):
        cov = np.linalg.inv(self.H)
        draw = self.rng.multivariate_normal(
            self.beta, self.posterior_scale ** 2 * 0.5 * (cov + cov.T))
        return int(np.argmax(self.F @ draw))
    def predict(self): return sigmoid(self.F @ self.beta)


class L2Policy:
    def __init__(self, features, C, epsilon, rng):
        self.F, self.C, self.epsilon, self.rng = features, C, epsilon, rng
        self.X, self.y, self.model = [], [], None
    def add(self, i, y): self.X.append(self.F[i]); self.y.append(y)
    def fit(self):
        if len(set(self.y)) > 1:
            self.model = LogisticRegression(C=self.C, max_iter=2000).fit(self.X, self.y)
    def predict(self):
        return np.full(len(self.F), .5) if self.model is None else self.model.predict_proba(self.F)[:, 1]
    def acquire(self):
        return int(self.rng.integers(len(self.F))) if self.rng.random() < self.epsilon else int(np.argmax(self.predict()))


class GPPolicy:
    def __init__(self, features, rng):
        self.F, self.rng, self.X, self.y, self.model = features, rng, [], [], None
        distances = pairwise_distances(features)
        self.length_scale = float(np.median(distances[distances > 0]))
    def add(self, i, y): self.X.append(self.F[i]); self.y.append(y)
    def fit(self):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            self.length_scale, length_scale_bounds="fixed")
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=.08, normalize_y=True).fit(self.X, self.y)
    def predict(self): return np.clip(self.model.predict(self.F), 0, 1)
    def acquire(self):
        mu, sd = self.model.predict(self.F, return_std=True)
        return int(np.argmax(mu + sd * self.rng.standard_normal(len(mu))))


class RFPolicy:
    def __init__(self, features, rng):
        self.F, self.rng, self.X, self.y, self.model = features, rng, [], [], None
    def add(self, i, y): self.X.append(self.F[i]); self.y.append(y)
    def fit(self):
        self.model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                            max_features="sqrt", random_state=int(self.rng.integers(2**31))).fit(self.X, self.y)
    def predict(self): return self.model.predict_proba(self.F)[:, list(self.model.classes_).index(1)]
    def acquire(self):
        tree = self.model.estimators_[int(self.rng.integers(len(self.model.estimators_)))]
        prob = tree.predict_proba(self.F)
        if prob.shape[1] == 1: return int(self.rng.integers(len(self.F)))
        return int(np.argmax(prob[:, list(tree.classes_).index(1)] + 1e-8 * self.rng.standard_normal(len(self.F))))


def one_run(rep, args):
    base = np.random.default_rng(10000 + rep)
    Z = base.normal(size=(args.actions, args.dim))
    Z -= Z.mean(0); Z /= Z.std(0)
    Z *= args.radius / np.sqrt(args.dim)
    F = np.column_stack([np.ones(args.actions), Z])
    direction = base.normal(size=args.dim); direction /= np.linalg.norm(direction)
    theta_star = np.r_[0.0, args.alpha * direction / args.radius]
    p_true = sigmoid(F @ theta_star)
    p_star = float(p_true.max())
    warm_idx = base.integers(args.actions, size=args.warm)
    warm_u = base.random(args.warm)
    warm_y = (warm_u < p_true[warm_idx]).astype(int)
    if len(set(warm_y)) < 2:
        warm_y[0], warm_y[1] = 0, 1

    policies = []
    for j in range(5):
        rng = np.random.default_rng(rep * 1000 + 37 + j)
        if j == 0: pol = LaplacePolicy(F, args.lam, rng, posterior_scale=args.ts_scale)
        elif j == 1: pol = L2Policy(F, 1 / args.lam, args.epsilon, rng)
        elif j == 2: pol = LaplacePolicy(F, args.lam, rng, quadratic=True,
                                         posterior_scale=args.ts_scale)
        elif j == 3: pol = GPPolicy(Z, rng)
        else: pol = RFPolicy(Z, rng)
        for i, y in zip(warm_idx, warm_y): pol.add(int(i), int(y))
        pol.fit(); policies.append(pol)

    rows, cumulative = [], np.zeros(5)
    for t in range(1, args.rounds + 1):
        for j, pol in enumerate(policies):
            idx = pol.acquire()
            regret = p_star - p_true[idx]
            cumulative[j] += regret
            y = int(pol.rng.random() < p_true[idx])
            pol.add(idx, y)
            # GP refits less often, matching the discrete benchmark implementation.
            if j != 3 or t % args.gp_refit_every == 0: pol.fit()
            pred = pol.predict()
            recommendation_probability = float(p_true[int(np.argmax(pred))])
            pred_centered = pred - pred.mean()
            true_centered = p_true - p_true.mean()
            pred_normed = pred_centered / (np.linalg.norm(pred_centered) + 1e-12)
            true_normed = true_centered / (np.linalg.norm(true_centered) + 1e-12)
            function_cosine = float(np.dot(pred_normed, true_normed))
            function_norm_mse = float(np.mean((pred_normed - true_normed) ** 2))
            rows.append(dict(rep=rep, round=t, method=NAMES[j],
                             oracle_share=p_star,
                             probability_rmse=np.sqrt(np.mean((pred - p_true) ** 2)),
                             cos_sim=function_cosine, norm_mse=function_norm_mse,
                             recommendation_probability=recommendation_probability,
                             recommendation_simple_regret=p_star - recommendation_probability,
                             chosen_probability=p_true[idx], instantaneous_regret=regret,
                             cumulative_regret=cumulative[j],
                             average_cumulative_regret=cumulative[j] / t,
                             theta_cosine=(np.dot(pol.beta, theta_star) /
                                (np.linalg.norm(pol.beta) * np.linalg.norm(theta_star) + 1e-12)) if j == 0 else np.nan))
    return rows


def plot(df, out, args):
    # Sum within each replication before computing the mean and uncertainty.
    # Reconstruct from per-round losses to support existing metrics.csv files.
    df = df.sort_values(["method", "rep", "round"]).copy()
    df["cumulative_regret"] = df.groupby(["method", "rep"])["instantaneous_regret"].cumsum()
    groups = [
        ("discrete_five_method_convergence_regret", "Convergence and Regret Across Discrete Policies",
         [("cumulative_regret", "Cumulative Regret", r"$R_t = \sum_{s=1}^{t}(p^* - p_s)$", False),
          ("norm_mse", "Normalized MSE", "Normalized MSE", False)]),
        ("discrete_five_method_share", "Share and Oracle Share",
         [("chosen_probability", "Share and Oracle Share", "Share", True)]),
        ("discrete_five_method_cosine", "Cosine Similarity",
         [("cos_sim", "Cosine Similarity", "Cosine Similarity", False)]),
    ]
    for filename, title, metrics in groups:
        single = len(metrics) == 1
        fig, axes = plt.subplots(1, len(metrics), figsize=(7.5 if single else 12, 4.8), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (metric, panel_title, ylabel, smooth) in zip(axes, metrics):
            for name, color in zip(NAMES, COLORS):
                x = df[df.method == name].pivot(index="round", columns="rep", values=metric)
                if smooth:
                    x = x.rolling(15, center=True, min_periods=1).mean()
                mean = x.mean(1)
                se = x.std(1) / np.sqrt(x.shape[1])
                ax.plot(mean.index, mean, label=name, color=color, lw=2)
                ax.fill_between(mean.index, mean-se, mean+se, color=color, alpha=.12)
            if metric == "chosen_probability":
                oracle = df.pivot_table(index="round", columns="rep", values="oracle_share").mean(1)
                ax.plot(oracle.index, oracle, color="black", ls="--", lw=1.8,
                        label="Oracle share")
            if not single:
                ax.set_title(panel_title)
            ax.set_xlabel("Round"); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945),
                   ncol=2 if single else 3, frameon=False, fontsize=9)
        fig.suptitle(title, y=.992, fontsize=14)
        fig.subplots_adjust(left=.11 if single else .085, right=.985, bottom=.12, top=.74, wspace=.25)
        fig.savefig(out / f"{filename}.png", dpi=250)
        fig.savefig(out / f"{filename}.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=228)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--warm", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--ts_scale", type=float, default=.25)
    ap.add_argument("--epsilon", type=float, default=.1)
    ap.add_argument("--gp_refit_every", type=int, default=5)
    ap.add_argument("--output", default="results/pub_fig/discrete_five_method_validation")
    ap.add_argument("--plot-only", action="store_true",
                    help="Redraw the regret/MSE pair and separate share and cosine figures from existing metrics.csv")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(pd.read_csv(out / "metrics.csv"), out, args)
        return
    rows = []
    for rep in range(args.reps):
        print(f"rep {rep+1}/{args.reps}", flush=True)
        rows.extend(one_run(rep, args))
    df = pd.DataFrame(rows); df.to_csv(out / "metrics.csv", index=False)
    plot(df, out, args)
    summary = df[df["round"] > args.rounds - 50].groupby("method")[["chosen_probability", "oracle_share", "cos_sim", "instantaneous_regret", "norm_mse", "average_cumulative_regret", "probability_rmse"]].mean()
    summary.to_csv(out / "summary_last50.csv")
    print(summary.sort_values("average_cumulative_regret"))


if __name__ == "__main__":
    main()
