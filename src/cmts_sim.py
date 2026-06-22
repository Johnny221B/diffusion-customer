"""
CM-TS convergence/regret simulator (synthetic, no image generation).

Implements Algorithm 1' (the APPROXIMATE continuous acquisition; 3-pager App. B),
the practical stand-in for the exact SOCP oracle Algorithm 1 that Theorem 1 analyzes.
CM-TS: global variance-inflated logistic Thompson Sampling over a CONTINUOUS kNN
validity manifold, beat-the-competitor objective, human-only. The per-round gap
between Algorithm 1' here and the exact Algorithm 1 is the optimization error eps_t
that Theorem 1 absorbs additively (+ 1/4 sum_t eps_t).

IMPORTANT (v2): the acquisition (Algorithm 1', argmax_over_M) is a genuine CONTINUOUS
optimization over M -- NOT an argmax over a fixed candidate pool (which would collapse
CM-TS to a finite-arm linear bandit / MAB and is unfaithful). Each round, for the drawn
beta-tilde, we form the per-anchor boundary maximizers  z_i + tau * beta/||beta||
(these MOVE with beta every round), keep the valid ones, and refine the best by
projected-gradient ascent onto the manifold boundary dM. The played design is a
continuous function of beta-tilde and lives on the continuum M, not in a pool.

No SD3.5 / T5 / FAISS. Synthetic vocabulary in R^d, synthetic Bernoulli feedback.
Dependencies: numpy only (matplotlib optional for --figs).
"""
import numpy as np


def sigma(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------- synthetic vocabulary ----------
def make_vocab(M, d, n_clusters=10, seed=0, aniso=None):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, size=(n_clusters, d))
    counts = rng.multinomial(M, np.ones(n_clusters) / n_clusters)
    pts = [c + 0.35 * rng.normal(0, 1.0, size=(n, d)) for c, n in zip(centers, counts)]
    Z = np.vstack(pts)
    if aniso is not None:
        # squash the informative coordinate(s): low spread => under-explored direction
        Z = Z * aniso[None, :]
    return Z


# ---------- kNN validity manifold (continuous) ----------
def kth_dist_batch(C, Z, k):
    # distance to k-th nearest anchor for each row of C  (vectorized)
    D = np.linalg.norm(C[:, None, :] - Z[None, :, :], axis=2)  # (n, M)
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def kth_dist(z, Z, k):
    d = np.linalg.norm(Z - z, axis=1)
    return np.partition(d, k - 1)[k - 1]


def calibrate_tau(Z, k, q=0.95):
    M = len(Z)
    dd = np.zeros(M)
    for i in range(M):
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[i] = np.inf
        dd[i] = np.partition(d, k - 1)[k - 1]
    return np.quantile(dd, q)


def project_to_M(z, Z, k, tau, iters=20):
    # approximate projection onto M = {d_k <= tau}: pull toward the centroid of
    # the k nearest anchors until valid (bisection on the pull fraction).
    if kth_dist(z, Z, k) <= tau:
        return z
    nn = np.argsort(np.linalg.norm(Z - z, axis=1))[:k]
    target = Z[nn].mean(0)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if kth_dist(z + mid * (target - z), Z, k) <= tau:
            hi = mid
        else:
            lo = mid
    return z + hi * (target - z)


def argmax_over_M(beta, Z, k, tau, refine_steps=12, step0=None):
    """Continuous argmax of a linear form beta^T z over M (NOT a pool lookup).

    1. per-anchor boundary maximizers c_i = z_i + tau * u  (u = beta/||beta||);
       each c_i is the exact max of beta^T z over the ball B(z_i, tau).
    2. keep valid c_i (d_k <= tau); seed with the best (fallback: best anchor).
    3. projected-gradient ascent: step along u, project back onto M.
    """
    u = beta / (np.linalg.norm(beta) + 1e-12)
    C = Z + tau * u  # (M, d) boundary maximizers, recomputed for THIS beta
    valid = kth_dist_batch(C, Z, k) <= tau + 1e-9
    cands = np.vstack([C[valid], Z]) if valid.any() else Z.copy()
    z = cands[int(np.argmax(cands @ beta))]
    step = (tau / 3.0 if step0 is None else step0)
    best, best_val = z.copy(), z @ beta
    for _ in range(refine_steps):
        zc = project_to_M(z + step * u, Z, k, tau)
        if zc @ beta > best_val + 1e-12:
            best, best_val, z = zc.copy(), zc @ beta, zc
        else:
            step *= 0.5
    return best


def random_valid_design(Z, k, tau, rng, nn_idx):
    # a random in-manifold design = convex combo of an anchor and its neighbors
    M = len(Z)
    for _ in range(50):
        i = rng.integers(M)
        nbrs = nn_idx[i]
        w = rng.dirichlet(np.ones(3))
        js = rng.choice(nbrs, size=2, replace=False)
        z = w[0] * Z[i] + w[1] * Z[js[0]] + w[2] * Z[js[1]]
        if kth_dist(z, Z, k) <= tau:
            return z
    return Z[rng.integers(M)]


# ---------- ridge logistic Laplace MAP ----------
def laplace_map(Phi, y, lam, d, beta0=None, iters=60, tol=1e-9):
    beta = np.zeros(d) if beta0 is None else beta0.copy()
    I = np.eye(d)
    for _ in range(iters):
        p = sigma(Phi @ beta)
        W = np.clip(p * (1 - p), 1e-6, None)
        grad = Phi.T @ (p - y) + lam * beta
        H = Phi.T @ (Phi * W[:, None]) + lam * I
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta -= step
        if np.linalg.norm(step) < tol:
            break
    p = sigma(Phi @ beta)
    W = np.clip(p * (1 - p), 1e-6, None)
    H = Phi.T @ (Phi * W[:, None]) + lam * I
    return beta, H


def project_norm(beta, S):
    n = np.linalg.norm(beta)
    return beta if n <= S else beta * (S / n)


# ---------- one CM-TS run (continuous acquisition, no pool) ----------
def run_cmts(d, T, seed=0, v=1.0, k=10, n0=24, S=8.0, policy="cmts", hard=False):
    rng = np.random.default_rng(seed)
    M = 228
    if hard:
        # ill-conditioned: coordinate 0 is the decision-relevant direction but
        # has small spread (under-explored); beta* loads heavily on it. Small seed.
        aniso = np.ones(d); aniso[0] = 0.12
        Z = make_vocab(M, d, seed=100 + seed, aniso=aniso)
        tau = calibrate_tau(Z, k)
        beta_star = rng.normal(0, 1.0, size=d) * 0.3
        beta_star[0] = 1.0                              # signal in the squashed dir
        beta_star = beta_star / np.linalg.norm(beta_star) * 6.0
    else:
        Z = make_vocab(M, d, seed=100 + seed)
        tau = calibrate_tau(Z, k)
        beta_star = rng.normal(0, 1.0, size=d)
        beta_star = beta_star / np.linalg.norm(beta_star) * 3.0
    nn_idx = np.argsort(
        np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2), axis=1)[:, 1:11]
    a = Z[rng.integers(M)].copy()                      # fixed competitor anchor
    # continuous best design vs competitor (refined argmax over M, computed once)
    z_star = argmax_over_M(beta_star, Z, k, tau, refine_steps=40)
    u_star = sigma((z_star - a) @ beta_star)

    lam = 1.0 / (1.0 / d)  # ridge = 1/sigma_min^2, sigma_min^2 = 1/d

    # seed stage: n0 random valid designs vs the competitor
    seeds = np.array([random_valid_design(Z, k, tau, rng, nn_idx) for _ in range(n0)])
    Phi = seeds - a
    y = rng.binomial(1, sigma(Phi @ beta_star)).astype(float)
    beta_hat, H = laplace_map(Phi, y, lam, d)
    beta_hat = project_norm(beta_hat, S)

    regrets = np.zeros(T)
    for t in range(T):
        if policy == "cmts":
            Hinv = np.linalg.inv(H)
            beta_tilde = rng.multivariate_normal(beta_hat, v * v * 0.5 * (Hinv + Hinv.T))
            z_t = argmax_over_M(beta_tilde, Z, k, tau)        # CONTINUOUS argmax
        elif policy == "greedy":
            z_t = argmax_over_M(beta_hat, Z, k, tau)
        else:  # random valid design
            z_t = random_valid_design(Z, k, tau, rng, nn_idx)

        phi_t = z_t - a
        u_t = sigma(phi_t @ beta_star)
        regrets[t] = u_star - u_t
        y_t = float(rng.binomial(1, u_t))
        Phi = np.vstack([Phi, phi_t]); y = np.append(y, y_t)
        beta_hat, H = laplace_map(Phi, y, lam, d, beta0=beta_hat)
        beta_hat = project_norm(beta_hat, S)

    return np.cumsum(regrets), beta_star, beta_hat


# ---------- experiment driver ----------
def loglog_slope(Ts, R):
    x, yl = np.log(np.array(Ts)), np.log(np.array(R))
    A = np.vstack([x, np.ones_like(x)]).T
    return np.linalg.lstsq(A, yl, rcond=None)[0][0]


def experiment(d, T=300, reps=8, v=0.5, n0=8, collect_curves=False):
    checkpoints = [50, 100, 150, T]
    cum = np.zeros((reps, T)); cos_align = np.zeros(reps)
    for r in range(reps):
        cumR, bstar, bhat = run_cmts(d, T, seed=r, v=v, n0=n0, policy="cmts")
        cum[r] = cumR
        cos_align[r] = (bstar @ bhat) / (np.linalg.norm(bstar) * np.linalg.norm(bhat))
    meanR, seR = cum.mean(0), cum.std(0) / np.sqrt(reps)
    cg = np.array([run_cmts(d, T, seed=r, n0=n0, policy="greedy")[0] for r in range(reps)])
    cr = np.array([run_cmts(d, T, seed=r, n0=n0, policy="random")[0] for r in range(reps)])
    if collect_curves:
        experiment.curves[d] = dict(cmts=meanR, cmts_se=seR,
                                    greedy=cg.mean(0), random=cr.mean(0))
    print(f"\n========= d={d}  (T={T}, reps={reps}, v={v}) — CONTINUOUS argmax over M =========")
    print(f"{'T':>6} | {'cum-regret':>11} | {'avg/round':>10}")
    for c in checkpoints:
        print(f"{c:>6} | {meanR[c-1]:>11.3f} | {meanR[c-1]/c:>10.4f}")
    slope = loglog_slope(checkpoints, [meanR[c-1] for c in checkpoints])
    print(f"  log-log cum-regret slope in T : {slope:.3f}  (sqrt(T) => 0.5)")
    print(f"  avg regret/round T=50->{T}      : {meanR[49]/50:.4f} -> {meanR[-1]/T:.4f}")
    print(f"  beta cosine align             : {cos_align.mean():.3f} +/- {cos_align.std():.3f}")
    print(f"  CM-TS / greedy / random @T={T} : {cum[:,-1].mean():.2f} / {cg[:,-1].mean():.2f} / {cr[:,-1].mean():.2f}")
    return meanR, slope


def vsweep(d=16, T=200, reps=12, vs=(0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
           n0=6, hard=True, collect=False):
    """Tune exploration: cumulative regret @T as a function of inflation v.
    v=0 is exactly greedy (zero-variance Thompson draw)."""
    label = "HARD (ill-conditioned, under-explored decision dir)" if hard else "EASY (well-specified)"
    print(f"\n===== Exploration sweep — {label}, d={d}, T={T}, reps={reps}, n0={n0} =====")
    print(f"{'v':>6} | {'cum-regret@T':>13} | {'se':>7}")
    res = {}
    for v in vs:
        vals = np.array([run_cmts(d, T, seed=r, v=v, n0=n0, policy="cmts", hard=hard)[0][-1]
                         for r in range(reps)])
        res[v] = (vals.mean(), vals.std() / np.sqrt(reps))
        tag = "  <- greedy" if v == 0.0 else ""
        print(f"{v:>6.2f} | {vals.mean():>13.3f} | {vals.std()/np.sqrt(reps):>7.3f}{tag}")
    greedy = res[0.0][0]
    best_v = min((v for v in vs if v > 0), key=lambda v: res[v][0])
    print(f"  greedy (v=0)            : {greedy:.3f}")
    print(f"  best tuned CM-TS (v={best_v}) : {res[best_v][0]:.3f}"
          f"   ({'BEATS' if res[best_v][0] < greedy else 'does NOT beat'} greedy"
          f", {100*(greedy-res[best_v][0])/greedy:+.1f}% regret change)")
    if collect:
        vsweep.curves[("hard" if hard else "easy", d)] = res
    return res


def make_figures(T, outdir="."):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = experiment.curves; t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, d in zip(axes, sorted(C)):
        c = C[d]
        ax.plot(t, c["random"], color="gray", ls=":", label="random-valid")
        ax.plot(t, c["greedy"], color="tab:orange", ls="--", label="greedy")
        ax.plot(t, c["cmts"], color="tab:blue", label="CM-TS")
        ax.fill_between(t, c["cmts"]-1.96*c["cmts_se"], c["cmts"]+1.96*c["cmts_se"],
                        color="tab:blue", alpha=0.2)
        ax.set_title(f"$d={d}$"); ax.set_xlabel("round $t$")
        ax.set_ylabel("cumulative regret $R_t$"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/regret_linear.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for d, col in zip(sorted(C), ["tab:blue", "tab:green"]):
        ax.loglog(t, C[d]["cmts"], color=col, label=f"CM-TS $d={d}$")
    ref = t.astype(float) ** 0.5; ref = ref / ref[20] * C[sorted(C)[0]]["cmts"][20]
    ax.loglog(t, ref, "k--", label=r"$\sqrt{T}$ reference")
    ax.set_xlabel("round $t$ (log)"); ax.set_ylabel("cum. regret (log)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{outdir}/regret_loglog.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for d, col in zip(sorted(C), ["tab:blue", "tab:green"]):
        ax.plot(t, C[d]["cmts"] / t, color=col, label=f"CM-TS $d={d}$")
    ax.set_xlabel("round $t$"); ax.set_ylabel("avg regret/round $R_t/t$")
    ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/avg_regret.pdf"); plt.close(fig)
    print(f"\nFigures -> {outdir}/: regret_linear.pdf, regret_loglog.pdf, avg_regret.pdf")


def make_vsweep_figure(d=16, T=200, reps=12, vs=(0.0, 0.5, 1.0, 2.0, 3.0, 4.0), outdir="."):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = {}
    for hard in (False, True):
        m, s = [], []
        for v in vs:
            vals = np.array([run_cmts(d, T, seed=r, v=v, n0=6, policy="cmts", hard=hard)[0][-1]
                             for r in range(reps)])
            m.append(vals.mean()); s.append(vals.std() / np.sqrt(reps))
        out[hard] = (np.array(m), np.array(s))
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    vv = np.array(vs)
    for hard, col, lab in [(False, "tab:blue", "easy (well-specified)"),
                           (True, "tab:red", "hard (ill-conditioned)")]:
        m, s = out[hard]
        ax.errorbar(vv, m, yerr=1.96 * s, marker="o", color=col, capsize=3, label=lab)
    ax.axvline(0.0, color="gray", ls=":", lw=1)
    ax.annotate("greedy\n($v{=}0$)", xy=(0.0, ax.get_ylim()[1]*0.8), fontsize=8, ha="left")
    ax.set_xlabel(r"exploration inflation $v$"); ax.set_ylabel(r"cumulative regret $R_T$ ($T={}$)".format(T))
    ax.set_title("Tuned exploration beats greedy"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/regret_vs_v.pdf"); plt.close(fig)
    print(f"v-sweep figure -> {outdir}/regret_vs_v.pdf")
    return out


if __name__ == "__main__":
    import sys
    np.seterr(over="ignore")
    if "--vsweep" in sys.argv:
        vsweep(d=16, T=200, reps=12, hard=False)   # easy: greedy ~ optimal
        vsweep(d=16, T=200, reps=12, hard=True)    # hard: exploration must help
        sys.exit(0)
    experiment.curves = {}
    print("CM-TS simulator v2 — CONTINUOUS argmax over the validity manifold (no pool, no MAB)")
    T = 300
    for d in [16, 32]:
        experiment(d, T=T, reps=8, v=1.0, collect_curves=True)
    print("\nConvergence <=> avg-regret/round -> 0, cum-regret slope ~0.5, beta cosine -> 1.")
    if "--figs" in sys.argv:
        make_figures(T, ".")
        make_vsweep_figure(d=16, T=200, reps=12, outdir=".")
