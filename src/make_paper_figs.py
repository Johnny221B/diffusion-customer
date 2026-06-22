"""
Comprehensive figure generator for the CM-TS 3-pager appendix.

SIM-phase (2026-05-29) rewrite: every number in Appendix C is reproduced from ONE
clean run with a SINGLE seed count (NSEED), and every reported quantity is derived
from SHARED arrays so that numbers appearing in more than one place are identical by
construction (no cross-figure inconsistency). The acquisition simulated is
Algorithm 1' (the approximate continuous oracle; see cmts_sim.run_cmts / argmax_over_M).

Produces 7 figures:
  1 regret_linear.pdf   cum regret vs t (CM-TS v=0.4 / greedy v=0 / random), +/-1 std, d=16 & 32
  2 regret_loglog.pdf   log-log cum regret + sqrt(T) reference
  3 avg_regret.pdf      average regret/round -> 0 (no-regret signature)
  4 regret_vs_v.pdf     exploration tuning: cum R_T vs v (easy & hard); v=0 is greedy
  5 regret_cdf.pdf      empirical CDF of final R_T over seeds: greedy tail vs CM-TS
  6 regret_vs_dim.pdf   cum R_T vs d (log-log): d^{3/2} dimension dependence
  7 seed_trajectories.pdf  per-seed R_t paths: greedy blow-ups vs bounded CM-TS (hard)

Shared-array guarantees (kill the C3 cross-figure inconsistencies):
  * greedy == the v=0 column of the v-sweep (same array), for both easy and hard.
  * CM-TS(default) == the v=0.4 column of the v-sweep (same array).
  * the hard-instance table, the CDF, and the trajectory figure all read the SAME
    hard greedy (v=0) and hard CM-TS (v=0.4) (NSEED,T) arrays.
  * the dimension-scaling d=16 / d=32 points reuse the easy-CM-TS means.
  * ONE seed count (NSEED) everywhere -> every caption seed-count is identical.

Writes results.txt with EVERY number used in the .tex captions/table; copy each
.tex number from that file (do not hand-edit numbers).
"""
import numpy as np
import cmts_sim as S

np.seterr(over="ignore")
NSEED, N0, T, VT = 40, 6, 200, 0.4          # single seed count, seed pairs, horizon, default inflation
V_GRID = [0.0, 0.25, 0.4, 0.5, 0.65, 0.8, 1.0]   # v=0 is greedy; VT=0.4 is the default CM-TS
DIMS = [8, 16, 32, 48]
CACHE = "figdata.npz"


def runs(d, v, reps, hard=False, policy="cmts"):
    """(reps, T) cumulative-regret curves at fixed (d, v, instance)."""
    out = np.zeros((reps, T))
    for r in range(reps):
        out[r] = S.run_cmts(d, T, seed=r, v=v, n0=N0, policy=policy, hard=hard)[0]
    return out


def compute():
    data = {}
    # ---- d=16 easy v-sweep: full (NSEED,T) curves for every v (v=0 greedy, 0.4 default) ----
    for v in V_GRID:
        data[f"easy16_v{v}"] = runs(16, v, NSEED, hard=False)
    # ---- d=16 hard v-sweep: full curves for every v ----
    for v in V_GRID:
        data[f"hard16_v{v}"] = runs(16, v, NSEED, hard=True)
    # ---- d=16 easy random baseline ----
    data["easy16_random"] = runs(16, VT, NSEED, hard=False, policy="random")
    # ---- d=32 easy: cmts (v=0.4), greedy (v=0), random ----
    data["easy32_cmts"] = runs(32, VT, NSEED, hard=False)
    data["easy32_greedy"] = runs(32, 0.0, NSEED, hard=False)
    data["easy32_random"] = runs(32, VT, NSEED, hard=False, policy="random")
    # ---- dimension scaling (CM-TS default v): d=8,48 fresh; d=16,32 reuse easy cmts ----
    data["dim8"] = runs(8, VT, NSEED, hard=False)
    data["dim48"] = runs(48, VT, NSEED, hard=False)
    np.savez(CACHE, vgrid=np.array(V_GRID), dims=np.array(DIMS), nseed=NSEED, T=T, **data)
    return data


def loglog_slope(curve_mean, cps=(50, 100, 150, 200)):
    x = np.log(np.array(cps)); y = np.log(np.array([curve_mean[c - 1] for c in cps]))
    return float(np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0][0])


def band(ax, t, A, color, label):
    m = A.mean(0); s = A.std(0)
    ax.plot(t, m, color=color, label=label)
    ax.fill_between(t, m - s, m + s, color=color, alpha=0.18)


def figures(data):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = np.arange(1, T + 1)
    R = {}  # the results dict copied into the .tex

    # shared arrays (single source of truth)
    cmts16, greedy16, rand16 = data["easy16_v0.4"], data["easy16_v0.0"], data["easy16_random"]
    cmts32, greedy32, rand32 = data["easy32_cmts"], data["easy32_greedy"], data["easy32_random"]
    hard_greedy, hard_cmts = data["hard16_v0.0"], data["hard16_v0.4"]

    # ---------- 1 regret_linear ----------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, d, c, g, rd in [(axes[0], 16, cmts16, greedy16, rand16),
                            (axes[1], 32, cmts32, greedy32, rand32)]:
        band(ax, t, rd, "gray", "random-valid")
        band(ax, t, g, "tab:orange", "greedy ($v{=}0$)")
        band(ax, t, c, "tab:blue", "CM-TS ($v{=}0.4$)")
        ax.set_title(f"$d={d}$"); ax.set_xlabel("round $t$")
        ax.set_ylabel("cumulative regret $R_t$"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("regret_linear.pdf"); plt.close(fig)

    R["main_cmts16"] = cmts16[:, -1].mean();  R["main_greedy16"] = greedy16[:, -1].mean()
    R["main_random16"] = rand16[:, -1].mean()
    R["main_cmts32"] = cmts32[:, -1].mean();  R["main_greedy32"] = greedy32[:, -1].mean()
    R["main_random32"] = rand32[:, -1].mean()
    R["mult_rand_over_cmts16"] = R["main_random16"] / R["main_cmts16"]
    R["mult_rand_over_cmts32"] = R["main_random32"] / R["main_cmts32"]
    R["mult_greedy_over_cmts16"] = R["main_greedy16"] / R["main_cmts16"]
    R["mult_greedy_over_cmts32"] = R["main_greedy32"] / R["main_cmts32"]

    # ---------- 2 regret_loglog ----------
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for d, m, col in [(16, cmts16.mean(0), "tab:blue"), (32, cmts32.mean(0), "tab:green")]:
        ax.loglog(t, m, color=col, label=f"CM-TS $d={d}$")
    ref = t.astype(float) ** 0.5; ref = ref / ref[20] * cmts16.mean(0)[20]
    ax.loglog(t, ref, "k--", label=r"$\sqrt{T}$ reference")
    ax.set_xlabel("round $t$ (log)"); ax.set_ylabel("cum. regret (log)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("regret_loglog.pdf"); plt.close(fig)
    R["Tslope16"] = loglog_slope(cmts16.mean(0));  R["Tslope32"] = loglog_slope(cmts32.mean(0))

    # ---------- 3 avg_regret ----------
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for d, m, col in [(16, cmts16.mean(0), "tab:blue"), (32, cmts32.mean(0), "tab:green")]:
        ax.plot(t, m / t, color=col, label=f"CM-TS $d={d}$")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("round $t$")
    ax.set_ylabel("avg regret/round $R_t/t$"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("avg_regret.pdf"); plt.close(fig)
    R["avg16_t50"] = cmts16.mean(0)[49] / 50;  R["avg16_t200"] = cmts16.mean(0)[-1] / 200
    R["avg32_t50"] = cmts32.mean(0)[49] / 50;  R["avg32_t200"] = cmts32.mean(0)[-1] / 200

    # ---------- 4 regret_vs_v (tuning) ----------
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    vv = np.array(V_GRID)
    for inst, col, lab in [("easy16", "tab:blue", "easy (well-specified)"),
                           ("hard16", "tab:red", "hard (ill-conditioned)")]:
        m = np.array([data[f"{inst}_v{v}"][:, -1].mean() for v in V_GRID])
        se = np.array([data[f"{inst}_v{v}"][:, -1].std() / np.sqrt(NSEED) for v in V_GRID])
        ax.errorbar(vv, m, yerr=1.96 * se, marker="o", color=col, capsize=3, label=lab)
        gz = m[0]                                    # v=0 == greedy (same array)
        ibest = 1 + int(np.argmin(m[1:])); vbest, mbest = V_GRID[ibest], m[ibest]
        R[f"tune_{inst}_greedy"] = gz; R[f"tune_{inst}_best"] = mbest; R[f"tune_{inst}_bestv"] = vbest
    ax.axvline(0.0, color="gray", ls=":", lw=1)
    ax.text(0.02, 0.95, "greedy = $v{=}0$", transform=ax.transAxes, fontsize=8, va="top")
    ax.set_xlabel(r"exploration inflation $v$"); ax.set_ylabel(rf"cumulative regret $R_{{{T}}}$")
    ax.set_title("Tuned exploration beats greedy"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("regret_vs_v.pdf"); plt.close(fig)

    # ---------- 5 regret_cdf (hard instance; SAME arrays as the table) ----------
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for arr, col, lab in [(hard_greedy, "tab:orange", "greedy ($v{=}0$)"),
                          (hard_cmts, "tab:blue", "CM-TS ($v{=}0.4$)")]:
        fr = np.sort(arr[:, -1]); cdf = np.arange(1, len(fr) + 1) / len(fr)
        ax.step(fr, cdf, color=col, where="post", label=lab)
    ax.set_xlabel(rf"final regret $R_{{{T}}}$ (hard instance)")
    ax.set_ylabel("empirical CDF over replications"); ax.legend(fontsize=8)
    ax.set_title("Greedy's catastrophic tail vs CM-TS robustness")
    fig.tight_layout(); fig.savefig("regret_cdf.pdf"); plt.close(fig)
    # hard-instance table + cdf numbers (one source: hard_greedy / hard_cmts final column)
    for tag, arr in [("greedy", hard_greedy), ("cmts", hard_cmts)]:
        fin = arr[:, -1]
        R[f"hard_{tag}_mean"] = fin.mean(); R[f"hard_{tag}_p90"] = np.percentile(fin, 90)
        R[f"hard_{tag}_max"] = fin.max()

    # ---------- 6 regret_vs_dim ----------
    dim_means = {8: data["dim8"][:, -1].mean(), 16: cmts16[:, -1].mean(),
                 32: cmts32[:, -1].mean(), 48: data["dim48"][:, -1].mean()}
    dims = np.array(DIMS, float); m = np.array([dim_means[d] for d in DIMS])
    se = np.array([data["dim8"][:, -1].std(), cmts16[:, -1].std(), cmts32[:, -1].std(),
                   data["dim48"][:, -1].std()]) / np.sqrt(NSEED)
    A = np.vstack([np.log(dims), np.ones_like(dims)]).T
    R["dslope"] = float(np.linalg.lstsq(A, np.log(m), rcond=None)[0][0])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.errorbar(dims, m, yerr=1.96 * se, marker="o", color="tab:purple", capsize=3, label="CM-TS")
    ref = dims ** 1.5; ref = ref / ref[1] * m[1]
    ax.plot(dims, ref, "k--", label=r"$d^{3/2}$ worst-case slope")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dimension $d$ (log)"); ax.set_ylabel(rf"cumulative regret $R_{{{T}}}$ (log)")
    ax.set_title(rf"$R_T$ vs $d$: slope {R['dslope']:.2f} $\ll$ $d^{{3/2}}$", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("regret_vs_dim.pdf", bbox_inches="tight"); plt.close(fig)

    # ---------- 7 seed_trajectories (hard; SAME arrays as table/cdf) ----------
    # displayed full-width in the paper; fonts sized to stay legible at that scale
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    for ax, arr, lab, col in [(axes[0], hard_greedy, "greedy ($v{=}0$)", "tab:orange"),
                              (axes[1], hard_cmts, "CM-TS ($v{=}0.4$)", "tab:blue")]:
        for r in range(min(25, len(arr))):
            ax.plot(t, arr[r], color=col, alpha=0.30, lw=0.7)
        ax.plot(t, arr.mean(0), color="k", lw=2.4, label="mean over replications")
        ax.set_title(lab, fontsize=14)
        ax.set_xlabel("round $t$", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=11, loc="upper left")
    axes[0].set_ylabel(r"cumulative regret $R_t$ (per replication)", fontsize=12)
    fig.suptitle("Hard instance: greedy blows up on a fraction of replications; CM-TS stays bounded", fontsize=13)
    fig.tight_layout(); fig.savefig("seed_trajectories.pdf", bbox_inches="tight"); plt.close(fig)

    # ---------- results dict ----------
    R["NSEED"] = NSEED; R["T"] = T; R["N0"] = N0; R["VT"] = VT
    with open("results.txt", "w") as f:
        f.write(f"# CM-TS Appendix-C numbers — ONE run, NSEED={NSEED}, T={T}, n0={N0}, v(default)={VT}\n")
        for k in sorted(R):
            f.write(f"{k} = {float(R[k]):.4f}\n")
    print("results: (copy each .tex number from here)")
    for k in sorted(R):
        print(f"  {k} = {float(R[k]):.4f}")
    # explicit tie-out / reconciliation assertions (printed for the changelog)
    print("\ntie-outs:")
    print(f"  avg16_t200*T = {R['avg16_t200']*T:.3f}  vs main_cmts16 = {R['main_cmts16']:.3f}")
    print(f"  avg32_t200*T = {R['avg32_t200']*T:.3f}  vs main_cmts32 = {R['main_cmts32']:.3f}")
    print(f"  tune_hard16_greedy = {R['tune_hard16_greedy']:.3f}  vs hard_greedy_mean = {R['hard_greedy_mean']:.3f}  (must match: same v=0 array)")
    print(f"  tune_hard16_best(v={R['tune_hard16_bestv']}) = {R['tune_hard16_best']:.3f}  vs hard_cmts_mean = {R['hard_cmts_mean']:.3f}  (match iff best v=0.4)")
    print(f"  tune_easy16_greedy = {R['tune_easy16_greedy']:.3f}  vs main_greedy16 = {R['main_greedy16']:.3f}  (must match: same v=0 array)")
    print(f"  Tslope16 = {R['Tslope16']:.3f} ; Tslope32 = {R['Tslope32']:.3f} ; dslope = {R['dslope']:.3f}  (distinct regressions)")
    print("figures written: regret_linear, regret_loglog, avg_regret, regret_vs_v, regret_cdf, regret_vs_dim, seed_trajectories")
    return R


if __name__ == "__main__":
    import os, sys
    if "--fresh" in sys.argv and os.path.exists(CACHE):
        os.remove(CACHE)
    if os.path.exists(CACHE):
        print("loading cache", CACHE)
        data = dict(np.load(CACHE))
    else:
        print(f"computing one clean run (NSEED={NSEED}); ~10-15 min...")
        data = compute()
    figures(data)
