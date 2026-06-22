"""78 - lambda={16,100} diagnostic curves (5 seeds each), per-round trends.

Answers "why does large lambda raise the win-rate, even though it shrinks ||beta||?"
by plotting, per round t (averaged across the completed sims):

  (a) hard win-rate  P(ds < D_B)            raw + rolling-mean + linear fit
  (b) MODEL BELIEF vs TRUE EXPECTATION:
        predicted_p  = sigma(beta^T phi)     (dashed) -- surrogate's belief
        true_p_soft  = sigma(alpha(D_B-ds))  (solid)  -- the real soft success
      the GAP is the point: large lambda -> ||beta|| small -> predicted_p ~ 0.5
      (flat, under-confident) while true_p_soft still climbs -> the rise is driven
      by exploration/geometry, not by the surrogate sharpening.
  (c) Thompson covariance spectrum: cov_eig_max & cov_eig_min over rounds (log y).
      large lambda -> both shrink AND converge (max/min -> 1, isotropic) -> the
      per-round draw direction becomes near-random (wide isotropic exploration).
  (d) best-so-far ds_to_R (cumulative min), with the D_B threshold line.

Usage:  conda run -n diverse --no-capture-output python scripts/78_lam16_100_diag.py [STAMP]
        (STAMP defaults to the newest cmts_lam16_* dir)
"""
import glob, os, re, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D_B = 0.4704190492630005     # competitor (black,18) threshold; y=1 iff ds_to_R < D_B
WIN = 15                     # rolling window for smoothed curves
OUT = "outputs/vsweep_curves"
os.makedirs(OUT, exist_ok=True)

# -- resolve STAMP --
if len(sys.argv) > 1:
    STAMP = sys.argv[1]
else:
    cands = sorted(glob.glob("outputs/cmts_lam16_v4.0_d16_B8_T200_*"))
    if not cands:
        sys.exit("no cmts_lam16_* runs found; pass STAMP explicitly")
    STAMP = re.search(r'_T200_(\d{4}_\d{4})$', cands[-1]).group(1)
print(f"STAMP = {STAMP}")

LAMS = [("16", "tab:gray", "saturated baseline"),
        ("100", "tab:green", "de-saturated")]


def load_lam(lam):
    """Return per-round (averaged over sims) series for one lambda, or None."""
    d = f"outputs/cmts_lam{lam}_v4.0_d16_B8_T200_{STAMP}"
    files = sorted(glob.glob(f"{d}/sim*/trajectory.csv"))
    wr, soft, pred, emax, emin, best = [], [], [], [], [], []
    for f in files:
        m = pd.read_csv(f).query("phase == 'main'")
        if m.empty:
            continue
        wr.append(m.groupby("t")["y_hard"].mean())   # HARD winrate (soft y is the training label)
        soft.append(m.groupby("t")["true_p_soft"].mean())
        pred.append(m.groupby("t")["predicted_p"].mean())
        emax.append(m.groupby("t")["cov_eig_max"].first())
        emin.append(m.groupby("t")["cov_eig_min"].first())
        best.append(m.groupby("t")["ds_to_R"].min().cummin())
    if not wr:
        return None
    agg = lambda L: pd.concat(L, axis=1).mean(axis=1)
    return dict(n=len(wr), wr=agg(wr), soft=agg(soft), pred=agg(pred),
                emax=agg(emax), emin=agg(emin), best=agg(best))


fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
(axWR, axGAP), (axEIG, axBEST) = axes
rows = []
for lam, c, note in LAMS:
    R = load_lam(lam)
    if R is None:
        print(f"  lam={lam}: no data yet (skipping)")
        continue
    t = R["wr"].index.values.astype(float)
    lbl = f"$\\lambda$={lam} (n={R['n']}) — {note}"

    # (a) hard win-rate
    axWR.plot(t, R["wr"].values, color=c, alpha=0.18, lw=0.9)
    axWR.plot(t, R["wr"].rolling(WIN, min_periods=1, center=True).mean().values,
              color=c, lw=2.2, label=lbl)
    a, b = np.polyfit(t, R["wr"].values, 1)
    axWR.plot(t, a * t + b, color=c, lw=1.0, ls="--", alpha=0.8)

    # (b) model belief (dashed) vs true soft success (solid)
    axGAP.plot(t, R["soft"].rolling(WIN, min_periods=1, center=True).mean().values,
               color=c, lw=2.2, label=f"$\\lambda$={lam} true_p_soft")
    axGAP.plot(t, R["pred"].rolling(WIN, min_periods=1, center=True).mean().values,
               color=c, lw=1.6, ls="--", alpha=0.85,
               label=f"$\\lambda$={lam} predicted_p")

    # (c) covariance spectrum (log y)
    axEIG.plot(t, R["emax"].values, color=c, lw=2.2, label=f"$\\lambda$={lam} eig_max")
    axEIG.plot(t, R["emin"].values, color=c, lw=1.4, ls=":", label=f"$\\lambda$={lam} eig_min")

    # (d) best-so-far distance
    axBEST.plot(R["best"].index.values, R["best"].values, color=c, lw=2.2, label=f"$\\lambda$={lam}")

    rows.append((lam, R["n"], R["wr"].iloc[:20].mean(), R["wr"].iloc[-20:].mean(),
                 a * 100, R["soft"].iloc[:20].mean(), R["soft"].iloc[-20:].mean(),
                 R["pred"].iloc[-20:].mean(),
                 R["emax"].iloc[-20:].mean(), R["emin"].iloc[-20:].mean(),
                 float(R["best"].iloc[-1])))

axWR.set_title("(a) hard win-rate  $P(\\mathrm{ds}<D_B)$ per round")
axWR.set_xlabel("round $t$"); axWR.set_ylabel("win-rate (avg over sims)")
axWR.axhline(0.5, color="gray", ls=":", lw=1); axWR.grid(alpha=0.3)
axWR.legend(title="prior precision $\\lambda$", fontsize=9)

axGAP.set_title("(b) model belief vs true expectation\nsolid=true_p_soft  dashed=predicted_p")
axGAP.set_xlabel("round $t$"); axGAP.set_ylabel("probability")
axGAP.axhline(0.5, color="gray", ls=":", lw=1); axGAP.grid(alpha=0.3)
axGAP.legend(fontsize=8)

axEIG.set_title("(c) Thompson cov spectrum (log)\nlam big -> small & isotropic (eig_max~eig_min)")
axEIG.set_xlabel("round $t$"); axEIG.set_ylabel("eigenvalue of $cov=v^2 H^{-1}$")
axEIG.set_yscale("log"); axEIG.grid(alpha=0.3, which="both")
axEIG.legend(fontsize=8)

axBEST.set_title("(d) best-so-far ds-to-$R$ (lower=closer)")
axBEST.set_xlabel("round $t$"); axBEST.set_ylabel("best-so-far ds_to_R (avg over sims)")
axBEST.axhline(D_B, color="gray", ls=":", lw=1)
axBEST.annotate(f"$D_B$={D_B:.4f}", (0, D_B), fontsize=8, color="gray", va="bottom")
axBEST.grid(alpha=0.3); axBEST.legend(title="prior precision $\\lambda$", fontsize=9)

fig.suptitle(f"CM-TS $\\lambda$=16 vs 100 diagnostics — why large $\\lambda$ raises win-rate "
             f"(v=4.0, d=16, T=200, B=8, {STAMP})", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(OUT, f"lam16_100_diag_{STAMP}.png")
fig.savefig(png, dpi=140); plt.close(fig)

hdr = ("lam", "nSim", "wr_f20", "wr_l20", "wr_slope/100r",
       "soft_f20", "soft_l20", "pred_l20", "eigmax_l20", "eigmin_l20", "best_ds")
print("\n" + " ".join(f"{h:>11}" for h in hdr))
for r in rows:
    print(f"{r[0]:>11} {r[1]:>11} {r[2]:>11.3f} {r[3]:>11.3f} {r[4]:>+11.3f} "
          f"{r[5]:>11.3f} {r[6]:>11.3f} {r[7]:>11.3f} {r[8]:>11.2e} {r[9]:>11.2e} {r[10]:>11.4f}")
print(f"\nsaved: {png}")
