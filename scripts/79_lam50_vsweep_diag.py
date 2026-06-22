"""79 - lambda=50 FIXED, v={4,12} sweep diagnostic curves (5 seeds each).

"广度 vs 方向" probe (see scripts/73l_lam50_vsweep.sh). lam=50 keeps beta de-saturated
(direction alive, like lam100); v scales the Thompson cov radius (coverage, like lam16's
breadth). Question: does v=12 (3x radius) recover lam16's coverage WITHOUT losing lam50's
direction -> does best-so-far go below lam100's 0.27 and does true_p_soft finally show slope?

Per round t (averaged across completed sims):
  (a) hard win-rate  P(ds < D_B)            raw + rolling-mean + linear fit
  (b) MODEL BELIEF vs TRUE EXPECTATION:
        predicted_p  = sigma(beta^T phi)     (dashed) -- surrogate's belief
        true_p_soft  = sigma(alpha(D_B-ds))  (solid)  -- real soft success (alpha=30 here)
  (c) Thompson covariance spectrum: cov_eig_max & cov_eig_min over rounds (log y).
      bigger v -> larger eigenvalues (wider draw). Watch eig_max/eig_min (anisotropy).
  (d) best-so-far ds_to_R (cumulative min), with the D_B threshold line.
      The money panel: did wider exploration push best below lam100's 0.27?

Usage:  conda run -n diverse --no-capture-output python scripts/79_lam50_vsweep_diag.py [STAMP]
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

if len(sys.argv) > 1:
    STAMP = sys.argv[1]
else:
    cands = sorted(glob.glob("outputs/cmts_lam50_v4.0_d16_B8_T200_*"))
    if not cands:
        sys.exit("no cmts_lam50_v4.0_* runs found; pass STAMP explicitly")
    STAMP = re.search(r'_T200_(\d{4}_\d{4})$', cands[-1]).group(1)
print(f"STAMP = {STAMP}")

# (v_tag, color, note)
CONFIGS = [("v4.0", "tab:blue",   "narrow (baseline radius)"),
           ("v12.0", "tab:orange", "wide (3x radius)")]


def load_cfg(v):
    """Return per-round (averaged over sims) series for one v, or None."""
    d = f"outputs/cmts_lam50_{v}_d16_B8_T200_{STAMP}"
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
for v, c, note in CONFIGS:
    R = load_cfg(v)
    if R is None:
        print(f"  {v}: no data yet (skipping)")
        continue
    t = R["wr"].index.values.astype(float)
    lbl = f"{v} (n={R['n']}) — {note}"

    axWR.plot(t, R["wr"].values, color=c, alpha=0.18, lw=0.9)
    axWR.plot(t, R["wr"].rolling(WIN, min_periods=1, center=True).mean().values,
              color=c, lw=2.2, label=lbl)
    a, b = np.polyfit(t, R["wr"].values, 1)
    axWR.plot(t, a * t + b, color=c, lw=1.0, ls="--", alpha=0.8)

    axGAP.plot(t, R["soft"].rolling(WIN, min_periods=1, center=True).mean().values,
               color=c, lw=2.2, label=f"{v} true_p_soft")
    axGAP.plot(t, R["pred"].rolling(WIN, min_periods=1, center=True).mean().values,
               color=c, lw=1.6, ls="--", alpha=0.85, label=f"{v} predicted_p")

    axEIG.plot(t, R["emax"].values, color=c, lw=2.2, label=f"{v} eig_max")
    axEIG.plot(t, R["emin"].values, color=c, lw=1.4, ls=":", label=f"{v} eig_min")

    axBEST.plot(R["best"].index.values, R["best"].values, color=c, lw=2.2, label=v)

    rows.append((v, R["n"], R["wr"].iloc[:20].mean(), R["wr"].iloc[-20:].mean(),
                 a * 100, R["soft"].iloc[:20].mean(), R["soft"].iloc[-20:].mean(),
                 R["pred"].iloc[-20:].mean(),
                 R["emax"].iloc[-20:].mean(), R["emin"].iloc[-20:].mean(),
                 float(R["best"].iloc[-1])))

axWR.set_title("(a) hard win-rate  $P(\\mathrm{ds}<D_B)$ per round")
axWR.set_xlabel("round $t$"); axWR.set_ylabel("win-rate (avg over sims)")
axWR.axhline(0.5, color="gray", ls=":", lw=1); axWR.grid(alpha=0.3)
axWR.legend(title="exploration radius $v$", fontsize=9)

axGAP.set_title("(b) model belief vs true expectation\nsolid=true_p_soft  dashed=predicted_p ($\\alpha$=30)")
axGAP.set_xlabel("round $t$"); axGAP.set_ylabel("probability")
axGAP.axhline(0.5, color="gray", ls=":", lw=1); axGAP.grid(alpha=0.3)
axGAP.legend(fontsize=8)

axEIG.set_title("(c) Thompson cov spectrum (log)\nbigger v -> larger draw radius")
axEIG.set_xlabel("round $t$"); axEIG.set_ylabel("eigenvalue of $cov=v^2\\,0.5(H^{-1}+H^{-T})$")
axEIG.set_yscale("log"); axEIG.grid(alpha=0.3, which="both")
axEIG.legend(fontsize=8)

axBEST.set_title("(d) best-so-far ds-to-$R$ (lower=closer)")
axBEST.set_xlabel("round $t$"); axBEST.set_ylabel("best-so-far ds_to_R (avg over sims)")
axBEST.axhline(D_B, color="gray", ls=":", lw=1)
axBEST.annotate(f"$D_B$={D_B:.4f}", (0, D_B), fontsize=8, color="gray", va="bottom")
axBEST.axhline(0.2702, color="tab:green", ls=":", lw=1)
axBEST.annotate("lam100 best=0.2702", (0, 0.2702), fontsize=8, color="tab:green", va="bottom")
axBEST.grid(alpha=0.3); axBEST.legend(title="exploration radius $v$", fontsize=9)

fig.suptitle(f"CM-TS $\\lambda$=50 fixed, $v$={{4,12}} sweep — 广度 vs 方向 "
             f"($\\alpha$=30, d=16, T=200, B=8, {STAMP})", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(OUT, f"lam50_vsweep_diag_{STAMP}.png")
fig.savefig(png, dpi=140); plt.close(fig)

hdr = ("v", "nSim", "wr_f20", "wr_l20", "wr_slope/100r",
       "soft_f20", "soft_l20", "pred_l20", "eigmax_l20", "eigmin_l20", "best_ds")
print("\n" + " ".join(f"{h:>11}" for h in hdr))
for r in rows:
    print(f"{r[0]:>11} {r[1]:>11} {r[2]:>11.3f} {r[3]:>11.3f} {r[4]:>+11.3f} "
          f"{r[5]:>11.3f} {r[6]:>11.3f} {r[7]:>11.3f} {r[8]:>11.2e} {r[9]:>11.2e} {r[10]:>11.4f}")
print(f"\nsaved: {png}")
