"""89 - lambda={0.5,1,10,50} sweep diagnostic, v=1.0, alpha=15, competitor bright/18 (D_B=0.4705, 3pct).

Same 4 panels as 82/84 but SIX lambdas, competitor bright/18 (D_B=0.4705, 3rd pct -> win-rate
headroom), and v=1.0 (Thompson cov x16 vs 73q v=0.25; recover exploration that v=0.25 killed).
GOAL: with exploration tamed by small v, locate the lambda where beta DE-SATURATES (norm leaves S=8)
AND a rising win-rate trend appears. Companion beta-norm chart written alongside (like scripts/83).

Usage:  conda run -n diverse --no-capture-output python scripts/89_lamsweep4_bright_a15_v1_diag.py [STAMP]
"""
import glob, os, re, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D_B = 0.4705198      # competitor (bright,18); y=1 iff ds_to_R < D_B
BLACK_BEST = 0.2803          # black/18 run (0613_0004) v12 best-so-far, cross-run reference
S_CLIP = 8.0
WIN = 15
OUT = "outputs/vsweep_curves"
os.makedirs(OUT, exist_ok=True)

if len(sys.argv) > 1:
    STAMP = sys.argv[1]
else:
    cands = sorted(glob.glob("outputs/cmts_lam50_v1.0_bbright_a15_d16_B8_T200_*"))
    if not cands:
        sys.exit("no cmts_lam*_v1.0_bbright_a15_* runs found; pass STAMP explicitly")
    STAMP = re.search(r'_T200_(\d{4}_\d{4})$', cands[-1]).group(1)
print(f"STAMP = {STAMP}")

# lambda -> (label, color); ordered small..large
CMAP = plt.get_cmap("viridis")
LAMS = ["0.5", "1", "10", "50"]
COLORS = {lam: CMAP(i / (len(LAMS) - 1)) for i, lam in enumerate(LAMS)}
LABELS = {"0.5": "$\\lambda$=0.5", "1": "$\\lambda$=1", "5": "$\\lambda$=5",
          "10": "$\\lambda$=10", "50": "$\\lambda$=50"}


def load_cfg(lam):
    d = f"outputs/cmts_lam{lam}_v1.0_bbright_a15_d16_B8_T200_{STAMP}"
    files = sorted(glob.glob(f"{d}/sim*/trajectory.csv"))
    wr, soft, pred, emax, emin, best, bn = [], [], [], [], [], [], []
    for f in files:
        m = pd.read_csv(f).query("phase == 'main'")
        if m.empty:
            continue
        wr.append(m.groupby("t")["y_hard"].mean())
        soft.append(m.groupby("t")["true_p_soft"].mean())
        pred.append(m.groupby("t")["predicted_p"].mean())
        emax.append(m.groupby("t")["cov_eig_max"].first())
        emin.append(m.groupby("t")["cov_eig_min"].first())
        best.append(m.groupby("t")["ds_to_R"].min().cummin())
        bn.append(m.groupby("t")["beta_norm"].first())
    if not wr:
        return None
    agg = lambda L: pd.concat(L, axis=1).mean(axis=1)
    return dict(n=len(wr), wr=agg(wr), soft=agg(soft), pred=agg(pred),
                emax=agg(emax), emin=agg(emin), best=agg(best), bn=agg(bn))


CFG = {lam: load_cfg(lam) for lam in LAMS}

# ---------- figure 1: 4-panel diagnostics ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
(axWR, axGAP), (axEIG, axBEST) = axes
rows = []
for lam in LAMS:
    R = CFG[lam]
    if R is None:
        print(f"  lam{lam}: no data yet (skipping)")
        continue
    c = COLORS[lam]; t = R["wr"].index.values.astype(float)
    lbl = f"{LABELS[lam]} (n={R['n']})"

    axWR.plot(t, R["wr"].rolling(WIN, min_periods=1, center=True).mean().values,
              color=c, lw=2.0, label=lbl)
    a, b = np.polyfit(t, R["wr"].values, 1)
    axWR.plot(t, a * t + b, color=c, lw=0.9, ls="--", alpha=0.6)

    axGAP.plot(t, R["soft"].rolling(WIN, min_periods=1, center=True).mean().values,
               color=c, lw=2.0, label=f"{LABELS[lam]} true")
    axGAP.plot(t, R["pred"].rolling(WIN, min_periods=1, center=True).mean().values,
               color=c, lw=1.3, ls="--", alpha=0.8)

    axEIG.plot(t, R["emax"].values, color=c, lw=2.0, label=f"{LABELS[lam]} max")
    axEIG.plot(t, R["emin"].values, color=c, lw=1.1, ls=":")

    axBEST.plot(R["best"].index.values, R["best"].values, color=c, lw=2.0, label=LABELS[lam])

    rows.append((lam, R["n"], R["wr"].iloc[:20].mean(), R["wr"].iloc[-20:].mean(),
                 a * 100, R["soft"].iloc[-20:].mean(), R["pred"].iloc[-20:].mean(),
                 R["emax"].iloc[-20:].mean(), float(R["bn"].iloc[-1]), float(R["best"].iloc[-1])))

axWR.set_title("(a) hard win-rate  $P(\\mathrm{ds}<D_B)$ per round (solid=roll15, dashed=lin fit)")
axWR.set_xlabel("round $t$"); axWR.set_ylabel("win-rate (avg over sims)")
axWR.axhline(0.5, color="gray", ls=":", lw=1); axWR.grid(alpha=0.3)
axWR.legend(title="prior precision $\\lambda$", fontsize=8, ncol=2)

axGAP.set_title("(b) belief vs truth  solid=true_p_soft  dashed=predicted_p ($\\alpha$=15)")
axGAP.set_xlabel("round $t$"); axGAP.set_ylabel("probability")
axGAP.axhline(0.5, color="gray", ls=":", lw=1); axGAP.grid(alpha=0.3)
axGAP.legend(fontsize=8, ncol=2)

axEIG.set_title("(c) Thompson cov spectrum (log)  solid=eig_max dotted=eig_min")
axEIG.set_xlabel("round $t$"); axEIG.set_ylabel("eigenvalue of $cov=v^2\\,0.5(H^{-1}+H^{-T})$")
axEIG.set_yscale("log"); axEIG.grid(alpha=0.3, which="both")
axEIG.legend(fontsize=8, ncol=2)

axBEST.set_title("(d) best-so-far ds-to-$R$ (lower=closer)")
axBEST.set_xlabel("round $t$"); axBEST.set_ylabel("best-so-far ds_to_R (avg over sims)")
axBEST.axhline(D_B, color="gray", ls=":", lw=1)
axBEST.annotate(f"$D_B$={D_B:.4f} (bright/18)", (0, D_B), fontsize=8, color="gray", va="bottom")
axBEST.axhline(BLACK_BEST, color="k", ls=":", lw=1)
axBEST.annotate(f"prev best={BLACK_BEST}", (0, BLACK_BEST), fontsize=8, color="k", va="bottom")
axBEST.grid(alpha=0.3); axBEST.legend(title="prior precision $\\lambda$", fontsize=8, ncol=2)

fig.suptitle(f"CM-TS $\\lambda$-sweep {{0.5..50}}, v=1.0, B=bright/18 "
             f"($D_B$=0.4705, 3pct; $\\alpha$=15, d=16, T=200, B=8, {STAMP})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(OUT, f"lamsweep4_bright_a15_v1_diag_{STAMP}.png")
fig.savefig(png, dpi=140); plt.close(fig)

# ---------- figure 2: beta-norm vs round (saturation check) ----------
fig2, ax = plt.subplots(figsize=(9, 5.5))
for lam in LAMS:
    R = CFG[lam]
    if R is None:
        continue
    t = R["bn"].index.values.astype(float)
    ax.plot(t, R["bn"].values, color=COLORS[lam], lw=2.0, label=f"{LABELS[lam]} (n={R['n']})")
ax.axhline(S_CLIP, color="k", ls="--", lw=1.2, label="S=8 (norm clip)")
ax.set_title(f"per-round $\\|\\hat\\beta_t\\|$ — v=1.0, B=bright/18, $\\alpha$=15 ({STAMP})\n"
             "where does beta leave the S=8 clip (de-saturate)?")
ax.set_xlabel("round $t$"); ax.set_ylabel("$\\|\\hat\\beta_t\\|$")
ax.grid(alpha=0.3); ax.legend(title="prior precision $\\lambda$", fontsize=9)
fig2.tight_layout()
png2 = os.path.join(OUT, f"beta_norm_lamsweep4_bright_{STAMP}.png")
fig2.savefig(png2, dpi=140); plt.close(fig2)

hdr = ("lam", "nSim", "wr_f20", "wr_l20", "wr_slope/100r",
       "soft_l20", "pred_l20", "eigmax_l20", "beta_norm_end", "best_ds")
print("\n" + " ".join(f"{h:>13}" for h in hdr))
for r in rows:
    print(f"{r[0]:>13} {r[1]:>13} {r[2]:>13.3f} {r[3]:>13.3f} {r[4]:>+13.3f} "
          f"{r[5]:>13.3f} {r[6]:>13.3f} {r[7]:>13.2e} {r[8]:>13.3f} {r[9]:>13.4f}")
print(f"\nsaved: {png}")
print(f"saved: {png2}")
