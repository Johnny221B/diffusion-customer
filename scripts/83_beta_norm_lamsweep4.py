"""83 - per-round ||beta_hat|| line chart for the 4-lambda small-lambda sweep (73p).

beta_norm is already logged every round in trajectory.csv -> no worker change needed.
Plots ||beta_hat_t|| vs round for lambda in {1/3, 1/2, 1, 5} (wine/34, v=1, alpha=12),
averaged over the 2 seeds, with the S=8 norm-clip line.

Usage:  conda run -n diverse --no-capture-output python scripts/83_beta_norm_lamsweep4.py [STAMP]
"""
import glob, os, re, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S_CLIP = 8.0
OUT = "outputs/vsweep_curves"
os.makedirs(OUT, exist_ok=True)

if len(sys.argv) > 1:
    STAMP = sys.argv[1]
else:
    cands = sorted(glob.glob("outputs/cmts_lam5.0_v1.0_bwine_a12_d16_B8_T200_*"))
    if not cands:
        sys.exit("no cmts_lam*_v1.0_bwine_a12_* runs found; pass STAMP explicitly")
    STAMP = re.search(r'_T200_(\d{4}_\d{4})$', cands[-1]).group(1)
print(f"STAMP = {STAMP}")

# (lam_tag in dir, legend label, color)
CONFIGS = [("0.3333", "λ=1/3", "tab:purple"),
           ("0.5",    "λ=1/2", "tab:blue"),
           ("1.0",    "λ=1",   "tab:green"),
           ("5.0",    "λ=5",   "tab:red")]

plt.figure(figsize=(9.5, 6))
rows = []
for lam, lbl, c in CONFIGS:
    d = f"outputs/cmts_lam{lam}_v1.0_bwine_a12_d16_B8_T200_{STAMP}"
    fs = sorted(glob.glob(f"{d}/sim*/trajectory.csv"))
    series = []
    for f in fs:
        m = pd.read_csv(f).query("phase == 'main'")
        if m.empty:
            continue
        series.append(m.groupby("t")["beta_norm"].first())
    if not series:
        print(f"  {lbl}: no data yet (skipping)")
        continue
    mean = pd.concat(series, axis=1).mean(axis=1)
    t = mean.index.values
    plt.plot(t, mean.values, color=c, lw=2.2, label=f"{lbl} (n={len(series)})")
    rows.append((lbl, len(series), mean.iloc[0], mean.iloc[-1], mean.max()))

plt.axhline(S_CLIP, color="k", ls="--", lw=1.2, label=f"S={S_CLIP} (norm clip)")
plt.xlabel("round $t$")
plt.ylabel(r"$\|\hat\beta_t\|$  (estimate norm)")
plt.title(f"per-round $\\|\\hat\\beta\\|$ — small-$\\lambda$ sweep, wine/34, v=1, $\\alpha$=12  ({STAMP})")
plt.grid(alpha=0.3)
plt.legend(title="prior precision $\\lambda$", fontsize=9)
plt.tight_layout()
png = os.path.join(OUT, f"beta_norm_lamsweep4_{STAMP}.png")
plt.savefig(png, dpi=140); plt.close()

print(f"\n{'lambda':>8}{'nSeed':>7}{'norm_t0':>10}{'norm_t199':>11}{'norm_max':>10}")
for r in rows:
    print(f"{r[0]:>8}{r[1]:>7}{r[2]:>10.3f}{r[3]:>11.3f}{r[4]:>10.3f}")
print(f"\nsaved: {png}")
