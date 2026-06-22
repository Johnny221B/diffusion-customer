"""M=5-averaged per-v trend. Pools ALL sim*/trajectory.csv per v, averages the
per-round win-rate across trajectories first (round t -> mean over sims of that
round's mean y), then reports first20 vs last20 and a linear slope. This is the
real decision metric: a single trajectory is a staircase; the upward trend (like
the discrete setting) only appears after averaging across trajectories."""
import glob, os, re
import numpy as np
import pandas as pd

hdr = (f"{'v':>5} {'nSim':>5} {'wr_first20':>10} {'wr_last20':>9} {'slope/100r':>10} "
       f"{'best_ds':>8} {'ds@last20':>9} {'rounds':>7}")
print(hdr); print("-" * len(hdr))
for d in sorted(glob.glob("outputs/cmts_v*_*_0603_0137"),
                key=lambda p: float(re.search(r'_v([0-9.]+)_', p).group(1))):
    v = re.search(r'_v([0-9.]+)_', d).group(1)
    files = sorted(glob.glob(f"{d}/sim*/trajectory.csv"))
    if not files:
        print(f"{v:>5}   (no csv)"); continue
    wr_per_sim, ds_per_sim, best_ds_all = [], [], []
    for f in files:
        df = pd.read_csv(f); m = df[df.phase == "main"]
        if m.empty:
            continue
        wr_per_sim.append(m.groupby("t")["y"].mean())          # round win-rate, this sim
        ds_per_sim.append(m.groupby("t")["ds_to_R"].mean())
        best_ds_all.append(m["ds_to_R"].min())
    if not wr_per_sim:
        print(f"{v:>5}   (no main rows)"); continue
    # align on common rounds, average across sims
    wr = pd.concat(wr_per_sim, axis=1).mean(axis=1)
    ds = pd.concat(ds_per_sim, axis=1).mean(axis=1)
    R = len(wr)
    first20 = wr.iloc[:20].mean(); last20 = wr.iloc[-20:].mean()
    t = wr.index.values.astype(float)
    slope = np.polyfit(t, wr.values, 1)[0] * 100 if R > 2 else float("nan")
    print(f"{v:>5} {len(wr_per_sim):>5} {first20:>10.3f} {last20:>9.3f} {slope:>+10.3f} "
          f"{np.min(best_ds_all):>8.4f} {ds.iloc[-20:].mean():>9.4f} {R:>7}")
print("\nAveraged across sims -> this is the discrete-style learning curve.")
print("Pick the v with the clearest sustained positive slope (wr_last20 > wr_first20).")
