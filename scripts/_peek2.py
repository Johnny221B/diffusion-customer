"""Full-horizon per-v trend read (uses sim000's complete T=200 trajectory).
Rolling-20 win-rate at the start vs end + a linear slope over rounds tells genuine
learning (sustained climb) apart from regression-to-mean (early jump then flat)."""
import glob, os, re
import numpy as np
import pandas as pd

hdr = (f"{'v':>5} {'wr_first20':>10} {'wr_last20':>9} {'slope/100r':>10} "
       f"{'best_ds':>8} {'ds@last20':>9} {'rounds':>7}")
print(hdr); print("-" * len(hdr))
for d in sorted(glob.glob("outputs/cmts_v*_*_0603_0137"),
                key=lambda p: float(re.search(r'_v([0-9.]+)_', p).group(1))):
    v = re.search(r'_v([0-9.]+)_', d).group(1)
    f = f"{d}/sim000/trajectory.csv"
    if not os.path.exists(f):
        print(f"{v:>5}   (no csv)"); continue
    df = pd.read_csv(f); m = df[df.phase == "main"]
    wr = m.groupby("t")["y"].mean()                  # per-round win-rate
    ds = m.groupby("t")["ds_to_R"].mean()
    R = len(wr)
    first20 = wr.iloc[:20].mean(); last20 = wr.iloc[-20:].mean()
    # linear slope of round win-rate vs t, scaled to "per 100 rounds"
    t = wr.index.values.astype(float)
    slope = np.polyfit(t, wr.values, 1)[0] * 100 if R > 2 else float("nan")
    print(f"{v:>5} {first20:>10.3f} {last20:>9.3f} {slope:>+10.3f} "
          f"{m['ds_to_R'].min():>8.4f} {ds.iloc[-20:].mean():>9.4f} {R:>7}")
print("\ngenuine learning = positive slope AND wr_last20 > wr_first20 sustained.")
print("greedy/regression = early-high then flat/declining (negative or ~0 slope).")
