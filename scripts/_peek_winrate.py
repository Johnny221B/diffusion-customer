import pandas as pd, numpy as np
f = "outputs/cmts_lam100_v4.0_d16_B8_T200_0608_1146/sim000/trajectory.csv"
df = pd.read_csv(f)
DB = 0.4704190492630005
m = df[df.phase == "main"].copy()

# 1) is y exactly 1[ds < D_B] ? check for any violation
viol = m[(m.ds_to_R < DB) != (m.y == 1)]
print(f"rows total={len(m)}  label violations (y != 1[ds<D_B]): {len(viol)}")
if len(viol):
    print(viol[["t", "b", "ds_to_R", "y"]].head())

# 2) per-round: mean ds vs win-rate (fraction below D_B) vs median ds
g = m.groupby("t").agg(mean_ds=("ds_to_R", "mean"),
                       median_ds=("ds_to_R", "median"),
                       winrate=("y", "mean"),
                       min_ds=("ds_to_R", "min"),
                       max_ds=("ds_to_R", "max"))
print(f"\nD_B = {DB:.4f}")
print("overall: mean_ds=%.4f  median_ds=%.4f  winrate=%.3f" %
      (m.ds_to_R.mean(), m.ds_to_R.median(), m.y.mean()))

# 3) the apparent paradox: rounds where MEAN ds > D_B yet winrate > 0.5
para = g[(g.mean_ds > DB) & (g.winrate > 0.5)]
print(f"\nrounds with mean_ds > D_B  AND  winrate > 0.5 : {len(para)} / {len(g)}")
print(para.head(8).to_string(float_format=lambda x: f"{x:.4f}"))

# 4) distribution skew: how far above D_B do the 'losing' samples sit
los = m[m.y == 0].ds_to_R
win = m[m.y == 1].ds_to_R
print(f"\nlosers (y=0): n={len(los)}  ds mean={los.mean():.4f}  max={los.max():.4f}")
print(f"winners(y=1): n={len(win)}  ds mean={win.mean():.4f}  min={win.min():.4f}")
