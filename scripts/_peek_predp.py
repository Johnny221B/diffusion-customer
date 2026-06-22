import pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
f = "outputs/cmts_lam100_v4.0_d16_B8_T200_0608_1146/sim000/trajectory.csv"
m = pd.read_csv(f).query("phase=='main'").dropna(subset=["predicted_p"])
DB = 0.4704190492630005
m = m.copy()
m["true_y"] = (m.ds_to_R < DB).astype(int)

# 1) does higher predicted_p go with smaller distance / more wins?
sp_ds = spearmanr(m.predicted_p, m.ds_to_R)
sp_y  = spearmanr(m.predicted_p, m.true_y)
print(f"corr(predicted_p, ds_to_R)  Spearman={sp_ds.correlation:+.3f}  (expect NEGATIVE if working)")
print(f"corr(predicted_p, true_y)   Spearman={sp_y.correlation:+.3f}  (expect POSITIVE if working)")

# 2) calibration: bin predicted_p, show empirical win-rate per bin
m["bin"] = pd.cut(m.predicted_p, [0,.3,.45,.55,.7,.85,1.0])
cal = m.groupby("bin", observed=True).agg(n=("true_y","size"),
        pred_mean=("predicted_p","mean"), emp_winrate=("true_y","mean"))
print("\ncalibration (predicted vs actual):")
print(cal.to_string(float_format=lambda x: f"{x:.3f}"))

# 3) the boundary band the user is in
band = m[(m.predicted_p>0.45)&(m.predicted_p<0.55)]
print(f"\npredicted_p in [0.45,0.55]: n={len(band)}  actual win-rate={band.true_y.mean():.3f}"
      f"  (a coin-flip band -> ~0.5 expected)")
