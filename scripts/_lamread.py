import json, glob, os, numpy as np, pandas as pd
for lam in [16,100,1000,10000]:
    dirs=sorted(glob.glob(f"outputs/cmts_lam{lam}_v4.0_d16_B8_T200_0608_1146/sim*"))
    mp=[]; cb=[]; first=[]; last=[]; tscos=[]; bn=[]
    for s in dirs:
        sj=os.path.join(s,"summary.json")
        if os.path.exists(sj):
            j=json.load(open(sj))
            mp.append(j.get("median_predicted_p")); cb.append(j.get("mean_cos_beta_prev"))
            bn.append(j.get("final_beta_norm"))
        tc=os.path.join(s,"trajectory.csv")
        if os.path.exists(tc):
            df=pd.read_csv(tc)
            rw=df.groupby("t")["y"].mean()
            if len(rw)>=40:
                first.append(rw.iloc[:20].mean()); last.append(rw.iloc[-20:].mean())
            if "ts_cos_mean" in df.columns: tscos.append(df["ts_cos_mean"].mean())
    def m(x):
        x=[v for v in x if v is not None]; 
        return np.mean(x) if x else float('nan')
    print(f"lam={lam:>5} (n={len(mp)})  pred_p={m(mp):.3f}  cos_beta_prev={m(cb):.3f}  "
          f"beta_norm={m(bn):.3f}  ts_cos={m(tscos):.3f}  "
          f"winrate {np.mean(first):.3f}->{np.mean(last):.3f} (d={np.mean(last)-np.mean(first):+.3f})")
