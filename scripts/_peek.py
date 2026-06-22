import glob, os, re, pandas as pd
hdr = f"{'v':>5} {'wr_t0-4':>8} {'wr_late':>8} {'bn_t0':>7} {'bn_now':>7} {'best_ds':>8} {'rounds':>7}"
print(hdr); print("-"*len(hdr))
for d in sorted(glob.glob("outputs/cmts_v*_*_0603_0137"),
                key=lambda p: float(re.search(r'_v([0-9.]+)_', p).group(1))):
    v = re.search(r'_v([0-9.]+)_', d).group(1)
    f = f"{d}/sim000/trajectory.csv"
    if not os.path.exists(f):
        print(f"{v:>5}   (no csv yet)"); continue
    df = pd.read_csv(f); m = df[df.phase == "main"]
    if len(m) == 0:
        print(f"{v:>5}   (warm only)"); continue
    wr = m.groupby("t")["y"].mean()
    early = wr.iloc[:5].mean(); late = wr.iloc[-5:].mean()
    bn0 = m[m.t == m.t.min()]["beta_norm"].iloc[0]
    bn1 = m[m.t == m.t.max()]["beta_norm"].iloc[0]
    print(f"{v:>5} {early:>8.2f} {late:>8.2f} {bn0:>7.2f} {bn1:>7.2f} "
          f"{m['ds_to_R'].min():>8.4f} {int(m.t.max())+1:>7}")
