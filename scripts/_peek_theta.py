import numpy as np, pickle, os
p = "outputs/cmts_lam1000_v4.0_d16_B8_T200_0608_1146/sim000"
print("files:", sorted(os.listdir(p)))
try:
    z = np.load(p + "/posterior.npz")
    print("posterior.npz keys:", z.files)
    for k in z.files:
        print("  ", k, z[k].shape, z[k].dtype)
except Exception as e:
    print("posterior err:", repr(e))
try:
    d = pickle.load(open(p + "/_ckpt.pkl", "rb"))
    print("ckpt type:", type(d).__name__)
    if isinstance(d, dict):
        for k, v in d.items():
            print("  ", k, "->", getattr(v, "shape", v if isinstance(v, (int, float, str)) else type(v).__name__))
except Exception as e:
    print("ckpt err:", repr(e))
