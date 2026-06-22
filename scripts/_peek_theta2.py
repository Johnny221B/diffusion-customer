import numpy as np, os

def unit(v): return v / (np.linalg.norm(v) + 1e-12)

for lam in [16, 1000]:
    p = f"outputs/cmts_lam{lam}_v4.0_d16_B8_T200_0608_1146/sim000"
    z = np.load(p + "/posterior.npz")
    Phi, y, beta, zc = z["Phi"], z["y"], z["beta_hat"], z["z_comp"]
    n_warm = 24
    Z = Phi[n_warm:]            # main-phase solved z's: 200*8 = 1600 x 16
    Y = y[n_warm:]
    norms = np.linalg.norm(Z, axis=1)
    u_beta = unit(beta)
    cos_beta = Z @ u_beta / (norms + 1e-12)          # alignment of pick w/ fitted dir
    cos_comp = Z @ unit(zc) / (norms + 1e-12)        # alignment w/ competitor z
    # round-to-round movement: reshape to (200, 8, 16), take per-round mean pick
    R = Z.reshape(-1, 8, 16)
    round_mean = R.mean(axis=1)                       # 200 x 16
    step = np.linalg.norm(np.diff(round_mean, axis=0), axis=1)   # 199 movements
    # how spread the 8 picks within a round are (mean pairwise to round centroid)
    within = np.linalg.norm(R - round_mean[:, None, :], axis=2).mean()
    print(f"=== lam={lam} ===")
    print(f"  ||z||:        min {norms.min():.2f}  mean {norms.mean():.2f}  max {norms.max():.2f}")
    print(f"  ||beta_hat||  {np.linalg.norm(beta):.3f}")
    print(f"  cos(z, beta): mean {cos_beta.mean():+.3f}  std {cos_beta.std():.3f}   (1 => pick points along fitted dir)")
    print(f"  cos(z, zcomp):mean {cos_comp.mean():+.3f}  std {cos_comp.std():.3f}")
    print(f"  within-round spread (mean dist of 8 picks to centroid): {within:.2f}")
    print(f"  round-to-round centroid move: mean {step.mean():.2f}  (early {step[:20].mean():.2f} -> late {step[-20:].mean():.2f})")
    print(f"  win-rate first20 {Y.reshape(-1,8).mean(1)[:20].mean():.3f} -> last20 {Y.reshape(-1,8).mean(1)[-20:].mean():.3f}")
    print()
