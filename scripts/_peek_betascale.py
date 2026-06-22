import numpy as np, glob, os
v = 4.0
print(f"{'lam':>6} {'||beta_hat||':>12} {'TS_std/dim':>11} {'||TS_noise||':>12} {'noise/||b||':>11}  regime")
for lam in [16, 100, 1000, 10000]:
    dirs = sorted(glob.glob(f"outputs/cmts_lam{lam}_v4.0_d16_B8_T200_0608_1146/sim*"))
    bn, sd, ratio = [], [], []
    for s in dirs:
        f = os.path.join(s, "posterior.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        beta, H = z["beta_hat"], z["H"]
        d = len(beta)
        cov = v**2 * np.linalg.inv(H)          # Thompson posterior covariance
        per_dim_std = np.sqrt(np.mean(np.diag(cov)))
        noise_norm = np.sqrt(np.trace(cov))    # expected ||beta_tilde - beta_hat||
        nb = np.linalg.norm(beta)
        bn.append(nb); sd.append(per_dim_std); ratio.append(noise_norm / (nb + 1e-12))
    if not bn:
        continue
    nb, s_, nn = np.mean(bn), np.mean(sd), np.mean([r for r in ratio])
    noise_norm_mean = np.mean([sd[i]*np.sqrt(16) for i in range(len(sd))])
    reg = "fit dominates" if nn < 0.5 else ("noise dominates -> ~random dir" if nn > 1.5 else "comparable")
    print(f"{lam:>6} {nb:>12.3f} {s_:>11.3f} {noise_norm_mean:>12.3f} {nn:>11.2f}  {reg}")
