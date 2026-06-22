"""
Recompute config A trajectories from saved images (the killed full run).

Config A (cmts_A_d16_v0.4_S8.0_0602_0350) was externally stopped at t~120/200;
trajectory.csv never written and stdout was buffered away. But all images are on
disk. Here we re-score each saved image with DreamSim vs the fixed reference R,
rebuild y_t = 1[ds_t < D_B], and emit recomputed trajectory.csv + a best-so-far
learning-curve figure to check whether the metric rises over the longer horizon.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from src.scorer import DreamSimScorer  # noqa: E402

RUN = "outputs/cmts_A_d16_v0.4_S8.0_0602_0350"
POOL = "outputs/strict_pool_s228_0429_0119"
D_B = 0.4704190492630005     # dreams[black, seed=18], from config A config_partial*.json
DEVICE = "cuda:0"


def step_index(fname):
    b = os.path.basename(fname)
    return int(b[1:].split(".")[0])    # 'w00'->0(warm), 't120'->120(main)


def main():
    print(f"Loading DreamSim on {DEVICE} ...")
    scorer = DreamSimScorer(device=DEVICE)
    R_img = Image.open(os.path.join(POOL, "reference.png")).convert("RGB")
    R = scorer.preprocess(R_img)

    sim_dirs = sorted(glob.glob(os.path.join(RUN, "sim*")))
    all_main = []
    for sd in sim_dirs:
        sim = os.path.basename(sd)
        img_dir = os.path.join(sd, "images")
        rows = []
        # warm
        for f in sorted(glob.glob(os.path.join(img_dir, "w*.png")), key=step_index):
            ds = float(scorer.model(R, scorer.preprocess(
                Image.open(f).convert("RGB"))).item())
            i = step_index(f)
            rows.append({"t": i - 24, "phase": "warm", "ds_to_R": ds,
                         "y": int(ds < D_B)})
        # main
        for f in sorted(glob.glob(os.path.join(img_dir, "t*.png")), key=step_index):
            ds = float(scorer.model(R, scorer.preprocess(
                Image.open(f).convert("RGB"))).item())
            t = step_index(f)
            rows.append({"t": t, "phase": "main", "ds_to_R": ds,
                         "y": int(ds < D_B)})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(sd, "trajectory_recomputed.csv"), index=False)
        m = df[df.phase == "main"].sort_values("t").reset_index(drop=True)
        m["sim"] = sim
        m["ds_runmin"] = m["ds_to_R"].cummin()
        m["cum_hit"] = m["y"].expanding().mean()
        all_main.append(m)
        print(f"  {sim}: T_done={len(m):3d}  best_ds={m.ds_to_R.min():.4f} "
              f"(@t={int(m.ds_to_R.values.argmin())})  hit_rate={m.y.mean():.3f}  "
              f"warm_hit={df[df.phase=='warm'].y.mean():.3f}")

    big = pd.concat(all_main, ignore_index=True)
    big.to_csv(os.path.join(RUN, "all_trajectories_recomputed.csv"), index=False)

    # ---- aggregate best-so-far across the 4 trajectories (min common length) ----
    Tmin = min(len(m) for m in all_main)
    runmin = np.stack([m["ds_runmin"].values[:Tmin] for m in all_main])  # (4, Tmin)
    cumhit = np.stack([m["cum_hit"].values[:Tmin] for m in all_main])
    t = np.arange(1, Tmin + 1)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    for m in all_main:
        ax[0].plot(m["t"] + 1, m["ds_runmin"], color="tab:blue", alpha=0.30, lw=1)
    ax[0].plot(t, runmin.mean(0), color="tab:blue", lw=2.4, label="mean best-so-far")
    ax[0].axhline(D_B, color="tab:red", ls="--", lw=1.2, label=f"$D_B$={D_B:.3f}")
    ax[0].set_xlabel("main round $t$"); ax[0].set_ylabel("best-so-far DreamSim (lower=better)")
    ax[0].set_title("Config A (d=16, v=0.4): best-so-far"); ax[0].legend(fontsize=9)
    ax[0].invert_yaxis()

    for m in all_main:
        ax[1].plot(m["t"] + 1, m["cum_hit"], color="tab:green", alpha=0.30, lw=1)
    ax[1].plot(t, cumhit.mean(0), color="tab:green", lw=2.4, label="mean cum hit-rate")
    ax[1].axhline(0.5, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("main round $t$"); ax[1].set_ylabel("cumulative hit-rate $1[ds<D_B]$")
    ax[1].set_title("Config A: cumulative win-rate"); ax[1].legend(fontsize=9)
    ax[1].set_ylim(0, 1.02)

    fig.tight_layout()
    out = os.path.join(RUN, "learning_curve_recomputed.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\nFigure -> {out}")

    # ---- numeric verdict ----
    print("\n=== AGGREGATE (mean over 4 trajectories) ===")
    print(f"  best-so-far @ t=1   : {runmin.mean(0)[0]:.4f}")
    print(f"  best-so-far @ t={Tmin:<3d}: {runmin.mean(0)[-1]:.4f}")
    print(f"  improvement         : {runmin.mean(0)[0]-runmin.mean(0)[-1]:.4f}")
    # where does best-so-far stop improving (mean curve)?
    mc = runmin.mean(0)
    final = mc[-1]
    plateau_t = int(np.argmax(mc <= final + 1e-4)) + 1
    print(f"  mean best-so-far first within 1e-4 of final at t={plateau_t}")
    print(f"  overall best_ds any traj: {big.ds_to_R.min():.4f}")
    print(f"  mean main hit-rate      : {np.mean([m.y.mean() for m in all_main]):.3f}")


if __name__ == "__main__":
    main()
