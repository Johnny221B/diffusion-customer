#!/usr/bin/env python3
"""Reduced warm-up on FIXED historical observations, not an online policy rerun.

Uses the first 4/8/24 warm observations and all original subsequent observations.
Acquisition was performed by the original n0=24 policy. No new regret is computed.
"""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmts_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.cmts_sim import laplace_map, project_norm, sigma


def main():
    source = PROJECT / "outputs/continuous_diagnostics/continuous_mse_regret_v05"
    meta = json.loads((source / "evaluation.json").read_text())
    cfg = meta["config"]
    run = Path(meta["run"])
    if not run.is_absolute():
        run = PROJECT / run
    out = source / "warmup_offline_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    test = np.load(source / "test_set.npz")
    truth = test["true_probability"]
    normalized = lambda a: (a-a.mean()) / (np.linalg.norm(a-a.mean())+1e-12)
    true_norm = normalized(truth)
    records = []
    for seed in meta["sims"]:
        state = np.load(run / f"sim{seed:03d}" / "posterior.npz")
        history = pd.read_csv(run / f"sim{seed:03d}" / "trajectory.csv")
        np.testing.assert_array_equal(state["y"], history.y)
        X_test = test["z"] - state["z_comp"]
        T = (len(state["y"])-cfg["n0"])//cfg["B"]
        for warm in [4, 8, 24]:
            indices = np.r_[np.arange(warm), np.arange(cfg["n0"], len(state["y"]))]
            X, y = state["Phi"][indices], state["y"][indices]
            beta = None
            for t in range(T+1):
                n = warm+t*cfg["B"]
                beta, _ = laplace_map(X[:n], y[:n], cfg["lam"], cfg["d"], beta0=beta)
                beta = project_norm(beta, cfg["S"])
                pred = sigma(X_test @ beta)
                records.append(dict(sim=seed, warm=warm, round=t,
                                    norm_mse=float(np.mean((normalized(pred)-true_norm)**2)),
                                    probability_mse=float(np.mean((pred-truth)**2)),
                                    beta_norm=float(np.linalg.norm(beta))))
                if warm == cfg["n0"] and t > 0:
                    batch = history[(history.phase=="main") & (history.t==t-1)]
                    np.testing.assert_allclose(sigma(state["Phi"][batch.index] @ beta),
                                               batch.predicted_p, atol=1e-6, rtol=1e-5)
            print(f"sim{seed:03d}, warm={warm}, replayed {T} rounds", flush=True)
            pd.DataFrame(records).to_csv(out / "metrics.csv", index=False)
    df = pd.DataFrame(records)
    summary = []
    for (seed, warm), g in df.groupby(["sim", "warm"]):
        first, last = g.iloc[0], g.tail(100)
        summary.append(dict(sim=seed, warm=warm, rounds=int(g["round"].max()),
                            initial_norm_mse=first.norm_mse, last100_norm_mse=last.norm_mse.mean(),
                            initial_probability_mse=first.probability_mse,
                            last100_probability_mse=last.probability_mse.mean()))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sim = df[df.sim==1]
    for ax, metric, title in zip(axes, ["norm_mse", "probability_mse"],
                                ["Normalized MSE", "Probability MSE"]):
        for warm, color in zip([4, 8, 24], ["#E69F00", "#009E73", "#0072B2"]):
            part = sim[sim.warm==warm]
            ax.plot(part["round"], part[metric], color=color, lw=1.6, label=f"Warm-up: {warm}")
            ax.scatter([0], [part.iloc[0][metric]], color=color, s=24, zorder=3)
        ax.set(title=title, ylabel=title, xlabel="Historical round")
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("Warm-up diagnostic: sim001 on fixed historical observations", y=.97)
    fig.text(.5,.02,"Offline refitting only: selected designs are unchanged; this is not a new online trajectory.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.8, wspace=.28)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"warmup_sim001_offline.{ext}", dpi=250)
    plt.close(fig)
    (out / "README.md").write_text(
        "# Warm-up sensitivity on fixed historical observations\n\n"
        "This CPU diagnostic retains the first 4, 8, or all 24 original warm-start "
        "observations, then replays the same subsequent training observations. "
        "The MAP objective, lambda=100, norm clip S=8, and update procedure are unchanged. "
        "Round 0 is evaluated before the first main-round batch. The shared independent "
        "test set is unchanged (N=120). Baseline predictions are verified against logs.\n\n"
        "The historical observations were selected by the original 24-warm-start policy. "
        "This is NOT a counterfactual online run, does not test new Thompson selections, "
        "and cannot produce an alternative regret curve. A genuine reduced-warm-start "
        "online run requires new rendering and evaluation of the newly selected designs. "
        "No original experiment or acquisition code was modified.\n")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
