#!/usr/bin/env python3
"""Plot the selected continuous run through round 400 using saved observations."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmts_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def main():
    out = PROJECT / "outputs/continuous_diagnostics/continuous_tau1.25_sim000_T400"
    metadata = json.loads((out / "evaluation.json").read_text())
    source = PROJECT / "outputs/cmts_tau_sweep_20260922_a8_v05_lam100/tau1.25/sim000/trajectory.csv"
    history = pd.read_csv(source)
    main = history.loc[(history.phase == "main") & history.t.between(0, 399)].copy()
    batches = main.groupby("t").true_p_soft
    np.testing.assert_array_equal(batches.size().index, np.arange(400))
    np.testing.assert_array_equal(batches.size().values, np.full(400, metadata["config"]["B"]))
    reference = metadata["oracle"]["probability"]
    df = pd.DataFrame({"round": np.arange(1, 401), "soft_win_rate": batches.mean().values})
    df["soft_win_rate_ma20"] = df.soft_win_rate.rolling(20, min_periods=20).mean()
    df["cumulative_regret"] = (reference * batches.size() - batches.sum()).cumsum().values
    saved = pd.read_csv(out / "metrics.csv")
    np.testing.assert_allclose(df.cumulative_regret, saved.loc[saved["round"].between(1, 400), "cumulative_regret"], rtol=1e-10, atol=1e-10)
    df.to_csv(out / "soft_win_rate_regret.csv", index=False)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    fig.suptitle("Learning Performance in the Continuous Design Space", fontsize=16)
    color = "#0072B2"
    axes[0].plot(df["round"], df.soft_win_rate, color=color, alpha=.25, lw=.8, label="Per round")
    axes[0].plot(df["round"], df.soft_win_rate_ma20, color=color, lw=2, label="20-round average")
    axes[0].axhline(reference, color="0.35", ls="--", lw=1.2, label="Approx. optimum")
    axes[0].set(title="Soft Win Rate", ylabel="Win probability", ylim=(0, 1.02))
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[1].plot(np.r_[0, df["round"]], np.r_[0, df.cumulative_regret], color=color, lw=2)
    axes[1].set(title="Cumulative Regret", ylabel="Cumulative regret", ylim=(0, None))
    for ax in axes:
        ax.set(xlabel="Round", xlim=(0, 400), xticks=np.arange(0, 401, 100))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"soft_win_rate_regret.{ext}", dpi=300)
    plt.close(fig)
    (out / "soft_win_rate_regret_notes.json").write_text(json.dumps({
        "source": str(source), "selected_trajectory": "tau1.25, sim000",
        "rounds": 400, "batch_size": metadata["config"]["B"],
        "reference_probability": reference,
        "regret_definition": "Sum of reference_probability - true_p_soft over all 8 designs per round; warm-up excluded; approximate fixed reference, not certified global optimum.",
        "smoothing": "Trailing 20-round mean, starts at round 20; raw per-round means also shown.",
        "selection_note": "Selected illustrative run, not a multi-seed average."
    }, indent=2))
    print(out / "soft_win_rate_regret.png")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
