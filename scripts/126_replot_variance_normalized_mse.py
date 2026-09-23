#!/usr/bin/env python3
"""Replot saved probability MSE divided by fixed test-target variance."""
import argparse
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluation", type=Path,
                    default=PROJECT / "outputs/continuous_diagnostics/continuous_tau1.25_sim000")
    args = ap.parse_args()
    source = args.evaluation
    df = pd.read_csv(source / "metrics.csv").sort_values("round")
    truth = np.load(source / "test_set.npz")["true_probability"]
    variance = float(np.mean((truth-truth.mean())**2))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Variance-normalized MSE requires nonconstant test probabilities")
    df["variance_normalized_mse"] = df.probability_mse / variance
    # Constant prediction equal to the test-target mean defines the unit baseline.
    np.testing.assert_allclose(np.mean((np.full_like(truth,truth.mean())-truth)**2)/variance,1.)
    metadata = json.loads((source / "evaluation.json").read_text())
    report = dict(definition="mean((predicted_probability-true_probability)^2) / mean((true_probability-mean(true_probability))^2)",
                  target_variance=variance,test_count=len(truth),
                  initial=df.iloc[0].to_dict(),final=df.iloc[-1].to_dict(),
                  first20=df[df["round"].between(1,20)].mean().to_dict(),
                  last20=df.tail(20).mean().to_dict(),
                  minimum=df.loc[df.variance_normalized_mse.idxmin()].to_dict(),
                  interpretation="Fixed positive denominator: identical trend to ordinary probability MSE. No data, fitting, or regret reference changed.")
    df.to_csv(source / "variance_normalized_metrics.csv",index=False)
    (source / "variance_normalized_evaluation.json").write_text(json.dumps(report,indent=2))
    cfg = metadata["config"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].plot(df["round"],df.variance_normalized_mse,color="#0072B2",lw=1.8)
    axes[0].axhline(1,color="0.4",ls="--",lw=1,label="Constant test-mean predictor")
    axes[0].legend(frameon=False,fontsize=9)
    axes[0].set(title="Variance-normalized MSE",ylabel=r"$\mathrm{MSE}\,/\,\mathrm{Var}(p)$")
    axes[1].plot(df["round"],df.cumulative_regret,color="#0072B2",lw=1.8)
    axes[1].set(title="Approximate Cumulative Regret",ylabel="Approximate Cumulative Regret")
    for ax in axes:
        ax.set_xlabel(f"Round ({cfg['B']} designs per round)")
        ax.grid(alpha=.25)
    fig.suptitle(rf"Selected continuous trajectory: $\tau\times{cfg['tau_scale']:g}$, sim{metadata['selected_sim']:03d}",y=.97)
    fig.text(.5,.02,f"Same fixed test set (N={len(truth)}); same observed-search reference p={metadata['oracle']['probability']:.4f}; single selected run.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.085,right=.985,bottom=.16,top=.8,wspace=.27)
    for ext in ["png","pdf"]:
        fig.savefig(source / f"variance_normalized_mse_regret.{ext}",dpi=250)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
