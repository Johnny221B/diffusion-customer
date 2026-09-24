#!/usr/bin/env python3
"""Create the paper panel of representative designs from one CM-TS run."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "outputs/cmts_a8_v1_lam100_bbright_d16_B8_T1000_0629_0842/sim003"
ROUND1_RUN = PROJECT / "outputs/cmts_a8_v1_lam100_recover_round1_sim003/sim003"
REFERENCE = PROJECT / "outputs/strict_pool_s228_0429_0119/reference.png"
COMPETITOR = PROJECT / "outputs/multiseed_s228_M40_0510_0241/imgs/127_bright_seed18.png"
OUTPUT = PROJECT / "results/pub_fig/continuous_generated_designs"

ALPHA = 8.0
D_B = 0.4704614281654358
# Representative saved candidates from the same simulation trajectory. The
# implementation uses zero-based t, so paper round r corresponds to t=r-1.
SELECTIONS = [
    (1, 0, 6),
    (150, 149, 0),
    (450, 449, 3),
    (650, 649, 2),
    (1150, 1149, 1),
    (1300, 1299, 7),
]


def probability(distance: float) -> float:
    return float(1.0 / (1.0 + np.exp(-ALPHA * (D_B - distance))))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trajectory = pd.read_csv(RUN / "trajectory.csv")
    round1_trajectory = pd.read_csv(ROUND1_RUN / "trajectory.csv")

    panels = [
        (REFERENCE, "(a) Reference", 0.0, probability(0.0)),
        (COMPETITOR, "(b) Competitor: bright/18", D_B, 0.5),
    ]
    records = []
    for paper_round, t, candidate_index in SELECTIONS:
        source_run = ROUND1_RUN if paper_round == 1 else RUN
        source_trajectory = round1_trajectory if paper_round == 1 else trajectory
        row = source_trajectory[
            (source_trajectory["phase"] == "main")
            & (source_trajectory["t"] == t)
            & (source_trajectory["b"] == candidate_index)
        ].iloc[0]
        image = source_run / "images" / f"t{t:03d}_b{candidate_index}.png"
        panels.append((image, f"Round {paper_round}", float(row.ds_to_R), float(row.true_p_soft)))
        records.append({
            "round": paper_round,
            "stored_t": t,
            "candidate_index": candidate_index,
            "dreamsim_distance": float(row.ds_to_R),
            "simulator_probability": float(row.true_p_soft),
            "image": str(image.relative_to(PROJECT)),
        })

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.65))
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    for ax, label, (path, title, distance, prob) in zip(axes.flat, labels, panels):
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.78")
            spine.set_linewidth(0.8)
        ax.set_title(f"{label} {title.split(') ', 1)[-1]}", fontsize=11.5, fontweight="semibold", pad=7)
        ax.set_xlabel(rf"$D={distance:.4f}$, $p={prob:.3f}$", fontsize=10.5, labelpad=6)

    fig.subplots_adjust(left=0.025, right=0.995, top=0.975, bottom=0.055,
                        wspace=0.08, hspace=0.22)
    png = OUTPUT / "continuous_generated_designs.png"
    pdf = OUTPUT / "continuous_generated_designs.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUTPUT / "continuous_generated_designs_values.csv", index=False)

    # Two benchmarks in the first column; evolution proceeds row-wise
    # through the remaining six cells. Retain all three late-round examples
    # from the original closer-candidates comparison.
    late_row = trajectory[(trajectory.phase == "main") &
                          (trajectory.t == 999) & (trajectory.b == 1)].iloc[0]
    evolution = panels[2:5] + [
        (RUN / "images/t999_b1.png", "Round 1000",
         float(late_row.ds_to_R), float(late_row.true_p_soft)),
    ] + panels[6:8]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.8))
    fig.subplots_adjust(left=.035, right=.985, top=.865, bottom=.07,
                        wspace=.14, hspace=.32)
    first = axes[0, 0].get_position()
    last = axes[1, 0].get_position()
    box = FancyBboxPatch(
        (first.x0-.016, last.y0-.048), first.width+.032,
        first.y1-last.y0+.112,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=fig.transFigure, facecolor="#E6F2FA",
        edgecolor="#AACDE3", linewidth=1.2, zorder=-1)
    fig.add_artist(box)
    fig.text((first.x0+first.x1)/2, .953, "Benchmark",
             ha="center", fontsize=15, fontweight="semibold", color="#285A78")
    right = axes[0, 1].get_position()
    fig.text((right.x0+axes[0, 3].get_position().x1)/2, .953,
             "Design evolution", ha="center", fontsize=15, fontweight="semibold")
    placements = [(axes[0,0], panels[0]),
                  (axes[1,0], (COMPETITOR, "Competitor", D_B, .5))]
    placements += list(zip(axes[:,1:].flat, evolution))
    for ax, (path, title, distance, prob) in placements:
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.78"); spine.set_linewidth(.8)
        ax.set_title(title.split(") ", 1)[-1], fontsize=12,
                     fontweight="semibold", pad=7)
        ax.set_xlabel(rf"$D={distance:.4f}$, $p={prob:.3f}$",
                      fontsize=10.5, labelpad=6)
    fig.savefig(OUTPUT / "closer_candidates_comparison.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "closer_candidates_comparison.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
