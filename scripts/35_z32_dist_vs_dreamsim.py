"""
Experiment: z-space distance vs DreamSim distance with dim_z=32.

Same setup as script 34 but with 32-dim z instead of 128-dim.
Lower dimensions = more variance in angular distances between random vectors.

For each R:
- Reference z = R * x0_dir_32 (best direction in 32-dim)
- Sample z = R * random_direction_32
- X-axis = ||z - z_ref||
- Y-axis = DreamSim distance

Also computes per-seed correlations.

4 R values x 4 seeds x 25 z samples = 400 images.
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_seed", type=int, default=1810772)
    parser.add_argument("--dim_z", type=int, default=32)
    args = parser.parse_args()

    device = args.device
    dim_z = args.dim_z
    run_dir = f"outputs/z{dim_z}_dist_corr_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    R_values = [50, 200, 500, 800]
    seeds = [945737, 1763690, 1082459, 1755275]
    n_samples = 25

    # --- Projection matrix W: 4096 x dim_z ---
    W_raw = np.random.RandomState(42).randn(4096, dim_z).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)
    print(f"W shape: {W_np.shape}")

    # --- Find best x0 direction in dim_z space ---
    # We need a good reference direction. Use the same approach: sample many directions,
    # pick the one that generates the image closest to z=0 baseline.
    # For simplicity, just use a random direction as x0_dir (since the old x0 was in 128-dim).
    # We'll use the first principal direction from the old x0 projected into this space.
    # Actually, let's just pick a random unit vector as reference direction.
    rng_x0 = np.random.RandomState(99)
    x0_dir = rng_x0.randn(dim_z).astype(np.float32)
    x0_dir = x0_dir / np.linalg.norm(x0_dir)
    print(f"x0_dir shape: {x0_dir.shape}, ||x0_dir||={np.linalg.norm(x0_dir):.4f}")

    # --- Load models ---
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)

    print("Loading DreamSim...")
    dreamsim_scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Sample z directions in dim_z ---
    rng = np.random.RandomState(0)
    directions = []
    for _ in range(n_samples):
        x = rng.randn(dim_z).astype(np.float32)
        x = x / np.linalg.norm(x)
        directions.append(x)

    # --- Run experiment per R ---
    import pandas as pd
    all_results = []

    for R in R_values:
        print(f"\n=== R={R} ===")

        # Reference: x0 direction scaled to norm R
        z_ref = x0_dir * R
        z_ref_4096 = (W_torch @ torch.from_numpy(z_ref).to(device, dtype=torch.float16)).unsqueeze(0)
        embeds_ref = gen.encode_batch_insert(prompt, z_ref_4096)
        ref_imgs = gen.generate_batch(embeds_ref, [args.ref_seed])
        ref_img = ref_imgs[0]
        ref_img.save(os.path.join(run_dir, f"reference_R{R}.png"))
        ref_tensor_ds = dreamsim_scorer.preprocess(ref_img)
        print(f"  Reference generated (z=x0_dir*{R})")

        for seed in seeds:
            print(f"  seed={seed}, generating {n_samples} images...")

            for z_idx, d in enumerate(directions):
                z = d * R  # random direction in dim_z, norm R
                z_dist = float(np.linalg.norm(z - z_ref))

                z_4096 = (W_torch @ torch.from_numpy(z).to(device, dtype=torch.float16)).unsqueeze(0)
                embeds = gen.encode_batch_insert(prompt, z_4096)
                imgs = gen.generate_batch(embeds, [seed])
                img = imgs[0]

                ds_dist = dreamsim_scorer.model(ref_tensor_ds, dreamsim_scorer.preprocess(img)).item()

                all_results.append({
                    "R": R,
                    "seed": seed,
                    "z_idx": z_idx,
                    "z_dist_to_ref": z_dist,
                    "dreamsim_dist": ds_dist,
                })
                img.close()

        ref_img.close()

    # --- Save CSV ---
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(run_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to {csv_path}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors_plt = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Plot 1: per-R subplots (all seeds)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ax_idx, R in enumerate(R_values):
        ax = axes[ax_idx]
        subset_R = df[df["R"] == R]
        for s_idx, seed in enumerate(seeds):
            subset = subset_R[subset_R["seed"] == seed]
            ax.scatter(subset["z_dist_to_ref"], subset["dreamsim_dist"],
                       c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")
        corr = np.corrcoef(subset_R["z_dist_to_ref"], subset_R["dreamsim_dist"])[0, 1]
        ax.set_title(f"R={R}\nPearson r={corr:.4f}")
        ax.set_xlabel(f"||z - z_ref|| (L2 in {dim_z}-dim)")
        if ax_idx == 0:
            ax.set_ylabel("DreamSim Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Z-space Distance vs DreamSim (dim_z={dim_z})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "correlation_by_R.png"), dpi=150)

    # Plot 2: per-seed per-R subplots
    fig2, axes2 = plt.subplots(4, 4, figsize=(20, 16))
    for r_idx, R in enumerate(R_values):
        for s_idx, seed in enumerate(seeds):
            ax = axes2[s_idx][r_idx]
            subset = df[(df["R"] == R) & (df["seed"] == seed)]
            ax.scatter(subset["z_dist_to_ref"], subset["dreamsim_dist"],
                       c=colors_plt[s_idx], alpha=0.7, s=40)
            corr = np.corrcoef(subset["z_dist_to_ref"], subset["dreamsim_dist"])[0, 1]
            ax.set_title(f"R={R}, seed={seed}\nr={corr:.4f}", fontsize=9)
            if s_idx == 3:
                ax.set_xlabel(f"||z - z_ref||")
            if r_idx == 0:
                ax.set_ylabel("DreamSim")
            ax.grid(True, alpha=0.3)
    fig2.suptitle(f"Per-Seed Correlation (dim_z={dim_z})", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(run_dir, "correlation_per_seed.png"), dpi=150)

    # --- Summary ---
    print(f"\n=== Per-R Correlation (all seeds) ===")
    for R in R_values:
        subset = df[df["R"] == R]
        corr = np.corrcoef(subset["z_dist_to_ref"], subset["dreamsim_dist"])[0, 1]
        print(f"  R={R:4d}: r={corr:.4f}, z_dist={subset['z_dist_to_ref'].mean():.2f}±{subset['z_dist_to_ref'].std():.2f}, "
              f"DS={subset['dreamsim_dist'].mean():.4f}±{subset['dreamsim_dist'].std():.4f}")

    print(f"\n=== Per-R Per-Seed Correlation ===")
    for R in R_values:
        print(f"R={R}:")
        for seed in seeds:
            subset = df[(df["R"] == R) & (df["seed"] == seed)]
            corr = np.corrcoef(subset["z_dist_to_ref"], subset["dreamsim_dist"])[0, 1]
            print(f"  seed={seed}: r={corr:.4f}")


if __name__ == "__main__":
    main()
