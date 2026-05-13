"""
Experiment: z-space distance vs DreamSim distance.

For each R:
- Reference z = R * (x0 / ||x0||), i.e. x0 direction scaled to norm R
- Reference image = generated with z_ref
- Sample z = R * random_direction (all on same sphere)
- X-axis = ||z - z_ref|| (distance on sphere, varies by direction)
- Y-axis = DreamSim(generated_image, reference_image)

4 R values x 4 seeds x 25 z samples = 400 images.
One scatter plot per R, 4 colors for seeds.
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
    parser.add_argument("--init_npz", type=str, required=True, help="Path to best_x0.npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_seed", type=int, default=1810772)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/z_dist_corr_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    R_values = [50, 200, 500, 800]
    seeds = [945737, 1763690, 1082459, 1755275]
    n_samples = 25

    # --- Load best x0 ---
    init_data = np.load(args.init_npz)
    x0 = init_data["x0"]  # (128,)
    x0_dir = x0 / np.linalg.norm(x0)  # unit direction of x0
    print(f"Loaded x0, ||x0||={np.linalg.norm(x0):.2f}")

    # --- Projection matrix W ---
    W_raw = np.random.RandomState(42).randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)

    # --- Load models ---
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)

    print("Loading DreamSim...")
    dreamsim_scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Sample z directions ---
    rng = np.random.RandomState(0)
    directions = []
    for _ in range(n_samples):
        x = rng.randn(128).astype(np.float32)
        x = x / np.linalg.norm(x)
        directions.append(x)

    # --- Run experiment per R ---
    import pandas as pd
    all_results = []

    for R in R_values:
        print(f"\n=== R={R} ===")

        # Reference: x0 direction scaled to norm R
        z_ref_128 = x0_dir * R
        z_ref_4096 = (W_torch @ torch.from_numpy(z_ref_128).to(device, dtype=torch.float16)).unsqueeze(0)
        embeds_ref = gen.encode_batch_insert(prompt, z_ref_4096)
        ref_imgs = gen.generate_batch(embeds_ref, [args.ref_seed])
        ref_img = ref_imgs[0]
        ref_img.save(os.path.join(run_dir, f"reference_R{R}.png"))
        ref_tensor_ds = dreamsim_scorer.preprocess(ref_img)
        print(f"  Reference generated (z=x0_dir*{R})")

        for seed in seeds:
            print(f"  seed={seed}, generating {n_samples} images...")

            for z_idx, d in enumerate(directions):
                z_128 = d * R  # random direction, norm R
                z_dist = float(np.linalg.norm(z_128 - z_ref_128))

                z_4096 = (W_torch @ torch.from_numpy(z_128).to(device, dtype=torch.float16)).unsqueeze(0)
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
        ax.set_xlabel("||z - z_ref|| (L2 in 128-dim)")
        if ax_idx == 0:
            ax.set_ylabel("DreamSim Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Z-space Distance to Reference vs DreamSim Distance (same R sphere)", fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join(run_dir, "correlation_by_R.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Scatter plot saved to {plot_path}")

    # --- Summary ---
    print("\n=== Per-R Correlation Summary ===")
    for R in R_values:
        subset = df[df["R"] == R]
        corr = np.corrcoef(subset["z_dist_to_ref"], subset["dreamsim_dist"])[0, 1]
        print(f"  R={R:4d}: r={corr:.4f}, z_dist={subset['z_dist_to_ref'].mean():.2f}±{subset['z_dist_to_ref'].std():.2f}, "
              f"DS dist={subset['dreamsim_dist'].mean():.4f}±{subset['dreamsim_dist'].std():.4f}")


if __name__ == "__main__":
    main()
