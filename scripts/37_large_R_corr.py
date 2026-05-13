"""
Test correlation at extreme R values.
dim_z=128, DreamSim scorer, same sphere setup.
Two plots per R: L2 distance and cosine similarity to z_ref.
Includes Pearson p-value on each subplot.
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from scipy import stats

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_seed", type=int, default=1810772)
    args = parser.parse_args()

    device = args.device
    dim_z = 128
    run_dir = f"outputs/large_R_corr_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    R_values = [1000, 3000, 5000, 8000, 10000]
    seeds = [945737, 1763690, 1082459, 1755275]
    n_samples = 25

    # Projection matrix W
    W_raw = np.random.RandomState(42).randn(4096, dim_z).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)

    # Reference direction
    rng_x0 = np.random.RandomState(99)
    x0_dir = rng_x0.randn(dim_z).astype(np.float32)
    x0_dir = x0_dir / np.linalg.norm(x0_dir)

    # Load models
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    dreamsim_scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # Sample directions
    rng = np.random.RandomState(0)
    directions = []
    for _ in range(n_samples):
        x = rng.randn(dim_z).astype(np.float32)
        x = x / np.linalg.norm(x)
        directions.append(x)

    # Run
    import pandas as pd
    all_results = []

    for R in R_values:
        print(f"\n=== R={R} ===")
        z_ref = x0_dir * R
        z_ref_4096 = (W_torch @ torch.from_numpy(z_ref).to(device, dtype=torch.float16)).unsqueeze(0)
        embeds_ref = gen.encode_batch_insert(prompt, z_ref_4096)
        ref_imgs = gen.generate_batch(embeds_ref, [args.ref_seed])
        ref_img = ref_imgs[0]
        ref_img.save(os.path.join(run_dir, f"reference_R{R}.png"))
        ref_tensor_ds = dreamsim_scorer.preprocess(ref_img)

        for seed in seeds:
            print(f"  seed={seed}...")
            for z_idx, d in enumerate(directions):
                z = d * R
                z_dist = float(np.linalg.norm(z - z_ref))
                cos_sim = float(np.dot(d, x0_dir))  # cosine similarity between directions

                z_4096 = (W_torch @ torch.from_numpy(z).to(device, dtype=torch.float16)).unsqueeze(0)
                embeds = gen.encode_batch_insert(prompt, z_4096)
                imgs = gen.generate_batch(embeds, [seed])
                img = imgs[0]

                ds_dist = dreamsim_scorer.model(ref_tensor_ds, dreamsim_scorer.preprocess(img)).item()
                all_results.append({
                    "R": R, "seed": seed, "z_idx": z_idx,
                    "z_dist_to_ref": z_dist, "cos_sim_to_ref": cos_sim,
                    "dreamsim_dist": ds_dist,
                })
                img.close()
        ref_img.close()

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(run_dir, "data.csv"), index=False)

    # --- Plot 1: L2 distance ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors_plt = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    n_R = len(R_values)

    fig, axes = plt.subplots(1, n_R, figsize=(5 * n_R, 5), sharey=True)
    for ax_idx, R in enumerate(R_values):
        ax = axes[ax_idx]
        subset_R = df[df["R"] == R]
        for s_idx, seed in enumerate(seeds):
            subset = subset_R[subset_R["seed"] == seed]
            ax.scatter(subset["z_dist_to_ref"], subset["dreamsim_dist"],
                       c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")
        r_val, p_val = stats.pearsonr(subset_R["z_dist_to_ref"], subset_R["dreamsim_dist"])
        ax.set_title(f"R={R}\nr={r_val:.4f}, p={p_val:.4f}")
        ax.set_xlabel("||z - z_ref|| (L2)")
        if ax_idx == 0:
            ax.set_ylabel("DreamSim Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("L2 Distance vs DreamSim (dim=128)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "l2_dist_vs_dreamsim.png"), dpi=150)

    # --- Plot 2: Cosine similarity ---
    fig2, axes2 = plt.subplots(1, n_R, figsize=(5 * n_R, 5), sharey=True)
    for ax_idx, R in enumerate(R_values):
        ax = axes2[ax_idx]
        subset_R = df[df["R"] == R]
        for s_idx, seed in enumerate(seeds):
            subset = subset_R[subset_R["seed"] == seed]
            ax.scatter(subset["cos_sim_to_ref"], subset["dreamsim_dist"],
                       c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")
        r_val, p_val = stats.pearsonr(subset_R["cos_sim_to_ref"], subset_R["dreamsim_dist"])
        ax.set_title(f"R={R}\nr={r_val:.4f}, p={p_val:.4f}")
        ax.set_xlabel("Cosine Similarity (z, z_ref)")
        if ax_idx == 0:
            ax.set_ylabel("DreamSim Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig2.suptitle("Cosine Similarity vs DreamSim (dim=128)", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(run_dir, "cos_sim_vs_dreamsim.png"), dpi=150)

    # --- Summary ---
    print("\n=== Summary (L2 distance) ===")
    print(f"{'R':>6s} | {'r':>8s} | {'p-value':>10s} | per-seed r")
    for R in R_values:
        subset = df[df["R"] == R]
        r_val, p_val = stats.pearsonr(subset["z_dist_to_ref"], subset["dreamsim_dist"])
        per = []
        for seed in seeds:
            s = subset[subset["seed"] == seed]
            r_s, _ = stats.pearsonr(s["z_dist_to_ref"], s["dreamsim_dist"])
            per.append(f"{r_s:.3f}")
        print(f"{R:6d} | {r_val:8.4f} | {p_val:10.4f} | {', '.join(per)}")

    print("\n=== Summary (Cosine similarity) ===")
    print(f"{'R':>6s} | {'r':>8s} | {'p-value':>10s} | per-seed r")
    for R in R_values:
        subset = df[df["R"] == R]
        r_val, p_val = stats.pearsonr(subset["cos_sim_to_ref"], subset["dreamsim_dist"])
        per = []
        for seed in seeds:
            s = subset[subset["seed"] == seed]
            r_s, _ = stats.pearsonr(s["cos_sim_to_ref"], s["dreamsim_dist"])
            per.append(f"{r_s:.3f}")
        print(f"{R:6d} | {r_val:8.4f} | {p_val:10.4f} | {', '.join(per)}")


if __name__ == "__main__":
    main()
