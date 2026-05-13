"""
Experiment: Correlation between prompt embedding distance and DreamSim distance.

X-axis: cosine distance of prompt embedding (with z inserted) vs baseline (z=0)
        - prompt embedding = mean-pooled [prompt_a, z, prompt_b] sequence
Y-axis: DreamSim distance of generated image vs reference image

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


def cosine_distance(v1, v2):
    """Cosine distance between two 1-D tensors."""
    sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    return 1.0 - sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/prompt_txt_emb_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    R_values = [50, 200, 500, 800]
    seeds = [1810772, 945737, 1763690, 1082459]
    n_samples = 25

    # --- Projection matrix W ---
    W_raw = np.random.RandomState(42).randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)

    # --- Load models ---
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)

    print("Loading DreamSim...")
    dreamsim_scorer = DreamSimScorer(device=device)
    ref_tensor_ds = dreamsim_scorer.preprocess(args.ref_image)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Baseline prompt embedding (z=0) ---
    z_zero = torch.zeros(1, 4096, device=device, dtype=torch.float16)
    embeds_zero = gen.encode_batch_insert(prompt, z_zero)
    # embeds_zero[0] shape: (1, L, 4096) - positive prompt embedding
    baseline_emb = embeds_zero[0][0].float().mean(dim=0)  # mean-pool -> (4096,)

    # --- Sample z directions ---
    rng = np.random.RandomState(0)
    directions = []
    for _ in range(n_samples):
        x = rng.randn(128).astype(np.float32)
        x = x / np.linalg.norm(x)
        directions.append(x)

    # --- Run ---
    import pandas as pd
    all_results = []

    for R in R_values:
        print(f"\n=== R={R} ===")

        for seed in seeds:
            print(f"  seed={seed}, generating {n_samples} images...")

            for z_idx, d in enumerate(directions):
                z_128 = d * R
                z_4096 = (W_torch @ torch.from_numpy(z_128).to(device, dtype=torch.float16)).unsqueeze(0)

                embeds = gen.encode_batch_insert(prompt, z_4096)

                # Prompt embedding distance (mean-pooled)
                prompt_emb = embeds[0][0].float().mean(dim=0)  # (4096,)
                prompt_dist = cosine_distance(prompt_emb, baseline_emb)

                # Generate image
                imgs = gen.generate_batch(embeds, [seed])
                img = imgs[0]

                # DreamSim distance
                ds_dist = dreamsim_scorer.model(ref_tensor_ds, dreamsim_scorer.preprocess(img)).item()

                all_results.append({
                    "R": R,
                    "seed": seed,
                    "z_idx": z_idx,
                    "prompt_emb_dist": prompt_dist,
                    "dreamsim_dist": ds_dist,
                })
                img.close()

    # --- Save CSV ---
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(run_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to {csv_path}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for ax_idx, R in enumerate(R_values):
        ax = axes[ax_idx]
        subset_R = df[df["R"] == R]

        for s_idx, seed in enumerate(seeds):
            subset = subset_R[subset_R["seed"] == seed]
            ax.scatter(subset["prompt_emb_dist"], subset["dreamsim_dist"],
                       c=colors[s_idx], alpha=0.6, s=40, label=f"seed={seed}")

        corr = np.corrcoef(subset_R["prompt_emb_dist"], subset_R["dreamsim_dist"])[0, 1]
        ax.set_title(f"R={R}\nPearson r={corr:.4f}")
        ax.set_xlabel("Prompt Embedding Distance (cosine)")
        if ax_idx == 0:
            ax.set_ylabel("DreamSim Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Prompt Embedding Distance vs DreamSim Distance (by R)", fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join(run_dir, "correlation_by_R.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    # --- Summary ---
    print("\n=== Per-R Correlation Summary ===")
    for R in R_values:
        subset = df[df["R"] == R]
        corr = np.corrcoef(subset["prompt_emb_dist"], subset["dreamsim_dist"])[0, 1]
        print(f"  R={R:4d}: r={corr:.4f}, prompt_dist={subset['prompt_emb_dist'].mean():.4f}±{subset['prompt_emb_dist'].std():.4f}, "
              f"DS dist={subset['dreamsim_dist'].mean():.4f}±{subset['dreamsim_dist'].std():.4f}")


if __name__ == "__main__":
    main()
