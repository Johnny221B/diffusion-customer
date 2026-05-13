"""
Experiment: z-space distance vs CLIP image embedding distance.

Same setup as script 35 but using CLIP image embedding distance as Y-axis
instead of DreamSim.

For each R:
- Reference z = R * x0_dir (best direction)
- Reference image = generated with z_ref
- Sample z = R * random_direction
- X-axis = ||z - z_ref|| (L2 in z-space)
- Y-axis = CLIP image embedding distance (generated vs reference)

Tests both dim_z=32 and dim_z=128.
4 R values x 4 seeds x 25 z samples = 400 images per dim.
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


class CLIPImageEmbedder:
    def __init__(self, device="cuda", model_name="openai/clip-vit-base-patch32"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.inference_mode()
    def get_image_embedding(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        inputs = self.processor(images=[img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def cosine_distance(self, emb1, emb2):
        sim = (emb1 * emb2).sum(dim=-1).item()
        return 1.0 - sim


def run_experiment(gen, dreamsim_scorer, clip_embedder, device, dim_z, R_values, seeds, n_samples, ref_seed, run_dir, prompt):
    """Run experiment for a given dim_z."""
    sub_dir = os.path.join(run_dir, f"dim{dim_z}")
    os.makedirs(sub_dir, exist_ok=True)

    # Projection matrix W
    W_raw = np.random.RandomState(42).randn(4096, dim_z).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)

    # Reference direction
    rng_x0 = np.random.RandomState(99)
    x0_dir = rng_x0.randn(dim_z).astype(np.float32)
    x0_dir = x0_dir / np.linalg.norm(x0_dir)

    # Sample directions
    rng = np.random.RandomState(0)
    directions = []
    for _ in range(n_samples):
        x = rng.randn(dim_z).astype(np.float32)
        x = x / np.linalg.norm(x)
        directions.append(x)

    import pandas as pd
    all_results = []

    for R in R_values:
        print(f"\n  [dim={dim_z}] R={R}")

        # Reference image
        z_ref = x0_dir * R
        z_ref_4096 = (W_torch @ torch.from_numpy(z_ref).to(device, dtype=torch.float16)).unsqueeze(0)
        embeds_ref = gen.encode_batch_insert(prompt, z_ref_4096)
        ref_imgs = gen.generate_batch(embeds_ref, [ref_seed])
        ref_img = ref_imgs[0]
        ref_tensor_ds = dreamsim_scorer.preprocess(ref_img)
        ref_emb_clip = clip_embedder.get_image_embedding(ref_img)

        for seed in seeds:
            for z_idx, d in enumerate(directions):
                z = d * R
                z_dist = float(np.linalg.norm(z - z_ref))

                z_4096 = (W_torch @ torch.from_numpy(z).to(device, dtype=torch.float16)).unsqueeze(0)
                embeds = gen.encode_batch_insert(prompt, z_4096)
                imgs = gen.generate_batch(embeds, [seed])
                img = imgs[0]

                # CLIP distance
                img_emb = clip_embedder.get_image_embedding(img)
                clip_dist = clip_embedder.cosine_distance(ref_emb_clip, img_emb)

                # DreamSim distance (for comparison)
                ds_dist = dreamsim_scorer.model(ref_tensor_ds, dreamsim_scorer.preprocess(img)).item()

                all_results.append({
                    "dim_z": dim_z,
                    "R": R,
                    "seed": seed,
                    "z_idx": z_idx,
                    "z_dist_to_ref": z_dist,
                    "clip_dist": clip_dist,
                    "dreamsim_dist": ds_dist,
                })
                img.close()

        ref_img.close()

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(sub_dir, "data.csv"), index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_seed", type=int, default=1810772)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/z_vs_clip_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    R_values = [50, 200, 500, 800]
    seeds = [945737, 1763690, 1082459, 1755275]
    n_samples = 25

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # Load models
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    dreamsim_scorer = DreamSimScorer(device=device)
    print("Loading CLIP...")
    clip_embedder = CLIPImageEmbedder(device=device)

    import pandas as pd

    # Run for both dims
    dfs = []
    for dim_z in [32, 128]:
        print(f"\n{'='*40}\nRunning dim_z={dim_z}\n{'='*40}")
        df = run_experiment(gen, dreamsim_scorer, clip_embedder, device,
                           dim_z, R_values, seeds, n_samples, args.ref_seed, run_dir, prompt)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    colors_plt = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for dim_z in [32, 128]:
        df_dim = df_all[df_all["dim_z"] == dim_z]

        # Plot: z_dist vs CLIP dist (per R)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        for ax_idx, R in enumerate(R_values):
            ax = axes[ax_idx]
            subset_R = df_dim[df_dim["R"] == R]
            for s_idx, seed in enumerate(seeds):
                subset = subset_R[subset_R["seed"] == seed]
                ax.scatter(subset["z_dist_to_ref"], subset["clip_dist"],
                           c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")
            corr = np.corrcoef(subset_R["z_dist_to_ref"], subset_R["clip_dist"])[0, 1]
            ax.set_title(f"R={R}\nPearson r={corr:.4f}")
            ax.set_xlabel(f"||z - z_ref|| ({dim_z}-dim)")
            if ax_idx == 0:
                ax.set_ylabel("CLIP Image Embedding Distance")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Z-distance vs CLIP Distance (dim_z={dim_z})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"clip_corr_dim{dim_z}.png"), dpi=150)

        # Plot: z_dist vs DreamSim (per R) for comparison
        fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        for ax_idx, R in enumerate(R_values):
            ax = axes2[ax_idx]
            subset_R = df_dim[df_dim["R"] == R]
            for s_idx, seed in enumerate(seeds):
                subset = subset_R[subset_R["seed"] == seed]
                ax.scatter(subset["z_dist_to_ref"], subset["dreamsim_dist"],
                           c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")
            corr = np.corrcoef(subset_R["z_dist_to_ref"], subset_R["dreamsim_dist"])[0, 1]
            ax.set_title(f"R={R}\nPearson r={corr:.4f}")
            ax.set_xlabel(f"||z - z_ref|| ({dim_z}-dim)")
            if ax_idx == 0:
                ax.set_ylabel("DreamSim Distance")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig2.suptitle(f"Z-distance vs DreamSim (dim_z={dim_z})", fontsize=14)
        fig2.tight_layout()
        fig2.savefig(os.path.join(run_dir, f"dreamsim_corr_dim{dim_z}.png"), dpi=150)

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY: Z-distance Correlation Comparison")
    print("="*60)
    for dim_z in [32, 128]:
        df_dim = df_all[df_all["dim_z"] == dim_z]
        print(f"\ndim_z={dim_z}:")
        print(f"  {'R':>4s} | {'CLIP r':>8s} | {'DreamSim r':>10s} | {'CLIP per-seed':>40s}")
        for R in R_values:
            subset = df_dim[df_dim["R"] == R]
            corr_clip = np.corrcoef(subset["z_dist_to_ref"], subset["clip_dist"])[0, 1]
            corr_ds = np.corrcoef(subset["z_dist_to_ref"], subset["dreamsim_dist"])[0, 1]
            per_seed = []
            for seed in seeds:
                s = subset[subset["seed"] == seed]
                r = np.corrcoef(s["z_dist_to_ref"], s["clip_dist"])[0, 1]
                per_seed.append(f"{r:.3f}")
            print(f"  {R:4d} | {corr_clip:8.4f} | {corr_ds:10.4f} | {', '.join(per_seed)}")


if __name__ == "__main__":
    main()
