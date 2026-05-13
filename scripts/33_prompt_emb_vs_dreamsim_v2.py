"""
Experiment: Prompt embedding distance vs DreamSim distance.

Reference = image generated with z=0, seed=1810772
X-axis: cosine distance between prompt embedding (with z) and reference prompt embedding (z=0)
Y-axis: DreamSim distance between generated image and reference image

Also produces visual comparison: reference + min/max DreamSim images per R.

4 R values x 4 seeds x 25 z samples = 400 images.
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


def cosine_distance_batch(v1, v2):
    """Cosine distance between two 1-D tensors."""
    sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    return 1.0 - sim


def get_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_seed", type=int, default=1810772)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/prompt_emb_v2_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    R_values = [50, 200, 500, 800]
    seeds = [945737, 1763690, 1082459, 1755275]
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

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Generate reference image (z=0, ref_seed) ---
    print(f"Generating reference image (z=0, seed={args.ref_seed})...")
    z_zero = torch.zeros(1, 4096, device=device, dtype=torch.float16)
    embeds_ref = gen.encode_batch_insert(prompt, z_zero)
    ref_imgs = gen.generate_batch(embeds_ref, [args.ref_seed])
    ref_img = ref_imgs[0]
    ref_img.save(os.path.join(run_dir, "reference.png"))
    ref_tensor_ds = dreamsim_scorer.preprocess(ref_img)

    # Reference prompt embedding: mean-pool the full sequence
    ref_prompt_emb = embeds_ref[0][0].float().mean(dim=0)  # (4096,)

    print("Reference image generated and saved.")

    # --- Sample z directions ---
    rng = np.random.RandomState(0)
    directions = []
    for _ in range(n_samples):
        x = rng.randn(128).astype(np.float32)
        x = x / np.linalg.norm(x)
        directions.append(x)

    # --- Run experiment ---
    import pandas as pd
    all_results = []
    # Store images for visual comparison: key = (R, seed, z_idx)
    best_worst_tracker = {}  # key=R -> list of (ds_dist, seed, z_idx, prompt_dist)

    for R in R_values:
        print(f"\n=== R={R} ===")
        best_worst_tracker[R] = []

        for seed in seeds:
            print(f"  seed={seed}, generating {n_samples} images...")

            for z_idx, d in enumerate(directions):
                z_128 = d * R
                z_4096 = (W_torch @ torch.from_numpy(z_128).to(device, dtype=torch.float16)).unsqueeze(0)

                embeds = gen.encode_batch_insert(prompt, z_4096)

                # Prompt embedding distance to reference
                prompt_emb = embeds[0][0].float().mean(dim=0)
                prompt_dist = cosine_distance_batch(prompt_emb, ref_prompt_emb)

                # Generate image
                imgs = gen.generate_batch(embeds, [seed])
                img = imgs[0]

                # DreamSim distance to reference
                ds_dist = dreamsim_scorer.model(ref_tensor_ds, dreamsim_scorer.preprocess(img)).item()

                all_results.append({
                    "R": R,
                    "seed": seed,
                    "z_idx": z_idx,
                    "prompt_emb_dist": prompt_dist,
                    "dreamsim_dist": ds_dist,
                })
                best_worst_tracker[R].append((ds_dist, seed, z_idx, prompt_dist))
                img.close()

    # --- Save CSV ---
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(run_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to {csv_path}")

    # --- Visual comparison: for each R, show reference + min/max DreamSim ---
    print("\nGenerating visual comparison grids...")
    font = get_font(13)
    font_title = get_font(16)
    thumb = 300

    for R in R_values:
        entries = best_worst_tracker[R]
        entries_sorted = sorted(entries, key=lambda x: x[0])

        # Pick top-3 closest and top-3 furthest
        closest_3 = entries_sorted[:3]
        furthest_3 = entries_sorted[-3:]

        # Grid: 1 row: [reference, closest1, closest2, closest3, gap, furthest1, furthest2, furthest3]
        n_imgs = 7  # ref + 3 closest + 3 furthest
        label_h = 50
        grid_w = n_imgs * thumb
        grid_h = 40 + thumb + label_h
        grid = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(grid)
        draw.text((10, 8), f"R={R}: Reference vs Closest/Furthest (DreamSim)", fill="black", font=font_title)

        # Reference
        ref_resized = ref_img.resize((thumb, thumb), Image.LANCZOS)
        grid.paste(ref_resized, (0, 40))
        draw.text((4, 40 + thumb + 2), "REFERENCE (z=0)", fill="red", font=font)

        # Closest 3
        for i, (ds_d, seed, z_idx, p_d) in enumerate(closest_3):
            z_128 = directions[z_idx] * R
            z_4096 = (W_torch @ torch.from_numpy(z_128).to(device, dtype=torch.float16)).unsqueeze(0)
            embeds = gen.encode_batch_insert(prompt, z_4096)
            imgs = gen.generate_batch(embeds, [seed])
            img = imgs[0]

            x = (i + 1) * thumb
            img_resized = img.resize((thumb, thumb), Image.LANCZOS)
            grid.paste(img_resized, (x, 40))
            draw.text((x + 2, 40 + thumb + 2), f"CLOSEST #{i+1}", fill="green", font=font)
            draw.text((x + 2, 40 + thumb + 18), f"DS={ds_d:.4f} seed={seed}", fill="gray", font=get_font(11))
            draw.text((x + 2, 40 + thumb + 32), f"prompt_d={p_d:.6f}", fill="gray", font=get_font(11))
            img.close()

        # Furthest 3
        for i, (ds_d, seed, z_idx, p_d) in enumerate(furthest_3):
            z_128 = directions[z_idx] * R
            z_4096 = (W_torch @ torch.from_numpy(z_128).to(device, dtype=torch.float16)).unsqueeze(0)
            embeds = gen.encode_batch_insert(prompt, z_4096)
            imgs = gen.generate_batch(embeds, [seed])
            img = imgs[0]

            x = (i + 4) * thumb
            img_resized = img.resize((thumb, thumb), Image.LANCZOS)
            grid.paste(img_resized, (x, 40))
            draw.text((x + 2, 40 + thumb + 2), f"FURTHEST #{i+1}", fill="red", font=font)
            draw.text((x + 2, 40 + thumb + 18), f"DS={ds_d:.4f} seed={seed}", fill="gray", font=get_font(11))
            draw.text((x + 2, 40 + thumb + 32), f"prompt_d={p_d:.6f}", fill="gray", font=get_font(11))
            img.close()

        grid_path = os.path.join(run_dir, f"visual_R{R}.png")
        grid.save(grid_path, quality=95)
        print(f"  Saved {grid_path}")

    # --- Scatter plots ---
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
            ax.scatter(subset["prompt_emb_dist"], subset["dreamsim_dist"],
                       c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")

        corr = np.corrcoef(subset_R["prompt_emb_dist"], subset_R["dreamsim_dist"])[0, 1]
        ax.set_title(f"R={R}\nPearson r={corr:.4f}")
        ax.set_xlabel("Prompt Embedding Distance (cosine to z=0 ref)")
        if ax_idx == 0:
            ax.set_ylabel("DreamSim Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Prompt Embedding Distance vs DreamSim Distance", fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join(run_dir, "correlation_by_R.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Scatter plot saved to {plot_path}")

    # --- Summary ---
    print("\n=== Per-R Correlation Summary ===")
    for R in R_values:
        subset = df[df["R"] == R]
        corr = np.corrcoef(subset["prompt_emb_dist"], subset["dreamsim_dist"])[0, 1]
        print(f"  R={R:4d}: r={corr:.4f}, prompt_dist={subset['prompt_emb_dist'].mean():.6f}±{subset['prompt_emb_dist'].std():.6f}, "
              f"DS dist={subset['dreamsim_dist'].mean():.4f}±{subset['dreamsim_dist'].std():.4f}")

    ref_img.close()


if __name__ == "__main__":
    main()
