"""
Experiment: Correlation between CLIP image embedding distance and DreamSim distance.

For N different z vectors, each with M different seeds (same embedding, different images),
measure both CLIP embedding distance and DreamSim distance to the reference image.

Output:
1. Per-z visual grids: reference + generated images, each labeled with (CLIP dist, DreamSim dist)
2. Scatter plot: X = CLIP dist, Y = DreamSim dist
3. CSV raw data
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor

from src.sd35_batch_generator import SD35EmbeddingGenerator
from src.scorer import DreamSimScorer


class CLIPImageEmbedder:
    """Extract CLIP image embeddings and compute cosine distance."""

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


def get_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_z_grid(ref_img_path, images, clip_dists, ds_dists, seeds, z_idx, z_norm,
                cols=10, thumb=200, label_h=40):
    """
    One grid per z: top-left is the reference, rest are generated images.
    Each image is labeled with CLIP dist and DreamSim dist.
    """
    n = len(images)
    rows = 1 + (n // cols) + (1 if n % cols else 0)  # +1 for ref row

    font = get_font(13)
    font_title = get_font(16)
    title_h = 36
    cell_h = thumb + label_h

    grid_w = cols * thumb
    grid_h = title_h + rows * cell_h
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    # Title
    draw.text((10, 8), f"z[{z_idx}]  ||z||={z_norm:.1f}", fill="black", font=font_title)

    # Reference image in position (0,0)
    ref_img = Image.open(ref_img_path).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
    y0 = title_h
    grid.paste(ref_img, (0, y0))
    draw.text((4, y0 + thumb + 2), "REFERENCE", fill="red", font=font)

    # Generated images starting from position (1,0)
    for i in range(n):
        pos = i + 1  # skip ref position
        r = pos // cols
        c = pos % cols
        x = c * thumb
        y = title_h + r * cell_h

        img_resized = images[i].resize((thumb, thumb), Image.LANCZOS)
        grid.paste(img_resized, (x, y))

        label = f"C:{clip_dists[i]:.3f} D:{ds_dists[i]:.3f}"
        draw.text((x + 2, y + thumb + 2), label, fill="black", font=font)
        draw.text((x + 2, y + thumb + 18), f"seed={seeds[i]}", fill="gray", font=get_font(11))

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--comp_image", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--N", type=int, default=5, help="Number of different z vectors")
    parser.add_argument("--M", type=int, default=50, help="Number of seeds per z")
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/correlation_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # --- Build projection matrix W (same as main pipeline) ---
    W_raw = np.random.RandomState(42).randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)

    # --- Load models ---
    print("Loading SD3.5...")
    gen = SD35EmbeddingGenerator(args.model_path, device=device)

    print("Loading DreamSim...")
    dreamsim_scorer = DreamSimScorer(device=device)
    ref_tensor_ds = dreamsim_scorer.preprocess(args.ref_image)

    print("Loading CLIP...")
    clip_embedder = CLIPImageEmbedder(device=device)
    ref_emb_clip = clip_embedder.get_image_embedding(args.ref_image)

    # --- Competitor distances ---
    comp_emb_clip = clip_embedder.get_image_embedding(args.comp_image)
    comp_ds_dist = dreamsim_scorer.model(ref_tensor_ds, dreamsim_scorer.preprocess(args.comp_image)).item()
    comp_clip_dist = clip_embedder.cosine_distance(ref_emb_clip, comp_emb_clip)
    print(f"Competitor -> DreamSim dist: {comp_ds_dist:.4f}, CLIP dist: {comp_clip_dist:.4f}")

    # --- Seeds ---
    seeds = list(range(args.M))

    # --- Sample z vectors: z[0]=zero, z[1..N-1]=random with norm R=10 ---
    rng = np.random.RandomState(0)
    z_list = [np.zeros(128, dtype=np.float32)]
    for _ in range(args.N - 1):
        z = rng.randn(128).astype(np.float32)
        z = z / np.linalg.norm(z) * 10.0
        z_list.append(z)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Run experiment ---
    all_results = []

    for z_idx, z_128 in enumerate(z_list):
        z_4096 = W_torch @ torch.from_numpy(z_128).to(device, dtype=torch.float16)
        embeds = gen.encode_simple_insert(prompt, z_4096)
        z_norm = float(np.linalg.norm(z_128))

        print(f"\nz[{z_idx}] norm={z_norm:.2f}, generating {args.M} images...")

        z_images = []
        z_clip_dists = []
        z_ds_dists = []

        for seed in seeds:
            img = gen.generate(embeds, seed=seed)

            img_emb = clip_embedder.get_image_embedding(img)
            clip_dist = clip_embedder.cosine_distance(ref_emb_clip, img_emb)
            ds_dist = dreamsim_scorer.model(ref_tensor_ds, dreamsim_scorer.preprocess(img)).item()

            all_results.append((z_idx, seed, clip_dist, ds_dist))
            z_images.append(img)
            z_clip_dists.append(clip_dist)
            z_ds_dists.append(ds_dist)

        print(f"  CLIP dist: {np.mean(z_clip_dists):.4f} +/- {np.std(z_clip_dists):.4f}")
        print(f"  DreamSim dist: {np.mean(z_ds_dists):.4f} +/- {np.std(z_ds_dists):.4f}")

        # Save per-z grid
        grid = make_z_grid(
            args.ref_image, z_images, z_clip_dists, z_ds_dists, seeds,
            z_idx, z_norm, cols=10, thumb=200,
        )
        grid_path = os.path.join(run_dir, f"z{z_idx}_grid.png")
        grid.save(grid_path, quality=95)
        print(f"  Grid saved to {grid_path}")

        # Save individual images
        z_dir = os.path.join(run_dir, f"z{z_idx}_images")
        os.makedirs(z_dir, exist_ok=True)
        for i, img in enumerate(z_images):
            fname = f"seed{seeds[i]:03d}_clip{z_clip_dists[i]:.3f}_ds{z_ds_dists[i]:.3f}.png"
            img.save(os.path.join(z_dir, fname))
            img.close()

    # --- Save CSV ---
    import pandas as pd
    df = pd.DataFrame(all_results, columns=["z_idx", "seed", "clip_dist", "dreamsim_dist"])
    csv_path = os.path.join(run_dir, "correlation_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to {csv_path}")

    # --- Scatter plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, args.N))
    for z_idx in range(args.N):
        subset = df[df["z_idx"] == z_idx]
        label = f"z[{z_idx}]" + (" (zero)" if z_idx == 0 else "")
        ax.scatter(subset["clip_dist"], subset["dreamsim_dist"],
                   c=[colors[z_idx]], alpha=0.6, s=30, label=label)

    ax.axhline(y=comp_ds_dist, color="red", linestyle="--", alpha=0.5,
               label=f"Competitor DreamSim={comp_ds_dist:.3f}")
    ax.axvline(x=comp_clip_dist, color="red", linestyle=":", alpha=0.5,
               label=f"Competitor CLIP={comp_clip_dist:.3f}")

    all_clip = df["clip_dist"].values
    all_ds = df["dreamsim_dist"].values
    corr = np.corrcoef(all_clip, all_ds)[0, 1]
    ax.set_title(f"CLIP Embedding Distance vs DreamSim Distance\nPearson r = {corr:.4f}")
    ax.set_xlabel("CLIP Image Embedding Distance (cosine)")
    ax.set_ylabel("DreamSim Distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(run_dir, "correlation_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Scatter plot saved to {plot_path}")

    # --- Per-z summary ---
    print("\n=== Per-z Summary ===")
    for z_idx in range(args.N):
        subset = df[df["z_idx"] == z_idx]
        print(f"z[{z_idx}]: CLIP = {subset['clip_dist'].mean():.4f}±{subset['clip_dist'].std():.4f}, "
              f"DreamSim = {subset['dreamsim_dist'].mean():.4f}±{subset['dreamsim_dist'].std():.4f}")
    print(f"\nOverall Pearson r = {corr:.4f}")


if __name__ == "__main__":
    main()
