"""
R Sweep: For each R, sample 12 directions from N(0,I), scale to norm R,
generate images with a fixed seed, and produce a visual grid.

Rows = different R values, Columns = different x_i directions.
Uses SD35BatchEmbeddingGenerator for consistency with main pipeline.
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator


def make_grid(images_2d, R_values, n_samples, img_size=200):
    """
    images_2d: list of lists, images_2d[r_idx][sample_idx] = PIL Image
    Returns a single PIL image grid with R labels on the left.
    """
    pad = 4
    label_w = 80
    cols = n_samples
    rows = len(R_values)

    grid_w = label_w + cols * (img_size + pad) + pad
    grid_h = rows * (img_size + pad) + pad
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for r_idx, R in enumerate(R_values):
        y = pad + r_idx * (img_size + pad)
        draw.text((8, y + img_size // 2 - 10), f"R={R}", fill="black", font=font)

        for s_idx in range(cols):
            x = label_w + pad + s_idx * (img_size + pad)
            img = images_2d[r_idx][s_idx].resize((img_size, img_size), Image.LANCZOS)
            grid.paste(img, (x, y))

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--n_samples", type=int, default=12)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/R_sweep_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    R_values = [50, 80, 100, 150, 200, 250, 300]

    # --- Projection matrix W ---
    W_raw = np.random.RandomState(42).randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)

    # --- Sample 12 random directions (unit vectors) ---
    rng = np.random.RandomState(123)
    directions = []
    for _ in range(args.n_samples):
        x = rng.randn(128).astype(np.float32)
        x = x / np.linalg.norm(x)
        directions.append(x)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Load model ---
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)

    # --- z=0 baseline ---
    print("Generating z=0 baseline...")
    z_zero = torch.zeros(1, 4096, device=device, dtype=torch.float16)
    embeds_zero = gen.encode_batch_insert(prompt, z_zero)
    imgs = gen.generate_batch(embeds_zero, [args.seed])
    imgs[0].save(os.path.join(run_dir, "baseline_z0.png"))
    imgs[0].close()

    # --- Sweep ---
    images_2d = []

    for r_idx, R in enumerate(R_values):
        print(f"\nR={R}: generating {args.n_samples} images...")
        row_images = []
        row_dir = os.path.join(run_dir, f"R_{R:03d}")
        os.makedirs(row_dir, exist_ok=True)

        for s_idx, direction in enumerate(directions):
            z_128 = direction * R
            z_4096 = (W_torch @ torch.from_numpy(z_128).to(device, dtype=torch.float16)).unsqueeze(0)
            embeds = gen.encode_batch_insert(prompt, z_4096)
            imgs = gen.generate_batch(embeds, [args.seed])
            img = imgs[0]

            img.save(os.path.join(row_dir, f"x{s_idx:02d}.png"))
            row_images.append(img)

        images_2d.append(row_images)

    # --- Make grid ---
    print("\nMaking grid...")
    grid = make_grid(images_2d, R_values, args.n_samples)
    grid_path = os.path.join(run_dir, "R_sweep_grid.png")
    grid.save(grid_path, quality=95)
    print(f"Grid saved to {grid_path}")

    for row in images_2d:
        for img in row:
            img.close()


if __name__ == "__main__":
    main()
