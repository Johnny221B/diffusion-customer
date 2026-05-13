"""
Initialization: sample K random x vectors in 128-dim, generate images,
score with DreamSim against reference, and save the best x0 + seed.

Output:
- best_x0.npz: contains x0 (128-dim), seed, dreamsim_dist, all scores
- visual grid of top candidates vs reference
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


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


def make_topk_grid(ref_img_path, images, dists, indices, seed, thumb=200):
    """Grid showing reference + top-K candidates, labeled with DreamSim distance."""
    n = len(images)
    cols = min(n + 1, 11)  # ref + up to 10 candidates per row
    rows = 1 + (n // (cols - 1)) + (1 if n % (cols - 1) else 0)

    font = get_font(13)
    font_title = get_font(16)
    label_h = 36
    title_h = 36
    cell_h = thumb + label_h

    grid_w = cols * thumb
    grid_h = title_h + rows * cell_h
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    draw.text((10, 8), f"Top-{n} candidates (seed={seed})", fill="black", font=font_title)

    # Reference
    ref_img = Image.open(ref_img_path).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
    grid.paste(ref_img, (0, title_h))
    draw.text((4, title_h + thumb + 2), "REFERENCE", fill="red", font=font)

    # Candidates
    for i in range(n):
        pos = i + 1
        r = pos // cols
        c = pos % cols
        x = c * thumb
        y = title_h + r * cell_h

        img_resized = images[i].resize((thumb, thumb), Image.LANCZOS)
        grid.paste(img_resized, (x, y))

        rank_label = f"#{i+1} idx={indices[i]}"
        dist_label = f"DS={dists[i]:.4f}"
        draw.text((x + 2, y + thumb + 2), rank_label, fill="black", font=font)
        draw.text((x + 2, y + thumb + 16), dist_label, fill="gray", font=get_font(11))

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--K", type=int, default=100, help="Number of candidate x vectors")
    parser.add_argument("--R", type=float, default=10.0, help="Norm of x vectors")
    parser.add_argument("--seed", type=int, default=42, help="Fixed generation seed")
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/init_x0_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # --- Projection matrix W (same as main pipeline) ---
    W_raw = np.random.RandomState(42).randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)

    # --- Load models ---
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)

    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=device)
    ref_tensor = scorer.preprocess(args.ref_image)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Sample K random x vectors ---
    rng = np.random.RandomState(0)
    x_candidates = []
    for _ in range(args.K):
        x = rng.randn(128).astype(np.float32)
        x = x / np.linalg.norm(x) * args.R
        x_candidates.append(x)

    # --- Generate and score ---
    all_dists = []
    print(f"Generating {args.K} images with seed={args.seed}...")

    for i, x_128 in enumerate(x_candidates):
        z_4096 = (W_torch @ torch.from_numpy(x_128).to(device, dtype=torch.float16)).unsqueeze(0)  # (1, 4096)
        embeds = gen.encode_batch_insert(prompt, z_4096)
        imgs = gen.generate_batch(embeds, [args.seed])
        img = imgs[0]

        ds_dist = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        all_dists.append(ds_dist)
        img.close()

        if (i + 1) % 20 == 0:
            best_so_far = min(all_dists)
            print(f"  [{i+1}/{args.K}] current best DreamSim dist = {best_so_far:.4f}")

    all_dists = np.array(all_dists)

    # --- Find best ---
    best_idx = int(np.argmin(all_dists))
    best_x0 = x_candidates[best_idx]
    best_dist = all_dists[best_idx]

    print(f"\n=== Best candidate ===")
    print(f"  Index: {best_idx}")
    print(f"  DreamSim dist: {best_dist:.4f}")
    print(f"  ||x0||: {np.linalg.norm(best_x0):.2f}")
    print(f"  Seed: {args.seed}")

    # --- Save x0 ---
    # Build prior_mean for optimizer: [intercept=0, x0]
    prior_mean = np.concatenate(([0.0], best_x0)).astype(np.float32)

    save_path = os.path.join(run_dir, "best_x0.npz")
    np.savez(
        save_path,
        x0=best_x0,
        prior_mean=prior_mean,
        seed=args.seed,
        best_dist=best_dist,
        best_idx=best_idx,
        all_dists=all_dists,
        R=args.R,
        K=args.K,
    )
    print(f"Saved to {save_path}")

    # --- Generate top-20 images for visual grid ---
    top_k = 20
    sorted_indices = np.argsort(all_dists)[:top_k]
    top_images = []
    top_dists = []

    print(f"\nRe-generating top-{top_k} for visual grid...")
    for rank, idx in enumerate(sorted_indices):
        x_128 = x_candidates[idx]
        z_4096 = (W_torch @ torch.from_numpy(x_128).to(device, dtype=torch.float16)).unsqueeze(0)  # (1, 4096)
        embeds = gen.encode_batch_insert(prompt, z_4096)
        imgs = gen.generate_batch(embeds, [args.seed])
        img = imgs[0]

        # Save individual image
        img.save(os.path.join(run_dir, f"rank{rank:02d}_idx{idx}_ds{all_dists[idx]:.4f}.png"))
        top_images.append(img)
        top_dists.append(all_dists[idx])

    grid = make_topk_grid(args.ref_image, top_images, top_dists, sorted_indices, args.seed)
    grid_path = os.path.join(run_dir, "top_candidates_grid.png")
    grid.save(grid_path, quality=95)
    print(f"Grid saved to {grid_path}")

    for img in top_images:
        img.close()

    # --- Stats ---
    print(f"\n=== Distribution of DreamSim distances ===")
    print(f"  Min:  {all_dists.min():.4f}")
    print(f"  Max:  {all_dists.max():.4f}")
    print(f"  Mean: {all_dists.mean():.4f}")
    print(f"  Std:  {all_dists.std():.4f}")


if __name__ == "__main__":
    main()
