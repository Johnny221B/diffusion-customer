"""
Seed ranking: for each of 100 seeds, sample 10 random z vectors,
generate images, compute avg DreamSim distance to reference.
Report top-10 seeds with lowest avg distance.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=10, help="z samples per seed")
    parser.add_argument("--R", type=float, default=10.0)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/seed_ranking_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # --- Projection matrix W ---
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

    # --- Sample z vectors (shared across all seeds) ---
    rng = np.random.RandomState(0)
    z_128_list = []
    for _ in range(args.n_samples):
        z = rng.randn(128).astype(np.float32)
        z = z / np.linalg.norm(z) * args.R
        z_128_list.append(z)

    # Pre-compute z_4096 batch: (n_samples, 4096)
    z_128_batch = torch.from_numpy(np.array(z_128_list)).to(device, dtype=torch.float16)
    z_4096_batch = z_128_batch @ W_torch.T  # (n_samples, 4096)

    # Pre-encode prompt embeddings (same for all)
    embeds = gen.encode_batch_insert(prompt, z_4096_batch)

    # --- Generate seeds ---
    seed_rng = np.random.RandomState(777)
    seeds = seed_rng.randint(0, 2_000_000, size=args.n_seeds).tolist()

    # --- Run ---
    seed_avg_dists = {}

    for s_idx, seed in enumerate(seeds):
        # Generate batch of n_samples images, all with the same seed
        seed_list = [seed] * args.n_samples
        imgs = gen.generate_batch(embeds, seed_list)

        dists = []
        for img in imgs:
            d = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            dists.append(d)
            img.close()

        avg_d = float(np.mean(dists))
        seed_avg_dists[seed] = avg_d

        if (s_idx + 1) % 10 == 0:
            best_seed = min(seed_avg_dists, key=seed_avg_dists.get)
            print(f"  [{s_idx+1}/{args.n_seeds}] best so far: seed={best_seed}, avg_dist={seed_avg_dists[best_seed]:.4f}")

    # --- Rank ---
    ranked = sorted(seed_avg_dists.items(), key=lambda x: x[1])

    print(f"\n=== Top 10 Seeds (lowest avg DreamSim distance to reference) ===")
    for rank, (seed, avg_d) in enumerate(ranked[:10]):
        print(f"  #{rank+1}: seed={seed}, avg_dist={avg_d:.4f}")

    # --- Save CSV ---
    import pandas as pd
    df = pd.DataFrame(ranked, columns=["seed", "avg_dreamsim_dist"])
    csv_path = os.path.join(run_dir, "seed_ranking.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull ranking saved to {csv_path}")

    # --- Generate visual grid for top-10 seeds (z=0 baseline image per seed) ---
    print("\nGenerating top-10 seed baseline images (z=0)...")
    top_seeds = [s for s, _ in ranked[:10]]
    top_dists = [d for _, d in ranked[:10]]

    z_zero = torch.zeros(1, 4096, device=device, dtype=torch.float16)
    embeds_zero = gen.encode_batch_insert(prompt, z_zero)

    font = get_font(13)
    font_title = get_font(16)
    thumb = 248
    label_h = 40
    cols = 11  # ref + 10 seeds
    grid_w = cols * thumb
    grid_h = 36 + thumb + label_h
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((10, 8), "Top-10 Seeds (z=0 baseline)", fill="black", font=font_title)

    # Reference
    ref_img = Image.open(args.ref_image).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
    grid.paste(ref_img, (0, 36))
    draw.text((4, 36 + thumb + 2), "REFERENCE", fill="red", font=font)

    for i, seed in enumerate(top_seeds):
        imgs = gen.generate_batch(embeds_zero, [seed])
        img = imgs[0]
        img_resized = img.resize((thumb, thumb), Image.LANCZOS)
        x = (i + 1) * thumb
        grid.paste(img_resized, (x, 36))
        draw.text((x + 2, 36 + thumb + 2), f"seed={seed}", fill="black", font=font)
        draw.text((x + 2, 36 + thumb + 18), f"avg_DS={top_dists[i]:.4f}", fill="gray", font=get_font(11))
        img.save(os.path.join(run_dir, f"top{i+1}_seed{seed}.png"))
        img.close()

    grid_path = os.path.join(run_dir, "top10_seeds_grid.png")
    grid.save(grid_path, quality=95)
    print(f"Grid saved to {grid_path}")


if __name__ == "__main__":
    main()
