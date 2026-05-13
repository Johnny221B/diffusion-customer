"""
Step 1: Collect (z, DreamSim_distance) data pairs.
Step 2: Fit linear model to find informative z-directions.
Step 3: Validate by generating images along the learned direction.

Fixed seed, random z vectors, compute DreamSim to reference.
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_seed", type=int, default=1810772)
    parser.add_argument("--N", type=int, default=200, help="Number of z samples for training")
    parser.add_argument("--R", type=float, default=500.0, help="Norm of z vectors")
    parser.add_argument("--dim_z", type=int, default=128)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/learn_W_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # --- Projection matrix W (same as pipeline) ---
    W_raw = np.random.RandomState(42).randn(4096, args.dim_z).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)

    # --- Load models ---
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Generate reference image (z=0) ---
    print("Generating reference image (z=0)...")
    z_zero = torch.zeros(1, 4096, device=device, dtype=torch.float16)
    embeds_ref = gen.encode_batch_insert(prompt, z_zero)
    ref_imgs = gen.generate_batch(embeds_ref, [args.ref_seed])
    ref_img = ref_imgs[0]
    ref_img.save(os.path.join(run_dir, "reference.png"))
    ref_tensor = scorer.preprocess(ref_img)
    ref_img.close()

    # =========================================
    # Step 1: Collect data
    # =========================================
    print(f"\n=== Step 1: Collecting {args.N} (z, DreamSim) pairs ===")
    rng = np.random.RandomState(0)

    Z_data = np.zeros((args.N, args.dim_z), dtype=np.float32)
    y_data = np.zeros(args.N, dtype=np.float32)

    for i in range(args.N):
        z = rng.randn(args.dim_z).astype(np.float32)
        z = z / np.linalg.norm(z) * args.R

        z_4096 = (W_torch @ torch.from_numpy(z).to(device, dtype=torch.float16)).unsqueeze(0)
        embeds = gen.encode_batch_insert(prompt, z_4096)
        imgs = gen.generate_batch(embeds, [args.ref_seed])
        img = imgs[0]

        ds_dist = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        Z_data[i] = z
        y_data[i] = ds_dist
        img.close()

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{args.N}] DreamSim range: [{y_data[:i+1].min():.4f}, {y_data[:i+1].max():.4f}]")

    # Save data
    np.savez(os.path.join(run_dir, "train_data.npz"), Z=Z_data, y=y_data, R=args.R)
    print(f"  DreamSim: mean={y_data.mean():.4f}, std={y_data.std():.4f}, range=[{y_data.min():.4f}, {y_data.max():.4f}]")

    # =========================================
    # Step 2: Fit linear model
    # =========================================
    print(f"\n=== Step 2: Linear regression z -> DreamSim ===")

    # Normalize z to unit directions (since all have same norm R)
    Z_dirs = Z_data / args.R  # unit vectors

    # Ridge regression with cross-validation
    model = Ridge(alpha=1.0)
    cv_scores = cross_val_score(model, Z_dirs, y_data, cv=5, scoring='r2')
    print(f"  5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per-fold: {[f'{s:.4f}' for s in cv_scores]}")

    # Fit on all data
    model.fit(Z_dirs, y_data)
    y_pred = model.predict(Z_dirs)
    from sklearn.metrics import r2_score
    train_r2 = r2_score(y_data, y_pred)
    print(f"  Train R²: {train_r2:.4f}")

    # The learned weight vector
    w_learned = model.coef_  # (128,)
    w_norm = w_learned / np.linalg.norm(w_learned)
    print(f"  ||w||: {np.linalg.norm(w_learned):.4f}")

    # Save
    np.savez(os.path.join(run_dir, "learned_w.npz"),
             w=w_learned, w_norm=w_norm, intercept=model.intercept_,
             cv_r2_mean=cv_scores.mean(), cv_r2_std=cv_scores.std(),
             train_r2=train_r2)

    # =========================================
    # Step 3: Validate - generate along learned direction
    # =========================================
    print(f"\n=== Step 3: Validation ===")

    # Generate images along w direction and opposite
    scales = np.linspace(-args.R, args.R, 11)
    val_results = []

    print("  Generating images along learned direction w...")
    for scale in scales:
        z_val = w_norm * scale
        z_4096 = (W_torch @ torch.from_numpy(z_val).to(device, dtype=torch.float16)).unsqueeze(0)
        embeds = gen.encode_batch_insert(prompt, z_4096)
        imgs = gen.generate_batch(embeds, [args.ref_seed])
        img = imgs[0]
        ds_dist = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        val_results.append({"scale": float(scale), "dreamsim_dist": ds_dist, "direction": "learned_w"})
        img.close()

    # Also generate along a random direction for comparison
    random_dir = rng.randn(args.dim_z).astype(np.float32)
    random_dir = random_dir / np.linalg.norm(random_dir)

    print("  Generating images along random direction...")
    for scale in scales:
        z_val = random_dir * scale
        z_4096 = (W_torch @ torch.from_numpy(z_val).to(device, dtype=torch.float16)).unsqueeze(0)
        embeds = gen.encode_batch_insert(prompt, z_4096)
        imgs = gen.generate_batch(embeds, [args.ref_seed])
        img = imgs[0]
        ds_dist = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        val_results.append({"scale": float(scale), "dreamsim_dist": ds_dist, "direction": "random"})
        img.close()

    import pandas as pd
    df_val = pd.DataFrame(val_results)
    df_val.to_csv(os.path.join(run_dir, "validation.csv"), index=False)

    # =========================================
    # Step 4: Plot
    # =========================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: Predicted vs Actual
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(y_data, y_pred, alpha=0.5, s=20)
    ax.plot([y_data.min(), y_data.max()], [y_data.min(), y_data.max()], 'r--')
    ax.set_xlabel("Actual DreamSim Distance")
    ax.set_ylabel("Predicted DreamSim Distance")
    ax.set_title(f"Linear Model: Train R²={train_r2:.4f}\nCV R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation - learned vs random direction
    ax = axes[1]
    df_w = df_val[df_val["direction"] == "learned_w"]
    df_r = df_val[df_val["direction"] == "random"]
    ax.plot(df_w["scale"], df_w["dreamsim_dist"], 'b-o', label="Learned w direction")
    ax.plot(df_r["scale"], df_r["dreamsim_dist"], 'gray', marker='x', linestyle='--', label="Random direction")
    ax.set_xlabel("Scale (along direction)")
    ax.set_ylabel("DreamSim Distance")
    ax.set_title("DreamSim along Learned vs Random Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Weight vector magnitude per dimension
    ax = axes[2]
    sorted_idx = np.argsort(np.abs(w_learned))[::-1]
    ax.bar(range(len(w_learned)), np.abs(w_learned[sorted_idx]), alpha=0.7)
    ax.set_xlabel("Dimension (sorted by |w|)")
    ax.set_ylabel("|w_i|")
    ax.set_title("Learned Weight Vector Magnitudes")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Metric Learning (N={args.N}, R={args.R}, dim={args.dim_z})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "results.png"), dpi=150)
    print(f"\nPlot saved to {run_dir}/results.png")

    # Final summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"  Training samples: {args.N}")
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Learned direction DreamSim range: [{df_w['dreamsim_dist'].min():.4f}, {df_w['dreamsim_dist'].max():.4f}]")
    print(f"  Random direction DreamSim range:  [{df_r['dreamsim_dist'].min():.4f}, {df_r['dreamsim_dist'].max():.4f}]")


if __name__ == "__main__":
    main()
