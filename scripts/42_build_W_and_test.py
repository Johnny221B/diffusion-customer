"""
Step 1: Collect word embeddings from SD3.5 for many words
Step 2: PCA -> build new W (4096 x 128) from top-128 principal components
Step 3: Validate: sample z through new W, check correlation with DreamSim
Step 4: Test with opposing word pairs (reference vs competitor)
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from scipy import stats

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


# Large word list covering colors, materials, styles, shapes, objects
WORD_LIST = [
    # Colors
    "red", "blue", "green", "black", "white", "yellow", "orange", "purple",
    "pink", "brown", "gray", "silver", "gold", "bronze", "beige", "ivory",
    "crimson", "turquoise", "indigo", "maroon", "coral", "teal", "navy",
    "magenta", "olive", "tan", "cream", "charcoal", "burgundy", "lavender",
    # Materials
    "leather", "suede", "canvas", "rubber", "plastic", "metal", "wood",
    "silk", "cotton", "wool", "nylon", "mesh", "denim", "velvet", "satin",
    "linen", "polyester", "foam", "cork", "patent",
    # Styles
    "elegant", "sporty", "casual", "formal", "vintage", "modern", "retro",
    "classic", "minimalist", "bold", "sleek", "rugged", "luxurious", "cheap",
    "expensive", "premium", "basic", "fancy", "trendy", "traditional",
    # Shapes/Properties
    "chunky", "slim", "flat", "thick", "thin", "round", "pointed", "narrow",
    "wide", "tall", "short", "heavy", "light", "soft", "hard", "smooth",
    "rough", "glossy", "matte", "shiny", "transparent", "opaque",
    # Shoe types
    "boot", "sandal", "sneaker", "heel", "loafer", "slipper", "clog",
    "moccasin", "oxford", "derby", "pump", "wedge", "platform", "flip",
    # Nature/Elements
    "fire", "ocean", "ice", "snow", "rain", "sun", "moon", "star",
    "forest", "desert", "mountain", "river", "storm", "wind", "earth",
    # Abstract
    "fast", "slow", "warm", "cool", "bright", "dark", "loud", "quiet",
    "sharp", "dull", "new", "old", "clean", "dirty", "wet", "dry",
    # Brands/Culture
    "nike", "adidas", "puma", "running", "walking", "hiking", "dancing",
    "athletic", "outdoor", "indoor", "urban", "rural", "street", "luxury",
    # More adjectives
    "comfortable", "uncomfortable", "breathable", "waterproof", "durable",
    "fragile", "flexible", "rigid", "elastic", "tight", "loose",
    "colorful", "monochrome", "striped", "spotted", "plain", "patterned",
    "neon", "pastel", "vivid", "subtle", "dramatic", "understated",
]


def get_word_emb(pipe, word):
    """Extract single token embedding for a word."""
    with torch.no_grad():
        out = pipe.encode_prompt(prompt=word, prompt_2=word, prompt_3=word, negative_prompt="")
        out_empty = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", negative_prompt="")
    pe = out[0]
    ee = out_empty[0]
    L_w, L_e = pe.shape[1], ee.shape[1]
    if L_w > L_e:
        n = L_w - L_e
        return pe[0, :n, :].mean(dim=0).detach()
    else:
        ml = min(L_w, L_e)
        diffs = (pe[0, :ml] - ee[0, :ml]).norm(dim=1)
        idx = diffs.argmax().item()
        return pe[0, idx, :].detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--dim_z", type=int, default=128)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/learned_W_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # Load model
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # =========================================
    # Step 1: Collect word embeddings
    # =========================================
    print(f"\n=== Step 1: Collecting {len(WORD_LIST)} word embeddings ===")
    word_embs = []
    valid_words = []
    for i, word in enumerate(WORD_LIST):
        try:
            emb = get_word_emb(gen.pipe, word)
            word_embs.append(emb.float().cpu().numpy())
            valid_words.append(word)
        except Exception as e:
            print(f"  Skipping '{word}': {e}")
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(WORD_LIST)}] collected {len(valid_words)} embeddings")

    word_emb_matrix = np.stack(word_embs)  # (N_words, 4096)
    print(f"  Collected {len(valid_words)} word embeddings, shape={word_emb_matrix.shape}")
    print(f"  Norm range: [{np.linalg.norm(word_emb_matrix, axis=1).min():.2f}, {np.linalg.norm(word_emb_matrix, axis=1).max():.2f}]")

    # =========================================
    # Step 2: PCA -> build W
    # =========================================
    print(f"\n=== Step 2: PCA (top-{args.dim_z} components) ===")

    # Center the embeddings
    mean_emb = word_emb_matrix.mean(axis=0)

    pca = PCA(n_components=args.dim_z)
    pca.fit(word_emb_matrix)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    print(f"  Top-10 explained variance: {explained_var[:10]}")
    print(f"  Cumulative variance (top-{args.dim_z}): {cumulative_var[-1]:.4f}")
    print(f"  Top-10 cumulative: {cumulative_var[9]:.4f}")
    print(f"  Top-32 cumulative: {cumulative_var[31]:.4f}")
    print(f"  Top-64 cumulative: {cumulative_var[63]:.4f}")

    # W = PCA components transposed: (4096, dim_z)
    # Each column is a principal direction
    W_new = pca.components_.T.astype(np.float32)  # (4096, 128)
    W_new_torch = torch.from_numpy(W_new).to(dtype=torch.float16, device=device)

    # Save
    np.savez(os.path.join(run_dir, "W_pca.npz"),
             W=W_new, mean_emb=mean_emb,
             explained_variance_ratio=explained_var,
             valid_words=valid_words)
    print(f"  W saved: shape={W_new.shape}")

    # =========================================
    # Step 3: Validate correlation
    # =========================================
    print(f"\n=== Step 3: Validate z-distance vs DreamSim ===")

    # Use a word as reference direction in the new z-space
    ref_word = "leather"
    ref_emb_4096 = get_word_emb(gen.pipe, ref_word).float().cpu().numpy()
    # Project ref into z-space: z = W^T @ emb (since W columns are orthonormal)
    z_ref_128 = (ref_emb_4096 - mean_emb) @ W_new  # (128,)
    R = np.linalg.norm(z_ref_128)
    z_ref_dir = z_ref_128 / R
    print(f"  Reference '{ref_word}' projected: ||z||={R:.2f}")

    # Generate reference image
    z_ref_4096 = W_new_torch @ torch.from_numpy(z_ref_128).to(device, dtype=torch.float16)
    z_ref_4096 = (z_ref_4096 + torch.from_numpy(mean_emb).to(device, dtype=torch.float16)).unsqueeze(0)
    embeds_ref = gen.encode_batch_insert(prompt, z_ref_4096)
    ref_imgs = gen.generate_batch(embeds_ref, [args.seed])
    ref_img = ref_imgs[0]
    ref_img.save(os.path.join(run_dir, "ref_leather.png"))
    ref_tensor = scorer.preprocess(ref_img)

    # Sample random z in 128-dim, project through new W
    seeds = [945737, 1763690, 1082459, 1755275]
    n_samples = 25
    rng = np.random.RandomState(0)
    directions = [rng.randn(args.dim_z).astype(np.float32) for _ in range(n_samples)]
    directions = [d / np.linalg.norm(d) for d in directions]

    import pandas as pd
    results = []

    for seed in seeds:
        print(f"  seed={seed}...")
        for z_idx, d in enumerate(directions):
            z_128 = d * R  # same norm as ref
            z_dist = float(np.linalg.norm(z_128 - z_ref_128))
            cos_sim = float(np.dot(d, z_ref_dir))

            z_4096 = W_new_torch @ torch.from_numpy(z_128).to(device, dtype=torch.float16)
            z_4096 = (z_4096 + torch.from_numpy(mean_emb).to(device, dtype=torch.float16)).unsqueeze(0)
            embeds = gen.encode_batch_insert(prompt, z_4096)
            imgs = gen.generate_batch(embeds, [seed])
            img = imgs[0]

            ds_dist = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            results.append({
                "seed": seed, "z_idx": z_idx,
                "z_dist": z_dist, "cos_sim": cos_sim,
                "dreamsim_dist": ds_dist,
            })
            img.close()

    ref_img.close()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, "validation.csv"), index=False)

    # Correlation
    r_l2, p_l2 = stats.pearsonr(df["z_dist"], df["dreamsim_dist"])
    r_cos, p_cos = stats.pearsonr(df["cos_sim"], df["dreamsim_dist"])

    print(f"\n  L2 distance:  r={r_l2:.4f}, p={p_l2:.6f}")
    print(f"  Cosine sim:   r={r_cos:.4f}, p={p_cos:.6f}")

    # Per-seed
    print("  Per-seed (L2):")
    for seed in seeds:
        s = df[df["seed"] == seed]
        rs, ps = stats.pearsonr(s["z_dist"], s["dreamsim_dist"])
        print(f"    seed={seed}: r={rs:.4f}, p={ps:.4f}")

    # =========================================
    # Step 4: Plot
    # =========================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors_plt = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PCA variance
    ax = axes[0]
    ax.plot(cumulative_var, 'b-')
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"PCA: {len(valid_words)} words -> {args.dim_z} dims\nTotal var={cumulative_var[-1]:.4f}")
    ax.grid(True, alpha=0.3)

    # L2 distance vs DreamSim
    ax = axes[1]
    for s_idx, seed in enumerate(seeds):
        s = df[df["seed"] == seed]
        ax.scatter(s["z_dist"], s["dreamsim_dist"], c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")
    ax.set_xlabel("||z - z_ref|| (L2, 128-dim)")
    ax.set_ylabel("DreamSim Distance")
    ax.set_title(f"Learned W: L2 vs DreamSim\nr={r_l2:.4f}, p={p_l2:.4f}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Cosine similarity vs DreamSim
    ax = axes[2]
    for s_idx, seed in enumerate(seeds):
        s = df[df["seed"] == seed]
        ax.scatter(s["cos_sim"], s["dreamsim_dist"], c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")
    ax.set_xlabel("Cosine Similarity (z, z_ref)")
    ax.set_ylabel("DreamSim Distance")
    ax.set_title(f"Learned W: Cosine vs DreamSim\nr={r_cos:.4f}, p={p_cos:.4f}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("PCA-based W: Validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "validation_plot.png"), dpi=150)
    print(f"\nPlot saved to {run_dir}/validation_plot.png")


if __name__ == "__main__":
    main()
