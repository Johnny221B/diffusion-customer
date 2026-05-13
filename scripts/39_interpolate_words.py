"""
Experiment: Interpolate between two word embeddings in SD3.5's 4096-dim space.

Take two words with very different meanings, extract their token embeddings,
interpolate between them, insert each interpolated embedding as a token in the prompt,
generate images, and measure DreamSim distance to the z1 (word1) reference image.

Tests whether the embedding space itself supports smooth perceptual transitions.
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
        except:
            continue
    return ImageFont.load_default()


def get_word_embedding(pipe, word):
    """Extract the token embedding for a single word from SD3.5's text encoder.

    Returns the embedding of the word's token(s), NOT mean-pooled.
    Compares with empty prompt to isolate the word's contribution.
    """
    out = pipe.encode_prompt(
        prompt=word, prompt_2=word, prompt_3=word,
        negative_prompt=""
    )
    prompt_embeds = out[0]  # (1, L, 4096)

    out_empty = pipe.encode_prompt(
        prompt="", prompt_2="", prompt_3="",
        negative_prompt=""
    )
    empty_embeds = out_empty[0]  # (1, L_empty, 4096)

    # Find which token positions differ from empty prompt
    # The word tokens are the ones that are NOT in the empty prompt
    L_word = prompt_embeds.shape[1]
    L_empty = empty_embeds.shape[1]

    if L_word > L_empty:
        # Extra tokens are the word tokens
        # Take the first (L_word - L_empty) tokens as the word embedding
        n_word_tokens = L_word - L_empty
        word_emb = prompt_embeds[0, :n_word_tokens, :].mean(dim=0)  # (4096,)
        print(f"  '{word}': {n_word_tokens} word token(s), ||emb||={word_emb.norm().item():.4f}")
    else:
        # Same length - find max-diff token
        min_len = min(L_word, L_empty)
        diffs = (prompt_embeds[0, :min_len] - empty_embeds[0, :min_len]).norm(dim=1)
        max_diff_idx = diffs.argmax().item()
        word_emb = prompt_embeds[0, max_diff_idx, :]  # (4096,)
        print(f"  '{word}': max-diff at token {max_diff_idx}, diff={diffs[max_diff_idx].item():.4f}, ||emb||={word_emb.norm().item():.4f}")

    return word_emb, prompt_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--n_steps", type=int, default=21, help="Number of interpolation steps")
    parser.add_argument("--word1", type=str, default="fire")
    parser.add_argument("--word2", type=str, default="ocean")
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/interpolate_{args.word1}_{args.word2}_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # Load models
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Extract word embeddings ---
    print(f"Extracting embeddings for '{args.word1}' and '{args.word2}'...")
    emb1, _ = get_word_embedding(gen.pipe, args.word1)
    emb2, _ = get_word_embedding(gen.pipe, args.word2)

    cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    l2_dist = torch.norm(emb1 - emb2).item()
    print(f"  '{args.word1}' vs '{args.word2}': cosine_sim={cos_sim:.4f}, L2_dist={l2_dist:.4f}")
    print(f"  ||emb1||={torch.norm(emb1).item():.4f}, ||emb2||={torch.norm(emb2).item():.4f}")

    # --- Generate reference image (word1) ---
    print(f"\nGenerating reference image (z={args.word1})...")
    z1_4096 = emb1.unsqueeze(0).to(dtype=torch.float16)
    embeds_ref = gen.encode_batch_insert(prompt, z1_4096)
    ref_imgs = gen.generate_batch(embeds_ref, [args.seed])
    ref_img = ref_imgs[0]
    ref_img.save(os.path.join(run_dir, f"ref_{args.word1}.png"))
    ref_tensor = scorer.preprocess(ref_img)

    # Also generate word2 endpoint
    z2_4096 = emb2.unsqueeze(0).to(dtype=torch.float16)
    embeds_w2 = gen.encode_batch_insert(prompt, z2_4096)
    w2_imgs = gen.generate_batch(embeds_w2, [args.seed])
    w2_img = w2_imgs[0]
    w2_img.save(os.path.join(run_dir, f"ref_{args.word2}.png"))

    # --- Interpolate ---
    print(f"\nInterpolating {args.n_steps} steps from '{args.word1}' to '{args.word2}'...")
    alphas = np.linspace(0, 1, args.n_steps)
    results = []
    images = []

    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * emb1 + alpha * emb2  # (4096,)
        z_4096 = z_interp.unsqueeze(0).to(dtype=torch.float16)

        embeds = gen.encode_batch_insert(prompt, z_4096)
        imgs = gen.generate_batch(embeds, [args.seed])
        img = imgs[0]

        ds_dist = scorer.model(ref_tensor, scorer.preprocess(img)).item()

        # Also compute distance to word2 image
        ds_dist_w2 = scorer.model(scorer.preprocess(w2_img), scorer.preprocess(img)).item()

        # L2 distance in embedding space to z1
        emb_dist = torch.norm(z_interp - emb1).item()

        results.append({
            "alpha": float(alpha),
            "dreamsim_to_w1": ds_dist,
            "dreamsim_to_w2": ds_dist_w2,
            "emb_dist_to_w1": emb_dist,
        })

        img.save(os.path.join(run_dir, f"step{i:02d}_a{alpha:.2f}_ds{ds_dist:.4f}.png"))
        images.append(img)

        print(f"  alpha={alpha:.2f}: DS_to_{args.word1}={ds_dist:.4f}, DS_to_{args.word2}={ds_dist_w2:.4f}, emb_dist={emb_dist:.4f}")

    ref_img.close()
    w2_img.close()

    # --- Save CSV ---
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, "interpolation.csv"), index=False)

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: DreamSim vs alpha
    ax = axes[0]
    ax.plot(df["alpha"], df["dreamsim_to_w1"], 'b-o', label=f"DS to '{args.word1}'", markersize=4)
    ax.plot(df["alpha"], df["dreamsim_to_w2"], 'r-o', label=f"DS to '{args.word2}'", markersize=4)
    ax.set_xlabel(f"alpha (0={args.word1}, 1={args.word2})")
    ax.set_ylabel("DreamSim Distance")
    ax.set_title("DreamSim Distance vs Interpolation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Embedding distance vs DreamSim
    ax = axes[1]
    ax.scatter(df["emb_dist_to_w1"], df["dreamsim_to_w1"], c=df["alpha"], cmap='coolwarm', s=50)
    r_val, p_val = stats.pearsonr(df["emb_dist_to_w1"], df["dreamsim_to_w1"])
    ax.set_xlabel(f"Embedding L2 Distance to '{args.word1}'")
    ax.set_ylabel(f"DreamSim Distance to '{args.word1}'")
    ax.set_title(f"Embedding Dist vs DreamSim\nr={r_val:.4f}, p={p_val:.4f}")
    ax.grid(True, alpha=0.3)

    # Plot 3: Visual grid (subset of images)
    # Show 7 evenly spaced images
    ax = axes[2]
    ax.axis('off')
    step_indices = np.linspace(0, len(images) - 1, 7, dtype=int)
    thumb = 80
    for j, idx in enumerate(step_indices):
        img_small = images[idx].resize((thumb, thumb), Image.LANCZOS)
        img_arr = np.array(img_small)
        ax.imshow(img_arr, extent=[j * (thumb + 5), j * (thumb + 5) + thumb, 0, thumb])
        ax.text(j * (thumb + 5) + thumb // 2, -5, f"a={alphas[idx]:.1f}", ha='center', fontsize=8)
    ax.set_xlim(-5, 7 * (thumb + 5))
    ax.set_ylim(-15, thumb + 5)
    ax.set_title(f"'{args.word1}' → '{args.word2}'")

    fig.suptitle(f"Interpolation: '{args.word1}' → '{args.word2}' (seed={args.seed})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "interpolation_plot.png"), dpi=150)

    # --- Visual grid: all images in a row ---
    font = get_font(11)
    n = len(images)
    thumb_big = 200
    label_h = 40
    grid_w = n * thumb_big
    grid_h = thumb_big + label_h
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    for i, img in enumerate(images):
        img_r = img.resize((thumb_big, thumb_big), Image.LANCZOS)
        grid.paste(img_r, (i * thumb_big, 0))
        draw.text((i * thumb_big + 2, thumb_big + 2),
                  f"a={alphas[i]:.2f}", fill="black", font=font)
        draw.text((i * thumb_big + 2, thumb_big + 16),
                  f"DS={results[i]['dreamsim_to_w1']:.3f}", fill="gray", font=font)

    grid.save(os.path.join(run_dir, "visual_grid.png"), quality=95)

    for img in images:
        img.close()

    print(f"\nResults saved to {run_dir}/")
    print(f"Correlation (emb_dist vs DreamSim): r={r_val:.4f}, p={p_val:.4f}")


if __name__ == "__main__":
    main()
