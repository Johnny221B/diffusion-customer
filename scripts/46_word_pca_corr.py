"""
Repeat experiment 8 but sample z ONLY from the 172 real word embeddings.
Reference: leather (same as experiment 8)
Same seed for all generations.

For each PCA dim in [8, 16, 32, 64, 128]:
- Build W_pca from all 172 words
- Project each word's embedding to PCA space -> z_word
- X-axis: ||z_word - z_leather|| (L2 in PCA space)
- Y-axis: DreamSim(word_image, leather_image)

Expected: if the PCA space is meaningful and valid (words are real), correlation should appear.
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


WORD_LIST = [
    "neon", "fluorescent", "glowing", "bright", "vivid", "luminous", "electric",
    "radiant", "phosphorescent", "shiny", "glossy", "yellow", "lime",
    "chartreuse", "magenta", "cyan", "highlighter", "psychedelic",
    "leather", "suede", "matte", "dull", "dark", "brown", "tan", "rugged",
    "vintage", "weathered", "aged", "rustic", "earthy", "muted", "sepia",
    "cocoa", "mocha", "umber", "mahogany", "chestnut",
    "rubber", "plastic", "metal", "wood", "silk", "cotton", "wool", "nylon",
    "mesh", "denim", "velvet", "satin", "linen", "polyester", "foam", "cork",
    "patent", "canvas", "felt", "fur",
    "red", "blue", "green", "black", "white", "orange", "purple",
    "pink", "gray", "silver", "gold", "bronze", "beige", "ivory",
    "crimson", "turquoise", "indigo", "maroon", "coral", "teal", "navy",
    "olive", "burgundy", "lavender",
    "elegant", "sporty", "casual", "formal", "modern", "retro",
    "classic", "minimalist", "bold", "sleek", "luxurious", "premium",
    "fancy", "trendy", "traditional", "athletic", "expensive",
    "boot", "sandal", "sneaker", "heel", "loafer", "slipper", "clog",
    "moccasin", "oxford", "derby", "pump", "wedge", "platform",
    "chunky", "slim", "flat", "thick", "thin", "round", "pointed", "narrow",
    "wide", "tall", "short", "heavy", "light", "soft", "hard", "smooth",
    "rough", "transparent", "opaque",
    "fire", "ocean", "ice", "snow", "sun", "moon", "star",
    "forest", "desert", "mountain", "river", "storm", "wind",
    "nike", "adidas", "puma", "running", "walking", "hiking", "dancing",
    "outdoor", "indoor", "urban", "street",
    "comfortable", "breathable", "waterproof", "durable",
    "flexible", "rigid", "elastic", "tight", "loose",
    "colorful", "monochrome", "striped", "spotted", "plain", "patterned",
    "pastel", "subtle", "dramatic",
]


def get_word_emb(pipe, word):
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
    parser.add_argument("--ref_word", type=str, default="leather")
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/word_pca_corr_{args.ref_word}_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    def gen_image(emb_4096):
        emb_t = torch.from_numpy(emb_4096.astype(np.float32)).to(device, dtype=torch.float16).unsqueeze(0)
        embeds = gen.encode_batch_insert(prompt, emb_t)
        return gen.generate_batch(embeds, [args.seed])[0]

    # === Collect word embeddings (all 172 words, including ref) ===
    words = list(set(WORD_LIST))
    print(f"\n=== Collecting embeddings for {len(words)} words ===")
    embs = []
    for w in words:
        e = get_word_emb(gen.pipe, w).float().cpu().numpy()
        embs.append(e)
    embs = np.stack(embs)

    # === Generate reference image and get its embedding ===
    print(f"\n=== Reference word: '{args.ref_word}' ===")
    ref_emb = get_word_emb(gen.pipe, args.ref_word).float().cpu().numpy()
    ref_img = gen_image(ref_emb)
    ref_img.save(os.path.join(run_dir, "reference.png"))
    ref_tensor = scorer.preprocess(ref_img)

    # === Generate all word images with same seed and compute DreamSim ===
    print(f"\n=== Generating images for all {len(words)} words (seed={args.seed}) ===")
    dreams = []
    for i, w in enumerate(words):
        img = gen_image(embs[i])
        d = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        dreams.append(d)
        img.close()
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(words)}]")
    dreams = np.array(dreams)
    ref_img.close()

    import pandas as pd
    df_base = pd.DataFrame({"word": words, "dreamsim": dreams})
    df_base.to_csv(os.path.join(run_dir, "word_dreamsim.csv"), index=False)
    print(f"  DreamSim range: [{dreams.min():.4f}, {dreams.max():.4f}]")

    # Get reference index
    ref_idx = words.index(args.ref_word)

    # === PCA sweep ===
    pca_dims = [16, 32, 64, 128]
    mean_emb = embs.mean(axis=0)

    all_results = {}

    for dim in pca_dims:
        print(f"\n=== PCA dim = {dim} ===")
        pca = PCA(n_components=dim)
        pca.fit(embs - mean_emb)
        W = pca.components_.T  # (4096, dim)

        # Project each word to PCA space
        Z = (embs - mean_emb) @ W  # (N, dim)
        z_ref = Z[ref_idx]

        # Compute ||z - z_ref|| for each word
        z_dists = np.linalg.norm(Z - z_ref, axis=1)

        # Exclude reference itself
        mask = np.arange(len(words)) != ref_idx
        z_d = z_dists[mask]
        ds_d = dreams[mask]

        r, p = stats.pearsonr(z_d, ds_d)
        cum_var = pca.explained_variance_ratio_.sum()

        print(f"  cumulative variance: {cum_var:.4f}")
        print(f"  Pearson r={r:.4f}, p={p:.6f}")

        all_results[dim] = {
            "z_d": z_d, "ds_d": ds_d, "r": r, "p": p, "cum_var": cum_var,
            "words": [w for i, w in enumerate(words) if i != ref_idx],
        }

    # === Plot ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(pca_dims)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    for ax_idx, dim in enumerate(pca_dims):
        ax = axes[ax_idx]
        res = all_results[dim]
        ax.scatter(res["z_d"], res["ds_d"], alpha=0.5, s=25)
        ax.set_xlabel(f"||z - z_{args.ref_word}|| ({dim}-dim PCA)")
        if ax_idx == 0:
            ax.set_ylabel(f"DreamSim to {args.ref_word}")
        ax.set_title(f"PCA dim={dim}\nr={res['r']:.4f}, p={res['p']:.4f}\nvar={res['cum_var']:.3f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Word-restricted Z vs DreamSim (ref='{args.ref_word}', N={len(words)-1})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "correlation_by_pca_dim.png"), dpi=150)

    # Summary CSV
    summary = pd.DataFrame([
        {"pca_dim": d, "r": all_results[d]["r"], "p": all_results[d]["p"], "cum_var": all_results[d]["cum_var"]}
        for d in pca_dims
    ])
    summary.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    print(f"\n{summary}")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
