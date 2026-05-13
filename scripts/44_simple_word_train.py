"""
Simplest test: train a linear classifier on (word_embedding, label) pairs.

For each word in a large vocabulary:
1. Insert its embedding as a token in the prompt
2. Generate an image (fixed seed)
3. Compute DreamSim distance to the reference image (neon)
4. Compare to competitor distance -> binary label
5. Collect (word_emb, label) as training data

Then:
- Train logistic regression
- Evaluate accuracy
- Compare learned direction with the true (neon - leather) direction
- Predict on held-out words

If this works, BO can in principle work on word embeddings.
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from scipy import stats

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


# Large word list - mix of categories so the optimization has signal
WORD_LIST = [
    # Bright/neon-like (should be CLOSER to "neon")
    "neon", "fluorescent", "glowing", "bright", "vivid", "luminous", "electric",
    "radiant", "phosphorescent", "shiny", "glossy", "fluo", "yellow", "lime",
    "chartreuse", "magenta", "cyan", "highlighter", "psychedelic", "fluo",
    # Dark/leather-like (should be CLOSER to "leather")
    "leather", "suede", "matte", "dull", "dark", "brown", "tan", "rugged",
    "vintage", "weathered", "aged", "rustic", "earthy", "muted", "sepia",
    "cocoa", "mocha", "umber", "mahogany", "chestnut",
    # Materials
    "rubber", "plastic", "metal", "wood", "silk", "cotton", "wool", "nylon",
    "mesh", "denim", "velvet", "satin", "linen", "polyester", "foam", "cork",
    "patent", "canvas", "felt", "fur",
    # Colors
    "red", "blue", "green", "black", "white", "yellow", "orange", "purple",
    "pink", "brown", "gray", "silver", "gold", "bronze", "beige", "ivory",
    "crimson", "turquoise", "indigo", "maroon", "coral", "teal", "navy",
    "olive", "burgundy", "lavender",
    # Styles
    "elegant", "sporty", "casual", "formal", "modern", "retro",
    "classic", "minimalist", "bold", "sleek", "luxurious", "premium",
    "fancy", "trendy", "traditional", "athletic", "expensive",
    # Shoe types/parts
    "boot", "sandal", "sneaker", "heel", "loafer", "slipper", "clog",
    "moccasin", "oxford", "derby", "pump", "wedge", "platform",
    # Properties
    "chunky", "slim", "flat", "thick", "thin", "round", "pointed", "narrow",
    "wide", "tall", "short", "heavy", "light", "soft", "hard", "smooth",
    "rough", "transparent", "opaque",
    # Nature
    "fire", "ocean", "ice", "snow", "sun", "moon", "star",
    "forest", "desert", "mountain", "river", "storm", "wind",
    # Brands/Activities
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
    parser.add_argument("--ref_word", type=str, default="neon")
    parser.add_argument("--anchor_word", type=str, default="leather")
    parser.add_argument("--sensitivity", type=float, default=10.0)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/word_train_{args.anchor_word}_{args.ref_word}_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # Load
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    def gen_image(emb_4096):
        emb_t = torch.from_numpy(emb_4096.astype(np.float32)).to(device, dtype=torch.float16).unsqueeze(0)
        embeds = gen.encode_batch_insert(prompt, emb_t)
        return gen.generate_batch(embeds, [args.seed])[0]

    # === Reference and competitor ===
    print(f"\nGenerating reference '{args.ref_word}' and competitor (midpoint)...")
    ref_emb = get_word_emb(gen.pipe, args.ref_word).float().cpu().numpy()
    anchor_emb = get_word_emb(gen.pipe, args.anchor_word).float().cpu().numpy()
    midpoint_emb = 0.5 * ref_emb + 0.5 * anchor_emb

    ref_img = gen_image(ref_emb)
    ref_img.save(os.path.join(run_dir, "reference.png"))
    ref_tensor = scorer.preprocess(ref_img)

    comp_img = gen_image(midpoint_emb)
    comp_img.save(os.path.join(run_dir, "competitor.png"))
    dist_competitor = scorer.model(ref_tensor, scorer.preprocess(comp_img)).item()
    comp_img.close()
    print(f"  Competitor DreamSim: {dist_competitor:.4f}")

    # === Collect data for all words ===
    # Deduplicate and exclude reference itself
    words = list(set(WORD_LIST))
    if args.ref_word in words:
        words.remove(args.ref_word)
    print(f"\nGenerating images for {len(words)} words...")

    import pandas as pd
    data = []
    embs_list = []
    for i, word in enumerate(words):
        emb = get_word_emb(gen.pipe, word).float().cpu().numpy()
        img = gen_image(emb)
        d = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        p = 1.0 / (1.0 + np.exp(-np.clip(args.sensitivity * (dist_competitor - d), -20, 20)))
        # Use both binary label (Bernoulli) and continuous score
        y = 1 if np.random.rand() < p else 0
        data.append({"word": word, "dreamsim": d, "prob": p, "label": y})
        embs_list.append(emb)
        img.close()
        if (i + 1) % 30 == 0:
            n_pos = sum(d["label"] for d in data)
            print(f"  [{i+1}/{len(words)}] {n_pos} positive labels so far")

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(run_dir, "raw_data.csv"), index=False)
    embs_arr = np.stack(embs_list)  # (N, 4096)
    np.save(os.path.join(run_dir, "embeddings.npy"), embs_arr)

    print(f"\nLabel distribution: {df['label'].value_counts().to_dict()}")
    print(f"DreamSim range: [{df['dreamsim'].min():.4f}, {df['dreamsim'].max():.4f}]")
    print(f"Top 10 closest to '{args.ref_word}':")
    print(df.sort_values('dreamsim').head(10).to_string(index=False))
    print(f"\nFurthest 10 from '{args.ref_word}':")
    print(df.sort_values('dreamsim').tail(10).to_string(index=False))

    # === Train logistic regression ===
    print(f"\n=== Training logistic regression on (word_emb, label) ===")
    X = embs_arr  # (N, 4096)
    y_binary = df["label"].values
    y_continuous = -df["dreamsim"].values  # Higher = better

    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_binary, np.arange(len(X)), test_size=0.3, random_state=42, stratify=y_binary if len(set(y_binary)) > 1 else None
    )

    if len(set(y_binary)) < 2:
        print("  Only one class, skipping logistic regression")
        train_acc = test_acc = -1
        learned_dir = None
    else:
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        learned_dir = clf.coef_[0]  # (4096,)
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")

        # The learned direction should align with (ref - anchor)
        true_dir = ref_emb - anchor_emb
        cos = np.dot(learned_dir, true_dir) / (np.linalg.norm(learned_dir) * np.linalg.norm(true_dir))
        print(f"  Cosine(learned_dir, ref-anchor): {cos:.4f}")

    # Also fit Ridge on continuous DreamSim
    print(f"\n=== Ridge regression on continuous DreamSim ===")
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, -y_train if False else df.loc[idx_train, "dreamsim"].values)
    train_r2 = reg.score(X_train, df.loc[idx_train, "dreamsim"].values)
    test_r2 = reg.score(X_test, df.loc[idx_test, "dreamsim"].values)
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    ridge_dir = -reg.coef_  # negate so high = good
    cos_ridge = np.dot(ridge_dir, ref_emb - anchor_emb) / (
        np.linalg.norm(ridge_dir) * np.linalg.norm(ref_emb - anchor_emb))
    print(f"  Cosine(ridge_dir, ref-anchor): {cos_ridge:.4f}")

    # === Save everything ===
    np.savez(os.path.join(run_dir, "results.npz"),
             learned_dir=learned_dir if learned_dir is not None else np.zeros(0),
             ridge_dir=ridge_dir,
             ref_emb=ref_emb,
             anchor_emb=anchor_emb,
             train_acc=train_acc, test_acc=test_acc,
             train_r2=train_r2, test_r2=test_r2)

    # === Plot ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of DreamSim distances
    ax = axes[0]
    ax.hist(df["dreamsim"], bins=30, alpha=0.7)
    ax.axvline(x=dist_competitor, color='r', linestyle='--', label=f'Competitor = {dist_competitor:.3f}')
    ax.set_xlabel(f"DreamSim Distance to '{args.ref_word}'")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution over {len(words)} words")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Predicted vs actual (Ridge)
    y_pred = reg.predict(X_test)
    y_actual = df.loc[idx_test, "dreamsim"].values
    ax = axes[1]
    ax.scatter(y_actual, y_pred, alpha=0.6)
    ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
    ax.set_xlabel("Actual DreamSim")
    ax.set_ylabel("Predicted DreamSim")
    ax.set_title(f"Ridge: Test R²={test_r2:.4f}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "results.png"), dpi=150)
    print(f"\nResults saved to {run_dir}")

    ref_img.close()


if __name__ == "__main__":
    main()
