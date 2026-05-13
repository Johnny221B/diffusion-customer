"""
Test: feed (word_embedding, label) iid data to Bayesian Thompson update.

For each word in vocab:
1. Get embedding (4096-dim)
2. Project to 128-dim via PCA on the vocab itself
3. Insert embedding as token, generate image (fixed seed)
4. Compute DreamSim to reference (neon), compare with competitor (midpoint), get binary label
5. Feed (projected_emb, label) into LogisticThompsonOptimizer

After all data is fed:
- Check learned mu_map direction
- Compare with true direction (neon - leather) projected to 128-dim
- Predict reward for held-out words and check ranking
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from scipy import stats

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
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
    parser.add_argument("--ref_word", type=str, default="neon")
    parser.add_argument("--anchor_word", type=str, default="leather")
    parser.add_argument("--sensitivity", type=float, default=10.0)
    parser.add_argument("--dim_z", type=int, default=64)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/word_BO_{args.anchor_word}_{args.ref_word}_{datetime.now().strftime('%m%d_%H%M')}"
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

    # === Reference and competitor ===
    print(f"\n=== Setup: ref='{args.ref_word}', anchor='{args.anchor_word}' ===")
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

    oracle_share = 1.0 / (1.0 + np.exp(-np.clip(args.sensitivity * dist_competitor, -20, 20)))
    print(f"  Competitor DreamSim: {dist_competitor:.4f}")
    print(f"  Oracle share: {oracle_share*100:.2f}%")

    # === Collect all word embeddings and dreamsim ===
    words = list(set(WORD_LIST))
    if args.ref_word in words:
        words.remove(args.ref_word)
    print(f"\n=== Collecting data for {len(words)} words ===")

    embs = []
    dreams = []
    labels = []
    probs = []
    for i, word in enumerate(words):
        emb = get_word_emb(gen.pipe, word).float().cpu().numpy()
        img = gen_image(emb)
        d = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        p = 1.0 / (1.0 + np.exp(-np.clip(args.sensitivity * (dist_competitor - d), -20, 20)))
        y = 1 if np.random.rand() < p else 0
        embs.append(emb)
        dreams.append(d)
        probs.append(p)
        labels.append(y)
        img.close()
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(words)}] positives: {sum(labels)}")

    embs = np.stack(embs)  # (N, 4096)
    dreams = np.array(dreams)
    labels = np.array(labels)
    probs = np.array(probs)

    print(f"  Total positives: {sum(labels)}/{len(labels)}")
    print(f"  DreamSim range: [{dreams.min():.4f}, {dreams.max():.4f}]")

    import pandas as pd
    pd.DataFrame({
        "word": words, "dreamsim": dreams, "prob": probs, "label": labels
    }).to_csv(os.path.join(run_dir, "raw_data.csv"), index=False)

    # === Build PCA W from all word embeddings ===
    print(f"\n=== PCA: 4096 -> {args.dim_z} ===")
    mean_emb = embs.mean(axis=0)
    pca = PCA(n_components=args.dim_z)
    pca.fit(embs - mean_emb)
    W_np = pca.components_.T.astype(np.float32)  # (4096, dim_z)
    print(f"  Cumulative variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Project embeddings into PCA space
    Z = (embs - mean_emb) @ W_np  # (N, dim_z)
    print(f"  Z norm range: [{np.linalg.norm(Z, axis=1).min():.2f}, {np.linalg.norm(Z, axis=1).max():.2f}]")

    # True target direction in Z space
    z_target = (ref_emb - mean_emb) @ W_np  # neon's projection
    z_anchor = (anchor_emb - mean_emb) @ W_np  # leather's projection
    z_dir_true = z_target - z_anchor  # ideal direction to learn

    # === Feed all data to Bayesian Thompson optimizer ===
    print(f"\n=== Bayesian Logistic Regression via LogisticThompsonOptimizer ===")
    opt = LogisticThompsonOptimizer(dim_latent=args.dim_z, prior_var=3.0)

    # Feed all data
    for i in range(len(words)):
        opt.add_comparison_data(Z[i], [int(labels[i])])
    opt.update_posterior()

    print(f"  Hessian condition: {opt.condition_number:.2f}")
    print(f"  ||mu_map||: {np.linalg.norm(opt.mu_map):.4f}")

    # Learned direction (skip intercept = mu_map[0])
    learned_dir = opt.mu_map[1:]  # (dim_z,)

    # Compare with true direction
    cos_true = np.dot(learned_dir, z_dir_true) / (
        np.linalg.norm(learned_dir) * np.linalg.norm(z_dir_true) + 1e-9)
    print(f"\n  Cos(learned, z_target - z_anchor): {cos_true:.4f}")

    # Predict reward for each word: z @ learned_dir
    pred_reward = Z @ learned_dir + opt.mu_map[0]
    pred_prob = 1.0 / (1.0 + np.exp(-np.clip(pred_reward, -20, 20)))

    # Correlation with actual probability
    r_prob, p_prob = stats.pearsonr(pred_prob, probs)
    r_dream, p_dream = stats.pearsonr(pred_reward, -dreams)  # higher reward = lower distance
    print(f"  Pred prob vs actual prob: r={r_prob:.4f}, p={p_prob:.6f}")
    print(f"  Pred reward vs (-dreamsim): r={r_dream:.4f}, p={p_dream:.6f}")

    # Top words by predicted reward
    df = pd.DataFrame({
        "word": words, "dreamsim": dreams, "prob": probs, "label": labels,
        "pred_reward": pred_reward, "pred_prob": pred_prob,
    })
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    print(f"\n=== Top 15 predicted (highest reward) ===")
    print(df.sort_values("pred_reward", ascending=False).head(15)[["word", "dreamsim", "prob", "pred_reward"]].to_string(index=False))

    print(f"\n=== Bottom 15 predicted (lowest reward) ===")
    print(df.sort_values("pred_reward", ascending=False).tail(15)[["word", "dreamsim", "prob", "pred_reward"]].to_string(index=False))

    print(f"\n=== Top 10 actual (closest to {args.ref_word}) ===")
    print(df.sort_values("dreamsim").head(10)[["word", "dreamsim", "prob", "pred_reward"]].to_string(index=False))

    # === Plot ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(probs, pred_prob, alpha=0.6)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel("Actual probability")
    ax.set_ylabel("Predicted probability")
    ax.set_title(f"Pred vs Actual Prob\nr={r_prob:.4f}")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(dreams, pred_reward, alpha=0.6, c=labels, cmap='RdYlGn_r')
    ax.set_xlabel(f"DreamSim to {args.ref_word}")
    ax.set_ylabel("Predicted reward")
    ax.set_title(f"Pred Reward vs DreamSim\nr={r_dream:.4f}")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(['Cos(learned, true)'], [cos_true])
    ax.set_ylim([-1, 1])
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(f"Direction Alignment\ncos = {cos_true:.4f}")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Word-level Bayesian Update (dim_z={args.dim_z}, N={len(words)})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "results.png"), dpi=150)

    np.savez(os.path.join(run_dir, "data.npz"),
             Z=Z, embs=embs, mean_emb=mean_emb, W=W_np,
             dreams=dreams, labels=labels, probs=probs,
             learned_dir=learned_dir, z_dir_true=z_dir_true,
             cos_true=cos_true)

    print(f"\nDone. Results in {run_dir}")
    ref_img.close()


if __name__ == "__main__":
    main()
