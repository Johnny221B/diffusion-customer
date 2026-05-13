"""
Repeat experiment 9 with:
- Reference: red
- Anchor: green
- Competitor: 0.5 * red + 0.5 * green
- PCA to reduce dimension

Output:
1. Collect (word_emb, dreamsim, label) for all 172 words - one image per word, fixed seed
2. For each PCA dim in sweep: run scaling analysis (N vs success metrics)
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


def evaluate_opt(opt, Z_eval, dreams_eval, true_dir, top_k=10):
    learned_dir = opt.mu_map[1:]
    intercept = opt.mu_map[0]
    pred_reward = Z_eval @ learned_dir + intercept

    cos = np.dot(learned_dir, true_dir) / (
        np.linalg.norm(learned_dir) * np.linalg.norm(true_dir) + 1e-9)

    if np.std(pred_reward) > 0 and np.std(dreams_eval) > 0:
        r_pred, _ = stats.pearsonr(pred_reward, -dreams_eval)
    else:
        r_pred = 0.0

    pred_topk = set(np.argsort(pred_reward)[::-1][:top_k])
    true_topk = set(np.argsort(dreams_eval)[:top_k])
    topk_prec = len(pred_topk & true_topk) / top_k

    return {"cos": cos, "r_pred": r_pred, "topk_prec": topk_prec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--ref_word", type=str, default="red")
    parser.add_argument("--anchor_word", type=str, default="green")
    parser.add_argument("--sensitivity", type=float, default=10.0)
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/scaling_{args.ref_word}_vs_{args.anchor_word}_d{args.pca_dim}_{datetime.now().strftime('%m%d_%H%M')}"
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

    # Reference, anchor, competitor
    print(f"\n=== ref='{args.ref_word}', anchor='{args.anchor_word}' ===")
    ref_emb = get_word_emb(gen.pipe, args.ref_word).float().cpu().numpy()
    anchor_emb = get_word_emb(gen.pipe, args.anchor_word).float().cpu().numpy()
    midpoint_emb = 0.5 * ref_emb + 0.5 * anchor_emb

    ref_img = gen_image(ref_emb)
    ref_img.save(os.path.join(run_dir, "reference.png"))
    ref_tensor = scorer.preprocess(ref_img)

    anchor_img = gen_image(anchor_emb)
    anchor_img.save(os.path.join(run_dir, "anchor.png"))
    anchor_img.close()

    comp_img = gen_image(midpoint_emb)
    comp_img.save(os.path.join(run_dir, "competitor.png"))
    dist_competitor = scorer.model(ref_tensor, scorer.preprocess(comp_img)).item()
    comp_img.close()

    oracle_share = 1.0 / (1.0 + np.exp(-np.clip(args.sensitivity * dist_competitor, -20, 20)))
    print(f"  Competitor DreamSim: {dist_competitor:.4f}")
    print(f"  Oracle share: {oracle_share*100:.2f}%")

    # Collect all word data (fixed seed)
    words = list(set(WORD_LIST))
    if args.ref_word in words:
        words.remove(args.ref_word)
    print(f"\n=== Collecting {len(words)} words ===")
    np.random.seed(42)

    embs, dreams, labels, probs = [], [], [], []
    for i, w in enumerate(words):
        emb = get_word_emb(gen.pipe, w).float().cpu().numpy()
        img = gen_image(emb)
        d = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        p = 1.0 / (1.0 + np.exp(-np.clip(args.sensitivity * (dist_competitor - d), -20, 20)))
        y = 1 if np.random.rand() < p else 0
        embs.append(emb); dreams.append(d); probs.append(p); labels.append(y)
        img.close()
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(words)}] pos={sum(labels)}")

    embs = np.stack(embs)
    dreams = np.array(dreams); labels = np.array(labels); probs = np.array(probs)
    ref_img.close()
    print(f"  Total positives: {labels.sum()}/{len(labels)}")

    # PCA
    print(f"\n=== PCA dim={args.pca_dim} ===")
    mean_emb = embs.mean(axis=0)
    pca = PCA(n_components=args.pca_dim)
    pca.fit(embs - mean_emb)
    W = pca.components_.T.astype(np.float32)
    Z = (embs - mean_emb) @ W  # (N, pca_dim)
    z_ref = (ref_emb - mean_emb) @ W
    z_anchor = (anchor_emb - mean_emb) @ W
    z_dir_true = z_ref - z_anchor
    print(f"  Cum var: {pca.explained_variance_ratio_.sum():.4f}")

    # Save data
    import pandas as pd
    pd.DataFrame({
        "word": words, "dreamsim": dreams, "prob": probs, "label": labels
    }).to_csv(os.path.join(run_dir, "raw_data.csv"), index=False)

    np.savez(os.path.join(run_dir, "data.npz"),
             Z=Z, embs=embs, dreams=dreams, labels=labels, probs=probs,
             W=W, mean_emb=mean_emb, z_ref=z_ref, z_anchor=z_anchor, z_dir_true=z_dir_true)

    # Scaling analysis
    N_total = len(words)
    N_list = [10, 20, 30, 50, 80, 120, 150, N_total - 10]
    N_list = sorted(set([n for n in N_list if 5 <= n < N_total]))
    print(f"\n=== Scaling: N in {N_list}, {args.n_trials} trials each ===")

    rng = np.random.RandomState(42)
    scaling_results = []

    for N in N_list:
        cos_list, rpred_list, topk_list = [], [], []
        for trial in range(args.n_trials):
            idx_perm = rng.permutation(N_total)
            train_idx = idx_perm[:N]
            test_idx = idx_perm[N:]

            opt = LogisticThompsonOptimizer(dim_latent=args.pca_dim, prior_var=3.0)
            for i in train_idx:
                opt.add_comparison_data(Z[i], [int(labels[i])])
            opt.update_posterior()

            m = evaluate_opt(opt, Z[test_idx], dreams[test_idx], z_dir_true, top_k=args.top_k)
            cos_list.append(m["cos"])
            rpred_list.append(m["r_pred"])
            topk_list.append(m["topk_prec"])

        scaling_results.append({
            "N": N,
            "cos_mean": np.mean(cos_list), "cos_std": np.std(cos_list),
            "rpred_mean": np.mean(rpred_list), "rpred_std": np.std(rpred_list),
            "topk_mean": np.mean(topk_list), "topk_std": np.std(topk_list),
        })
        print(f"  N={N:4d}: cos={np.mean(cos_list):.3f}±{np.std(cos_list):.3f}, "
              f"r_pred={np.mean(rpred_list):.3f}±{np.std(rpred_list):.3f}, "
              f"top{args.top_k}={np.mean(topk_list):.3f}±{np.std(topk_list):.3f}")

    df = pd.DataFrame(scaling_results)
    df.to_csv(os.path.join(run_dir, "scaling.csv"), index=False)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.errorbar(df["N"], df["cos_mean"], yerr=df["cos_std"], marker='o', capsize=3)
    ax.set_xlabel("Number of iid samples"); ax.set_ylabel("Cos(learned_dir, true_dir)")
    ax.set_title("Direction Alignment"); ax.set_xscale("log")
    ax.axhline(y=0, color='k', alpha=0.3); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(df["N"], df["rpred_mean"], yerr=df["rpred_std"], marker='o', capsize=3, color='green')
    ax.set_xlabel("Number of iid samples"); ax.set_ylabel("Pearson(pred, -dreamsim)")
    ax.set_title("Reward Prediction"); ax.set_xscale("log")
    ax.axhline(y=0, color='k', alpha=0.3); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.errorbar(df["N"], df["topk_mean"], yerr=df["topk_std"], marker='o', capsize=3, color='red')
    ax.set_xlabel("Number of iid samples"); ax.set_ylabel(f"Top-{args.top_k} Precision")
    ax.set_title(f"Top-{args.top_k} Precision"); ax.set_xscale("log")
    ax.axhline(y=args.top_k / N_total, color='gray', linestyle='--', alpha=0.5, label='random')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"BO Scaling: {args.ref_word} vs {args.anchor_word} (PCA dim={args.pca_dim}, N_total={N_total})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "scaling.png"), dpi=150)
    print(f"\nDone. {run_dir}")


if __name__ == "__main__":
    main()
