"""
Validation experiment: Can BO learn to move from leather to neon?

Setup:
- W = PCA of word embeddings
- Anchor: token = leather_emb + W @ z (so z=0 => leather)
- Reference image: generated with neon_emb token
- Competitor image: generated with midpoint (0.5 leather + 0.5 neon) token
- Prior mean: 0 (i.e., start at leather)
- Same seed for all generations

If optimizer works, share should rise from ~0 toward 1 as z moves toward neon.
"""

import os
import time
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
from src.scorer import DreamSimScorer


WORD_LIST = [
    "red", "blue", "green", "black", "white", "yellow", "orange", "purple",
    "pink", "brown", "gray", "silver", "gold", "bronze", "beige", "ivory",
    "crimson", "turquoise", "indigo", "maroon", "coral", "teal", "navy",
    "magenta", "olive", "tan", "cream", "charcoal", "burgundy", "lavender",
    "leather", "suede", "canvas", "rubber", "plastic", "metal", "wood",
    "silk", "cotton", "wool", "nylon", "mesh", "denim", "velvet", "satin",
    "linen", "polyester", "foam", "cork", "patent",
    "elegant", "sporty", "casual", "formal", "vintage", "modern", "retro",
    "classic", "minimalist", "bold", "sleek", "rugged", "luxurious", "cheap",
    "expensive", "premium", "basic", "fancy", "trendy", "traditional",
    "chunky", "slim", "flat", "thick", "thin", "round", "pointed", "narrow",
    "wide", "tall", "short", "heavy", "light", "soft", "hard", "smooth",
    "rough", "glossy", "matte", "shiny", "transparent", "opaque",
    "boot", "sandal", "sneaker", "heel", "loafer", "slipper", "clog",
    "moccasin", "oxford", "derby", "pump", "wedge", "platform", "flip",
    "fire", "ocean", "ice", "snow", "rain", "sun", "moon", "star",
    "forest", "desert", "mountain", "river", "storm", "wind", "earth",
    "fast", "slow", "warm", "cool", "bright", "dark", "loud", "quiet",
    "sharp", "dull", "new", "old", "clean", "dirty", "wet", "dry",
    "nike", "adidas", "puma", "running", "walking", "hiking", "dancing",
    "athletic", "outdoor", "indoor", "urban", "rural", "street", "luxury",
    "comfortable", "uncomfortable", "breathable", "waterproof", "durable",
    "fragile", "flexible", "rigid", "elastic", "tight", "loose",
    "colorful", "monochrome", "striped", "spotted", "plain", "patterned",
    "neon", "pastel", "vivid", "subtle", "dramatic", "understated",
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


def get_logistic_prob(d_our, d_comp, sensitivity=5.0):
    return float(1.0 / (1.0 + np.exp(-np.clip(sensitivity * (d_comp - d_our), -20, 20))))


def get_logistic_label(d_our, d_comp, sensitivity=5.0):
    return 1 if np.random.rand() < get_logistic_prob(d_our, d_comp, sensitivity) else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--sensitivity", type=float, default=10.0)
    parser.add_argument("--R", type=float, default=48.0, help="Norm of W@z perturbation")
    parser.add_argument("--dim_z", type=int, default=128)
    parser.add_argument("--ref_word", type=str, default="neon")
    parser.add_argument("--anchor_word", type=str, default="leather")
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/validate_BO_{args.anchor_word}_{args.ref_word}_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "metrics.csv")
    pd.DataFrame(columns=[
        "epoch", "share", "avg_dist_to_ref", "avg_dist_to_anchor",
        "avg_utility", "regret", "hessian_min", "hessian_cond"
    ]).to_csv(csv_path, index=False)

    # Save config
    pd.Series(vars(args)).to_json(os.path.join(run_dir, "config.json"))

    # --- Load models ---
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # =========================================
    # 1. Build W from word embeddings (PCA)
    # =========================================
    print(f"\n=== Building W from {len(WORD_LIST)} word embeddings ===")
    word_embs = []
    for word in WORD_LIST:
        emb = get_word_emb(gen.pipe, word)
        word_embs.append(emb.float().cpu().numpy())
    word_emb_matrix = np.stack(word_embs)
    mean_emb = word_emb_matrix.mean(axis=0)

    # Center and PCA
    pca = PCA(n_components=args.dim_z)
    pca.fit(word_emb_matrix - mean_emb)
    W_np = pca.components_.T.astype(np.float32)  # (4096, 128)
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)
    print(f"  W shape: {W_np.shape}, total variance: {pca.explained_variance_ratio_.sum():.4f}")

    # =========================================
    # 2. Setup anchor (leather) and target (neon)
    # =========================================
    anchor_emb = get_word_emb(gen.pipe, args.anchor_word).float().cpu().numpy()
    target_emb = get_word_emb(gen.pipe, args.ref_word).float().cpu().numpy()
    midpoint_emb = 0.5 * anchor_emb + 0.5 * target_emb

    print(f"\n  Anchor '{args.anchor_word}' norm: {np.linalg.norm(anchor_emb):.2f}")
    print(f"  Target '{args.ref_word}' norm: {np.linalg.norm(target_emb):.2f}")
    print(f"  ||target - anchor||: {np.linalg.norm(target_emb - anchor_emb):.2f}")

    # The optimal z direction (in 128-dim) that should reach target from anchor:
    # token = anchor + W @ z; want token = target
    # so W @ z = target - anchor
    # z* = W^T @ (target - anchor) (since W has orthonormal columns)
    delta_target = target_emb - anchor_emb
    z_optimal = W_np.T @ delta_target  # (128,)
    print(f"  ||z_optimal||: {np.linalg.norm(z_optimal):.2f}")
    # Check reconstruction quality
    delta_recon = W_np @ z_optimal
    recon_error = np.linalg.norm(delta_target - delta_recon) / np.linalg.norm(delta_target)
    print(f"  PCA reconstruction error: {recon_error:.4f}")

    # =========================================
    # 3. Generate reference and competitor images
    # =========================================
    print(f"\n=== Generating reference (neon) and competitor (midpoint) ===")

    def gen_image(emb_4096):
        emb_t = torch.from_numpy(emb_4096.astype(np.float32)).to(device, dtype=torch.float16).unsqueeze(0)
        embeds = gen.encode_batch_insert(prompt, emb_t)
        imgs = gen.generate_batch(embeds, [args.seed])
        return imgs[0]

    ref_img = gen_image(target_emb)
    ref_img.save(os.path.join(run_dir, f"reference_{args.ref_word}.png"))
    ref_tensor = scorer.preprocess(ref_img)

    anchor_img = gen_image(anchor_emb)
    anchor_img.save(os.path.join(run_dir, f"anchor_{args.anchor_word}.png"))

    comp_img = gen_image(midpoint_emb)
    comp_img.save(os.path.join(run_dir, f"competitor_midpoint.png"))
    dist_competitor = scorer.model(ref_tensor, scorer.preprocess(comp_img)).item()

    dist_anchor = scorer.model(ref_tensor, scorer.preprocess(anchor_img)).item()

    print(f"  Anchor->Reference DreamSim: {dist_anchor:.4f}")
    print(f"  Competitor->Reference DreamSim: {dist_competitor:.4f}")

    oracle_share = 1.0 / (1.0 + np.exp(-np.clip(args.sensitivity * dist_competitor, -20, 20)))
    print(f"  Oracle share: {oracle_share*100:.2f}%")

    # If anchor (leather) is FURTHER from ref than competitor, initial share will be low
    # That's exactly what we want: optimizer must do work to win
    initial_share = 1.0 / (1.0 + np.exp(-np.clip(args.sensitivity * (dist_competitor - dist_anchor), -20, 20)))
    print(f"  Initial share (z=0, anchor): {initial_share*100:.2f}%")

    anchor_img.close()
    comp_img.close()

    # =========================================
    # 4. Optimizer setup
    # =========================================
    # prior_mean = 0 in z-space (corresponds to anchor)
    opt = LogisticThompsonOptimizer(dim_latent=args.dim_z, prior_mean=None)

    # No S_matrix (we don't need orthogonal projection since W already projects)
    S_matrix = None

    batch_seeds = [args.seed] * args.batch_size

    # =========================================
    # 5. Cold start
    # =========================================
    print(f"\n=== Cold start ===")
    cold_start = 5
    labels = set()
    cs_count = 0
    while cs_count < cold_start or len(labels) < 2:
        z = np.random.normal(0, 1, args.dim_z).astype(np.float32)
        z = z / max(np.linalg.norm(z), 1e-9) * args.R

        token = anchor_emb + W_np @ z
        img = gen_image(token)
        d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        y = get_logistic_label(d_our, dist_competitor, args.sensitivity)
        opt.add_comparison_data(z, [y])
        labels.add(y)
        cs_count += 1
        img.close()

    opt.update_posterior()
    print(f"  Cold start done. Labels: {labels}")

    # =========================================
    # 6. Main loop
    # =========================================
    print(f"\n=== Optimization: {args.num_epochs} epochs, batch={args.batch_size} ===")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        z_all, theta_list, mu_list = [], [], []

        for _ in range(args.batch_size):
            mu_i, theta_i = opt.sample_theta()
            # Normalize to norm R
            norm_th = np.linalg.norm(theta_i)
            z_i = (theta_i / max(norm_th, 1e-9) * args.R).astype(np.float32)
            z_all.append(z_i)
            theta_list.append(theta_i)
            mu_list.append(mu_i)

        z_all = np.array(z_all)  # (B, 128)
        z_torch = torch.from_numpy(z_all).to(device, dtype=torch.float16)

        # token = anchor + W @ z (per sample)
        # W shape (4096, 128), z shape (B, 128) -> tokens shape (B, 4096)
        anchor_t = torch.from_numpy(anchor_emb.astype(np.float32)).to(device, dtype=torch.float16)
        tokens = anchor_t.unsqueeze(0) + z_torch @ W_torch.T  # (B, 4096)

        embeds = gen.encode_batch_insert(prompt, tokens)
        imgs = gen.generate_batch(embeds, batch_seeds)

        # Score
        y_buf = np.zeros(args.batch_size, dtype=np.int64)
        d_buf = np.zeros(args.batch_size, dtype=np.float32)
        d_anchor_buf = np.zeros(args.batch_size, dtype=np.float32)
        p_buf = np.zeros(args.batch_size, dtype=np.float32)
        for i, img in enumerate(imgs):
            d = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            d_buf[i] = d
            p_buf[i] = get_logistic_prob(d, dist_competitor, args.sensitivity)
            y_buf[i] = get_logistic_label(d, dist_competitor, args.sensitivity)
            img.close()

        # Update
        for i in range(args.batch_size):
            opt.add_comparison_data(z_all[i], [int(y_buf[i])])
        opt.update_posterior()

        share = float(y_buf.mean())
        avg_d = float(d_buf.mean())
        regret = float(oracle_share - p_buf.mean())
        utilities = np.array([float(np.dot(z_all[i], theta_list[i]) + mu_list[i]) for i in range(args.batch_size)])

        row = {
            "epoch": epoch,
            "share": share,
            "avg_dist_to_ref": avg_d,
            "avg_dist_to_anchor": float(np.linalg.norm(z_all - 0, axis=1).mean()),
            "avg_utility": float(utilities.mean()),
            "regret": regret,
            "hessian_min": float(opt.min_eigenvalue),
            "hessian_cond": float(opt.condition_number),
        }
        pd.DataFrame([row]).to_csv(csv_path, mode='a', index=False, header=False)

        # Save sample images at key epochs
        if epoch == 1 or epoch % 25 == 0 or epoch == args.num_epochs:
            ep_dir = os.path.join(run_dir, f"epoch_{epoch:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            imgs_save = gen.generate_batch(embeds, batch_seeds)
            for i, img in enumerate(imgs_save):
                img.save(os.path.join(ep_dir, f"idx{i:02d}_d{d_buf[i]:.4f}.png"))
                img.close()

        sec = time.time() - t0
        print(f"  Epoch {epoch:03d} [{sec:.1f}s] share={share*100:.1f}% | dist={avg_d:.4f} "
              f"| oracle={oracle_share*100:.1f}% | regret={regret*100:.1f}%")

    ref_img.close()
    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
