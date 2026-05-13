"""
Single-GPU optimization with prior_mean initialization.
Uses the best x0 and seed from initialization search.
"""

import os
import time
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
from src.scorer import DreamSimScorer


def get_logistic_prob(d_our, d_comp, sensitivity=5.0):
    delta_d = d_comp - d_our
    return float(1.0 / (1.0 + np.exp(-np.clip(sensitivity * delta_d, -20, 20))))


def get_logistic_label(d_our, d_comp, sensitivity=5.0):
    prob = get_logistic_prob(d_our, d_comp, sensitivity)
    return 1 if np.random.rand() < prob else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--comp_image", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--init_npz", type=str, required=True, help="Path to best_x0.npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--sensitivity", type=float, default=5.0)
    parser.add_argument("--R", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=1810772)
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/opt_v30_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # --- Save config ---
    config = vars(args)
    config["run_dir"] = run_dir
    pd.Series(config).to_json(os.path.join(run_dir, "config.json"))

    # --- CSV ---
    csv_path = os.path.join(run_dir, "metrics.csv")
    pd.DataFrame(columns=[
        "epoch", "share", "avg_dist", "avg_utility", "oracle_share",
        "regret", "hessian_min_eig", "hessian_max_eig", "hessian_cond"
    ]).to_csv(csv_path, index=False)

    # --- Load init x0 ---
    init_data = np.load(args.init_npz)
    prior_mean = init_data["prior_mean"]  # (129,)
    print(f"Loaded prior_mean from {args.init_npz}, ||x0||={np.linalg.norm(prior_mean[1:]):.2f}")

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
    comp_tensor = scorer.preprocess(args.comp_image)
    dist_competitor = scorer.model(ref_tensor, comp_tensor).item()
    print(f"Competitor DreamSim dist to ref: {dist_competitor:.4f}")

    oracle_share = 1.0 / (1.0 + np.exp(-np.clip(args.sensitivity * dist_competitor, -20, 20)))
    print(f"Oracle share (upper bound): {oracle_share*100:.2f}%")

    # --- Prompt S_matrix ---
    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"
    prompt_outputs = gen.pipe.encode_prompt(prompt, prompt, prompt)
    prompt_high = prompt_outputs[0]
    tokens = gen.pipe.tokenizer(prompt, return_tensors="pt")
    valid_len = tokens.input_ids.shape[1]
    effective_len = min(valid_len, prompt_high.shape[1])
    active_tokens = prompt_high[0, :effective_len, :]
    S_matrix = (active_tokens @ W_torch).T.detach().cpu().float().numpy()

    # --- Seeds: all same seed ---
    batch_seeds = [args.seed] * args.batch_size

    # --- Optimizer with prior_mean ---
    opt = LogisticThompsonOptimizer(dim_latent=128, prior_mean=prior_mean)

    # --- Cold start ---
    cold_start = 5
    labels_collected = set()
    cold_count = 0
    print(f"\nCold start ({cold_start} samples)...")

    while cold_count < cold_start or len(labels_collected) < 2:
        z_raw = np.random.normal(0, 1.0, 128).astype(np.float32)
        z_latent = opt.solve_analytical_best(z_raw, R=args.R, S_matrix=S_matrix)

        z_4096 = (W_torch @ torch.from_numpy(z_latent).to(device, dtype=torch.float16)).unsqueeze(0)
        embeds = gen.encode_batch_insert(prompt, z_4096)
        imgs = gen.generate_batch(embeds, [args.seed])
        img = imgs[0]

        d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        p = get_logistic_prob(d_our, dist_competitor, args.sensitivity)
        y = 1 if np.random.rand() < p else 0

        opt.add_comparison_data(z_latent, [y])
        labels_collected.add(y)
        cold_count += 1
        img.close()

    opt.update_posterior()
    print(f"Cold start done. Labels collected: {labels_collected}")

    # --- Main loop ---
    print(f"\nStarting optimization: {args.num_epochs} epochs, batch_size={args.batch_size}, R={args.R}")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        # Sample z vectors
        z_128_all = []
        theta_list = []
        mu_list = []

        for _ in range(args.batch_size):
            mu_i, theta_i = opt.sample_theta()
            z_i = opt.solve_analytical_best(theta_i, R=args.R, S_matrix=S_matrix)
            z_128_all.append(z_i)
            theta_list.append(theta_i)
            mu_list.append(mu_i)

        z_128_all = np.array(z_128_all)
        theta_list = np.array(theta_list)
        mu_list = np.array(mu_list)

        # Project to 4096
        z_128_torch = torch.from_numpy(z_128_all).to(dtype=torch.float16, device=device)
        z_4096_all = z_128_torch @ W_torch.T  # (BS, 4096)

        # Generate batch
        embeds = gen.encode_batch_insert(prompt, z_4096_all)
        imgs = gen.generate_batch(embeds, batch_seeds)

        # Score
        y_buf = np.empty(args.batch_size, dtype=np.int64)
        d_buf = np.empty(args.batch_size, dtype=np.float32)
        p_buf = np.empty(args.batch_size, dtype=np.float32)

        for i, img in enumerate(imgs):
            d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            p_buf[i] = get_logistic_prob(d_our, dist_competitor, args.sensitivity)
            y_buf[i] = get_logistic_label(d_our, dist_competitor, args.sensitivity)
            d_buf[i] = d_our
            img.close()

        # Utility
        epoch_utilities = np.array([
            float(np.dot(z_128_all[i], theta_list[i]) + mu_list[i])
            for i in range(args.batch_size)
        ])
        current_utility = float(epoch_utilities.mean())

        # Update optimizer
        for i in range(args.batch_size):
            opt.add_comparison_data(z_128_all[i], [int(y_buf[i])])
        opt.update_posterior()

        # Metrics
        current_share = float(y_buf.mean())
        current_dist = float(d_buf.mean())
        current_exp_share = float(p_buf.mean())
        regret = float(oracle_share - current_exp_share)
        epoch_sec = time.time() - t0

        row = {
            "epoch": epoch,
            "share": current_share,
            "avg_dist": current_dist,
            "avg_utility": current_utility,
            "oracle_share": oracle_share,
            "regret": regret,
            "hessian_min_eig": float(opt.min_eigenvalue),
            "hessian_max_eig": float(opt.max_eigenvalue),
            "hessian_cond": float(opt.condition_number),
        }
        pd.DataFrame([row]).to_csv(csv_path, mode='a', index=False, header=False)

        # Save images at key epochs
        if epoch == 1 or epoch % 25 == 0 or epoch == args.num_epochs:
            epoch_dir = os.path.join(run_dir, f"epoch_{epoch:05d}")
            os.makedirs(epoch_dir, exist_ok=True)
            # Re-generate to save (since we closed them)
            imgs_save = gen.generate_batch(embeds, batch_seeds)
            for i, img in enumerate(imgs_save):
                img.save(os.path.join(epoch_dir, f"idx{i:02d}_dist{d_buf[i]:.4f}.png"))
                img.close()

        print(f"Epoch {epoch:03d} [{epoch_sec:.1f}s] | Share: {current_share*100:.1f}% "
              f"| Oracle: {oracle_share*100:.1f}% | Regret: {regret*100:.1f}% "
              f"| Avg Dist: {current_dist:.4f} | Utility: {current_utility:.4f}")

    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
