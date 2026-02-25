# scripts/18_verify_pipeline.py
import os
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from src.sd35_embedding_generator import SD35EmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
from src.scorer import DreamSimScorer

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def project_orthogonal(z_raw, S_matrix, eps=1e-6):
    """
    Project z_raw onto the orthogonal complement of span(S_matrix),
    where S_matrix has shape (128, L).
    """
    if S_matrix is None or S_matrix.shape[1] == 0:
        return z_raw

    S = S_matrix  # (128, L)
    gram = S.T @ S  # (L, L)
    gram_inv = np.linalg.inv(gram + eps * np.eye(gram.shape[0]))
    coeffs = (z_raw @ S) @ gram_inv      # (L,)
    z_proj = S @ coeffs                  # (128,)
    z_perp = z_raw - z_proj
    return z_perp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--comp_image", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=50000)
    parser.add_argument("--cold_start", type=int, default=5)
    parser.add_argument("--R", type=float, default=5.0)
    args = parser.parse_args()

    run_dir = f"outputs/conquest_v21_128d_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "metrics.csv")
    pd.DataFrame(columns=[
        "epoch", "share", "oracle_share", "regret", "cos_sim", "mse", "norm_mse"
    ]).to_csv(csv_path, index=False)

    gen = SD35EmbeddingGenerator("/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large")
    target_prompt = "Side profile of a white athletic shoe, facing left, no brand logos, centered on white plain background"

    scorer = DreamSimScorer(device="cuda")
    opt = LogisticThompsonOptimizer(dim_latent=128)

    ref_tensor = scorer.preprocess(args.ref_image)
    comp_tensor = scorer.preprocess(args.comp_image)
    _ = scorer.model(ref_tensor, comp_tensor).item()  # dist_competitor not used in verify labeling

    # --- 1) QR orthogonal matrix W (4096 x 128) ---
    W_raw = np.random.randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(device="cuda", dtype=torch.float16)

    # --- competitor -> z_comp (128D) via DreamSim visual embedding (1792) ---
    with torch.no_grad():
        comp_feat_raw = scorer.model.embed(comp_tensor)  # (1,1792) or (1792,)
        if comp_feat_raw.dim() == 2:
            comp_feat_raw = comp_feat_raw.squeeze(0)

        W_vis_raw = np.random.randn(1792, 128).astype(np.float32)
        W_vis_np, _ = np.linalg.qr(W_vis_raw)
        W_vis_torch = torch.from_numpy(W_vis_np).to("cuda", dtype=torch.float16)

        z_comp = (comp_feat_raw.to(W_vis_torch.dtype) @ W_vis_torch).detach().cpu().float().numpy().flatten()

    # --- 2) define ground-truth theta*: gamma_star is all-ones direction, then normalize theta to unit norm ---
    gamma_star = np.ones(128, dtype=np.float32)
    gamma_star /= (np.linalg.norm(gamma_star) + 1e-9)  # unit direction
    offset = 2.5

    alpha_true = - float(np.dot(gamma_star, z_comp)) - offset   # competitor on decision boundary (raw)

    true_theta_raw = np.concatenate(([alpha_true], gamma_star)).astype(np.float32)

    # normalize the whole theta to unit norm
    theta_norm = float(np.linalg.norm(true_theta_raw) + 1e-9)
    true_theta = true_theta_raw / theta_norm

    # IMPORTANT: keep labeling consistent with the normalized theta
    alpha_true = float(true_theta[0])
    gamma_star = true_theta[1:].astype(np.float32)

    print(f">>> Verify mode: gamma_star = ones (unit dir), theta* normalized to ||theta*||=1.")
    print(f">>> alpha_true(after norm)={alpha_true:.6f}, ||gamma_star||={np.linalg.norm(gamma_star):.6f}")
    print(f">>> ||z_comp||={np.linalg.norm(z_comp):.4f}")
    # >>> alpha_true(after norm)=-0.021914, ||gamma_star||=0.999760
    # >>> ||z_comp||=0.2384

    # --- 3) build S_matrix for prompt-orthogonal constraint ---
    with torch.no_grad():
        prompt_outputs = gen.pipe.encode_prompt(target_prompt, target_prompt, target_prompt)
        prompt_high = prompt_outputs[0]  # (1, L, 4096)

        tokens = gen.pipe.tokenizer(target_prompt, return_tensors="pt")
        valid_len = tokens.input_ids.shape[1]
        effective_len = min(valid_len, prompt_high.shape[1])

        active_tokens = prompt_high[0, :effective_len, :]  # (L_eff, 4096)
        S_matrix = (active_tokens @ W_torch).T.detach().cpu().float().numpy()  # (128, L_eff)
        print(f">>> Effective token length: {effective_len}. S_matrix shape: {S_matrix.shape}")

    fixed_R = float(args.R)  # e.g., 5.0 (faster) or 3.0 (smoother)
    batch = 20

    # --- 4) compute oracle share (best possible under same constraints) ---
    gamma_perp = project_orthogonal(gamma_star.copy(), S_matrix)
    if np.linalg.norm(gamma_perp) < 1e-9:
        # degenerate case: gamma lies in span(S); fall back to raw gamma
        gamma_perp = gamma_star.copy()

    z_oracle = fixed_R * (gamma_perp / (np.linalg.norm(gamma_perp) + 1e-9))
    prob_best = sigmoid(np.dot(z_oracle, gamma_star) + alpha_true)  # oracle probability
    print(f">>> prob_best (oracle): {prob_best:.4f}")

    # --- Phase 1: Cold Start ---
    # print(">>> Cold start (128D prompt-orthogonal exploration)...")
    # cold_start_count = 0
    # labels_collected = set()

    # while cold_start_count < args.cold_start or len(labels_collected) < 2:
    #     z_latent_raw = np.random.normal(0, 1.0, 128).astype(np.float32)
    #     z_latent = opt.solve_analytical_best(z_latent_raw, R=fixed_R, S_matrix=S_matrix)

    #     # hard-threshold label for early diversity
    #     y = 1 if (np.dot(z_latent, gamma_star) + alpha_true) > 0 else 0
    #     opt.add_comparison_data(z_latent, y)
    #     labels_collected.add(y)
    #     cold_start_count += 1

    # print(f">>> Cold start done: {cold_start_count} samples. Labels: {labels_collected}")
    opt.update_posterior()

    # --- Phase 2: Thompson Sampling ---
    print(">>> Iterations start.")
    running_share = []
    running_best_share = []
    
    del gen
    del scorer
    torch.cuda.empty_cache()

    for epoch in range(1, args.num_epochs + 1):
        batch_raw_labels = []
        batch_best_labels = []
        batch_z_list = []
        batch_probs = []

        for b in range(batch):
            intercept, theta_sampled = opt.sample_theta()
            z_cand = opt.solve_analytical_best(theta_sampled, R=fixed_R, S_matrix=S_matrix)

            # stochastic logistic labels from the ground-truth utility
            v_score = float(np.dot(z_cand, gamma_star) + alpha_true)
            prob = sigmoid(v_score)
            u = np.random.rand()               # 同一个随机数
            y_raw = 1 if u < prob else 0       # 当前策略的标签
            y_oracle = 1 if u < prob_best else 0  # oracle 标签（同一个 u）
            
            # if epoch == 1:
            #     z_projected = W_torch @ torch.from_numpy(z_cand).to(device="cuda", dtype=torch.float16)
            #     embeds = gen.encode_simple_concat(target_prompt, z_projected)
            #     img = gen.generate(embeds,seed=42+b)
            #     first_epoch_dir = os.path.join(run_dir, "epoch_001_all")
            #     os.makedirs(first_epoch_dir, exist_ok=True)
            #     img.save(os.path.join(first_epoch_dir, f"sample_{b:02d}_score{v_score:.3f}.png"))
            
            # if epoch == 500:
            #     z_projected = W_torch @ torch.from_numpy(z_cand).to(device="cuda", dtype=torch.float16)
            #     embeds = gen.encode_simple_concat(target_prompt, z_projected)
            #     img = gen.generate(embeds,seed=42+b)
            #     first_epoch_dir = os.path.join(run_dir, "epoch_500_all")
            #     os.makedirs(first_epoch_dir, exist_ok=True)
            #     img.save(os.path.join(first_epoch_dir, f"sample_{b:02d}_score{v_score:.3f}.png"))

            batch_raw_labels.append(y_raw)
            batch_best_labels.append(y_oracle)
            batch_z_list.append(z_cand)
            batch_probs.append(prob)

        # running_share.append(np.mean(batch_raw_labels))
        running_share.append(np.mean(batch_probs))
        running_best_share.append(np.mean(batch_best_labels))

        for b in range(batch):
            opt.add_comparison_data(batch_z_list[b], [batch_raw_labels[b]])

        opt.update_posterior()

        if epoch % 10 == 0:
            current_mu = opt.mu_map.astype(np.float32)

            # metrics
            cos_sim = float(np.dot(current_mu, true_theta) /
                            (np.linalg.norm(current_mu) * np.linalg.norm(true_theta) + 1e-9))

            mse = float(np.mean((current_mu - true_theta) ** 2))

            mu_normed = current_mu / (np.linalg.norm(current_mu) + 1e-9)
            theta_normed = true_theta / (np.linalg.norm(true_theta) + 1e-9)
            norm_mse = float(np.mean((mu_normed - theta_normed) ** 2))

            avg_share = float(np.mean(running_share))
            # oracle_share = float(np.mean(running_best_share))
            oracle_share = prob_best
            regret = float(oracle_share - avg_share)

            print(f"Epoch {epoch:05d} | Avg Share: {avg_share*100:5.1f}% | "
                  f"Oracle: {oracle_share*100:5.1f}% | Regret: {regret:.4f} | "
                  f"CosSim: {cos_sim:.4f} | MSE: {mse:.6f}")

            df_row = pd.DataFrame([{
                "epoch": epoch,
                "share": avg_share,
                "oracle_share": oracle_share,
                "regret": regret,
                "cos_sim": cos_sim,
                "mse": mse,
                "norm_mse": norm_mse
            }])
            df_row.to_csv(csv_path, mode='a', index=False, header=False)

            running_share = []
            running_best_share = []

if __name__ == "__main__":
    main()