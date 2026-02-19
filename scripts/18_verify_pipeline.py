# scripts/18_verify_pipeline.py
import os
import time
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from src.sd35_embedding_generator import SD35EmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
from src.scorer import DreamSimScorer

def get_noisy_label(d_our, d_comp, mean=0.0, var=0.02):
    eps1 = np.random.normal(mean, np.sqrt(var))
    eps2 = np.random.normal(mean, np.sqrt(var))
    return 1 if (d_our + eps1) < (d_comp + eps2) else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--comp_image", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=50000)
    parser.add_argument("--cold_start", type=int, default=5)
    args = parser.parse_args()

    run_dir = f"outputs/conquest_v18_128d_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)
    
    csv_path = os.path.join(run_dir, "metrics.csv")
    pd.DataFrame(columns=["epoch", "share", "cos_sim", "mse", "norm_mse"]).to_csv(csv_path, index=False)

    gen = SD35EmbeddingGenerator("/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large")
    target_prompt = "Side profile of a white athletic shoe, facing left, no brand logos, centered on white plain background"
    
    scorer = DreamSimScorer(device="cuda")
    opt = LogisticThompsonOptimizer(dim_latent=128)
    
    ref_tensor = scorer.preprocess(args.ref_image)
    comp_tensor = scorer.preprocess(args.comp_image)
    dist_competitor = scorer.model(ref_tensor, comp_tensor).item()

    # --- 1. QR 正交化投影矩阵 W (4096 x 128) ---
    W_raw = np.random.randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(device="cuda", dtype=torch.float16)
    
    # --- 验证模式：基于物理图片提取 z_comp (修复 dtype 报错) ---
    with torch.no_grad():
        # 1. 提取 1792 维视觉特征
        comp_feat_raw = scorer.model.embed(comp_tensor) # 通常返回 float32
        
        # 2. 定义 1792x128 的投影矩阵（放在循环外定义更好，这里演示位置）
        W_vis_raw = np.random.randn(1792, 128).astype(np.float32)
        W_vis_np, _ = np.linalg.qr(W_vis_raw)
        W_vis_torch = torch.from_numpy(W_vis_np).to("cuda", dtype=torch.float16)
        
        # 3. 核心修复点：使用 .to(W_vis_torch.dtype) 强制类型对齐
        # 将 float32 的特征转为 float16 再进行矩阵乘法
        z_comp = (comp_feat_raw.to(W_vis_torch.dtype) @ W_vis_torch).detach().cpu().float().numpy().flatten()

    # 4. 设定上帝视角权重并计算对齐截距 alpha
    gamma_star = np.random.randn(128).astype(np.float32)
    gamma_star /= np.linalg.norm(gamma_star)
    
    # alpha = - gamma^T * z_comp
    alpha_true = - np.dot(gamma_star, z_comp)
    true_theta = np.concatenate(([alpha_true], gamma_star))
    print(f">>> 物理闭环验证：基于图片特征提取了 z_comp (128d)，并设定 alpha={alpha_true:.4f}")
    
    # --- 2. 动态提取有效 Token 的低维表示 S_low ---
    with torch.no_grad():
        prompt_outputs = gen.pipe.encode_prompt(target_prompt, target_prompt, target_prompt)
        prompt_high = prompt_outputs[0] # (1, L, 4096)
        
        # 动态检测实际长度
        tokens = gen.pipe.tokenizer(target_prompt, return_tensors="pt")
        valid_len = tokens.input_ids.shape[1] 
        # 截取有效部分，避免 Padding 干扰
        effective_len = min(valid_len, prompt_high.shape[1])
        
        active_tokens = prompt_high[0, :effective_len, :] 
        S_matrix = (active_tokens @ W_torch).T.detach().cpu().float().numpy()
        print(f">>> 动态检测到有效 Token 长度: {effective_len}, 已纳入正交空间。")
        
    fixed_R = 5.0
    current_share = 0.0

    # --- Phase 1: Cold Start (也执行垂直化操作) ---
    print(">>> 启动冷启动 (128D Vertical Exploration)...")
    
    cold_start_count = 0
    labels_collected = set()
    while cold_start_count < args.cold_start or len(labels_collected)<2:
        z_latent_raw = np.random.normal(0, 1.0, 128).astype(np.float32)
        z_latent = opt.solve_analytical_best(z_latent_raw, R=fixed_R, S_matrix=S_matrix)
        
        z_projected = W_torch @ torch.from_numpy(z_latent).to(device="cuda", dtype=torch.float16)
        y = 1 if (np.dot(z_latent, gamma_star) + alpha_true) > 0 else 0
        opt.add_comparison_data(z_latent, y)
        labels_collected.add(y)
        cold_start_count += 1 

    print(f">>> 冷启动结束，共生成 {cold_start_count} 张图片。标签覆盖情况: {labels_collected}")
    opt.update_posterior()

    # --- Phase 2: Thompson Sampling (8-Sample Batch) ---
    results = []
    # prof_data = {
    #     "sample_and_solve": [],
    #     "add_data": [],
    #     "update_posterior": [],
    #     "total_epoch": []
    # }
    print(">>> 启动迭代流程")
    running_share = [] 

    for epoch in range(1, args.num_epochs + 1):
        batch_raw_labels = []
        batch_z_list = []

        for b in range(10):
            intercept, theta_sampled = opt.sample_theta()
            z_cand = opt.solve_analytical_best(theta_sampled, R=fixed_R, S_matrix=S_matrix)
            
            # 模拟标签
            v_score = np.dot(z_cand, gamma_star) + alpha_true
            prob = 1.0 / (1.0 + np.exp(-v_score))
            y_raw = 1 if np.random.rand() < prob else 0
            
            batch_raw_labels.append(y_raw)
            batch_z_list.append(z_cand)

        # 累计这 10 个 Epoch 的胜率表现
        running_share.append(np.mean(batch_raw_labels))
        # prof_data["sample_and_solve"].append(time.time() - t0)
        # t1 = time.time()

        # --- 核心逻辑：处理标签与更新 ---
        # true_labels = np.array(batch_raw_labels)
        # current_share = np.mean(true_labels) # 真实的胜率
        # avg_utility = np.mean(batch_utilities)
        
        # effective_labels = true_labels.copy()

        # 将数据喂给优化器
        for b in range(10):
            # 将 effective_labels[b] 包装成 [effective_labels[b]]
            opt.add_comparison_data(batch_z_list[b], [batch_raw_labels[b]])
        
        # prof_data["add_data"].append(time.time() - t1)
        # t2 = time.time()
        
        opt.update_posterior()
        # prof_data["update_posterior"].append(time.time() - t2)
        
        # prof_data["total_epoch"].append(time.time() - epoch_start)
        if epoch % 10 == 0:
            current_mu = opt.mu_map
            
            # 1. Cosine Similarity
            cos_sim = np.dot(current_mu, true_theta) / (np.linalg.norm(current_mu) * np.linalg.norm(true_theta) + 1e-9)
            
            # 2. MSE
            mse = np.mean((current_mu - true_theta)**2)
            
            # 3. Normalized MSE
            mu_normed = current_mu / (np.linalg.norm(current_mu) + 1e-9)
            theta_normed = true_theta / (np.linalg.norm(true_theta) + 1e-9)
            norm_mse = np.mean((mu_normed - theta_normed)**2)
            
            avg_share = np.mean(running_share)
            
            print(f"Epoch {epoch:05d} | Avg Share: {avg_share*100:4.1f}% | CosSim: {cos_sim:.4f} | MSE: {mse:.4f}")
            
            # 写入 CSV
            df_row = pd.DataFrame([{
                "epoch": epoch,
                "share": avg_share,
                "cos_sim": cos_sim,
                "mse": mse,
                "norm_mse": norm_mse
            }])
            df_row.to_csv(csv_path, mode='a', index=False, header=False)
            
            # 重置计数器
            running_share = []


if __name__ == "__main__":
    main()