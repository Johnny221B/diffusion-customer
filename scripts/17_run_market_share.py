# scripts/17_run_market_share.py
import os
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

def get_logistic_label(d_our, d_comp, sensitivity=1.0):
    delta_d = d_comp - d_our 
    v_score = sensitivity * delta_d
    prob = 1.0 / (1.0 + np.exp(-np.clip(v_score, -20, 20)))
    
    return 1 if np.random.rand() < prob else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--comp_image", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--cold_start", type=int, default=5)
    args = parser.parse_args()

    run_dir = f"outputs/conquest_v17_128d_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    gen = SD35EmbeddingGenerator("/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large")
    scorer = DreamSimScorer(device="cuda")
    opt = LogisticThompsonOptimizer(dim_latent=128)
    
    ref_tensor = scorer.preprocess(args.ref_image)
    comp_tensor = scorer.preprocess(args.comp_image)
    dist_competitor = scorer.model(ref_tensor, comp_tensor).item()

    # target_prompt = "Left-facing profile of one sneaker, centered, solid white background, no logos."
    # target_prompt = "A professional photo of a centered sneaker, no brand logos, clean white background"
    # target_prompt = "A photo of a centered and left-facing sneaker, no brand logos, clean white background"
    # target_prompt = "A photo of a single athletic shoe, no brand logos, centered on white background"
    target_prompt = "Side profile of a white athletic shoe, facing left, no brand logos, centered on white plain background"
    # target_prompt = "A single athletic shoe, side profile facing right, centered on clean white background, studio lighting, product photography, 8k, highly detailed, full shot"

    # --- 1. QR 正交化投影矩阵 W (4096 x 128) ---
    W_raw = np.random.randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(device="cuda", dtype=torch.float16)

    # --- 2. 提取 Prompt 的低维表示 S_low (用于垂直化操作) ---
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
        # 执行垂直化操作，确保探索方向在 Prompt 语义之外
        z_latent = opt.solve_analytical_best(z_latent_raw, R=fixed_R, S_matrix=S_matrix)
        
        z_projected = W_torch @ torch.from_numpy(z_latent).to(device="cuda", dtype=torch.float16)
        embeds = gen.encode_simple_concat(target_prompt, z_projected)
        img = gen.generate(embeds,seed=cold_start_count)
        
        d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        y = 1 if d_our < dist_competitor else 0
        opt.add_comparison_data(z_latent, y)
        labels_collected.add(y)
        cold_start_count += 1 
        
    # for i in range(args.cold_start):
    #     z_latent_raw = np.random.normal(0, 1.0, 128).astype(np.float32)
    #     # 手动执行一次垂直化以保证冷启动质量
    #     z_latent = opt.solve_analytical_best(z_latent_raw, R=fixed_R, S_matrix=S_matrix)
        
    #     z_projected = W_torch @ torch.from_numpy(z_latent).to(device="cuda", dtype=torch.float16)
    #     embeds = gen.encode_simple_concat(target_prompt, z_projected)
    #     img = gen.generate(embeds, seed=i)
        
    #     d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
    #     y = 1 if d_our < dist_competitor else 0
    #     opt.add_comparison_data(z_latent, [y])
    print(f">>> 冷启动结束，共生成 {cold_start_count} 张图片。标签覆盖情况: {labels_collected}")
    opt.update_posterior()

    # --- Phase 2: Thompson Sampling (8-Sample Batch) ---
    results = []
    print(">>> 启动迭代流程")
    for epoch in range(1, args.num_epochs + 1):
        # 动态调整探索强度
        opt.exploration_a = max(0.2, 1.3 - current_share)
        
        batch_utilities, batch_raw_labels, batch_dists, batch_imgs, batch_z_list = [], [], [], [], []

        for b in range(10):
            intercept, theta_sampled = opt.sample_theta()
            z_cand = opt.solve_analytical_best(theta_sampled, R=fixed_R, S_matrix=S_matrix) # 128
            
            # 执行投影并生成
            z_projected = W_torch @ torch.from_numpy(z_cand).to(device="cuda", dtype=torch.float16) # 4096
            embeds = gen.encode_simple_concat(target_prompt, z_projected)
            
            current_u = intercept + np.dot(theta_sampled, z_cand)
            batch_utilities.append(current_u)
            
            img = gen.generate(embeds, seed=42+b)
            # img = gen.generate(embeds, seed=epoch*100 + b)
            if epoch == 1:
                first_epoch_dir = os.path.join(run_dir, "epoch_001_all")
                os.makedirs(first_epoch_dir, exist_ok=True)
                img.save(os.path.join(first_epoch_dir, f"sample_{b:02d}_dist{d_our:.3f}.png"))
            
            if epoch == args.num_epochs:
                last_epoch_dir = os.path.join(run_dir, f"epoch_{epoch:03d}_all")
                os.makedirs(last_epoch_dir, exist_ok=True)
                img.save(os.path.join(last_epoch_dir, f"sample_{b:02d}_dist{d_our:.3f}.png"))
            
            d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            
            y_raw = get_logistic_label(d_our, dist_competitor)
            # y_true = 1 if d_our < d_comp else 0
            
            batch_raw_labels.append(y_raw)
            batch_dists.append(d_our)
            batch_imgs.append(img)
            batch_z_list.append(z_cand)

        # --- 核心逻辑：处理标签与更新 ---
        true_labels = np.array(batch_raw_labels)
        current_share = np.mean(true_labels) # 真实的胜率
        avg_dist = np.mean(batch_dists)
        avg_utility = np.mean(batch_utilities)
        best_idx = np.argmin(batch_dists)
        
        effective_labels = true_labels.copy()
        # 如果 Batch 全军覆没，强制给最近的那个打 1
        if np.all(true_labels == 0):
            best_in_batch_idx = np.argmin(batch_dists)
            effective_labels[best_in_batch_idx] = 1
            print(f"Epoch {epoch} | All zero batch. Forcing Label 1 for idx {best_in_batch_idx}")

        # 将数据喂给优化器
        for b in range(10):
            # 将 effective_labels[b] 包装成 [effective_labels[b]]
            opt.add_comparison_data(batch_z_list[b], [effective_labels[b]])
        
        opt.update_posterior()

        # 保存当前 Batch 表现最好的图片
        best_idx = np.argmin(batch_dists)
        worst_idx = np.argmax(batch_dists)
        best_u_idx = np.argmax(batch_utilities)
        batch_imgs[best_idx].save(os.path.join(run_dir, f"ep{epoch:03d}_share{int(current_share*100)}_best.png"))
        batch_imgs[worst_idx].save(os.path.join(run_dir, f"ep{epoch:03d}_share{int(current_share*100)}_worst.png"))
        
        print(f"Epoch {epoch:03d} | Share: {current_share*100:4.1f}% | "
              f"Dist (Best/Avg): {batch_dists[best_idx]:.4f}/{avg_dist:.4f} | "
              f"Best_utility: {batch_utilities[best_u_idx]:.4f} | "
              f"Avg Utility: {avg_utility:.4f}")
        
        results.append({
            'epoch': epoch, 
            'share': current_share, 
            'best_dist': batch_dists[best_idx], 
            'avg_dist': avg_dist,
            'avg_utility': avg_utility
        })

    pd.DataFrame(results).to_csv(os.path.join(run_dir, "results.csv"), index=False)

if __name__ == "__main__":
    main()