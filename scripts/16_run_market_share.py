# scripts/16_run_market_share.py
import os
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from src.sd35_embedding_generator import SD35EmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
from src.scorer import DreamSimScorer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--comp_image", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--cold_start", type=int, default=20)
    args = parser.parse_args()

    run_dir = f"outputs/conquest_v16_128d_{datetime.now().strftime('%m%d_%H%M')}"
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
    # target_prompt = "Left-facing profile of one single sneaker, centered in frame, no brand logos, clean white background"
    target_prompt = "A single athletic shoe, side profile facing right, centered on clean white background, studio lighting, product photography, 8k, highly detailed, full shot"

    # --- 1. QR 正交化投影矩阵 W (4096 x 128) ---
    W_raw = np.random.randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    W_torch = torch.from_numpy(W_np).to(device="cuda", dtype=torch.float16)

    # --- 2. 提取 Prompt 的低维表示 S_low (用于垂直化操作) ---
    # 修复设备冲突：确保都在 CUDA 上计算，最后再转 CPU numpy
    with torch.no_grad():
        prompt_high, _, _, _ = gen.pipe.encode_prompt(target_prompt, target_prompt, target_prompt)
        
        # 1. 提取有效 Token 的 Embedding (比如前 20 个)
        # prompt_high 形状是 (1, 77, 4096)
        active_tokens = prompt_high[0, :20, :] # (20, 4096)
        
        # 2. 投影到 128 维空间
        # 得到的 S_matrix_tensor 形状是 (20, 128)
        S_matrix_tensor = active_tokens @ W_torch
        
        # 3. 转置并回到 CPU numpy 为 (128, 20)   
        S_matrix = S_matrix_tensor.T.detach().cpu().float().numpy()
        
    fixed_R = 3.0
    current_share = 0.0

    # --- Phase 1: Cold Start (也执行垂直化操作) ---
    print(">>> 启动冷启动 (128D Vertical Exploration)...")
    for i in range(args.cold_start):
        z_latent_raw = np.random.normal(0, 1.0, 128).astype(np.float32)
        # 手动执行一次垂直化以保证冷启动质量
        z_latent = opt.solve_analytical_best(z_latent_raw, R=fixed_R, S_matrix=S_matrix)
        
        z_projected = W_torch @ torch.from_numpy(z_latent).to(device="cuda", dtype=torch.float16)
        embeds = gen.encode_sandwich(target_prompt, z_projected)
        img = gen.generate(embeds, seed=i)
        
        d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        y = 1 if d_our < dist_competitor else 0
        opt.add_comparison_data(z_latent, [y])
    opt.update_posterior()

    # --- Phase 2: Thompson Sampling (8-Sample Batch) ---
    results = []
    for epoch in range(1, args.num_epochs + 1):
        # 动态调整探索强度
        opt.exploration_a = max(0.2, 1.3 - current_share)
        
        batch_labels, batch_dists, batch_imgs, batch_z_list = [], [], [], []

        for b in range(10):
            # 关键：独立采样 theta
            theta_sampled = opt.sample_theta()
            # 核心：解出 z 并强制垂直于 S_low
            z_cand = opt.solve_analytical_best(theta_sampled, R=fixed_R, S_matrix=S_matrix)
            
            # 执行投影并生成
            z_projected = W_torch @ torch.from_numpy(z_cand).to(device="cuda", dtype=torch.float16)
            embeds = gen.encode_sandwich(target_prompt, z_projected)
            
            img = gen.generate(embeds, seed=42)
            # img = gen.generate(embeds, seed=epoch*100 + b)
            d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            
            y = 1 if d_our < dist_competitor else 0
            batch_labels.append(y)
            batch_dists.append(d_our)
            batch_imgs.append(img)
            batch_z_list.append(z_cand)

        # 记录数据并更新
        for b in range(10):
            opt.add_comparison_data(batch_z_list[b], [batch_labels[b]])
        opt.update_posterior()

        current_share = sum(batch_labels) / 10.0
        best_idx = np.argmin(batch_dists)
        batch_imgs[best_idx].save(os.path.join(run_dir, f"epoch_{epoch:03d}_share_{int(current_share*100)}.png"))
        
        print(f"Epoch {epoch:03d} | Share: {current_share*100}% | Dist: {batch_dists[best_idx]:.4f}")
        results.append({'epoch': epoch, 'share': current_share*100, 'dist': batch_dists[best_idx]})

    pd.DataFrame(results).to_csv(os.path.join(run_dir, "results.csv"), index=False)

if __name__ == "__main__":
    main()