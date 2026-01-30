import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

from src.sd35_embedding_generator import SD35EmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
from src.scorer import DreamSimScorer

def parse_args():
    parser = argparse.ArgumentParser(description="Market Conquest Pipeline - Detailed Logging")
    parser.add_argument("--ref_image", type=str, required=True, help="Ideal Design Reference")
    parser.add_argument("--comp_image", type=str, required=True, help="Competitor Fixed Design")
    parser.add_argument("--tag", type=str, default="market_conquest")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--cold_start", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8, help="8 images per epoch")
    parser.add_argument("--r_start", type=float, default=2.5)
    parser.add_argument("--r_end", type=float, default=0.6)
    parser.add_argument("--model_dir", type=str, default="/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large")
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = os.path.join("/home/linyuliu/jxmount/diffusion_custom/outputs", f"batch_{args.tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. 初始化
    gen = SD35EmbeddingGenerator(args.model_dir)
    scorer = DreamSimScorer(device="cuda")
    opt = LogisticThompsonOptimizer(dim_latent=64)
    
    ref_tensor = scorer.preprocess(args.ref_image)
    comp_tensor = scorer.preprocess(args.comp_image)
    # 计算竞争对手的固定感知距离
    dist_competitor = scorer.model(ref_tensor, comp_tensor).item()
    
    # 强化 Prompt：确保居中、无牌、无 Logo
    target_prompt = "A professional photo of a centered minimalist sneaker, no brand logos, no text, clean white background, studio lighting"

    W = np.random.randn(4096, 64).astype(np.float32)
    W /= np.linalg.norm(W, axis=0)

    results = []

    # --- Phase 1: Cold Start (按照 Kveton 2020 进行参数初始化) ---
    print(f">>> 启动市场冷启动: {args.cold_start} 轮")
    for i in range(args.cold_start):
        z_latent = np.random.normal(0, 1.0, 64).astype(np.float32)
        z_latent = args.r_start * (z_latent / (np.linalg.norm(z_latent) + 1e-9))
        p = np.random.uniform(50, 200)
        
        # 对称三明治生成: [Prompt, z, Prompt]
        embeds = gen.encode_sandwich(target_prompt, torch.from_numpy(W @ z_latent))
        img = gen.generate(embeds, seed=i)
        
        d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item() #
        y = 1 if d_our < dist_competitor else 0
        opt.add_comparison_data(np.concatenate([z_latent, [-p]]), [y])
        
        # 冷启动每隔 5 轮存一张图，作为市场探测记录
        if (i+1) % 5 == 0:
            img.save(os.path.join(run_dir, f"coldstart_{i+1:02d}_win_{y}.png"))

    opt.update_posterior()

    # --- Phase 2: Thompson Sampling 进化阶段 ---
    print(f">>> 开始竞争进化 (R: {args.r_start} -> {args.r_end})")
    for epoch in range(1, args.num_epochs + 1):
        current_R = max(args.r_end, args.r_start - (args.r_start - args.r_end) * (epoch / args.num_epochs))
        
        theta = opt.sample_theta() # 采样隐藏的线性效用函数参数
        z_cand, p_cand = opt.solve_analytical_best(theta, R=current_R)
        x_cand = np.concatenate([z_cand, [-p_cand]])
        
        batch_imgs, batch_dists, batch_labels = [], [], []
        
        # 核心：每轮生成 batch_size(8) 张图
        embeds = gen.encode_sandwich(target_prompt, torch.from_numpy(W @ z_cand))
        for b in range(args.batch_size):
            img = gen.generate(embeds, seed=epoch*100 + b)
            d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            
            y = 1 if d_our < dist_competitor else 0
            batch_imgs.append(img)
            batch_dists.append(d_our)
            batch_labels.append(y)

        # 批量向 Kveton 优化器提供反馈
        opt.add_comparison_data(x_cand, batch_labels)
        opt.update_posterior()

        # 统计本轮指标
        share = (sum(batch_labels) / args.batch_size) * 100
        min_dist_idx = np.argmin(batch_dists)
        best_dist = batch_dists[min_dist_idx]
        
        # 保存本轮“最贴近 ref”的图
        save_name = f"epoch_{epoch:03d}_share_{int(share)}_dist_{best_dist:.3f}.png"
        batch_imgs[min_dist_idx].save(os.path.join(run_dir, save_name))
        
        results.append({
            'epoch': epoch, 
            'share': share, 
            'min_dist': best_dist, 
            'avg_dist': np.mean(batch_dists),
            'R': current_R
        })
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d} | Market Share: {share:4.1f}% | Best Dist: {best_dist:.4f}")

    # 保存所有指标数据
    pd.DataFrame(results).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    print(f"\n>>> 任务完成！所有图片已保存在: {run_dir}")

if __name__ == "__main__":
    main()