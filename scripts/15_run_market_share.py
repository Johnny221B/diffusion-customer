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
    parser.add_argument("--update_horizon", type=int, default=3, help="每3轮更新一次参数")
    args = parser.parse_args()

    run_dir = f"outputs/conquest_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    gen = SD35EmbeddingGenerator("/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large")
    scorer = DreamSimScorer(device="cuda")
    opt = LogisticThompsonOptimizer(dim_latent=64)
    
    ref_tensor = scorer.preprocess(args.ref_image)
    comp_tensor = scorer.preprocess(args.comp_image)
    dist_competitor = scorer.model(ref_tensor, comp_tensor).item()

    target_prompt = "A professional product photo of a centered minimalist sneaker, no brand logos, no text, clean white background"
    # 控制一下

    np.random.seed(42)
    W = np.random.randn(4096, 64).astype(np.float32)
    W /= np.linalg.norm(W, axis=0)

    # --- Phase 1: Cold Start ---
    print(f">>> 启动冷启动...")
    for i in range(args.cold_start):
        z_latent = np.random.normal(0, 1.5, 64).astype(np.float32)
        img = gen.generate(gen.encode_sandwich(target_prompt, torch.from_numpy(W @ z_latent)), seed=i)
        d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
        y = 1 if d_our < dist_competitor else 0
        opt.add_comparison_data(np.concatenate([z_latent, [-150.0]]), [y])
    opt.update_posterior()

    # --- Phase 2: Thompson Sampling ---
    current_R = 5
    last_share = 0
    results = []

    for epoch in range(1, args.num_epochs + 1):
        # 1. 动态 alpha (Cooling)
        cooling = max(0.2, 1.0 - (epoch / args.num_epochs))
        theta = opt.sample_theta(cooling_factor=cooling)
        
        # 2. 求解并生成
        z_cand, p_cand = opt.solve_analytical_best(theta, R=current_R)
        embeds = gen.encode_sandwich(target_prompt, torch.from_numpy(W @ z_cand))
        # randomness源自z
        
        batch_labels, batch_dists, batch_imgs = [], [], []
        for b in range(8): # 每轮 8 张
            img = gen.generate(embeds, seed=epoch*100+b)
            d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            y = 1 if d_our < dist_competitor else 0
            batch_labels.append(y)
            batch_dists.append(d_our)
            batch_imgs.append(img)

        # 3. 记录与延迟更新
        opt.add_comparison_data(np.concatenate([z_cand, [-p_cand]]), batch_labels)
        if epoch % args.update_horizon == 0:
            opt.update_posterior()

        # 4. Adaptive R 策略
        share = sum(batch_labels) / 8.0
        # 如果份额提升，收缩半径进行精细化微调
        if share > 0.5:
            current_R = max(0.4, current_R * 0.85)
        elif share == 0:
            # 如果颗粒无收，反向扩张半径重新探索
            current_R = min(3.0, current_R * 1.1)

        # 5. 保存最优图
        best_idx = np.argmin(batch_dists)
        batch_imgs[best_idx].save(os.path.join(run_dir, f"epoch_{epoch:03d}_share_{int(share*100)}.png"))
        
        print(f"Epoch {epoch} | Share: {share*100}% | R: {current_R:.2f} | Dist: {batch_dists[best_idx]:.4f}")
        results.append({'epoch': epoch, 'share': share*100, 'dist': batch_dists[best_idx]})

    pd.DataFrame(results).to_csv(os.path.join(run_dir, "results.csv"), index=False)

if __name__ == "__main__":
    main()