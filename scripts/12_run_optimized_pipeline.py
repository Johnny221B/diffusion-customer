import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.sd35_embedding_generator import SD35EmbeddingGenerator
from src.thompson_optimizer import ThompsonOptimizer
from src.scorer import CLIPScorer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="optimized_run")
    parser.add_argument("--target_preference", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--dim_latent", type=int, default=64) # 优化空间维度
    parser.add_argument("--cold_start", type=int, default=20)
    parser.add_argument("--r_start", type=float, default=2.0)
    parser.add_argument("--r_end", type=float, default=0.4)
    parser.add_argument("--update_every", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_dir", type=str, default="/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--output_root", type=str, default="/home/linyuliu/jxmount/diffusion_custom/outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, f"batch_{args.tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. 投影矩阵 W: 将 64维 映射到 4096维
    np.random.seed(42)
    W = np.random.randn(4096, args.dim_latent).astype(np.float32)
    W = W / np.linalg.norm(W, axis=0) 

    gen = SD35EmbeddingGenerator(args.model_dir)
    scorer = CLIPScorer(device="cuda")
    
    # --- 修复位置：确保参数名与类定义一致 ---
    opt = ThompsonOptimizer(dim_latent=args.dim_latent)
    
    fixed_prefix = "A sneaker with no brand, centered in the image"
    fixed_suffix = "high quality, studio lighting, white background"
    metrics_log = []
    
    print(f">>> 启动冷启动: {args.cold_start} 轮")
    for i in range(args.cold_start):
        z_latent_rand = np.random.normal(0, 1.0, args.dim_latent).astype(np.float32)
        p_rand = np.random.uniform(50, 200)
        z_4096 = W @ z_latent_rand
        embeds = gen.encode_sandwich(fixed_prefix, fixed_suffix, torch.from_numpy(z_4096))
        img = gen.generate(embeds, seed=i)
        score = scorer(img, args.target_preference)
        opt.add_to_buffer(np.concatenate([z_latent_rand, [-p_rand]]), score)
        if (i+1) % args.update_every == 0:
            opt.update_from_buffer()

    print(f">>> 开始 Thompson 探索: {args.num_epochs} Epochs")
    for epoch in range(1, args.num_epochs + 1):
        current_R = max(args.r_end, args.r_start - (args.r_start - args.r_end) * (epoch / args.num_epochs))
        
        theta = opt.sample_theta()
        z_latent, best_p = opt.solve_analytical_best(theta, R=current_R)
        
        z_4096 = W @ z_latent 
        z_tensor = torch.from_numpy(z_4096)
        embeds = gen.encode_sandwich(fixed_prefix, fixed_suffix, z_tensor)
        
        batch_scores = []
        batch_imgs = []
        for b in range(args.batch_size):
            img = gen.generate(embeds, seed=epoch*100+b)
            s = scorer(img, args.target_preference)
            batch_imgs.append(img)
            batch_scores.append(s)
        
        best_idx = np.argmax(batch_scores)
        best_img, best_score = batch_imgs[best_idx], batch_scores[best_idx]
        best_img.save(os.path.join(run_dir, f"epoch_{epoch:03d}_best.png"))
        
        opt.add_to_buffer(np.concatenate([z_latent, [-best_p]]), best_score)
        if epoch % args.update_every == 0:
            opt.update_from_buffer()
        
        max_eig = opt.get_max_eigenvalue()
        metrics_log.append({'epoch': epoch, 'clip_score': best_score, 'max_eigenvalue': max_eig, 'R': current_R})
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Score: {best_score:.4f} | R: {current_R:.2f}")

    df = pd.DataFrame(metrics_log)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.plot(df['epoch'], df['clip_score']); plt.title('CLIP Score')
    plt.subplot(1, 2, 2); plt.plot(df['epoch'], df['max_eigenvalue']); plt.title('Uncertainty')
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "convergence_plot.png"))

if __name__ == "__main__":
    main()