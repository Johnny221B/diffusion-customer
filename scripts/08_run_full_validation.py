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
    parser = argparse.ArgumentParser(description="Structured Thompson Experiment")
    parser.add_argument("--tag", type=str, default="experiment", help="实验标签，用于文件夹命名")
    parser.add_argument("--target_preference", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--cold_start", type=int, default=20)
    parser.add_argument("--update_every", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--explore_r", type=float, default=0.5)
    parser.add_argument("--model_dir", type=str, default="/home/wan/guanting's/diffusion-customer/model/stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--output_root", type=str, default="/home/wan/guanting's/diffusion-customer/outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 路径命名逻辑：batch_[标签]_[时间戳]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, f"batch_{args.tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    gen = SD35EmbeddingGenerator(args.model_dir)
    scorer = CLIPScorer(model_name="openai/clip-vit-base-patch32", device="cuda")
    opt = ThompsonOptimizer(dim_z=4096)
    
    fixed_prefix = "A sneaker with no brand, centered in the image"
    fixed_suffix = "high quality, studio lighting, white background"
    metrics_log = []

    # 1. 冷启动 (20轮)
    print(f">>> 冷启动: {args.cold_start} 轮")
    for i in range(args.cold_start):
        z_rand = np.random.normal(0, 0.1, 4096)
        p_rand = np.random.uniform(50, 200)
        embeds = gen.encode_sandwich(fixed_prefix, fixed_suffix, torch.from_numpy(z_rand))
        img = gen.generate(embeds, seed=i)
        score = float(scorer(img, args.target_preference))
        opt.add_to_buffer(np.concatenate([z_rand, [-p_rand]]), score)
        if (i+1) % args.update_every == 0:
            opt.update_from_buffer()

    # 2. Thompson 循环
    print(f">>> 开始 Epochs: {args.num_epochs}")
    for epoch in range(1, args.num_epochs + 1):
        theta = opt.sample_theta()
        best_z, best_p = opt.solve_analytical_best(theta, R=args.explore_r)
        
        # 8张优选
        batch_scores = []
        batch_imgs = []
        z_tensor = torch.from_numpy(best_z)
        embeds = gen.encode_sandwich(fixed_prefix, fixed_suffix, z_tensor)
        
        for b in range(args.batch_size):
            img = gen.generate(embeds, seed=epoch*100+b)
            s = float(scorer(img, args.target_preference))
            batch_imgs.append(img)
            batch_scores.append(s)
        
        # 只保存这一轮选中的那张
        best_idx = np.argmax(batch_scores)
        best_img = batch_imgs[best_idx]
        best_score = batch_scores[best_idx]
        best_img.save(os.path.join(run_dir, f"epoch_{epoch:03d}_best.png"))
        
        opt.add_to_buffer(np.concatenate([best_z, [-best_p]]), best_score)
        if epoch % args.update_every == 0:
            opt.update_from_buffer()
        
        max_eig = opt.get_max_eigenvalue()
        metrics_log.append({'epoch': epoch, 'clip_score': best_score, 'max_eigenvalue': max_eig, 'price': best_p})
        if epoch % 10 == 0: print(f"Epoch {epoch} | Score: {best_score:.4f} | Eig: {max_eig:.4f}")

    # 保存数据与绘图
    df = pd.DataFrame(metrics_log)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(df['epoch'], df['clip_score'], 'b'); plt.title('CLIP Score')
    plt.subplot(1, 2, 2); plt.plot(df['epoch'], df['max_eigenvalue'], 'r'); plt.title('Sigma Max Eig')
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "convergence_plot.png"))
    print(f"完成！保存在: {run_dir}")

if __name__ == "__main__":
    main()