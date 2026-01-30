import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# 导入自定义组件
from src.sd35_embedding_generator import SD35EmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
from src.scorer import DreamSimScorer

def parse_args():
    parser = argparse.ArgumentParser(description="DreamSim Logistic Thompson Pipeline")
    # 路径配置
    parser.add_argument("--ref_image", type=str, required=True, help="用户理想的参考图路径")
    parser.add_argument("--tag", type=str, default="dreamsim_logistic")
    parser.add_argument("--model_dir", type=str, default="/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--output_root", type=str, default="/home/linyuliu/jxmount/diffusion_custom/outputs")
    
    # 实验超参数
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dim_latent", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--explore_r", type=float, default=1.5, help="Thompson 采样的探索半径")
    parser.add_argument("--update_every", type=int, default=1, help="逻辑回归建议每轮更新以保持敏感度")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 初始化环境与文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, f"batch_{args.tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 初始化组件
    gen = SD35EmbeddingGenerator(args.model_dir)
    scorer = DreamSimScorer(device="cuda") # 内部加载 dreamsim
    opt = LogisticThompsonOptimizer(dim_latent=args.dim_latent)
    
    # 固定投影矩阵 W (4096 x 64)
    np.random.seed(42)
    W = np.random.randn(4096, args.dim_latent).astype(np.float32)
    W = W / np.linalg.norm(W, axis=0)

    # 基础 Prompt 配置
    fixed_prefix = "A sneaker with no brand, centered in the image"
    fixed_suffix = "high quality, studio lighting, white background"
    
    # --- Phase 1: 建立初始基准 (Baseline / 图1) ---
    print(f">>> 正在生成初始基准图...")
    z_baseline_latent = np.random.normal(0, 0.5, args.dim_latent).astype(np.float32)
    p_baseline = 150.0 # 初始假设中等价格
    
    z_4096_init = W @ z_baseline_latent
    embeds_init = gen.encode_sandwich(fixed_prefix, fixed_suffix, torch.from_numpy(z_4096_init))
    img_baseline = gen.generate(embeds_init, seed=42)
    img_baseline.save(os.path.join(run_dir, "initial_baseline.png"))
    
    # 当前基准的特征向量 x1
    x_baseline = np.concatenate([z_baseline_latent, [-p_baseline]])
    
    metrics_log = []

    # --- Phase 2: Thompson 循环 (图2 vs 图1) ---
    print(f">>> 开始感知进化循环: {args.num_epochs} Epochs")
    for epoch in range(1, args.num_epochs + 1):
        
        # Thompson 采样 theta
        theta = opt.sample_theta()
        # 求解当前 theta 下的最佳潜在向量 (图2的参数)
        z_latent_gen, p_gen = opt.solve_analytical_best(theta, R=args.explore_r)
        x_gen = np.concatenate([z_latent_gen, [-p_gen]])
        
        z_4096_gen = W @ z_latent_gen
        embeds_gen = gen.encode_sandwich(fixed_prefix, fixed_suffix, torch.from_numpy(z_4096_gen))
        
        batch_images = []
        batch_distances = []
        
        # 每一轮生成 batch_size 张图片作为候选
        for b in range(args.batch_size):
            seed = epoch * 100 + b
            img_candidate = gen.generate(embeds_gen, seed=seed)
            
            # 使用 DreamSim 进行判定: img_candidate (图2) 是否比 img_baseline (图1) 更接近参考图
            # 通过模型计算 distance
            y, dist_to_ref = scorer.compare(args.ref_image, img_baseline, img_candidate)
            
            # 记录比较结果 (x2 - x1, y)
            opt.add_comparison(x_baseline, x_gen, y)
            
            batch_images.append(img_candidate)
            batch_distances.append(dist_to_ref)
        
        # 每一轮结束后更新后验 (Laplace 近似)
        if epoch % args.update_every == 0:
            opt.update_posterior()

        # 找到这 8 张图片中最接近参考图的一张
        best_idx_in_batch = np.argmin(batch_distances)
        best_dist = batch_distances[best_idx_in_batch]
        best_img = batch_images[best_idx_in_batch]
        
        # 策略：如果本轮生成的“最优点”确实比当前的基准图好，则更新基准
        # 这确保了 Pipeline 始终在向 Reference 爬升
        if best_dist < scorer.last_baseline_dist: # scorer 内部记录了上一次 baseline 的距离
            img_baseline = best_img
            x_baseline = x_gen
            status = "IMPROVED"
        else:
            status = "STABLE"

        # 保存本轮最优图
        best_img.save(os.path.join(run_dir, f"epoch_{epoch:03d}_{status}.png"))
        
        # 记录指标
        metrics_log.append({
            'epoch': epoch,
            'min_dist': best_dist,
            'status': status,
            'R': args.explore_r
        })
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Best Dist: {best_dist:.4f} | Status: {status}")

    # 保存数据并绘图
    df = pd.DataFrame(metrics_log)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['min_dist'], color='purple', label='DreamSim Distance')
    plt.gca().invert_yaxis() # 距离越小越好，反转 Y 轴直观显示“上升”
    plt.title('Perceptual Convergence to Reference')
    plt.xlabel('Epoch')
    plt.ylabel('DreamSim Distance (Lower is Better)')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "convergence_plot.png"))
    
    print(f"\n>>> 实验完成！结果保存在: {run_dir}")

if __name__ == "__main__":
    main()