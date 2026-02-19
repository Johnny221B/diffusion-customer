import os
import time
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator, SD35EmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
from src.scorer import DreamSimScorer
# from diffusers import StableDiffusion35Pipeline

# 强制使用 spawn 模式以适配 CUDA
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def get_logistic_label(d_our, d_comp, sensitivity=1.0):
    delta_d = d_comp - d_our 
    prob = 1.0 / (1.0 + np.exp(-np.clip(sensitivity * delta_d, -20, 20)))
    return 1 if np.random.rand() < prob else 0

def worker_fn(rank, task_queue, result_queue, model_path, ref_image_path, dist_competitor):
    """
    每个 GPU 进程的逻辑
    """
    device = f"cuda:{rank}"
    print(f"Worker {rank} loading model on {device}...")
    
    # # 1. 在子进程内加载 Pipeline
    # pipe = StableDiffusion35Pipeline.from_pretrained(
    #     model_path, 
    #     torch_dtype=torch.float16,
    #     variant="fp16"
    # ).to(device)
    # pipe.set_progress_bar_config(disable=True)
    
    # # 2. 实例化你提供的 Batch 生成类
    # batch_gen = SD35BatchEmbeddingGenerator(pipe, device=device)
    batch_gen = SD35BatchEmbeddingGenerator(model_path, device=device)
    scorer = DreamSimScorer(device=device)
    ref_tensor = scorer.preprocess(ref_image_path)
    
    while True:
        task = task_queue.get()
        if task is None: break # 接收到退出信号
        
        # 解包任务：128维向量、4096维向量、Seed列表、Prompt、灵敏度
        z_128_chunk, z_4096_chunk, seeds, prompt, sensitivity, save_this_epoch_dir = task
        
        # --- 使用你的 Batch 生成逻辑 ---
        # A. 批量 Encoding
        # z_4096_torch = torch.from_numpy(z_4096_chunk).to(device=device, dtype=torch.float16)
        # embeds_batch = batch_gen.encode_batch_concat(prompt, z_4096_torch)
        z_4096_torch = z_4096_chunk.to(device=device, dtype=torch.float16) 
        embeds_batch = batch_gen.encode_batch_concat(prompt, z_4096_torch)
        
        # B. 批量并行生成图片 (25张一组)
        images = batch_gen.generate_batch(embeds_batch, seeds)
        
        # C. 批量打分与反馈
        batch_results = []
        for i, img in enumerate(images):
            d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            y = get_logistic_label(d_our, dist_competitor, sensitivity)
            if save_this_epoch_dir is not None:
                # 命名格式：epoch_N_rank_R_seed_S.png
                img_name = f"rank{rank}_seed{seeds[i]}_dist{d_our:.4f}.png"
                img.save(os.path.join(save_this_epoch_dir, img_name))
            batch_results.append((z_128_chunk[i], y, d_our))
            img.close()
            
        result_queue.put(batch_results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--comp_image", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=100) # 总 Batch
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--cold_start", type=int, default=5)
    args = parser.parse_args()

    run_dir = f"outputs/dist_v20_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "metrics.csv")
    pd.DataFrame(columns=["epoch", "share", "avg_dist", "avg_utility"]).to_csv(csv_path, index=False)
    global_seeds = [42 + i for i in range(args.batch_size)]
    target_prompt = "Side profile of a white athletic shoe, facing left, no brand logos, centered on white plain background"

    # --- Master 初始化 ---
    opt = LogisticThompsonOptimizer(dim_latent=128)
    W_raw = np.random.randn(4096, 128).astype(np.float32)
    W_np, _ = np.linalg.qr(W_raw)
    # W_torch = torch.from_numpy(W_np).to(device="cuda", dtype=torch.float16)
    # 保持在 CPU 上，作为基础权重
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16)
    
    # 计算竞争对手距离 (先用一张卡算完即释放)
    temp_scorer = DreamSimScorer(device="cuda:0")
    temp_gen = SD35EmbeddingGenerator(args.model_path, device="cuda:0")
    with torch.no_grad():
        prompt_outputs = temp_gen.pipe.encode_prompt(target_prompt, target_prompt, target_prompt)
        prompt_high = prompt_outputs[0] 
        tokens = temp_gen.pipe.tokenizer(target_prompt, return_tensors="pt")
        valid_len = tokens.input_ids.shape[1] 
        effective_len = min(valid_len, prompt_high.shape[1])
        
        active_tokens = prompt_high[0, :effective_len, :] 
        # 注意：这里 W_torch 必须已经在 CPU/GPU 准备好
        S_matrix = (active_tokens @ W_torch.to("cuda")).T.detach().cpu().float().numpy()
        print(f">>> 动态检测到有效 Token 长度: {effective_len}，已纳入正交空间控制。")

    # --- 计算竞争者基准距离 ---
    ref_tensor = temp_scorer.preprocess(args.ref_image)
    comp_tensor = temp_scorer.preprocess(args.comp_image)
    dist_competitor = temp_scorer.model(ref_tensor, comp_tensor).item()

    # --- Phase 1: Cold Start (利用 temp_gen 快速初始化) ---
    print(">>> 启动冷启动...")
    cold_start_count = 0
    labels_collected = set()
    while cold_start_count < args.cold_start or len(labels_collected) < 2:
        z_latent_raw = np.random.normal(0, 1.0, 128).astype(np.float32)
        # 使用刚算好的 S_matrix 进行垂直化
        z_latent = opt.solve_analytical_best(z_latent_raw, R=5.0, S_matrix=S_matrix)
        
        # z_projected = W_torch.to("cuda") @ torch.from_numpy(z_latent).to("cuda", dtype=torch.float16)
        z_projected = W_torch.to("cuda") @ torch.from_numpy(z_latent).to("cuda", dtype=torch.float16)
        embeds = temp_gen.encode_simple_concat(target_prompt, z_projected)
        img = temp_gen.generate(embeds, seed=cold_start_count)
        
        d_our = temp_scorer.model(ref_tensor, temp_scorer.preprocess(img)).item()
        y = get_logistic_label(d_our, dist_competitor, args.sensitivity)
        
        opt.add_comparison_data(z_latent, [y])
        labels_collected.add(y)
        cold_start_count += 1
    
    # --- 彻底清理 Master 显存 ---
    del temp_gen
    del temp_scorer
    torch.cuda.empty_cache()
    opt.update_posterior()

    # --- 启动 4 卡并行 ---
    world_size = 4
    task_queues = [mp.Queue() for _ in range(world_size)]
    result_queue = mp.Queue()
    workers = []
    for r in range(world_size):
        p = mp.Process(target=worker_fn, args=(r, task_queues[r], result_queue, 
                                               args.model_path, args.ref_image, dist_competitor))
        p.start()
        workers.append(p)

    print(f">>> Multi-GPU Simulation Started: 100 images per epoch across 4 GPUs.")

    running_data = []
    for epoch in range(1, args.num_epochs + 1):
        # 1. Master 采样 100 个决策
        z_128_all = []
        save_this_epoch_dir = None
        if epoch == 1 or epoch % 500 == 0:
            save_this_epoch_dir = os.path.join(run_dir, f"epoch_{epoch:05d}")
            os.makedirs(save_this_epoch_dir, exist_ok=True)
            print(f">>> Epoch {epoch}: 开启图片保存模式，目录: {save_this_epoch_dir}")
        for _ in range(args.batch_size):
            mu, theta_s = opt.sample_theta()
            z_128_all.append(opt.solve_analytical_best(theta_s, R=5.0, S_matrix=S_matrix))
        z_128_all = np.array(z_128_all) # (100, 128)
        z_128_torch = torch.from_numpy(z_128_all).to(dtype=torch.float16)
        z_4096_all = z_128_torch @ W_torch.T
        # z_4096_all = torch.from_numpy(z_128_all).to(dtype=torch.float16) @ W_torch.T # (100, 4096)

        # 3. Scatter：分发任务 (每卡 25 个)
        chunk = args.batch_size // world_size
        for r in range(world_size):
            start, end = r*chunk, (r+1)*chunk
            worker_seeds = global_seeds[start:end]
            task = (z_128_all[start:end], 
                    z_4096_all[start:end], 
                    worker_seeds,
                    target_prompt, 
                    args.sensitivity,
                    save_this_epoch_dir)
            task_queues[r].put(task)

        # 4. Gather：收集 4 卡结果
        epoch_results = []
        for _ in range(world_size):
            epoch_results.extend(result_queue.get())
            
        # current_theta = opt.mu 
        epoch_utilities = []
        
        for z, y, d in epoch_results:
            opt.add_comparison_data(z, [y])
            # Utility 计算：z 和当前学到的 theta 的点积
            u = np.dot(z, theta_s) + mu
            epoch_utilities.append(u)

        # 5. 更新优化器 (向量化)
        for z, y, _ in epoch_results:
            opt.add_comparison_data(z, [y])
        opt.update_posterior() # 这里内部必须是向量化的高速版

        # 6. 监控与存档
        current_share = np.mean([r[1] for r in epoch_results])
        current_dist = np.mean([r[2] for r in epoch_results])
        current_utility = np.mean(epoch_utilities)
        running_data.append({"epoch": epoch, "share": current_share, "avg_dist": current_dist, "avg_utility": current_utility})

        if epoch % 5 == 0:
            df = pd.DataFrame(running_data)
            df.to_csv(csv_path, mode='a', index=False, header=False)
            print(f"Epoch {epoch:05d} | Share: {df['share'].mean()*100:4.1f}% | Avg Dist: {df['avg_dist'].mean():.4f} | Utility: {current_utility:.4f}")
            running_data = []

    # 停止信号
    for q in task_queues: q.put(None)
    for p in workers: p.join()

if __name__ == "__main__":
    main()