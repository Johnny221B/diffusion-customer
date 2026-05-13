# 20_multiGPU_market_share.py
import os
import time
import queue
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
import gc

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator, SD35EmbeddingGenerator
from src.thompson_optimizer import LogisticThompsonOptimizer
from src.scorer import DreamSimScorer
from src.seed_selector import select_seeds_by_clip
# from diffusers import StableDiffusion35Pipeline
# 图片的embedding的norm
# expection
# hessian matrix min eigenvalue / condition number
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def load_seeds_from_txt(txt_path, expected_n=None):
    seeds = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seeds.append(int(line))

    if expected_n is not None and len(seeds) < expected_n:
        raise ValueError(
            f"Seed file only contains {len(seeds)} seeds, but expected at least {expected_n}."
        )
    return seeds

def get_logistic_prob(d_our, d_comp, sensitivity=5.0):
    delta_d = d_comp - d_our
    prob = 1.0 / (1.0 + np.exp(-np.clip(sensitivity * delta_d, -20, 20)))
    return float(prob)

def get_logistic_label(d_our, d_comp, sensitivity=5.0):
    prob = get_logistic_prob(d_our, d_comp, sensitivity)
    return 1 if np.random.rand() < prob else 0

def worker_fn(rank, task_queue, result_queue, model_path, ref_image_path, dist_competitor):
    device = f"cuda:{rank}"
    print(f"Worker {rank} loading model on {device}...")
    
    batch_gen = SD35BatchEmbeddingGenerator(model_path, device=device)
    scorer = DreamSimScorer(device=device)
    ref_tensor = scorer.preprocess(ref_image_path)
    
    while True:
        task = task_queue.get()
        if task is None: break 
        
        idx_chunk, z_4096_chunk, seeds, prompt, sensitivity, save_this_epoch_dir = task
        
        z_4096_torch = z_4096_chunk.to(device=device, dtype=torch.float16) 
        embeds_batch = batch_gen.encode_batch_insert(prompt, z_4096_torch)
        images = batch_gen.generate_batch(embeds_batch, seeds)
        
        y_list = [0] * len(images)
        p_list = [0.0] * len(images)
        d_list = [0.0] * len(images)

        for i, img in enumerate(images):
            d_our = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            p = get_logistic_prob(d_our, dist_competitor, sensitivity)
            y = get_logistic_label(d_our, dist_competitor, sensitivity)

            if save_this_epoch_dir is not None:
                img_name = f"rank{rank}_idx{int(idx_chunk[i])}_seed{seeds[i]}_dist{d_our:.4f}.png"
                img.save(os.path.join(save_this_epoch_dir, img_name))

            p_list[i] = float(p)
            y_list[i] = int(y)
            d_list[i] = float(d_our)
            img.close()

        result_queue.put((idx_chunk, y_list, d_list, p_list))

def warmup_and_prepare(result_queue, model_path, ref_image, comp_image, batch_size, sensitivity, cold_start, target_prompt, run_dir):
    device = "cuda:0"

    temp_gen = None
    temp_scorer = None

    try:
        with torch.inference_mode():
            opt = LogisticThompsonOptimizer(dim_latent=128)

            W_raw = np.random.randn(4096, 128).astype(np.float32)
            W_np, _ = np.linalg.qr(W_raw)
            W_torch = torch.from_numpy(W_np).to(dtype=torch.float16, device=device)

            temp_scorer = DreamSimScorer(device=device)
            temp_gen = SD35EmbeddingGenerator(model_path, device=device)

            prompt_outputs = temp_gen.pipe.encode_prompt(target_prompt, target_prompt, target_prompt)
            prompt_high = prompt_outputs[0]

            tokens = temp_gen.pipe.tokenizer(target_prompt, return_tensors="pt")
            valid_len = tokens.input_ids.shape[1]
            effective_len = min(valid_len, prompt_high.shape[1])

            active_tokens = prompt_high[0, :effective_len, :]
            S_matrix = (active_tokens @ W_torch).T.detach().cpu().float().numpy()

            ref_tensor = temp_scorer.preprocess(ref_image)
            comp_tensor = temp_scorer.preprocess(comp_image)
            dist_competitor = temp_scorer.model(ref_tensor, comp_tensor).item()
            
            oracle_share = 1.0 / (1.0 + np.exp(-np.clip(sensitivity * dist_competitor, -20, 20)))
            print(f">>> Oracle share (theoretical upper bound): {oracle_share*100:.2f}%")
            
            # global_seeds = [42 + i for i in range(100)]
            # global_seeds = select_seeds_by_clip(
            #     temp_gen=temp_gen,
            #     prompt=target_prompt,
            #     run_dir=run_dir,
            #     candidate_n=300,
            #     top_k=batch_size,
            # )
            seed_txt_path = "/home/linyuliu/jxmount/diffusion_custom/outputs/dist_v22_0312_1234/seed_screening/selected_seeds.txt"
            global_seeds = load_seeds_from_txt(seed_txt_path, expected_n=batch_size)[:batch_size]
            print(f">>> Loaded {len(global_seeds)} seeds from {seed_txt_path}")

            cold_start_records = []
            cold_start_count = 0
            labels_collected = set()

            while cold_start_count < cold_start or len(labels_collected) < 2:
                z_latent_raw = np.random.normal(0, 1.0, 128).astype(np.float32)
                z_latent = opt.solve_analytical_best(z_latent_raw, R=10.0, S_matrix=S_matrix)

                z_projected = W_torch @ torch.from_numpy(z_latent).to(device, dtype=torch.float16)
                embeds = temp_gen.encode_simple_concat(target_prompt, z_projected)
                img = temp_gen.generate(embeds, seed=cold_start_count)

                d_our = temp_scorer.model(ref_tensor, temp_scorer.preprocess(img)).item()
                p = 1.0 / (1.0 + np.exp(-np.clip(sensitivity * (dist_competitor - d_our), -20, 20)))
                y = 1 if np.random.rand() < p else 0

                cold_start_records.append((z_latent.astype(np.float32), int(y)))
                labels_collected.add(y)
                cold_start_count += 1

                img.close()
                del z_projected, embeds, img

            result_queue.put({
                "W_np": W_np.astype(np.float32),
                "S_matrix": S_matrix.astype(np.float32),
                "dist_competitor": float(dist_competitor),
                "global_seeds": list(global_seeds),
                "cold_start_records": cold_start_records,
                "optimal":oracle_share
            })

    finally:
        # 显式清理
        del temp_gen
        del temp_scorer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--comp_image", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100) 
    parser.add_argument("--sensitivity", type=float, default=5.0)
    parser.add_argument("--cold_start", type=int, default=5)
    args = parser.parse_args()

    run_dir = f"outputs/dist_v23_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "metrics.csv")
    pd.DataFrame(columns=["epoch", "share", "avg_dist", "avg_utility", "oracle_share", "regret", "hessian_min_eig", "hessian_max_eig", "hessian_cond"]).to_csv(csv_path, index=False)
    # target_prompt = "Side profile of an athletic shoe, facing left, no brand logos, centered on white plain background"
    # target_prompt = "Product photo of a single athletic shoe, side profile, facing left, centered, full shoe visible, on a plain white background"
    target_prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"
    prompt_a = "Product photo of a single shoe with a specific style,"
    prompt_b = "full shoe visible, side profile, centered on a plain white background"

    # --- Master 初始化 ---
    prep_queue = mp.Queue()
    prep_proc = mp.Process(
        target=warmup_and_prepare,
        args=(
            prep_queue,
            args.model_path,
            args.ref_image,
            args.comp_image,
            args.batch_size,
            args.sensitivity,
            args.cold_start,
            target_prompt,
            run_dir,
        ),
    )
    prep_proc.start()

    prep_result = prep_queue.get()
    prep_proc.join()
    prep_proc.close()
    
    opt = LogisticThompsonOptimizer(dim_latent=128)

    W_np = prep_result["W_np"]
    W_torch = torch.from_numpy(W_np).to(dtype=torch.float16)

    S_matrix = prep_result["S_matrix"]
    dist_competitor = prep_result["dist_competitor"]
    global_seeds = prep_result["global_seeds"]
    cold_start_records = prep_result["cold_start_records"]
    oracle_share = prep_result["optimal"]

    for z_latent, y in cold_start_records:
        opt.add_comparison_data(z_latent, [y])
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
        t0 = time.time()
        z_128_all = []
        theta_list = []
        mu_list = []
        
        save_this_epoch_dir = None
        if epoch == 1 or epoch % 50 == 0:
            save_this_epoch_dir = os.path.join(run_dir, f"epoch_{epoch:05d}")
            os.makedirs(save_this_epoch_dir, exist_ok=True)
            print(f">>> Epoch {epoch}: 开启图片保存模式，目录: {save_this_epoch_dir}")
            
        for _ in range(args.batch_size):
            mu_i, theta_i = opt.sample_theta()
            z_i = opt.solve_analytical_best(theta_i, R=10.0, S_matrix=S_matrix)
            z_128_all.append(z_i)
            theta_list.append(theta_i)
            mu_list.append(mu_i)

        z_128_all = np.array(z_128_all) # (100, 128)
        theta_list = np.array(theta_list)  # (B, 128)
        mu_list = np.array(mu_list)  
        
        z_128_torch = torch.from_numpy(z_128_all).to(dtype=torch.float16)
        z_4096_all = z_128_torch @ W_torch.T

        # 3. Scatter：分发任务 (每卡 25 个)
        chunk = args.batch_size // world_size
        for r in range(world_size):
            start, end = r*chunk, (r+1)*chunk
            idx_chunk = np.arange(start, end, dtype=np.int32)  
            worker_seeds = global_seeds[start:end]
            task = (
                idx_chunk,
                z_4096_all[start:end].contiguous(),   # torch CPU tensor
                worker_seeds,
                target_prompt,
                args.sensitivity,
                save_this_epoch_dir
            )
            task_queues[r].put(task)

        # 4. Gather：收集 4 卡结果
        y_buf = np.empty(args.batch_size, dtype=np.int64)
        d_buf = np.empty(args.batch_size, dtype=np.float32)
        p_buf = np.empty(args.batch_size, dtype=np.float32)
        for _ in range(world_size):
            idx_chunk, y_list, d_list, p_list = result_queue.get()
            y_buf[idx_chunk] = np.array(y_list, dtype=np.int64)
            d_buf[idx_chunk] = np.array(d_list, dtype=np.float32)
            p_buf[idx_chunk] = np.array(p_list, dtype=np.float32)

        epoch_utilities = np.empty(args.batch_size, dtype=np.float32)
        for idx in range(args.batch_size):
            epoch_utilities[idx] = float(np.dot(z_128_all[idx], theta_list[idx]) + mu_list[idx])
        current_utility = float(epoch_utilities.mean())
  
        for i in range(args.batch_size):
            opt.add_comparison_data(z_128_all[i], [int(y_buf[i])])
        opt.update_posterior() 
        
        current_share = float(y_buf.mean())
        current_dist  = float(d_buf.mean())
        current_exp_share = float(p_buf.mean()) 
        regret = float(oracle_share - current_exp_share) 
        
        epoch_sec = float(time.time() - t0)
        print(f"epoch time:{epoch_sec:4.1f}")

        running_data.append({
            "epoch": epoch,
            "share": current_share,
            "avg_dist": current_dist,
            "avg_utility": current_utility,
            "oracle_share": oracle_share,
            "regret": regret,
            "hessian_min_eig": float(opt.min_eigenvalue),
            "hessian_max_eig": float(opt.max_eigenvalue),
            "hessian_cond": float(opt.condition_number),
        })

        if epoch % 1 == 0:
            df = pd.DataFrame(running_data)
            df.to_csv(csv_path, mode='a', index=False, header=False)
            print(
                f"Epoch {epoch:05d} | Share: {df['share'].mean()*100:4.1f}% "
                f"| Oracle: {oracle_share*100:4.1f}% | Regret: {df['regret'].mean()*100:4.1f}% "
                f"| Avg Dist: {df['avg_dist'].mean():.4f} | Utility: {current_utility:.4f}"
            )
            running_data = []

    for q in task_queues: q.put(None)
    for p in workers: p.join()

if __name__ == "__main__":
    main()