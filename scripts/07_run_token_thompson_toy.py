import os
import torch
import numpy as np
from src.sd35_embedding_generator import SD35EmbeddingGenerator
from src.thompson_optimizer import ThompsonOptimizer
from src.scorer import CLIPScorer # 使用您已有的 scorer

MODEL_DIR = "/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large"
OUT_DIR = "/home/linyuliu/jxmount/diffusion_custom/thompson_toy_v2"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    gen = SD35EmbeddingGenerator(MODEL_DIR)
    scorer = CLIPScorer(model_name="openai/clip-vit-base-patch32", device="cuda")
    
    # 定义维度
    dim_s = 2  # 结构化属性 (如是否品牌A, 是否亮色)
    dim_z = 16 # 我们探索的风格向量维度
    opt = ThompsonOptimizer(dim_s, dim_z)
    
    # 模拟用户偏好目标
    target_preference = "a luxury leather sneaker"

    # --- 第一步：初始化 (冷启动/生成基础数据) ---
    print("Starting Cold Start...")
    for i in range(3):
        s_rand = np.random.binomial(1, 0.5, dim_s)
        z_rand = np.random.normal(0, 0.1, dim_z)
        p_rand = np.random.uniform(50, 200) # 价格 p
        
        z_full = torch.zeros(4096)
        z_full[:dim_z] = torch.from_numpy(z_rand)
        
        embeds = gen.encode_sandwich("a professional sneaker", "high quality", z_full)
        img = gen.generate(embeds, seed=123+i)
        img.save(os.path.join(OUT_DIR, f"init_{i}.png"))
        
        score = float(scorer(img, target_preference))
        x = np.concatenate([s_rand, z_rand, [-p_rand]]) # 构造 x
        opt.update(x, score)
    
    # --- 第二步：Thompson 采样探索循环 ---
    print("Starting Thompson Loop...")
    for t in range(5):
        # 采样偏好参数 theta
        theta = opt.sample_theta()
        
        # 随机产生候选集 (模拟 Phase 0 的搜索过程)
        candidates_s = [np.random.binomial(1, 0.5, dim_s) for _ in range(10)]
        candidates_z = [np.random.normal(0, 0.2, dim_z) for _ in range(10)]
        candidates_p = [np.random.uniform(50, 200) for _ in range(10)]
        
        idx = opt.select_best_x(theta, candidates_s, candidates_z, candidates_p)
        
        best_z = candidates_z[idx]
        best_s = candidates_s[idx]
        best_p = candidates_p[idx]
        
        z_full = torch.zeros(4096)
        z_full[:dim_z] = torch.from_numpy(best_z)
        
        # 生成图片 (Phase 1)
        embeds = gen.encode_sandwich("a sneaker", "luxury style", z_full)
        img = gen.generate(embeds, seed=999) 
        img.save(os.path.join(OUT_DIR, f"round_{t}.png"))
        
        # 获取用户反馈 (Phase 3 模拟)
        score = float(scorer(img, target_preference))
        print(f"Round {t}: Score={score:.4f}, Selected Price={best_p:.2f}")
        
        # 更新后验 (Phase 4)
        x = np.concatenate([best_s, best_z, [-best_p]])
        opt.update(x, score)

if __name__ == "__main__":
    main()