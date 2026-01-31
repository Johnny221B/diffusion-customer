import os
import torch
import numpy as np
from src.sd35_embedding_generator import SD35EmbeddingGenerator
from src.thompson_optimizer import ThompsonOptimizer
from src.scorer import CLIPScorer

MODEL_DIR = "/home/wan/guanting's/diffusion-customer/model/stabilityai/stable-diffusion-3.5-large"
OUT_DIR = "/home/wan/guanting's/diffusion-customer/outputs/thompson_toy_v5"
os.makedirs(OUT_DIR, exist_ok=True)

def map_s_to_prompt(s):
    brand = "Nike style" if s[0] == 1 else "Adidas style"
    color = "white" if s[1] == 1 else "red"
    return f"a {color} {brand} sneaker"

def main():
    gen = SD35EmbeddingGenerator(MODEL_DIR)
    scorer = CLIPScorer(model_name="openai/clip-vit-base-patch32", device="cuda")
    
    dim_s, dim_z = 2, 4096
    opt = ThompsonOptimizer(dim_s, dim_z)
    target = "a white luxury leather sneaker" # 模拟用户偏好。随便来了一个偏好

    # 1. 冷启动 
    for i in range(5):
        s_rand = np.random.binomial(1, 0.5, dim_s)
        z_rand = np.random.normal(0, 0.2, dim_z)
        p_rand = np.random.uniform(50, 200)
        
        prefix = map_s_to_prompt(s_rand)
        z_full = torch.zeros(4096)
        z_full[:dim_z] = torch.from_numpy(z_rand)
        
        embeds = gen.encode_sandwich(prefix, "high quality", z_full)
        img = gen.generate(embeds, seed=100+i)
        img.save(os.path.join(OUT_DIR, f"init_{i}.png"))
        
        score = float(scorer(img, target))
        x = np.concatenate([s_rand, z_rand, [-p_rand]])
        opt.update(x, score)

    # 2. Thompson 显式解循环
    print("\nStarting Thompson Rounds with Analytical Solution...")
    for t in range(40):
        theta = opt.sample_theta()
        
        best_s, best_z, best_p = opt.solve_analytical_best(theta, R=0.5)
        
        # 动态 Prompt 映射：根据算法选出的 s 决定文字
        dynamic_prefix = map_s_to_prompt(best_s)
        
        z_full = torch.zeros(4096)
        z_full[:dim_z] = torch.from_numpy(best_z)
        
        embeds = gen.encode_sandwich(dynamic_prefix, "luxury leather style", z_full)
        img = gen.generate(embeds, seed=999)
        img.save(os.path.join(OUT_DIR, f"round_{t}.png"))
        
        score = float(scorer(img, target))
        print(f"Round {t}: Prompt='{dynamic_prefix}', Price={best_p:.1f}, Score={score:.4f}")
        
        x = np.concatenate([best_s, best_z, [-best_p]])
        opt.update(x, score)

if __name__ == "__main__":
    main()