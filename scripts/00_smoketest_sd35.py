# import os
# import torch
# from diffusers import StableDiffusion3Pipeline

# MODEL_DIR = "/home/wan/guanting's/diffusion-customer/model/stabilityai/stable-diffusion-3.5-large"
# OUT_DIR = "/home/wan/guanting's/diffusion-customer/outputs/smoke"
# os.makedirs(OUT_DIR, exist_ok=True)

# def main():
#     pipe = StableDiffusion3Pipeline.from_pretrained(
#         MODEL_DIR,
#         torch_dtype=torch.float16,
#     ).to("cuda")

#     # 如果你想更稳：可开启 offload（A6000 48GB 通常不必，但遇到 OOM 就开）
#     # diffusers 文档提到 SD3 pipeline 需要 float16，并且对多数硬件需要 offload :contentReference[oaicite:2]{index=2}
#     # pipe.enable_model_cpu_offload()

#     prompt = "product photo of a modern sneaker, studio lighting, white background, high detail"
#     gen = torch.Generator(device="cuda").manual_seed(123)

#     img = pipe(
#         prompt=prompt,
#         negative_prompt="lowres, blurry, worst quality",
#         guidance_scale=5.0,
#         num_inference_steps=24,
#         height=1024,
#         width=1024,
#         generator=gen,
#     ).images[0]

#     img.save(os.path.join(OUT_DIR, "smoke_sd35_large.png"))
#     print("saved:", os.path.join(OUT_DIR, "smoke_sd35_large.png"))

# if __name__ == "__main__":
#     main()

import torch
import os
from diffusers import StableDiffusion3Pipeline

# 参考你的路径配置
MODEL_DIR = "/home/wan/guanting's/diffusion-customer/model/stabilityai/stable-diffusion-3.5-large"
OUT_DIR = "/home/wan/guanting's/diffusion-customer/outputs/stability_test"
os.makedirs(OUT_DIR, exist_ok=True)

def test_token_embedding_stability():
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    # Reduce VRAM pressure for large SD3.5 models.
    pipe.enable_model_cpu_offload()
    
    prompt = "product photo of a modern sneaker, high detail"
    
    # 1. 获取原始的 prompt_embeds (这包含了 CLIP L, G 和 T5 的输出)
    with torch.no_grad():
        # 修复：显式传递 prompt_2 和 prompt_3，并处理 4 个返回值
        (
            prompt_embeds, 
            negative_prompt_embeds, 
            pooled_prompt_embeds, 
            negative_pooled_prompt_embeds
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt, 
            prompt_3=prompt,
            negative_prompt="lowres, blurry, worst quality" # 建议在这里也把负向提示词编码了
        )

    # 2. 模拟 Thompson 探索中的噪声抖动 (实验不同强度的 sigma)
    sigmas = [0.0, 0.01, 0.05, 0.1, 0.5]
    
    for sigma in sigmas:
        # 对 prompt_embeds 加入噪声
        # 注意：实际 Thompson 采样只会在 z_m 对应的特定 token 位置加噪声
        noise = torch.randn_like(prompt_embeds) * sigma
        perturbed_embeds = prompt_embeds + noise
        
        image = pipe(
            prompt_embeds=perturbed_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=512,  # 原为 1024
            width=512,   # 原为 1024
            num_inference_steps=24,
            guidance_scale=5.0,
            generator=torch.Generator("cuda").manual_seed(123)
        ).images[0]
        
        image.save(os.path.join(OUT_DIR, f"token_sigma_{sigma}.png"))
        print(f"Saved sigma {sigma} to {OUT_DIR}")

if __name__ == "__main__":
    test_token_embedding_stability()