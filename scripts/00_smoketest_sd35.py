import os
import torch
from diffusers import StableDiffusion3Pipeline

MODEL_DIR = "/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large"
OUT_DIR = "/home/linyuliu/jxmount/diffusion_custom/outputs/smoke"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
    ).to("cuda")

    # 如果你想更稳：可开启 offload（A6000 48GB 通常不必，但遇到 OOM 就开）
    # diffusers 文档提到 SD3 pipeline 需要 float16，并且对多数硬件需要 offload :contentReference[oaicite:2]{index=2}
    # pipe.enable_model_cpu_offload()

    prompt = "product photo of a modern sneaker, studio lighting, white background, high detail"
    gen = torch.Generator(device="cuda").manual_seed(123)

    img = pipe(
        prompt=prompt,
        negative_prompt="lowres, blurry, worst quality",
        guidance_scale=5.0,
        num_inference_steps=24,
        height=1024,
        width=1024,
        generator=gen,
    ).images[0]

    img.save(os.path.join(OUT_DIR, "smoke_sd35_large.png"))
    print("saved:", os.path.join(OUT_DIR, "smoke_sd35_large.png"))

if __name__ == "__main__":
    main()
