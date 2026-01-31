import os
import torch
from diffusers import StableDiffusion3Pipeline

BASE_DIR = "/home/wan/guanting's/diffusion-customer/model/stabilityai/stable-diffusion-3.5-large"
OUT_PATH = "/home/wan/guanting's/diffusion-customer/assets/ref.png"

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        BASE_DIR,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    prompt = "product photo of a sneaker, studio lighting, high detail, white background"
    img = pipe(
        prompt=prompt,
        guidance_scale=5.0,
        num_inference_steps=20,
        height=512,
        width=512,
    ).images[0]

    img.save(OUT_PATH)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
