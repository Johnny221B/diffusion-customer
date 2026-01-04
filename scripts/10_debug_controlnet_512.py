import os
import torch
from PIL import Image

from diffusers import StableDiffusion3Pipeline, StableDiffusion3ControlNetPipeline
from diffusers.models.controlnets import SD3ControlNetModel

from src.edge_utils import make_canny_edge

# -------------------------
# Paths / Config
# -------------------------
BASE_DIR = "/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large"
CN_DIR   = "/home/linyuliu/jxmount/diffusion_custom/models/controlnets/sd35_large_controlnet_canny"
REF_IMG_PATH = "/home/linyuliu/jxmount/diffusion_custom/assets/ref.png"
OUT_DIR  = "/home/linyuliu/jxmount/diffusion_custom/outputs/debug_512_controlnet_2gpu"

PROMPT = "product photo of a modern sneaker, studio lighting, white background, high detail"
NEG    = "lowres, blurry, worst quality, artifacts"

H = 512
W = 512
STEPS = 24
GUIDANCE = 5.0
SEED = 123456

DTYPE = torch.float16

# 手动指定两张卡
BASE_DEVICE = "cuda:0"
CN_DEVICE   = "cuda:1"


def save_img(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print("Saved:", path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------------
    # 0) Prepare edge at target resolution
    # -------------------------
    # 重要：先把 ref resize 到 (W,H)，再做 canny，避免“先 canny 再 resize”导致边缘变形/变稀
    ref = Image.open(REF_IMG_PATH).convert("RGB").resize((W, H))
    edge = make_canny_edge(ref, low=100, high=200).convert("RGB")
    save_img(edge, os.path.join(OUT_DIR, "control_edge_512.png"))

    # -------------------------
    # 1) Base pipeline on GPU0 (optional sanity check)
    # -------------------------
    print(f"\n[1] Loading BASE pipeline on {BASE_DEVICE} ...")
    base = StableDiffusion3Pipeline.from_pretrained(
        BASE_DIR,
        torch_dtype=DTYPE,
    ).to(BASE_DEVICE)

    gen0 = torch.Generator(device=BASE_DEVICE).manual_seed(SEED)
    print("[1] Generating base image (no ControlNet) ...")
    img_base = base(
        prompt=PROMPT,
        negative_prompt=NEG,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        height=H,
        width=W,
        generator=gen0,
    ).images[0]
    save_img(img_base, os.path.join(OUT_DIR, f"base_no_control_seed{SEED}.png"))

    # 释放 GPU0 显存（避免碎片化也顺便保险）
    print("[1] Freeing BASE pipeline GPU memory ...")
    del base
    torch.cuda.empty_cache()

    # -------------------------
    # 2) ControlNet pipeline on GPU1
    # -------------------------
    print(f"\n[2] Loading ControlNet on {CN_DEVICE} ...")
    controlnet = SD3ControlNetModel.from_pretrained(
        CN_DIR,
        torch_dtype=DTYPE,
    ).to(CN_DEVICE)

    print(f"[2] Loading ControlNet pipeline on {CN_DEVICE} ...")
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        BASE_DIR,
        controlnet=controlnet,
        torch_dtype=DTYPE,
    ).to(CN_DEVICE)

    def run_control(scale: float, tag: str):
        gen1 = torch.Generator(device=CN_DEVICE).manual_seed(SEED)
        print(f"[2] Generating ControlNet image: {tag}, scale={scale} ...")
        img = pipe(
            prompt=PROMPT,
            negative_prompt=NEG,
            control_image=edge,  # edge 已经是 512x512
            controlnet_conditioning_scale=scale,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            height=H,
            width=W,
            generator=gen1,
        ).images[0]
        save_img(img, os.path.join(OUT_DIR, f"{tag}_scale{scale}_seed{SEED}.png"))

    # A) scale=0 -> 等价于“关掉 ControlNet 影响”（最关键对照）
    run_control(0.0, "control")

    # B) scale=1.0 -> 你当前默认强度
    run_control(1.0, "control")

    # C) scale=0.3 -> 常见救塌缩强度
    run_control(0.3, "control")

    print("\nDONE. Please compare outputs in:", OUT_DIR)
    print(" - control_edge_512.png")
    print(" - base_no_control_seed*.png")
    print(" - control_scale0.0_seed*.png (ControlNet off)")
    print(" - control_scale1.0_seed*.png (your current)")
    print(" - control_scale0.3_seed*.png (weaker control)")


if __name__ == "__main__":
    main()
