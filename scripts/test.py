import torch
from PIL import Image
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models.controlnets import SD3ControlNetModel

BASE_DIR = "/home/wan/guanting's/diffusion-customer/model/stabilityai/stable-diffusion-3.5-large"
CN_DIR = "/home/wan/guanting's/diffusion-customer/model/controlnets/sd35_large_controlnet_canny"

controlnet = SD3ControlNetModel.from_pretrained(CN_DIR, torch_dtype=torch.float16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    BASE_DIR,
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

edge = Image.open("edge.png").convert("RGB")
img = pipe(
    prompt="product photo of a sneaker, studio lighting",
    control_image=edge,                       # :contentReference[oaicite:6]{index=6}
    controlnet_conditioning_scale=1.0,        # :contentReference[oaicite:7]{index=7}
    guidance_scale=5.0,
    num_inference_steps=24,
    height=1024,
    width=1024,
).images[0]
