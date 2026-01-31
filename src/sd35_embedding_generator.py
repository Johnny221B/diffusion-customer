import torch
from diffusers import StableDiffusion3Pipeline

class SD35EmbeddingGenerator:
    def __init__(self, model_dir, device="cuda", torch_dtype=torch.float16):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir, torch_dtype=torch_dtype
        )
        self.device = device
        if device == "cuda":
            # Reduce VRAM pressure for large SD3.5 models
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    @torch.no_grad()
    def encode_sandwich(self, prefix, suffix, z_vector):
        prompt = f"{prefix} , {suffix}"
        (p_embeds, n_p_embeds, pooled, n_p_pooled) = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="lowres, blurry, worst quality, messy"
        )
        # 注入 T5 向量到逗号位置 (idx 5)
        p_embeds_mod = p_embeds.clone()
        z_vector = z_vector.to(device=self.device, dtype=p_embeds.dtype)
        p_embeds_mod[0, 5, :] = z_vector
        return (p_embeds_mod, n_p_embeds, pooled, n_p_pooled)

    @torch.inference_mode()
    def generate(self, embeds, seed=123):
        generator = torch.Generator(self.device).manual_seed(seed)
        p, n_p, pooled, n_p_pooled = embeds
        image = self.pipe(
            prompt_embeds=p, negative_prompt_embeds=n_p,
            pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=n_p_pooled,
            num_inference_steps=28, guidance_scale=5.0,
            height=512, width=512, generator=generator
        ).images[0]
        return image