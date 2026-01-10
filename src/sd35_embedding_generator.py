import torch
from diffusers import StableDiffusion3Pipeline

class SD35EmbeddingGenerator:
    def __init__(self, model_dir, device="cuda", torch_dtype=torch.float16):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir, torch_dtype=torch_dtype
        ).to(device)
        self.device = device

    @torch.no_grad()
    def encode_sandwich(self, prefix, suffix, z_vector):
        """
        实现三明治格式注入: I_m = [t_prefix, z, t_suffix]
        z_vector: 形状为 (4096,) 的 T5 style 向量
        """
        prompt = f"{prefix} , {suffix}"
        
        # 获取基础编码
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="lowres, blurry, worst quality"
        )
        
        # 在 SD3.5 中，T5 的 token 位于 prompt_embeds [0:256] 区域
        # 我们将 z 注入到逗号所在的 index 5 位置（仅作为示例，实际可动态定位）
        injection_idx = 5 
        
        prompt_embeds_mod = prompt_embeds.clone()
        z_vector = z_vector.to(device=self.device, dtype=prompt_embeds.dtype)
        
        # 执行向量注入 (Phase 1)
        prompt_embeds_mod[0, injection_idx, :] = z_vector
        
        return (prompt_embeds_mod, negative_prompt_embeds, 
                pooled_prompt_embeds, negative_pooled_prompt_embeds)

    @torch.inference_mode()
    def generate(self, embeds, seed=123, height=512, width=512):
        generator = torch.Generator(self.device).manual_seed(seed)
        p, n_p, pooled, n_pooled = embeds
        image = self.pipe(
            prompt_embeds=p,
            negative_prompt_embeds=n_p,
            pooled_prompt_embeds=pooled,
            negative_pooled_prompt_embeds=n_pooled,
            num_inference_steps=24,
            guidance_scale=5.0,
            height=height,
            width=width,
            generator=generator
        ).images[0]
        return image