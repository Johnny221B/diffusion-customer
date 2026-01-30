import torch
from diffusers import StableDiffusion3Pipeline

class SD35EmbeddingGenerator:
    def __init__(self, model_dir, device="cuda", torch_dtype=torch.float16):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir, torch_dtype=torch_dtype
        ).to(device)
        self.device = device

    @torch.no_grad()
    def encode_sandwich(self, prompt, z_vector):
        """
        修复版对称三明治结构：确保正面和负面 Embeds 形状一致
        """
        # 1. 编码主 Prompt
        out = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="lowres, blurry, worst quality, logo, brand, text"
        )
        p_embeds, n_p_embeds, pooled, n_p_pooled = out
        
        # 2. 准备 z 向量
        z_vector = z_vector.to(device=self.device, dtype=p_embeds.dtype).view(1, 1, -1)
        
        # 3. 对齐正面和负面序列
        # 取前 30 个 token
        p_slice = p_embeds[:, :30, :]
        n_slice = n_p_embeds[:, :30, :] # 负面向量也取同样长度
        
        # 正面拼接: [30] + [1] + [30] = 61
        p_combined = torch.cat([p_slice, z_vector, p_slice], dim=1)
        
        # 负面拼接: 保持形状一致，中间位置填充零向量或取负面序列的对应部分
        # 这里最稳妥的是直接对负面向量也进行同样的 30+1+30 结构处理
        z_neg = torch.zeros_like(z_vector)
        n_combined = torch.cat([n_slice, z_neg, n_slice], dim=1)
        
        return (p_combined, n_combined, pooled, n_p_pooled)

    @torch.inference_mode()
    def generate(self, embeds, seed=123):
        generator = torch.Generator(self.device).manual_seed(seed)
        p, n_p, pooled, n_p_pooled = embeds
        # 调用时形状现在是 (1, 61, 4096) == (1, 61, 4096)
        return self.pipe(
            prompt_embeds=p, 
            negative_prompt_embeds=n_p,
            pooled_prompt_embeds=pooled, 
            negative_pooled_prompt_embeds=n_p_pooled,
            num_inference_steps=28, 
            guidance_scale=7.0,
            height=512, 
            width=512, 
            generator=generator
        ).images[0]