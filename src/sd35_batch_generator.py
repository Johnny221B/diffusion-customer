# src/sd35_batch_generator.py
import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline

class SD35BatchEmbeddingGenerator:
    # 删掉那个没用的 pipe_instance，并将 device 默认值设为 "cuda"
    def __init__(self, model_path, device="cuda", torch_dtype=torch.float16):
        self.device = device
        # 统一使用从外部传入的 device (例如 cuda:0, cuda:1 等)
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype,
            variant="fp16" 
        ).to(self.device) # <--- 关键：确保移动到指定的卡
        
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def encode_batch_concat(self, prompt, z_vectors_batch):
        """
        z_vectors_batch: (BS, 4096) 的 Tensor
        """
        bs = z_vectors_batch.shape[0]
        
        # 1. 基础 Encoding (获取单份 Embedding)
        out = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="logo, text, brand, blurry, low quality, multiples, pair,"
        )
        p_embeds, n_p_embeds, pooled, n_p_pooled = out
        
        # 2. 批量扩展与拼接 
        # 将基础 embedding 复制 BS 份
        p_embeds_batch = p_embeds.repeat(bs, 1, 1)    # (BS, L, 4096)
        n_p_embeds_batch = n_p_embeds.repeat(bs, 1, 1) # (BS, L, 4096)
        pooled_batch = pooled.repeat(bs, 1)           # (BS, 2048)
        n_p_pooled_batch = n_p_pooled.repeat(bs, 1)   # (BS, 2048)
        
        # 准备注入的 z 向量 (BS, 1, 4096)
        z_v = z_vectors_batch.unsqueeze(1).to(device=self.device, dtype=p_embeds.dtype)
        
        # 在 Token 序列维度 (dim=1) 拼接
        p_combined = torch.cat([p_embeds_batch, z_v], dim=1)
        n_combined = torch.cat([n_p_embeds_batch, torch.zeros_like(z_v)], dim=1)
        
        return (p_combined, n_combined, pooled_batch, n_p_pooled_batch)

    @torch.inference_mode()
    def generate_batch(self, embeds_batch, seeds):
        """
        并行生成 25 张图
        """
        p, n_p, pooled, n_p_pooled = embeds_batch
        # 准备批量 Generator
        generators = [torch.Generator(self.device).manual_seed(s) for s in seeds]
        
        # 调用 pipe 进行批量推理
        return self.pipe(
            prompt_embeds=p, 
            negative_prompt_embeds=n_p,
            pooled_prompt_embeds=pooled, 
            negative_pooled_prompt_embeds=n_p_pooled,
            num_inference_steps=30, 
            guidance_scale=7.5,
            height=512, width=512, 
            generator=generators,
            output_type="pil"
        ).images


class SD35EmbeddingGenerator:
    def __init__(self, model_path, device="cuda", torch_dtype=torch.float16):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype
        ).to(device)
        self.device = device

    @torch.no_grad()
    def encode_sandwich(self, prompt, z_vector):
        """
        Symmetric Sandwich: [Prompt_Slice, z, Prompt_Slice]
        """
        out = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="logo, text, brand, blurry, low quality, multiples, pair,"
        )
        p_embeds, n_p_embeds, pooled, n_p_pooled = out
        
        z_vector = z_vector.to(device=self.device, dtype=p_embeds.dtype).view(1, 1, -1)
        
        # Slice to 30 tokens to ensure shape alignment
        p_slice = p_embeds[:, :30, :]
        n_slice = n_p_embeds[:, :30, :]
        
        # Build 30 + 1 + 30 = 61 token sequence
        p_combined = torch.cat([p_slice, z_vector, p_slice], dim=1)
        # Negative sequence must match shape exactly
        n_combined = torch.cat([n_slice, torch.zeros_like(z_vector), n_slice], dim=1)
        
        return (p_combined, n_combined, pooled, n_p_pooled)
    
    @torch.no_grad()
    def encode_simple_concat(self, prompt, z_vector):
        """
        将 z 直接拼接到完整 Prompt Embedding 的末尾
        """
        out = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="logo, text, brand, blurry, low quality, multiples, pair,"
        )
        p_embeds, n_p_embeds, pooled, n_p_pooled = out
        
        # 确保维度匹配 (1, 1, 4096)
        z_v = z_vector.to(device=self.device, dtype=p_embeds.dtype).view(1, 1, -1)
        
        # 直接在 token 序列维度(dim=1)上拼接
        p_combined = torch.cat([p_embeds, z_v], dim=1)
        n_combined = torch.cat([n_p_embeds, torch.zeros_like(z_v)], dim=1)
        
        return (p_combined, n_combined, pooled, n_p_pooled)

    @torch.inference_mode()
    def generate(self, embeds, seed=123):
        generator = torch.Generator(self.device).manual_seed(seed)
        p, n_p, pooled, n_p_pooled = embeds
        return self.pipe(
            prompt_embeds=p, negative_prompt_embeds=n_p,
            pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=n_p_pooled,
            num_inference_steps=30, guidance_scale=7.5,
            height=512, width=512, generator=generator
        ).images[0]