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
        self._prompt_cache = {}

    @torch.no_grad()
    def encode_batch_concat(self, prompt, z_vectors_batch, negative_prompt=None):
        """
        z_vectors_batch: (BS, 4096) Tensor (CPU or GPU)
        """
        bs = z_vectors_batch.shape[0]
        if negative_prompt is None:
            # negative_prompt = "cropped, out of frame, cut off, close-up, zoomed in, off-center, corner, partial, multiple shoes, pair, logo, text, brand"
            negative_prompt = "cropped, out of frame, close-up, off-center, cut off"

        # ✅ 1) cache prompt encoding
        cache_key = (prompt, negative_prompt)
        if cache_key in self._prompt_cache:
            p_embeds, n_p_embeds, pooled, n_p_pooled = self._prompt_cache[cache_key]
        else:
            out = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                negative_prompt=negative_prompt
            )
            p_embeds, n_p_embeds, pooled, n_p_pooled = out
            self._prompt_cache[cache_key] = (p_embeds, n_p_embeds, pooled, n_p_pooled)

        # ✅ 2) expand 替代 repeat（避免真实复制）
        # p_embeds shape: (1, L, 4096)  -> expand to (BS, L, 4096)
        p_embeds_batch = p_embeds.expand(bs, -1, -1).contiguous()
        n_p_embeds_batch = n_p_embeds.expand(bs, -1, -1).contiguous()
        pooled_batch = pooled.expand(bs, -1).contiguous()
        n_p_pooled_batch = n_p_pooled.expand(bs, -1).contiguous()

        # z: (BS, 4096) -> (BS, 1, 4096)
        z_v = z_vectors_batch.unsqueeze(1).to(device=self.device, dtype=p_embeds.dtype)

        p_combined = torch.cat([p_embeds_batch, z_v], dim=1)
        n_combined = torch.cat([n_p_embeds_batch, torch.zeros_like(z_v)], dim=1)
        return (p_combined, n_combined, pooled_batch, n_p_pooled_batch)

    @torch.inference_mode()
    def generate_batch(self, embeds_batch, seeds):
        p, n_p, pooled, n_p_pooled = embeds_batch
        # 准备批量 Generator
        generators = [torch.Generator(self.device).manual_seed(s) for s in seeds]
        
        # 调用 pipe 进行批量推理
        return self.pipe(
            prompt_embeds=p, 
            negative_prompt_embeds=n_p,
            pooled_prompt_embeds=pooled, 
            negative_pooled_prompt_embeds=n_p_pooled,
            num_inference_steps=20, 
            guidance_scale=4.5,
            height=384, width=384, 
            generator=generators,
            output_type="pil"
        ).images


class SD35EmbeddingGenerator:
    def __init__(self, model_path, device="cuda", torch_dtype=torch.float16):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype
        ).to(device)
        self.device = device

    # @torch.no_grad()
    # def encode_sandwich(self, prompt, z_vector):
    #     """
    #     Symmetric Sandwich: [Prompt_Slice, z, Prompt_Slice]
    #     """
    #     out = self.pipe.encode_prompt(
    #         prompt=prompt, prompt_2=prompt, prompt_3=prompt,
    #         negative_prompt="cropped, out of frame, close-up, off-center, cut off"
    #     )
    #     p_embeds, n_p_embeds, pooled, n_p_pooled = out
        
    #     z_vector = z_vector.to(device=self.device, dtype=p_embeds.dtype).view(1, 1, -1)
        
    #     # Slice to 30 tokens to ensure shape alignment
    #     p_slice = p_embeds[:, :30, :]
    #     n_slice = n_p_embeds[:, :30, :]
        
    #     # Build 30 + 1 + 30 = 61 token sequence
    #     p_combined = torch.cat([p_slice, z_vector, p_slice], dim=1)
    #     # Negative sequence must match shape exactly
    #     n_combined = torch.cat([n_slice, torch.zeros_like(z_vector), n_slice], dim=1)
        
    #     return (p_combined, n_combined, pooled, n_p_pooled)
    
    @torch.no_grad()
    def encode_simple_concat(self, prompt, z_vector, negative_prompt=None):
        """
        将 z 直接拼接到完整 Prompt Embedding 的末尾
        """
        if negative_prompt is None:
            negative_prompt = "logo, text, brand, blurry, low quality, multiples, cropped, out of frame, close-up, off-center, cut off"
        out = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt=negative_prompt
        )
        p_embeds, n_p_embeds, pooled, n_p_pooled = out
        
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
            num_inference_steps=20, guidance_scale=4.5,
            height=384, width=384, generator=generator
        ).images[0]