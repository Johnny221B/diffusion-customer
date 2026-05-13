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
    
    @torch.no_grad()
    def encode_batch_insert(self, prompt, z_vectors_batch, negative_prompt=None):
        """
        将 z 插入到两段 prompt 的中间，而不是直接拼到最后。

        参数
        ----
        prompt : str
            这里保留接口兼容，但函数内部会使用预定义的前后半句。
        z_vectors_batch : Tensor
            shape = (BS, 4096), CPU 或 GPU 都可以
        negative_prompt : str or None
        """
        bs = z_vectors_batch.shape[0]

        if negative_prompt is None:
            negative_prompt = "cropped, out of frame, close-up, off-center, cut off, blurry"

        # 你想要的两段 prompt
        prompt_a = "Product photo of a single shoe with a style,"
        prompt_b = "full shoe visible, side profile, centered on a white background"
        # prompt_a = "Product photo of a single shoe with a specific style,"
        # prompt_b = "full shoe visible, side profile, centered on a plain white background"

        # 完整 prompt 只用来提供 pooled embedding
        full_prompt = f"{prompt_a} {prompt_b}"

        cache_key = (prompt_a, prompt_b, full_prompt, negative_prompt)

        if cache_key in self._prompt_cache:
            (
                p_a, p_b,
                n_a, n_b,
                pooled_full, n_p_pooled_full
            ) = self._prompt_cache[cache_key]
        else:
            # 前半句 token embeddings
            out_a = self.pipe.encode_prompt(
                prompt=prompt_a,
                prompt_2=prompt_a,
                prompt_3=prompt_a,
                negative_prompt=negative_prompt
            )
            p_a, n_a, _, _ = out_a

            # 后半句 token embeddings
            out_b = self.pipe.encode_prompt(
                prompt=prompt_b,
                prompt_2=prompt_b,
                prompt_3=prompt_b,
                negative_prompt=negative_prompt
            )
            p_b, n_b, _, _ = out_b

            # 完整 prompt 的 pooled embeddings
            out_full = self.pipe.encode_prompt(
                prompt=full_prompt,
                prompt_2=full_prompt,
                prompt_3=full_prompt,
                negative_prompt=negative_prompt
            )
            _, _, pooled_full, n_p_pooled_full = out_full

            self._prompt_cache[cache_key] = (
                p_a, p_b,
                n_a, n_b,
                pooled_full, n_p_pooled_full
            )

        # expand 到 batch
        p_a_batch = p_a.expand(bs, -1, -1).contiguous()
        p_b_batch = p_b.expand(bs, -1, -1).contiguous()
        n_a_batch = n_a.expand(bs, -1, -1).contiguous()
        n_b_batch = n_b.expand(bs, -1, -1).contiguous()

        pooled_batch = pooled_full.expand(bs, -1).contiguous()
        n_p_pooled_batch = n_p_pooled_full.expand(bs, -1).contiguous()

        # z: (BS, 4096) -> (BS, 1, 4096)
        z_v = z_vectors_batch.to(device=self.device, dtype=p_a.dtype).unsqueeze(1)

        # 正向：前半句 + z + 后半句
        p_combined = torch.cat([p_a_batch, z_v, p_b_batch], dim=1)

        # 负向：前半句负向 + 0向量 + 后半句负向
        n_combined = torch.cat([n_a_batch, torch.zeros_like(z_v), n_b_batch], dim=1)

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
            guidance_scale=5,
            height=496, width=496, 
            generator=generators,
            output_type="pil"
        ).images


class SD35EmbeddingGenerator:
    def __init__(self, model_path, device="cuda", torch_dtype=torch.float16):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype,
            variant="fp16"
        ).to(device)
        self.device = device
        self.pipe.set_progress_bar_config(disable=True)
        self._prompt_cache = {}

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
    
    @torch.no_grad()
    def encode_simple_insert(self, prompt, z_vector, negative_prompt=None):
        """
        单张版本：将 z 插入到两段 prompt 中间
        """
        if negative_prompt is None:
            negative_prompt = "cropped, out of frame, close-up, off-center, cut off, blurry"

        prompt_a = "Product photo of a single shoe with a style,"
        prompt_b = "full shoe visible, side profile, centered on a white background"
        full_prompt = f"{prompt_a} {prompt_b}"

        cache_key = (prompt_a, prompt_b, full_prompt, negative_prompt)

        if cache_key in self._prompt_cache:
            (
                p_a, p_b,
                n_a, n_b,
                pooled_full, n_p_pooled_full
            ) = self._prompt_cache[cache_key]
        else:
            out_a = self.pipe.encode_prompt(
                prompt=prompt_a,
                prompt_2=prompt_a,
                prompt_3=prompt_a,
                negative_prompt=negative_prompt
            )
            p_a, n_a, _, _ = out_a

            out_b = self.pipe.encode_prompt(
                prompt=prompt_b,
                prompt_2=prompt_b,
                prompt_3=prompt_b,
                negative_prompt=negative_prompt
            )
            p_b, n_b, _, _ = out_b

            out_full = self.pipe.encode_prompt(
                prompt=full_prompt,
                prompt_2=full_prompt,
                prompt_3=full_prompt,
                negative_prompt=negative_prompt
            )
            _, _, pooled_full, n_p_pooled_full = out_full

            self._prompt_cache[cache_key] = (
                p_a, p_b,
                n_a, n_b,
                pooled_full, n_p_pooled_full
            )

        z_v = z_vector.to(device=self.device, dtype=p_a.dtype).view(1, 1, -1)

        p_combined = torch.cat([p_a, z_v, p_b], dim=1)
        n_combined = torch.cat([n_a, torch.zeros_like(z_v), n_b], dim=1)

        return (p_combined, n_combined, pooled_full, n_p_pooled_full)

    @torch.inference_mode()
    def generate(self, embeds, seed=123):
        generator = torch.Generator(self.device).manual_seed(seed)
        p, n_p, pooled, n_p_pooled = embeds
        return self.pipe(
            prompt_embeds=p, negative_prompt_embeds=n_p,
            pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=n_p_pooled,
            num_inference_steps=20, guidance_scale=5,
            height=496, width=496, generator=generator
        ).images[0]