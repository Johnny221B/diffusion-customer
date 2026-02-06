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

    @torch.inference_mode()
    def generate(self, embeds, seed=123):
        generator = torch.Generator(self.device).manual_seed(seed)
        p, n_p, pooled, n_p_pooled = embeds
        return self.pipe(
            prompt_embeds=p, negative_prompt_embeds=n_p,
            pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=n_p_pooled,
            num_inference_steps=30, guidance_scale=9.0,
            height=512, width=512, generator=generator
        ).images[0]