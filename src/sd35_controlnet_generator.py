import torch
from PIL import Image
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models.controlnets import SD3ControlNetModel


class SD35ControlNetGenerator:
    def __init__(
        self,
        base_dir: str,
        controlnet_dir: str,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.device = device

        controlnet = SD3ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=torch_dtype)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            base_dir,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
        )
        self.pipe = pipe.to(device)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        seed: int | None = None,
        latents: torch.Tensor | None = None,   # NEW: explicit diffusion latents
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 24,
        guidance_scale: float = 5.0,
        controlnet_conditioning_scale: float = 0.5,  
        negative_prompt: str = "lowres, blurry, worst quality, artifacts",
    ) -> Image.Image:
        # control image resize
        control_image = control_image.convert("RGB").resize((width, height))

        # If user provides latents, we pass them directly.
        # Otherwise fall back to seed-driven randomness.
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
        )

        if latents is not None:
            # Ensure device/dtype match pipeline
            latents = latents.to(self.device)
            # keep dtype consistent with pipeline weights
            # (transformer exists on SD3; fall back to pipe.dtype if needed)
            dtype = getattr(getattr(self.pipe, "transformer", None), "dtype", None) or self.pipe.dtype
            latents = latents.to(dtype=dtype)
            out = self.pipe(**kwargs, latents=latents)
        else:
            if seed is None:
                raise ValueError("Either `latents` or `seed` must be provided.")
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            out = self.pipe(**kwargs, generator=gen)

        return out.images[0]
