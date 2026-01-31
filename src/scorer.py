from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel


class CLIPScorer:
    """
    Returns cosine similarity between image embedding and text embedding.
    Higher = more aligned.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name,use_safetensors=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()

    @torch.inference_mode()
    def __call__(self, img: Image.Image, text: str) -> float:
        inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # CLIP already produces normalized-ish embeddings sometimes; to be safe, normalize
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        score = (img_emb * txt_emb).sum(dim=-1).item()  # cosine similarity
        return float(score)


class ToyRedScorer:
    def __call__(self, img: Image.Image) -> float:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        # 红色相对强度：r - (g+b)/2
        score = float((r - 0.5 * (g + b)).mean())
        return score

class CLIPImageScorer:
    """
    Returns cosine similarity between two image embeddings.
    Higher = more similar.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, img: Image.Image, ref_img: Image.Image) -> float:
        inputs = self.processor(images=[img, ref_img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        img_emb = outputs.image_embeds[0]
        ref_emb = outputs.image_embeds[1]

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)

        score = (img_emb * ref_emb).sum(dim=-1).item()
        return float(score)
