from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

from torchvision import transforms
# 假设 dreamsim 文件夹在你的 python path 中
from dreamsim import dreamsim


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


import torch
from PIL import Image
from torchvision import transforms
# 确保 dreamsim 库在您的 Python 路径中
from dreamsim import dreamsim

class DreamSimScorer:
    def __init__(self, device="cuda"):
        self.device = device
        # 初始化 DreamSim 模型
        self.model, _ = dreamsim(pretrained=True, device=self.device)
        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), 
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def preprocess(self, img):
        """将 PIL Image 转换为 DreamSim 输入张量"""
        if isinstance(img, str):
            img = Image.open(img)
        img = img.convert('RGB')
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def get_distance(self, img_ref_tensor, img_cand):
        """计算图片与参考张量的感知距离"""
        cand_tensor = self.preprocess(img_cand)
        return self.model(img_ref_tensor, cand_tensor).item()