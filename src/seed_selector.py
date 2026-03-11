# src/seed_selector.py
import os
import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from src.sd35_batch_generator import SD35BatchEmbeddingGenerator, SD35EmbeddingGenerator

def border_is_all_white(img, border=6, white_thresh=240) -> bool:
    """
    判断图片最外圈 border 像素是否几乎全白。
    white_thresh 越小越严格（245 表示 RGB 都 >=245 才算白）。
    """
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    b = int(border)

    # 取四条边框区域
    top = arr[:b, :, :]
    bottom = arr[h-b:, :, :]
    left = arr[:, :b, :]
    right = arr[:, w-b:, :]

    # 白的条件：三个通道都 >= white_thresh
    def is_white(region):
        return np.all(region >= white_thresh)

    return is_white(top) and is_white(bottom) and is_white(left) and is_white(right)

def bbox_stats(img, white_thresh=240):
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    fg = np.any(arr < white_thresh, axis=-1)  # 前景：任一通道 < 阈值
    fg_ratio = float(fg.mean())
    if fg_ratio < 1e-6:
        return None  # 全白
    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_w = float(x1 - x0 + 1)
    bbox_h = float(y1 - y0 + 1)
    area_ratio = (bbox_w * bbox_h) / float(h * w)
    return {
        "h": h, "w": w,
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "bbox_w": bbox_w, "bbox_h": bbox_h,
        "area_ratio": float(area_ratio),
        "fg_ratio": fg_ratio,
    }

def full_shoe_likely(img, white_thresh=240,
                     min_area=0.08, max_area=0.75,
                     min_fg=0.012,
                     min_margin=12):
    st = bbox_stats(img, white_thresh=white_thresh)
    if st is None:
        return False
    h, w = st["h"], st["w"]
    # 前景不能太少（否则只是小碎片）
    if st["fg_ratio"] < min_fg:
        return False
    # bbox 面积不能太小/太大（太大多是 close-up；太小是只露一点点）
    if st["area_ratio"] < min_area or st["area_ratio"] > max_area:
        return False
    # bbox 不能贴边（防裁切/角落）
    if st["x0"] < min_margin or st["y0"] < min_margin or (w - 1 - st["x1"]) < min_margin or (h - 1 - st["y1"]) < min_margin:
        return False
    return True


def center_ok(img, white_thresh=240, min_margin=12, max_center_offset=0.12) -> bool:
    """
    （可选）更严格的“居中/完整”检查：
    - 找非白前景 bbox，要求离边缘 >= min_margin
    - bbox 中心距画面中心的偏移比例 <= max_center_offset
    """
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape

    fg = np.any(arr < white_thresh, axis=-1)  # 前景：任一通道 < 阈值
    if fg.mean() < 0.01:
        return False  # 太空白

    ys, xs = np.where(fg)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # 不贴边
    if x0 < min_margin or y0 < min_margin or (w - 1 - x1) < min_margin or (h - 1 - y1) < min_margin:
        return False

    # 居中程度
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    off_x = abs(cx - (w / 2.0)) / w
    off_y = abs(cy - (h / 2.0)) / h
    return (off_x <= max_center_offset) and (off_y <= max_center_offset)

@torch.no_grad()
def select_seeds_by_clip(
    temp_gen: SD35EmbeddingGenerator,
    prompt: str,
    run_dir: str,
    candidate_n: int = 300,
    top_k: int = 100,
    seed_low: int = 0,
    seed_high: int = 2_000_000,
    rng_seed: int = 12345,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    clip_batch_size: int = 16,
    border_px: int = 8,
    border_penalty: float = 15.0,
):
    out_dir = os.path.join(run_dir, "seed_screening")
    sel_dir = os.path.join(out_dir, "selected")
    rej_dir = os.path.join(out_dir, "rejected")
    os.makedirs(sel_dir, exist_ok=True)
    os.makedirs(rej_dir, exist_ok=True)
    
    check_prompt = "A single shoe fully visible, centered, not cropped, product photo on white background"

    device = temp_gen.device  # e.g. "cuda:0"
    negative_prompt = "cropped, close-up"

    # 1) sample candidate seeds (unique)
    rng = np.random.default_rng(rng_seed)
    seeds = rng.choice(np.arange(seed_low, seed_high, dtype=np.int64), size=candidate_n, replace=False).tolist()

    # 2) generate images with z=0
    z0 = torch.zeros(4096, dtype=torch.float16, device=device)

    images = []
    for s in seeds:
        embeds = temp_gen.encode_simple_concat(prompt, z0, negative_prompt)
        img = temp_gen.generate(embeds, seed=int(s))
        images.append(img)

    # 3) CLIP scoring (batch)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_model_name)

    scores = []
    for st in range(0, len(images), clip_batch_size):
        batch_imgs = images[st:st + clip_batch_size]
        inputs = clip_proc(
            text=[check_prompt],              # ✅ 只给 1 条文本
            images=batch_imgs,
            return_tensors="pt",
            padding=True
        ).to(device)

        out = clip_model(**inputs)

        # logits_per_image: (bs, 1)
        # logits = out.logits_per_image[:, 0]  # ✅ (bs,)
        # scores.extend([float(x) for x in logits.detach().cpu()])
        logits = out.logits_per_image[:, 0].detach().float().cpu().numpy()

        # ✅ border penalty：外圈不是全白的，直接减 10 分
        for i, img in enumerate(batch_imgs):
            s = float(logits[i])

            # 1) 边框白：不满足就重罚
            if not border_is_all_white(img, border=border_px, white_thresh=240):
                s -= float(border_penalty)

            # 2) 完整性：不满足就更重罚（关键）
            if not full_shoe_likely(img, white_thresh=240,
                                    min_area=0.06, max_area=0.70,
                                    min_fg=0.012, min_margin=8):
                s -= 50.0   # ✅ 建议先用 50（非常强的惩罚，保证 top100 里几乎没有半截）

            scores.append(s)
    # scores = np.array(scores, dtype=np.float32)
    # order = np.argsort(scores)[::-1]  # high->low

    # top_idx = set(order[:top_k].tolist())
    # top_seeds = [int(seeds[i]) for i in order[:top_k]]
    # top_scores = [float(scores[i]) for i in order[:top_k]]

    # # 4) save images into selected/rejected
    # for i, (s, img) in enumerate(zip(seeds, images)):
    #     sc = float(scores[i])
    #     if i in top_idx:
    #         save_path = os.path.join(sel_dir, f"seed_{int(s)}_clip{sc:.4f}.png")
    #     else:
    #         save_path = os.path.join(rej_dir, f"seed_{int(s)}_clip{sc:.4f}.png")
    #     img.save(save_path)
    #     img.close()
    
    # ✅ 先保证长度一致
    if len(scores) != len(seeds):
        raise RuntimeError(f"Length mismatch: seeds={len(seeds)}, images={len(images)}, scores={len(scores)}")

    df = pd.DataFrame({
        "seed": [int(s) for s in seeds],
        "clip_score": [float(x) for x in scores],
    })

    df_sorted = df.sort_values("clip_score", ascending=False).reset_index(drop=True)

    # ✅ 防止 top_k > candidate_n
    top_k_eff = min(top_k, len(df_sorted))
    df_top = df_sorted.head(top_k_eff)

    top_seeds = df_top["seed"].tolist()
    top_scores = df_top["clip_score"].tolist()

    top_set = set(top_seeds)

    # 保存图片到 selected/rejected（按 seed 判断，不用索引对齐）
    for s, img, sc in zip(seeds, images, scores):
        s = int(s)
        sc = float(sc)
        if s in top_set:
            save_path = os.path.join(sel_dir, f"seed_{s}_clip{sc:.4f}.png")
        else:
            save_path = os.path.join(rej_dir, f"seed_{s}_clip{sc:.4f}.png")
        img.save(save_path)
        img.close()

    # 保存表
    df.to_csv(os.path.join(out_dir, "all_seed_scores.csv"), index=False)
    df_top.to_csv(os.path.join(out_dir, "selected_seeds.csv"), index=False)
    with open(os.path.join(out_dir, "selected_seeds.txt"), "w") as f:
        for s in top_seeds:
            f.write(f"{s}\n")

    # 5) save scores + seed lists
    # pd.DataFrame({"seed": [int(x) for x in seeds], "clip_score": scores}).to_csv(
    #     os.path.join(out_dir, "all_seed_scores.csv"), index=False
    # )
    # pd.DataFrame({"seed": top_seeds, "clip_score": top_scores}).to_csv(
    #     os.path.join(out_dir, "selected_seeds.csv"), index=False
    # )
    # with open(os.path.join(out_dir, "selected_seeds.txt"), "w") as f:
    #     for s in top_seeds:
    #         f.write(f"{s}\n")

    return top_seeds