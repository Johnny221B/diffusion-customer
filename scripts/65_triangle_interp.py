"""
Experiment: Triangle (3-word) interpolation in SD3.5's 4096-dim token-embedding space.

#39 的两点插值升到三点。取三个概念 A / B / C 的 token embedding，张成一个
仿射平面。三角形内部用重心坐标 (w_a, w_b, w_c), w_a+w_b+w_c=1 参数化（2 个自由度）。

产出两类东西：
  1) 三角形上密集采样 (barycentric grid, 分辨率 K) -> 每点生成一张图，
     按重心坐标 2D 位置拼成三角缩略图 collage（肉眼看哪片区域开始崩）。
     另存一张 ternary 散点：每点按「最近的顶点图」上色（语义归属分区）。
  2) 取内部点 P（默认重心），分别向 A / B / C 三个顶点连线，每条线就退化成
     #39 那种一维插值。画 3 张折线图，横轴 alpha、纵轴 DreamSim，
     每张同时画到 A/B/C 三个顶点的 DreamSim（看走向一个顶点时是否同时远离另两个）。

DreamSim 距离都是「到三个顶点生成图」的距离（不是到 canonical R）。
"""

import os
import sys
import argparse
import itertools
import torch
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# 让 `from src...` 在 `python scripts/65_...py` 直跑时也能 import 到项目根（同 63）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator  # noqa: E402
from src.scorer import DreamSimScorer  # noqa: E402


def get_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def get_word_embedding(pipe, word):
    """Extract the token embedding for a single word from SD3.5's text encoder.

    与 #39 一致：用空 prompt 做差分，定位 word 引入的 token 位置。
    """
    out = pipe.encode_prompt(prompt=word, prompt_2=word, prompt_3=word, negative_prompt="")
    prompt_embeds = out[0]  # (1, L, 4096)
    out_empty = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", negative_prompt="")
    empty_embeds = out_empty[0]

    L_word, L_empty = prompt_embeds.shape[1], empty_embeds.shape[1]
    if L_word > L_empty:
        n_word_tokens = L_word - L_empty
        word_emb = prompt_embeds[0, :n_word_tokens, :].mean(dim=0)  # (4096,)
        print(f"  '{word}': {n_word_tokens} word token(s), ||emb||={word_emb.norm().item():.4f}")
    else:
        min_len = min(L_word, L_empty)
        diffs = (prompt_embeds[0, :min_len] - empty_embeds[0, :min_len]).norm(dim=1)
        idx = diffs.argmax().item()
        word_emb = prompt_embeds[0, idx, :]
        print(f"  '{word}': max-diff at token {idx}, diff={diffs[idx].item():.4f}, ||emb||={word_emb.norm().item():.4f}")
    return word_emb.float()  # 保持 float32 做重心坐标运算


# 三角形顶点的 2D 画布坐标：A 顶部，B 左下，C 右下
VA = np.array([0.5, np.sqrt(3) / 2.0])
VB = np.array([0.0, 0.0])
VC = np.array([1.0, 0.0])


def bary_to_xy(w):
    """(w_a, w_b, w_c) -> 2D 画布坐标。"""
    return w[0] * VA + w[1] * VB + w[2] * VC


def barycentric_grid(K):
    """分辨率 K 的规则三角网格，返回所有 (w_a, w_b, w_c) 且 w_i = n_i/K, sum=K。"""
    pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            k = K - i - j
            pts.append((i / K, j / K, k / K))
    return pts


def gen_images_for_weights(gen, embs3, weights, seed, batch_size=8):
    """对一批重心权重生成图。

    embs3: (3, 4096) float32 torch on device.
    weights: list of (w_a, w_b, w_c).
    返回 PIL image 列表，顺序与 weights 一致。同一 seed 隔离 z 的影响。
    """
    images = []
    for start in range(0, len(weights), batch_size):
        chunk = weights[start:start + batch_size]
        W = torch.tensor(chunk, dtype=torch.float32, device=embs3.device)  # (bs, 3)
        Z = (W @ embs3).to(dtype=torch.float16)  # (bs, 4096)
        embeds = gen.encode_batch_insert("", Z)
        imgs = gen.generate_batch(embeds, [seed] * len(chunk))
        images.extend(imgs)
        print(f"    generated {start + len(chunk)}/{len(weights)}")
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--word_a", type=str, default="leather")
    parser.add_argument("--word_b", type=str, default="neon")
    parser.add_argument("--word_c", type=str, default="canvas")
    parser.add_argument("--K", type=int, default=10, help="dense grid resolution; #points = (K+1)(K+2)/2")
    parser.add_argument("--n_line_steps", type=int, default=11, help="alpha steps for each P->vertex line")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = args.device
    A, B, C = args.word_a, args.word_b, args.word_c
    stamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = f"outputs/triangle_{A}_{B}_{C}_{stamp}"
    os.makedirs(run_dir, exist_ok=True)
    grid_dir = os.path.join(run_dir, "grid_imgs")
    os.makedirs(grid_dir, exist_ok=True)

    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=device)

    # === 1. 三个顶点 embedding + 顶点参考图 ===
    print(f"\n=== Extracting embeddings: A={A}, B={B}, C={C} ===")
    emb_a = get_word_embedding(gen.pipe, A)
    emb_b = get_word_embedding(gen.pipe, B)
    emb_c = get_word_embedding(gen.pipe, C)
    embs3 = torch.stack([emb_a, emb_b, emb_c]).to(device)  # (3, 4096)

    # 顶点两两关系
    for (n1, e1), (n2, e2) in itertools.combinations(
            [(A, emb_a), (B, emb_b), (C, emb_c)], 2):
        cos = torch.nn.functional.cosine_similarity(e1[None], e2[None]).item()
        l2 = torch.norm(e1 - e2).item()
        print(f"  {n1} vs {n2}: cos={cos:.4f}, L2={l2:.4f}")

    print("\n=== Vertex reference images ===")
    vert_imgs = gen_images_for_weights(
        gen, embs3,
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        args.seed, args.batch_size)
    vert_tensors = [scorer.preprocess(im) for im in vert_imgs]
    for name, im in zip([A, B, C], vert_imgs):
        im.save(os.path.join(run_dir, f"vertex_{name}.png"))

    def ds_to_vertices(img):
        t = scorer.preprocess(img)
        return [scorer.model(vt, t).item() for vt in vert_tensors]

    # === 2. 密集三角网格 ===
    weights = barycentric_grid(args.K)
    print(f"\n=== Dense grid: K={args.K}, {len(weights)} points ===")
    grid_imgs = gen_images_for_weights(gen, embs3, weights, args.seed, args.batch_size)

    import pandas as pd
    rows = []
    for idx, (w, img) in enumerate(zip(weights, grid_imgs)):
        ds = ds_to_vertices(img)
        nearest = int(np.argmin(ds))  # 0=A,1=B,2=C
        fname = f"g{idx:03d}_a{w[0]:.2f}_b{w[1]:.2f}_c{w[2]:.2f}.png"
        img.save(os.path.join(grid_dir, fname))
        rows.append({
            "idx": idx, "w_a": w[0], "w_b": w[1], "w_c": w[2],
            "ds_to_A": ds[0], "ds_to_B": ds[1], "ds_to_C": ds[2],
            "nearest_vertex": [A, B, C][nearest],
            "argmax_weight": [A, B, C][int(np.argmax(w))],
        })
    df_grid = pd.DataFrame(rows)
    df_grid["semantic_aligned"] = df_grid["nearest_vertex"] == df_grid["argmax_weight"]
    df_grid.to_csv(os.path.join(run_dir, "grid.csv"), index=False)
    align_rate = df_grid["semantic_aligned"].mean()
    print(f"  semantic alignment (nearest vertex == argmax weight): {align_rate:.3f}")

    # --- 三角缩略图 collage ---
    S = 1200
    thumb = max(40, int(S / (args.K + 2)))
    margin = thumb
    canvas_w = S + 2 * margin
    canvas_h = int(S * np.sqrt(3) / 2) + 2 * margin
    collage = Image.new("RGB", (canvas_w, canvas_h), "white")
    for w, img in zip(weights, grid_imgs):
        xy = bary_to_xy(w)
        px = int(margin + xy[0] * S)
        py = int(margin + (np.sqrt(3) / 2 - xy[1]) * S)  # 翻转 y，让 A 在上
        th = img.resize((thumb, thumb), Image.LANCZOS)
        collage.paste(th, (px - thumb // 2, py - thumb // 2))
    draw = ImageDraw.Draw(collage)
    font = get_font(22)
    for name, vw in [(A, (1, 0, 0)), (B, (0, 1, 0)), (C, (0, 0, 1))]:
        xy = bary_to_xy(vw)
        px = int(margin + xy[0] * S)
        py = int(margin + (np.sqrt(3) / 2 - xy[1]) * S)
        draw.text((px, py), name, fill="red", font=font)
    collage.save(os.path.join(run_dir, "grid_collage.png"))

    # === 3. P -> 三顶点 折线图 ===
    P = embs3.mean(0)  # 重心 = (e_a+e_b+e_c)/3
    alphas = np.linspace(0, 1, args.n_line_steps)
    line_data = {}  # vertex_name -> dict
    for vname, vemb in [(A, emb_a.to(device)), (B, emb_b.to(device)), (C, emb_c.to(device))]:
        print(f"\n=== Line P -> {vname} ===")
        # 这条线不在重心坐标网格上，直接在 4096 维做一维插值
        Zs = [((1 - a) * P + a * vemb) for a in alphas]
        imgs = []
        for start in range(0, len(Zs), args.batch_size):
            chunk = Zs[start:start + args.batch_size]
            Zb = torch.stack(chunk).to(dtype=torch.float16)
            embeds = gen.encode_batch_insert("", Zb)
            imgs.extend(gen.generate_batch(embeds, [args.seed] * len(chunk)))
        ds_mat = np.array([ds_to_vertices(im) for im in imgs])  # (n, 3)
        line_data[vname] = ds_mat
        for im in imgs:
            im.close()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = {A: "tab:blue", B: "tab:orange", C: "tab:green"}
    for ax, vname in zip(axes, [A, B, C]):
        ds_mat = line_data[vname]
        for ci, cname in enumerate([A, B, C]):
            ax.plot(alphas, ds_mat[:, ci], marker="o", markersize=4,
                    color=colors[cname], label=f"DS to {cname}")
        ax.set_xlabel(f"alpha  (0 = centroid P, 1 = {vname})")
        ax.set_title(f"P -> {vname}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("DreamSim distance")
    fig.suptitle(f"Centroid->vertex lines: {A}/{B}/{C} (seed={args.seed})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "centroid_lines.png"), dpi=150)

    # --- ternary 语义分区散点 ---
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    cmap = {A: "tab:blue", B: "tab:orange", C: "tab:green"}
    for _, r in df_grid.iterrows():
        xy = bary_to_xy((r["w_a"], r["w_b"], r["w_c"]))
        ax2.scatter(xy[0], xy[1], color=cmap[r["nearest_vertex"]],
                    s=60, edgecolors="white", linewidths=0.5)
    tri = np.array([VA, VB, VC, VA])
    ax2.plot(tri[:, 0], tri[:, 1], "k-", lw=1)
    for name, vw in [(A, VA), (B, VB), (C, VC)]:
        ax2.text(vw[0], vw[1], f"  {name}", fontsize=12, fontweight="bold")
    ax2.set_title(f"Semantic partition (color = nearest vertex)\nalignment={align_rate:.3f}")
    ax2.axis("equal"); ax2.axis("off")
    fig2.tight_layout()
    fig2.savefig(os.path.join(run_dir, "ternary_partition.png"), dpi=150)

    for im in grid_imgs + vert_imgs:
        im.close()

    print(f"\nDone. Results in {run_dir}/")
    print("  - grid_collage.png       (三角缩略图，肉眼看哪里崩)")
    print("  - ternary_partition.png  (语义归属分区)")
    print("  - centroid_lines.png     (P->A/B/C 三张折线)")
    print("  - grid.csv               (每个网格点的 DreamSim + 对齐标记)")


if __name__ == "__main__":
    main()
