"""
Triangle halo: 在 65_triangle_interp 同一仿射平面内，往三角形外部扩散。

设计 (见对话 2026-05-27):
  - 仿射平面 = span(e_A, e_B, e_C)，仍保持 w_a + w_b + w_c = 1，但允许 w_i < 0。
  - 对每个顶点 V_i，在 canvas 2D 里做极坐标扇形：
      r ∈ R_LIST （默认 0.10, 0.20, 0.35, 0.55，单位 = 三角形边长 1）
      θ ∈ THETA_LIST（默认 -90°..+90° 步长 30°，0° = 外向法线）
    (x, y) = V_i + r·(cos θ · n_i + sin θ · t_i)
  - (x, y) → barycentric (w_a, w_b, w_c)（一两个为负）→
    Z = w_a e_A + w_b e_B + w_c e_C → SD3.5 生成。
  - 复用 65 的旧 run：内圈 K=10 grid_imgs + vertex_*.png + grid.csv 不重跑。

产出：
  - halo_imgs/h_v<X>_r<rr>_t<tt>.png      （生成的新图）
  - halo_grid.csv                          （每点的 w/x/y/r/θ + DS-to-A/B/C）
  - composite_collage.png                  （内圈 thumb + 三个 halo 扇形 + 三角形轮廓）
  - radial_ds.png                          （3 panel × DS-to-self vs r，按角度上色）
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator  # noqa: E402
from src.scorer import DreamSimScorer  # noqa: E402


# === 三角形顶点（canvas 坐标，与 65_triangle_interp.py 完全一致） ===
VA = np.array([0.5, np.sqrt(3) / 2.0])
VB = np.array([0.0, 0.0])
VC = np.array([1.0, 0.0])
CENTROID = (VA + VB + VC) / 3.0


def get_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def get_word_embedding(pipe, word):
    """与 65_triangle_interp.py 完全一致的差分抽取。"""
    out = pipe.encode_prompt(prompt=word, prompt_2=word, prompt_3=word, negative_prompt="")
    prompt_embeds = out[0]
    out_empty = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", negative_prompt="")
    empty_embeds = out_empty[0]
    L_word, L_empty = prompt_embeds.shape[1], empty_embeds.shape[1]
    if L_word > L_empty:
        n_word_tokens = L_word - L_empty
        word_emb = prompt_embeds[0, :n_word_tokens, :].mean(dim=0)
        print(f"  '{word}': {n_word_tokens} word token(s), ||emb||={word_emb.norm().item():.4f}")
    else:
        min_len = min(L_word, L_empty)
        diffs = (prompt_embeds[0, :min_len] - empty_embeds[0, :min_len]).norm(dim=1)
        idx = diffs.argmax().item()
        word_emb = prompt_embeds[0, idx, :]
        print(f"  '{word}': max-diff at token {idx}, diff={diffs[idx].item():.4f}, ||emb||={word_emb.norm().item():.4f}")
    return word_emb.float()


def cart_to_bary(p):
    """(x, y) -> (w_a, w_b, w_c)，sum=1, 允许负值。"""
    # 解 w_a*VA + w_b*VB + w_c*VC = p, w_a+w_b+w_c=1
    # 用 VC 做参考：[VA-VC, VB-VC] @ [w_a; w_b] = p - VC
    M = np.column_stack([VA - VC, VB - VC])  # 2x2
    w_ab = np.linalg.solve(M, p - VC)
    w_a, w_b = w_ab
    w_c = 1.0 - w_a - w_b
    return np.array([w_a, w_b, w_c])


def vertex_frame(vert):
    """给一个顶点返回 (n_outward, t_tangent_CCW)。"""
    n = vert - CENTROID
    n = n / np.linalg.norm(n)
    t = np.array([-n[1], n[0]])  # 90° CCW
    return n, t


def build_halo_points(R_list, THETA_list):
    """对 (A,B,C) 三个顶点扫扇形。返回 list of dict。"""
    pts = []
    for vname, V in [("A", VA), ("B", VB), ("C", VC)]:
        n, t = vertex_frame(V)
        for r in R_list:
            for th_deg in THETA_list:
                th = np.deg2rad(th_deg)
                p = V + r * (np.cos(th) * n + np.sin(th) * t)
                bary = cart_to_bary(p)
                pts.append({
                    "vertex": vname, "r": r, "theta_deg": th_deg,
                    "x": p[0], "y": p[1],
                    "w_a": bary[0], "w_b": bary[1], "w_c": bary[2],
                })
    return pts


def gen_images_for_weights(gen, embs3, weights, seed, batch_size=8):
    images = []
    for start in range(0, len(weights), batch_size):
        chunk = weights[start:start + batch_size]
        W = torch.tensor(chunk, dtype=torch.float32, device=embs3.device)
        Z = (W @ embs3).to(dtype=torch.float16)
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
    parser.add_argument("--orig_run_dir", type=str,
                        default="outputs/triangle_leather_neon_canvas_0521_0252",
                        help="复用此 dir 的 grid_imgs/ + vertex_*.png + grid.csv 当内圈参照")
    parser.add_argument("--word_a", type=str, default="leather")
    parser.add_argument("--word_b", type=str, default="neon")
    parser.add_argument("--word_c", type=str, default="canvas")
    parser.add_argument("--r_list", type=float, nargs="+",
                        default=[0.10, 0.20, 0.35, 0.55])
    parser.add_argument("--theta_list", type=float, nargs="+",
                        default=[-90, -60, -30, 0, 30, 60, 90])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    A, B, C = args.word_a, args.word_b, args.word_c
    stamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = f"outputs/triangle_halo_{A}_{B}_{C}_{stamp}"
    os.makedirs(run_dir, exist_ok=True)
    halo_dir = os.path.join(run_dir, "halo_imgs")
    os.makedirs(halo_dir, exist_ok=True)

    # === 0. 检查 orig run dir ===
    if not os.path.isdir(args.orig_run_dir):
        raise FileNotFoundError(f"orig_run_dir not found: {args.orig_run_dir}")
    orig_grid_csv = os.path.join(args.orig_run_dir, "grid.csv")
    orig_grid_dir = os.path.join(args.orig_run_dir, "grid_imgs")
    if not (os.path.isfile(orig_grid_csv) and os.path.isdir(orig_grid_dir)):
        raise FileNotFoundError(f"orig_run_dir missing grid.csv or grid_imgs/: {args.orig_run_dir}")
    df_orig = pd.read_csv(orig_grid_csv)
    print(f"Reusing orig run: {args.orig_run_dir}  ({len(df_orig)} internal grid points)")

    # === 1. 加载 SD3.5 + DreamSim ===
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=args.device)
    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=args.device)

    # === 2. 重新抽 3 个顶点 embedding（确定性，无需保存）===
    print(f"\n=== Re-extracting embeddings: A={A}, B={B}, C={C} ===")
    emb_a = get_word_embedding(gen.pipe, A)
    emb_b = get_word_embedding(gen.pipe, B)
    emb_c = get_word_embedding(gen.pipe, C)
    embs3 = torch.stack([emb_a, emb_b, emb_c]).to(args.device)  # (3, 4096)

    # 顶点参考图直接从 orig run 读
    vert_imgs = {}
    for vname, wname in [("A", A), ("B", B), ("C", C)]:
        p = os.path.join(args.orig_run_dir, f"vertex_{wname}.png")
        vert_imgs[vname] = Image.open(p).convert("RGB")
    vert_tensors = {k: scorer.preprocess(im) for k, im in vert_imgs.items()}

    def ds_to_vertices(img):
        t = scorer.preprocess(img)
        return [scorer.model(vert_tensors[k], t).item() for k in ("A", "B", "C")]

    # === 3. 生成 halo 图 ===
    pts = build_halo_points(args.r_list, args.theta_list)
    print(f"\n=== Halo: {len(pts)} points "
          f"({len(args.r_list)} radii × {len(args.theta_list)} angles × 3 vertices) ===")
    weights = [(p["w_a"], p["w_b"], p["w_c"]) for p in pts]
    halo_imgs = gen_images_for_weights(gen, embs3, weights,
                                       args.seed, args.batch_size)

    rows = []
    for i, (p, img) in enumerate(zip(pts, halo_imgs)):
        ds = ds_to_vertices(img)
        fname = f"h_v{p['vertex']}_r{p['r']:.2f}_t{int(p['theta_deg']):+04d}.png"
        img.save(os.path.join(halo_dir, fname))
        rows.append({
            **p,
            "fname": fname,
            "ds_to_A": ds[0], "ds_to_B": ds[1], "ds_to_C": ds[2],
        })
    df_halo = pd.DataFrame(rows)
    df_halo.to_csv(os.path.join(run_dir, "halo_grid.csv"), index=False)

    # === 4. Composite collage（PIL）===
    # bbox: x ∈ [-0.55, 1.55], y ∈ [-0.55, 1.42]
    all_x = np.concatenate([df_halo["x"].values, [0, 1]])
    all_y = np.concatenate([df_halo["y"].values, [0, np.sqrt(3) / 2]])
    pad = 0.1
    x_lo, x_hi = float(all_x.min()) - pad, float(all_x.max()) + pad
    y_lo, y_hi = float(all_y.min()) - pad, float(all_y.max()) + pad
    W_units = x_hi - x_lo
    H_units = y_hi - y_lo

    PX_PER_UNIT = 720
    thumb = 70
    canvas_w = int(W_units * PX_PER_UNIT)
    canvas_h = int(H_units * PX_PER_UNIT)
    collage = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(collage)

    def to_px(x, y):
        px = int((x - x_lo) * PX_PER_UNIT)
        py = int((y_hi - y) * PX_PER_UNIT)  # 翻转 y
        return px, py

    # 画三角形轮廓
    tri_pts = [to_px(*VA), to_px(*VB), to_px(*VC), to_px(*VA)]
    draw.line(tri_pts, fill=(60, 60, 60), width=3)

    # 内圈 thumbs（来自 orig run）
    for _, r in df_orig.iterrows():
        fname = f"g{int(r['idx']):03d}_a{r['w_a']:.2f}_b{r['w_b']:.2f}_c{r['w_c']:.2f}.png"
        path = os.path.join(orig_grid_dir, fname)
        if not os.path.isfile(path):
            continue
        im = Image.open(path).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        x = r["w_a"] * VA[0] + r["w_b"] * VB[0] + r["w_c"] * VC[0]
        y = r["w_a"] * VA[1] + r["w_b"] * VB[1] + r["w_c"] * VC[1]
        px, py = to_px(x, y)
        collage.paste(im, (px - thumb // 2, py - thumb // 2))

    # halo thumbs（带细色框 = 所属顶点）
    border_col = {"A": (30, 80, 200), "B": (220, 120, 30), "C": (40, 160, 60)}
    for _, r in df_halo.iterrows():
        path = os.path.join(halo_dir, r["fname"])
        im = Image.open(path).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        px, py = to_px(r["x"], r["y"])
        # 用一个临时画布加细框
        framed = Image.new("RGB", (thumb + 4, thumb + 4), border_col[r["vertex"]])
        framed.paste(im, (2, 2))
        collage.paste(framed, (px - (thumb + 4) // 2, py - (thumb + 4) // 2))

    # 顶点标注
    font = get_font(34)
    for vname, wname, V in [("A", A, VA), ("B", B, VB), ("C", C, VC)]:
        px, py = to_px(*V)
        draw.text((px + 18, py - 18), f"{vname}={wname}", fill=border_col[vname], font=font)

    collage.save(os.path.join(run_dir, "composite_collage.png"))
    print(f"  composite_collage.png  ({canvas_w}×{canvas_h})")

    # === 5. 径向 DS 曲线 ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    vname_to_word = {"A": A, "B": B, "C": C}
    vname_to_dscol = {"A": "ds_to_A", "B": "ds_to_B", "C": "ds_to_C"}
    n_th = len(args.theta_list)
    cmap = cm.get_cmap("viridis", n_th)

    for ax, vname in zip(axes, ["A", "B", "C"]):
        sub = df_halo[df_halo["vertex"] == vname]
        self_col = vname_to_dscol[vname]
        for i, th in enumerate(sorted(set(sub["theta_deg"]))):
            line = sub[sub["theta_deg"] == th].sort_values("r")
            # 在 r=0 处串上顶点自身的 DS（=0）作为锚
            rs = np.concatenate([[0.0], line["r"].values])
            ds = np.concatenate([[0.0], line[self_col].values])
            ax.plot(rs, ds, marker="o", markersize=4,
                    color=cmap(i), label=f"θ={int(th):+d}°")
        ax.set_xlabel("radius r  (from vertex, in triangle-side units)")
        ax.set_title(f"V_{vname} = {vname_to_word[vname]}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("DreamSim distance to own vertex image")
    fig.suptitle(f"Radial halo: DS-to-self vs r (seed={args.seed})  "
                 f"[{A} / {B} / {C}]", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "radial_ds.png"), dpi=150)
    plt.close(fig)
    print(f"  radial_ds.png")

    for im in halo_imgs:
        im.close()
    for im in vert_imgs.values():
        im.close()

    print(f"\nDone. Results in {run_dir}/")


if __name__ == "__main__":
    main()
