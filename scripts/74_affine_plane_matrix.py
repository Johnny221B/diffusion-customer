"""Render a publication-ready affine-plane matrix around a ternary interpolation.

The plane is parameterized as

    z(u, v) = e_B + u (e_C - e_B) + v (e_A - e_B),

so B=(0,0), C=(1,0), and A=(0,1).  Points satisfying u>=0, v>=0,
u+v<=1 are convex combinations inside the original triangle.  Existing images
from script 65 are reused whenever a matrix point lies on its K=10 grid.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator  # noqa: E402


def font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else
             ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"])
    for name in names:
        path = Path("/usr/share/fonts/truetype/dejavu") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def word_embedding(pipe, word):
    out = pipe.encode_prompt(prompt=word, prompt_2=word, prompt_3=word,
                             negative_prompt="")[0]
    empty = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="",
                               negative_prompt="")[0]
    if out.shape[1] > empty.shape[1]:
        return out[0, :out.shape[1] - empty.shape[1]].mean(dim=0).float()
    length = min(out.shape[1], empty.shape[1])
    idx = (out[0, :length] - empty[0, :length]).norm(dim=1).argmax().item()
    return out[0, idx].float()


def load_original_grid(run_dir):
    lookup = {}
    with (run_dir / "grid.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            idx = int(row["idx"])
            wa, wb, wc = (float(row[k]) for k in ("w_a", "w_b", "w_c"))
            name = f"g{idx:03d}_a{wa:.2f}_b{wb:.2f}_c{wc:.2f}.png"
            lookup[(round(wa, 6), round(wb, 6), round(wc, 6))] = run_dir / "grid_imgs" / name
    return lookup


def centered_text(draw, xy, text, text_font, fill=(25, 25, 25)):
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2,
               xy[1] - (box[3] - box[1]) / 2), text,
              font=text_font, fill=fill)


def render_matrix(records, values, words, output_dir, thumb=300):
    """Render without text over any generated image."""
    n = len(values)
    left, right, top, bottom = 210, 90, 190, 210
    width, height = left + n * thumb + right, top + n * thumb + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    by_uv = {(round(r["u"], 6), round(r["v"], 6)): r for r in records}
    for row, v in enumerate(reversed(values)):
        for col, u in enumerate(values):
            rec = by_uv[(round(u, 6), round(v, 6))]
            with Image.open(rec["path"]) as source:
                tile = source.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
            x, y = left + col * thumb, top + row * thumb
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + thumb, y + thumb), outline=(225, 225, 225), width=2)

    # Triangle vertices in matrix coordinates; line is the only overlay.
    def point(u, v):
        col = (u - values[0]) / (values[1] - values[0])
        row = (values[-1] - v) / (values[1] - values[0])
        return (int(left + (col + 0.5) * thumb), int(top + (row + 0.5) * thumb))

    a, b, c = point(0, 1), point(0, 0), point(1, 0)
    draw.line([a, b, c, a], fill=(35, 35, 35), width=4, joint="curve")
    vertex_colors = ((45, 75, 125), (190, 105, 25), (45, 125, 70))
    for p, color in zip((a, b, c), vertex_colors):
        r = 11
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=color)

    tick_font, axis_font, label_font = font(42), font(48), font(48, bold=True)
    for col, u in enumerate(values):
        centered_text(draw, (left + (col + .5) * thumb, top + n * thumb + 52),
                      f"{u:.1f}", tick_font)
    for row, v in enumerate(reversed(values)):
        centered_text(draw, (left - 62, top + (row + .5) * thumb), f"{v:.1f}", tick_font)
    centered_text(draw, (left + n * thumb / 2, height - 55),
                  f"u  ({words[1]} → {words[2]})", axis_font)
    # Rotated y label in its own margin.
    y_label = Image.new("RGBA", (1100, 90), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    centered_text(yd, (550, 45), f"v  ({words[1]} → {words[0]})", axis_font)
    y_label = y_label.rotate(90, expand=True)
    canvas.paste(y_label, (12, int((height - y_label.height) / 2)), y_label)

    # Vertex names live in a legend above the matrix, never over generated images.
    legend_y = 62
    legend_xs = (left + n * thumb * .25,
                 left + n * thumb * .50,
                 left + n * thumb * .75)
    for letter, name, color, lx in zip(("A", "B", "C"), words,
                                       vertex_colors, legend_xs):
        r = 11
        draw.ellipse((lx - 145 - r, legend_y - r,
                      lx - 145 + r, legend_y + r), fill=color)
        centered_text(draw, (lx, legend_y), f"{letter}: {name}", label_font)

    png = output_dir / "affine_plane_matrix_publication.png"
    pdf = output_dir / "affine_plane_matrix_publication.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, "PDF", resolution=300.0)
    return png, pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--orig_run_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--word_a", default="leather")
    parser.add_argument("--word_b", default="denim")
    parser.add_argument("--word_c", default="suede")
    parser.add_argument("--lo", type=float, default=-0.2)
    parser.add_argument("--hi", type=float, default=1.2)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    orig = Path(args.orig_run_dir).resolve()
    output = orig / "publication_affine_plane"
    image_dir = output / "matrix_imgs"
    image_dir.mkdir(parents=True, exist_ok=True)
    existing = load_original_grid(orig)
    values = np.round(np.arange(args.lo, args.hi + args.step / 2, args.step), 6).tolist()

    records, missing = [], []
    for v in values:
        for u in values:
            wa, wb, wc = v, 1.0 - u - v, u
            key = tuple(round(x, 6) for x in (wa, wb, wc))
            target = image_dir / f"u{u:+.1f}_v{v:+.1f}.png"
            source = existing.get(key)
            if source and source.is_file():
                path, origin = source, "reused"
            elif target.is_file():
                path, origin = target, "generated"
            else:
                path, origin = target, "generated"
                missing.append((u, v, (wa, wb, wc), target))
            records.append({"u": u, "v": v, "w_a": wa, "w_b": wb, "w_c": wc,
                            "path": str(path), "origin": origin})

    if missing:
        print(f"Generating {len(missing)} new affine-plane images; reusing {len(records)-len(missing)}.")
        gen = SD35BatchEmbeddingGenerator(args.model_path, device=args.device)
        embeddings = torch.stack([
            word_embedding(gen.pipe, args.word_a),
            word_embedding(gen.pipe, args.word_b),
            word_embedding(gen.pipe, args.word_c),
        ]).to(args.device)
        for start in range(0, len(missing), args.batch_size):
            chunk = missing[start:start + args.batch_size]
            weights = torch.tensor([x[2] for x in chunk], dtype=torch.float32,
                                   device=args.device)
            z = (weights @ embeddings).to(torch.float16)
            prompt_embeds = gen.encode_batch_insert("", z)
            images = gen.generate_batch(prompt_embeds, [args.seed] * len(chunk))
            for image, item in zip(images, chunk):
                image.save(item[3])
                image.close()
            print(f"  {start + len(chunk)}/{len(missing)}")

    with (output / "affine_plane_grid.json").open("w") as handle:
        json.dump([{k: v for k, v in r.items() if k != "path"} |
                   {"file": os.path.relpath(r["path"], output)} for r in records],
                  handle, indent=2)
    png, pdf = render_matrix(records, values,
                             (args.word_a, args.word_b, args.word_c), output)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
