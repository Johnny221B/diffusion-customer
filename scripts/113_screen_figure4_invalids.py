"""Generate a contact sheet of farther off-support Figure 4 candidates."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def get_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="outputs/paper_figures/figure4/figure4_geometry.npz")
    ap.add_argument("--model_path", default="models/stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--output_dir", default="outputs/paper_figures/figure4/invalid_screen")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1810772)
    args = ap.parse_args()

    geom = np.load(args.geometry, allow_pickle=True)
    Z, tau = geom["Z"], float(geom["tau"])
    components, mean = geom["pca_components"], geom["pca_mean"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    samples = []
    for i, angle in enumerate(np.arange(0, 360, 30)):
        target_ratio = 2.2 if i % 2 == 0 else 2.8
        direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        radii = np.linspace(0, 130, 2601)
        points2 = radii[:, None] * direction[None]
        points16 = np.zeros((len(radii), Z.shape[1]))
        points16[:, :2] = points2
        distances = cdist(points16, Z)
        dk = np.partition(distances, 9, axis=1)[:, 9]
        idx = int(np.argmin(np.abs(dk / tau - target_ratio)))
        z = points16[idx]
        samples.append({"id": f"C{i+1:02d}", "angle": int(angle),
                        "target_ratio": target_ratio, "ratio": float(dk[idx] / tau),
                        "pc1": float(z[0]), "pc2": float(z[1]), "z": z})

    missing = [s for s in samples if not (output / f"{s['id']}.png").is_file()]
    if missing:
        import torch
        from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
        gen = SD35BatchEmbeddingGenerator(args.model_path, device=args.device)
        latent = np.stack([s["z"] for s in missing])
        lifted = latent @ components + mean
        for start in range(0, len(missing), 4):
            chunk = missing[start:start + 4]
            tensor = torch.tensor(lifted[start:start + 4], dtype=torch.float16,
                                  device=args.device)
            encoded = gen.encode_batch_insert("", tensor)
            images = gen.generate_batch(encoded, [args.seed] * len(chunk))
            for sample, image in zip(chunk, images):
                image.save(output / f"{sample['id']}.png")
                image.close()

    tile, label_h, cols, rows = 260, 62, 4, 3
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    regular, bold = get_font(22), get_font(25, True)
    for i, sample in enumerate(samples):
        col, row = i % cols, i // cols
        x, y = col * tile, row * (tile + label_h)
        with Image.open(output / f"{sample['id']}.png") as source:
            image = source.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(205, 205, 205), width=2)
        draw.text((x + 8, y + tile + 3), sample["id"], font=bold, fill=(35, 35, 35))
        draw.text((x + 62, y + tile + 6),
                  f"angle={sample['angle']}°   d10/τ={sample['ratio']:.2f}",
                  font=regular, fill=(70, 70, 70))
    sheet.save(output / "invalid_candidates_contact_sheet.png", dpi=(300, 300))
    with (output / "invalid_candidates.json").open("w") as handle:
        json.dump([{k: v for k, v in s.items() if k != "z"} for s in samples], handle, indent=2)
    print(output / "invalid_candidates_contact_sheet.png")


if __name__ == "__main__":
    main()
