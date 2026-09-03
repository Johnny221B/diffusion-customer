"""Re-layout an existing triangle-halo run as non-overlapping paper panels."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def get_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size)


def center(draw, xy, text, font, fill=(25, 25, 25)):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2,
               xy[1] - (box[3] - box[1]) / 2), text, font=font, fill=fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halo_run_dir", required=True)
    ap.add_argument("--triangle_run_dir", required=True)
    ap.add_argument("--word_a", default="leather")
    ap.add_argument("--word_b", default="neon")
    ap.add_argument("--word_c", default="canvas")
    ap.add_argument("--tile", type=int, default=150)
    args = ap.parse_args()

    halo = Path(args.halo_run_dir).resolve()
    triangle = Path(args.triangle_run_dir).resolve()
    output = halo / "publication"
    output.mkdir(exist_ok=True)

    with (halo / "halo_grid.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    radii = sorted({float(row["r"]) for row in rows})
    angles = sorted({float(row["theta_deg"]) for row in rows})
    words = {"A": args.word_a, "B": args.word_b, "C": args.word_c}
    colors = {"A": (45, 75, 125), "B": (190, 105, 25), "C": (45, 125, 70)}
    lookup = {(r["vertex"], float(r["r"]), float(r["theta_deg"])): r for r in rows}

    tile = args.tile
    cols, image_rows = len(angles), len(radii) + 1
    left, right, top, bottom = 145, 35, 150, 125
    gap = 90
    panel_w = left + cols * tile + right
    width = 3 * panel_w + 2 * gap
    height = top + image_rows * tile + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(34, True)
    axis_font = get_font(27)
    tick_font = get_font(23)

    for pi, vertex in enumerate(("A", "B", "C")):
        x0 = pi * (panel_w + gap)
        grid_x = x0 + left
        grid_y = top
        color = colors[vertex]

        center(draw, (grid_x + cols * tile / 2, 48),
               f"{vertex}: {words[vertex]}", title_font, color)
        center(draw, (grid_x + cols * tile / 2, 93),
               "direction from vertex, θ", axis_font)

        # Vertex image appears once, centered in the r=0 row.
        vertex_path = triangle / f"vertex_{words[vertex]}.png"
        with Image.open(vertex_path) as source:
            image = source.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        middle_col = cols // 2
        vx, vy = grid_x + middle_col * tile, grid_y
        canvas.paste(image, (vx, vy))
        draw.rectangle((vx, vy, vx + tile, vy + tile), outline=color, width=5)
        center(draw, (x0 + left - 62, vy + tile / 2), "vertex", tick_font, color)

        for col, angle in enumerate(angles):
            center(draw, (grid_x + (col + .5) * tile, grid_y - 25),
                   f"{angle:+.0f}°", tick_font)
        for ri, radius in enumerate(radii, start=1):
            y = grid_y + ri * tile
            center(draw, (x0 + left - 62, y + tile / 2), f"{radius:.2f}", tick_font)
            for col, angle in enumerate(angles):
                row = lookup[(vertex, radius, angle)]
                path = halo / "halo_imgs" / row["fname"]
                with Image.open(path) as source:
                    image = source.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                x = grid_x + col * tile
                canvas.paste(image, (x, y))
                draw.rectangle((x, y, x + tile, y + tile), outline=(225, 225, 225), width=2)

        # A compact symbol avoids colliding with the numeric radius labels.
        center(draw, (x0 + 20, grid_y + image_rows * tile / 2),
               "r", title_font)
        # Separator lies only in whitespace between panels.
        if pi < 2:
            sx = x0 + panel_w + gap // 2
            draw.line((sx, 25, sx, height - 25), fill=(210, 210, 210), width=2)

    png = output / "vertex_extrapolation_matrices.png"
    pdf = output / "vertex_extrapolation_matrices.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, "PDF", resolution=300)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
