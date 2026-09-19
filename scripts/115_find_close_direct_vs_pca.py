#!/usr/bin/env python3
"""Render low-reconstruction-error anchors and find a close direct/PCA pair."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


POOL = PROJECT / "outputs/strict_pool_s228_0429_0119/embeddings.npz"
DIRECT_DIR = PROJECT / "outputs/multiseed_s228_M40_0510_0241/imgs"
OUT = PROJECT / "results/pub_fig/embedding_pca_comparison_close"
SEED = 34
DIM = 16
N_CANDIDATES = 12


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(POOL, allow_pickle=True)
    embs = data["embs"].astype(np.float64)
    words = data["words"].astype(str)
    pca = PCA(n_components=DIM, random_state=0).fit(embs)
    recon = pca.inverse_transform(pca.transform(embs))
    rel = np.linalg.norm(embs - recon, axis=1) / np.linalg.norm(embs, axis=1)
    candidates = np.argsort(rel)[:N_CANDIDATES]

    generator = SD35BatchEmbeddingGenerator(
        PROJECT / "models/stabilityai/stable-diffusion-3.5-large", device="cuda:0"
    )
    scorer = DreamSimScorer(device="cuda:0")
    tensor = torch.tensor(recon[candidates], dtype=torch.float16, device="cuda:0")
    encoded = generator.encode_batch_insert("", tensor)
    rendered = generator.generate_batch(encoded, [SEED] * len(candidates))

    rows = []
    for idx, pca_image in zip(candidates, rendered):
        word = words[idx]
        direct_path = DIRECT_DIR / f"{idx:03d}_{word}_seed{SEED}.png"
        direct = Image.open(direct_path).convert("RGB")
        distance = float(scorer.model(scorer.preprocess(direct), scorer.preprocess(pca_image)).item())
        pca_path = OUT / f"{idx:03d}_{word}_pca{DIM}_seed{SEED}.png"
        pca_image.save(pca_path)
        rows.append({"index": int(idx), "word": word, "embedding_relative_l2_error": float(rel[idx]),
                     "dreamsim_direct_vs_pca": distance, "direct": str(direct_path), "pca": str(pca_path)})
        direct.close()

    rows.sort(key=lambda row: row["dreamsim_direct_vs_pca"])
    (OUT / "candidate_metrics.json").write_text(json.dumps(rows, indent=2))
    best = rows[0]
    direct = Image.open(best["direct"]).convert("RGB")
    pca_image = Image.open(best["pca"]).convert("RGB")
    tile = 496
    sheet = Image.new("RGB", (tile * 2, 620), "white")
    sheet.paste(direct, (0, 72)); sheet.paste(pca_image, (tile, 72))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
    body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    draw.text((tile // 2, 30), f"(a) Direct: {best['word']}", anchor="mm", fill="black", font=font)
    draw.text((tile + tile // 2, 30), f"(b) PCA-{DIM} reconstruction", anchor="mm", fill="black", font=font)
    draw.text((tile, 590), f"DreamSim(a, b) = {best['dreamsim_direct_vs_pca']:.4f}",
              anchor="mm", fill="black", font=body)
    sheet.save(OUT / f"best_{best['word']}_direct_vs_pca{DIM}.png", dpi=(300, 300))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
