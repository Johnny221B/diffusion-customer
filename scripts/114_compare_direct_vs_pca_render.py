"""Compare rendering a discrete word embedding directly vs after PCA reconstruction."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator  # noqa: E402
from src.scorer import DreamSimScorer  # noqa: E402


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_npz", default="outputs/strict_pool_s228_0429_0119/embeddings.npz")
    ap.add_argument("--direct_image", default="outputs/multiseed_s228_M40_0510_0241/imgs/011_canvas_seed34.png")
    ap.add_argument("--reference", default="outputs/strict_pool_s228_0429_0119/reference.png")
    ap.add_argument("--model_path", default="models/stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--output_dir", default="results/pub_fig/embedding_pca_comparison")
    ap.add_argument("--word", default="canvas")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pool = np.load(args.pool_npz, allow_pickle=True)
    embs = pool["embs"].astype(np.float64)
    words = [str(w) for w in pool["words"]]
    idx = words.index(args.word)

    pca = PCA(n_components=args.dim, random_state=0).fit(embs)
    z = pca.transform(embs[idx:idx + 1])
    reconstructed = pca.inverse_transform(z)[0]
    original = embs[idx]
    l2 = float(np.linalg.norm(original - reconstructed))
    rel_l2 = float(l2 / np.linalg.norm(original))
    cosine = float(np.dot(original, reconstructed) /
                   (np.linalg.norm(original) * np.linalg.norm(reconstructed)))

    direct_out = output / f"a_{args.word}_direct_seed{args.seed:02d}.png"
    pca_out = output / f"b_{args.word}_pca{args.dim}_seed{args.seed:02d}.png"
    shutil.copy2(args.direct_image, direct_out)

    gen = SD35BatchEmbeddingGenerator(args.model_path, device=args.device)
    scorer = DreamSimScorer(device=args.device)
    tensor = torch.tensor(reconstructed[None], dtype=torch.float16, device=args.device)
    encoded = gen.encode_batch_insert("", tensor)
    pca_image = gen.generate_batch(encoded, [args.seed])[0]
    pca_image.save(pca_out)

    direct_image = Image.open(direct_out).convert("RGB")
    reference = Image.open(args.reference).convert("RGB")
    t_direct = scorer.preprocess(direct_image)
    t_pca = scorer.preprocess(pca_image)
    t_ref = scorer.preprocess(reference)
    image_ds = float(scorer.model(t_direct, t_pca).item())
    direct_ref_ds = float(scorer.model(t_ref, t_direct).item())
    pca_ref_ds = float(scorer.model(t_ref, t_pca).item())

    metrics = {
        "word": args.word, "seed": args.seed, "pca_dim": args.dim,
        "n_anchors": len(words),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "embedding_l2_reconstruction_error": l2,
        "embedding_relative_l2_error": rel_l2,
        "embedding_cosine_original_vs_reconstructed": cosine,
        "dreamsim_direct_vs_pca": image_ds,
        "dreamsim_direct_to_reference": direct_ref_ds,
        "dreamsim_pca_to_reference": pca_ref_ds,
    }
    with (output / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    tile, top, bottom = 496, 76, 66
    sheet = Image.new("RGB", (tile * 2, top + tile + bottom), "white")
    draw = ImageDraw.Draw(sheet)
    sheet.paste(direct_image, (0, top))
    sheet.paste(pca_image.convert("RGB"), (tile, top))
    draw.line((tile, top, tile, top + tile), fill=(215, 215, 215), width=2)
    draw.text((tile // 2, 28), "(a) Direct embedding", anchor="mm",
              fill=(25, 25, 25), font=font(27, True))
    draw.text((tile + tile // 2, 28), f"(b) PCA-{args.dim} reconstruction",
              anchor="mm", fill=(25, 25, 25), font=font(27, True))
    draw.text((tile, top + tile + 30),
              f"DreamSim(a, b) = {image_ds:.4f}", anchor="mm",
              fill=(40, 40, 40), font=font(24))
    sheet.save(output / f"{args.word}_direct_vs_pca{args.dim}.png", dpi=(300, 300))

    direct_image.close(); reference.close(); pca_image.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
