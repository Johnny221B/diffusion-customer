"""Render matched original/PCA pairs for a qualitative reconstruction figure."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
OUT = ROOT / 'outputs/pca_reconstruction_review'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = np.load(ROOT / 'outputs/strict_pool_s228_0429_0119/embeddings.npz', allow_pickle=True)
    embeddings = pool['embs'].astype(np.float64)
    idx = list(pool['words'].astype(str)).index('brown')
    pca = PCA(n_components=16, random_state=0).fit(embeddings)
    reconstructed = pca.inverse_transform(pca.transform(embeddings[idx:idx+1]))[0]
    generator = SD35BatchEmbeddingGenerator(str(ROOT / 'models/stabilityai/stable-diffusion-3.5-large'), device='cuda:0')
    vectors = torch.tensor(np.stack([embeddings[idx], reconstructed]), dtype=torch.float16, device='cuda:0')
    encoded = generator.encode_batch_insert('', vectors)
    for seed in [1, 13, 17, 18]:
        images = generator.generate_batch(encoded, [seed, seed])
        for name, im in zip(['normal', 'pca16'], images):
            im.save(OUT / f'brown_seed{seed:02d}_{name}.png')
        print(f'Completed seed {seed}', flush=True)
    (OUT / 'render_metadata.json').write_text(json.dumps(dict(word='brown', seeds=[1,13,17,18], pca_dim=16, pca_anchor_count=len(embeddings), steps=20, guidance_scale=5, image_size=496, same_seed_within_pair=True), indent=2))

if __name__ == '__main__':
    main()
