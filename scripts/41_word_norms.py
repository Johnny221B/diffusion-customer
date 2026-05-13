import torch
from src.sd35_batch_generator import SD35BatchEmbeddingGenerator

gen = SD35BatchEmbeddingGenerator('models/stabilityai/stable-diffusion-3.5-large', device='cuda:0')

def get_word_emb(pipe, word):
    out = pipe.encode_prompt(prompt=word, prompt_2=word, prompt_3=word, negative_prompt='')
    out_empty = pipe.encode_prompt(prompt='', prompt_2='', prompt_3='', negative_prompt='')
    pe = out[0]
    ee = out_empty[0]
    L_w, L_e = pe.shape[1], ee.shape[1]
    if L_w > L_e:
        n = L_w - L_e
        return pe[0, :n, :].mean(dim=0).detach()
    else:
        ml = min(L_w, L_e)
        diffs = (pe[0, :ml] - ee[0, :ml]).norm(dim=1)
        idx = diffs.argmax().item()
        return pe[0, idx, :].detach()

words = ['red', 'blue', 'green', 'black', 'white', 'gold',
         'leather', 'neon', 'fire', 'ocean', 'ice', 'metal',
         'sport', 'elegant', 'vintage', 'modern', 'chunky', 'slim',
         'boot', 'sandal', 'sneaker', 'heel', 'flat', 'running',
         'nike', 'adidas', 'puma', 'luxury', 'cheap', 'expensive']

embs = {}
print(f'{"Word":>12s} | {"Norm":>8s}')
print('-' * 25)
for w in words:
    e = get_word_emb(gen.pipe, w)
    n = e.norm().item()
    embs[w] = e
    print(f'{w:>12s} | {n:8.2f}')

print(f'\n{"=== Pairwise Cosine Similarity ==="}')
pairs = [
    ('red', 'blue'), ('red', 'green'), ('black', 'white'),
    ('leather', 'neon'), ('fire', 'ocean'), ('fire', 'ice'),
    ('sport', 'elegant'), ('vintage', 'modern'),
    ('boot', 'sandal'), ('sneaker', 'heel'), ('boot', 'sneaker'),
    ('chunky', 'slim'), ('luxury', 'cheap'),
    ('nike', 'adidas'), ('running', 'elegant'),
]
print(f'{"Pair":>25s} | {"CosSim":>8s} | {"L2 Dist":>8s}')
print('-' * 50)
for w1, w2 in pairs:
    e1, e2 = embs[w1], embs[w2]
    cs = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
    l2 = (e1 - e2).norm().item()
    print(f'{w1+" vs "+w2:>25s} | {cs:8.4f} | {l2:8.2f}')
