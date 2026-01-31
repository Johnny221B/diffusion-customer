import os
import json
import numpy as np
from PIL import Image

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from src.sd35_controlnet_generator import SD35ControlNetGenerator
from src.edge_utils import make_canny_edge
from src.scorer import ToyRedScorer

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


BASE_DIR = "/home/wan/guanting's/diffusion-customer/model/stabilityai/stable-diffusion-3.5-large"
CN_DIR   = "/home/wan/guanting's/diffusion-customer/model/controlnets/sd35_large_controlnet_canny"
OUT_DIR  = "/home/wan/guanting's/diffusion-customer/outputs/bo_run"
os.makedirs(OUT_DIR, exist_ok=True)

REF_IMG_PATH = "/home/wan/guanting's/diffusion-customer/assets/ref.png"  # 改成你自己的参考鞋图路径


def ucb(mu: np.ndarray, sigma: np.ndarray, beta: float) -> np.ndarray:
    return mu + beta * sigma


def main():
    prompt = "product photo of a modern sneaker, studio lighting, white background, high detail"
    beta = 1.5                 # UCB 探索强度
    rounds = 5                 # 迭代轮数
    candidate_pool = 64        # 每轮评估多少候选 seed（越大越“像”连续优化，但越慢）

    # 1) 初始化 generator / control image / scorer
    gen = SD35ControlNetGenerator(BASE_DIR, CN_DIR)
    ref = Image.open(REF_IMG_PATH)
    edge = make_canny_edge(ref, low=100, high=200)
    scorer = ToyRedScorer()

    # 2) GP：用 seed 作为 1D 输入（形状 (n,1)）
    #    kernel: 常数*RBF + 白噪声
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    # 3) 初始随机探索：先随机跑几张
    rng = np.random.default_rng(0)
    X = []
    y = []

    init_seeds = rng.integers(low=0, high=2**31-1, size=4, dtype=np.int64)

    log = []
    best = {"score": -1e9, "seed": None, "img_path": None}

    def run_one(seed: int, step: int, tag: str):
        nonlocal best
        img = gen.generate(prompt=prompt, control_image=edge, seed=int(seed))
        img_path = os.path.join(OUT_DIR, f"{step:03d}_{tag}_seed{seed}.png")
        img.save(img_path)
        score = float(scorer(img))

        rec = {"step": step, "tag": tag, "seed": int(seed), "score": score, "img_path": img_path}
        log.append(rec)

        if score > best["score"]:
            best = {"score": score, "seed": int(seed), "img_path": img_path}
        print(f"[{step:03d}] {tag} seed={seed} score={score:.6f} best={best['score']:.6f}")
        return score, img_path

    step = 0
    for s in init_seeds:
        score, _ = run_one(int(s), step, "init")
        X.append([float(s)])
        y.append(score)
        step += 1

    # 4) BO 迭代
    for t in range(rounds):
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)

        gp.fit(X_np, y_np)

        # 候选 seed 池：每轮随机采样一批
        cand_seeds = rng.integers(low=0, high=2**31-1, size=candidate_pool, dtype=np.int64)
        X_cand = cand_seeds.astype(np.float64).reshape(-1, 1)

        mu, std = gp.predict(X_cand, return_std=True)
        acq = ucb(mu, std, beta=beta)

        best_idx = int(np.argmax(acq))
        seed_next = int(cand_seeds[best_idx])

        score_next, _ = run_one(seed_next, step, "bo")
        X.append([float(seed_next)])
        y.append(score_next)
        step += 1

        # 可选：把每轮的信息写盘
        with open(os.path.join(OUT_DIR, "log.json"), "w") as f:
            json.dump({"log": log, "best": best}, f, indent=2)

    print("\nDONE. Best:", best)


if __name__ == "__main__":
    main()
