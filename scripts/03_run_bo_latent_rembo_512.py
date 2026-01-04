import os
import json
import numpy as np
import torch
from PIL import Image

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from src.sd35_controlnet_generator import SD35ControlNetGenerator
from src.edge_utils import make_canny_edge
from src.scorer import ToyRedScorer

BASE_DIR = "/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large"
CN_DIR   = "/home/linyuliu/jxmount/diffusion_custom/models/controlnets/sd35_large_controlnet_canny"
OUT_DIR  = "/home/linyuliu/jxmount/diffusion_custom/outputs/bo_latent_rembo_512"
REF_IMG_PATH = "/home/linyuliu/jxmount/diffusion_custom/assets/ref.png"

os.makedirs(OUT_DIR, exist_ok=True)


def ucb(mu: np.ndarray, sigma: np.ndarray, beta: float) -> np.ndarray:
    return mu + beta * sigma


def make_A_constrained(n_z: int, n_h: int, rng: np.random.Generator) -> np.ndarray:
    """
    Construct A in R^{n_z x n_h} such that A^T A = (n_z/n_h) I approximately,
    using QR on a random Gaussian matrix then scaling columns.
    """
    M = rng.standard_normal((n_z, n_h)).astype(np.float64)
    Q, _ = np.linalg.qr(M)  # Q: (n_z, n_h), orthonormal columns => Q^T Q = I
    A = Q * np.sqrt(n_z / n_h)
    return A.astype(np.float32)


def sample_h_on_sphere(n_h: int, rng: np.random.Generator) -> np.ndarray:
    """
    Typical-set friendly: sample h ~ N(0,I) then normalize to ||h|| = sqrt(n_h).
    This makes ||h||^2 ~= n_h, which implies ||z||^2 ~= n_z when A^T A=(n_z/n_h)I.
    """
    h = rng.standard_normal((n_h,)).astype(np.float32)
    norm = np.linalg.norm(h) + 1e-8
    h = h / norm * np.sqrt(n_h)
    return h


def main():
    prompt = "product photo of a modern sneaker, studio lighting, white background, high detail"
    height = width = 512

    # BO params
    n_h = 16
    beta = 1.5
    rounds = 5
    candidate_pool = 64
    init_points = 4

    rng = np.random.default_rng(0)

    # 1) generator / edge / scorer
    gen = SD35ControlNetGenerator(BASE_DIR, CN_DIR, torch_dtype=torch.float16)
    ref = Image.open(REF_IMG_PATH).convert("RGB").resize((width, height))
    edge = make_canny_edge(ref, low=100, high=200).convert("RGB")
    edge.save(os.path.join(OUT_DIR, "edge_512.png"))

    scorer = ToyRedScorer()

    # 2) infer latent shape from pipeline (robust)
    pipe = gen.pipe
    # SD3 uses VAE latent scaling; try best-effort to infer scale factor
    vae_scale = getattr(pipe, "vae_scale_factor", None) or 8
    h_lat = height // vae_scale
    w_lat = width // vae_scale

    # latent channels: SD3 pipelines typically expose transformer.config.in_channels
    C_lat = None
    if getattr(pipe, "transformer", None) is not None and getattr(pipe.transformer, "config", None) is not None:
        C_lat = getattr(pipe.transformer.config, "in_channels", None)
    if C_lat is None and getattr(pipe, "unet", None) is not None:
        C_lat = getattr(pipe.unet.config, "in_channels", None)
    if C_lat is None:
        raise RuntimeError("Cannot infer latent channels. Please inspect pipe.transformer.config.in_channels.")

    n_z = int(C_lat * h_lat * w_lat)
    print(f"[INFO] latent tensor shape = (1, {C_lat}, {h_lat}, {w_lat}), n_z={n_z}, n_h={n_h}")

    # 3) build constrained A once (offline prep)
    A = make_A_constrained(n_z=n_z, n_h=n_h, rng=rng)

    # 4) GP (fixed kernel, no hyperparam optimization)
    kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None)

    X = []  # list of h vectors
    y = []

    log = []
    best = {"score": -1e9, "img_path": None}

    def h_to_latents(h: np.ndarray) -> torch.Tensor:
        z = A @ h.astype(np.float32)  # (n_z,)
        lat = torch.from_numpy(z).reshape(1, C_lat, h_lat, w_lat)
        return lat

    def run_one(h: np.ndarray, step: int, tag: str):
        nonlocal best
        latents = h_to_latents(h)
        img = gen.generate(
            prompt=prompt,
            control_image=edge,
            latents=latents,
            height=height,
            width=width,
            controlnet_conditioning_scale=0.5,
        )
        img_path = os.path.join(OUT_DIR, f"{step:03d}_{tag}.png")
        img.save(img_path)

        score = float(scorer(img))
        rec = {"step": step, "tag": tag, "score": score, "img_path": img_path, "h": h.tolist()}
        log.append(rec)

        if score > best["score"]:
            best = {"score": score, "img_path": img_path}

        print(f"[{step:03d}] {tag} score={score:.6f} best={best['score']:.6f}")
        return score

    # 5) offline init points (random h on typical sphere)
    step = 0
    for _ in range(init_points):
        h = sample_h_on_sphere(n_h, rng)
        score = run_one(h, step, "init")
        X.append(h)
        y.append(score)
        step += 1

    # 6) online BO loop over h
    for t in range(rounds):
        X_np = np.stack(X, axis=0).astype(np.float64)  # (n, n_h)
        y_np = np.array(y, dtype=np.float64)

        gp.fit(X_np, y_np)

        # sample candidate hs
        H_cand = np.stack([sample_h_on_sphere(n_h, rng) for _ in range(candidate_pool)], axis=0).astype(np.float64)
        mu, std = gp.predict(H_cand, return_std=True)
        acq = ucb(mu, std, beta=beta)

        best_idx = int(np.argmax(acq))
        h_next = H_cand[best_idx].astype(np.float32)

        score_next = run_one(h_next, step, "bo")
        X.append(h_next)
        y.append(score_next)
        step += 1

        with open(os.path.join(OUT_DIR, "log.json"), "w") as f:
            json.dump({"log": log, "best": best}, f, indent=2)

    print("\nDONE. Best:", best)


if __name__ == "__main__":
    main()
