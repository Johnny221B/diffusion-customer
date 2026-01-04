# scripts/04_run_bo_latent_rembo_continuous_512.py
import os
import json
import numpy as np
import torch
from PIL import Image

from src.sd35_controlnet_generator import SD35ControlNetGenerator
from src.edge_utils import make_canny_edge
from src.scorer import CLIPScorer

# -------------------------
# Paths / Config
# -------------------------
BASE_DIR = "/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large"
CN_DIR   = "/home/linyuliu/jxmount/diffusion_custom/models/controlnets/sd35_large_controlnet_canny"
REF_IMG_PATH = "/home/linyuliu/jxmount/diffusion_custom/assets/ref.png"
OUT_DIR  = "/home/linyuliu/jxmount/diffusion_custom/outputs/bo_latent_rembo_continuous_512_clip_hp"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = "product photo of a modern sneaker, studio lighting, white background, high detail"
NEG    = "lowres, blurry, worst quality, artifacts"

H = W = 512
CONTROL_SCALE = 0.5
# BO / REMBO
N_H = 16
INIT_POINTS = 16     # 你要求 init 增加到 16
ROUNDS = 8
BETA = 1.5

# CLIP device：默认 CPU（避免显存占用）
CLIP_DEVICE = "cpu"
CLIP_MODEL  = "openai/clip-vit-base-patch32"

# GP超参学习设置
HP_UPDATE_EVERY = 1 
HP_OPT_STEPS = 120
HP_LR = 0.08

# 合理的 bounds（log 空间 clamp）
VAR_BOUNDS   = (1e-4, 1e3)
LS_BOUNDS    = (1e-3, 1e2)
NOISE_BOUNDS = (1e-6, 1e-1)


# -------------------------
# Utilities: constrained REMBO matrix A
# -------------------------
def make_A_constrained(n_z: int, n_h: int, rng: np.random.Generator) -> np.ndarray:
    """Construct A so that A^T A ≈ (n_z/n_h) I using QR + scaling."""
    M = rng.standard_normal((n_z, n_h)).astype(np.float64)
    Q, _ = np.linalg.qr(M)  # orthonormal columns
    A = Q * np.sqrt(n_z / n_h)
    return A.astype(np.float32)


def sample_h_on_sphere(n_h: int, rng: np.random.Generator) -> np.ndarray:
    """Sample h and normalize to ||h|| = sqrt(n_h) (typical-set friendly)."""
    v = rng.standard_normal((n_h,)).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v * np.sqrt(n_h)


# -------------------------
# Torch GP: RBF kernel + log marginal likelihood hyperparam learning
# -------------------------
def rbf_kernel(X1: torch.Tensor, X2: torch.Tensor, var: torch.Tensor, ls: torch.Tensor) -> torch.Tensor:
    # dist^2 = x^2 + x'^2 - 2xx'
    x1_sq = (X1 ** 2).sum(dim=1, keepdim=True)
    x2_sq = (X2 ** 2).sum(dim=1, keepdim=True).T
    dist2 = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)
    return var * torch.exp(-0.5 * dist2 / (ls ** 2))


def gp_cholesky(X: torch.Tensor, y: torch.Tensor, var: float, ls: float, noise: float):
    """
    Returns Cholesky factor L and alpha = K^{-1}y computed via cholesky_solve.
    """
    var_t = torch.tensor(var, dtype=X.dtype, device=X.device)
    ls_t = torch.tensor(ls, dtype=X.dtype, device=X.device)
    noise_t = torch.tensor(noise, dtype=X.dtype, device=X.device)

    K = rbf_kernel(X, X, var_t, ls_t)
    n = K.shape[0]
    K = K + (noise_t + 1e-8) * torch.eye(n, dtype=X.dtype, device=X.device)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
    return L, alpha


def gp_predict_one(x: torch.Tensor, X: torch.Tensor, L: torch.Tensor, alpha: torch.Tensor,
                   var: float, ls: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GP posterior mean/var for one x.
    """
    var_t = torch.tensor(var, dtype=X.dtype, device=X.device)
    ls_t  = torch.tensor(ls,  dtype=X.dtype, device=X.device)

    x1 = x.unsqueeze(0)  # (1,d)
    k = rbf_kernel(X, x1, var_t, ls_t).squeeze(1)  # (n,)
    mu = (k * alpha).sum()

    # predictive variance: k_xx - k^T K^{-1} k
    # K^{-1}k via cholesky_solve
    v = torch.cholesky_solve(k.unsqueeze(1), L).squeeze(1)
    k_xx = var_t
    var_post = k_xx - (k * v).sum()
    var_post = torch.clamp(var_post, min=1e-12)
    return mu, var_post


def log_marginal_likelihood(X: torch.Tensor, y: torch.Tensor,
                            log_var: torch.Tensor, log_ls: torch.Tensor, log_noise: torch.Tensor) -> torch.Tensor:
    """
    LML = -0.5 y^T K^{-1}y - 0.5 log|K| - n/2 log(2pi)
    """
    var = torch.exp(log_var)
    ls = torch.exp(log_ls)
    noise = torch.exp(log_noise)

    K = rbf_kernel(X, X, var, ls)
    n = K.shape[0]
    K = K + (noise + 1e-8) * torch.eye(n, dtype=X.dtype, device=X.device)

    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)

    data_fit = -0.5 * (y * alpha).sum()
    logdet = -torch.log(torch.diagonal(L)).sum() * 2.0 * 0.5  # will fix below

    # Correct logdet: log|K| = 2 * sum(log(diag(L)))
    logdet = -0.5 * (2.0 * torch.log(torch.diagonal(L)).sum())

    const = -0.5 * n * np.log(2.0 * np.pi)
    return data_fit + logdet + const


def fit_gp_hyperparams(X_np: np.ndarray, y_np: np.ndarray,
                       init_var: float, init_ls: float, init_noise: float,
                       steps: int = 120, lr: float = 0.08,
                       device: str = "cpu") -> tuple[float, float, float]:
    """
    Optimize (var, ls, noise) by maximizing GP log marginal likelihood (on CPU).
    """
    dev = torch.device(device)
    dtype = torch.float64

    X = torch.from_numpy(X_np).to(dev, dtype=dtype)
    y = torch.from_numpy(y_np).to(dev, dtype=dtype)

    # log-params
    log_var = torch.tensor(np.log(init_var), device=dev, dtype=dtype, requires_grad=True)
    log_ls = torch.tensor(np.log(init_ls), device=dev, dtype=dtype, requires_grad=True)
    log_noise = torch.tensor(np.log(init_noise), device=dev, dtype=dtype, requires_grad=True)

    # bounds in log space
    log_var_min, log_var_max = np.log(VAR_BOUNDS[0]), np.log(VAR_BOUNDS[1])
    log_ls_min, log_ls_max = np.log(LS_BOUNDS[0]), np.log(LS_BOUNDS[1])
    log_noise_min, log_noise_max = np.log(NOISE_BOUNDS[0]), np.log(NOISE_BOUNDS[1])

    opt = torch.optim.Adam([log_var, log_ls, log_noise], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        lml = log_marginal_likelihood(X, y, log_var, log_ls, log_noise)
        loss = -lml
        loss.backward()
        opt.step()

        # clamp
        with torch.no_grad():
            log_var.clamp_(log_var_min, log_var_max)
            log_ls.clamp_(log_ls_min, log_ls_max)
            log_noise.clamp_(log_noise_min, log_noise_max)

    var = float(torch.exp(log_var).item())
    ls = float(torch.exp(log_ls).item())
    noise = float(torch.exp(log_noise).item())
    return var, ls, noise


# -------------------------
# Continuous maximize UCB over sphere using Adam (multi-start)
# -------------------------
def maximize_ucb_continuous(X_np: np.ndarray, y_np: np.ndarray,
                            beta: float, var: float, ls: float, noise: float,
                            n_h: int,
                            restarts: int = 16,
                            steps: int = 90,
                            lr: float = 0.05,
                            device: str = "cpu") -> np.ndarray:
    """
    Find h_next = argmax UCB(h) over ||h||=sqrt(n_h) by optimizing unconstrained v and projecting.
    """
    dev = torch.device(device)
    dtype = torch.float64

    X = torch.from_numpy(X_np).to(dev, dtype=dtype)
    y = torch.from_numpy(y_np).to(dev, dtype=dtype)

    # Precompute GP cholesky factor and alpha for current hypers
    L, alpha = gp_cholesky(X, y, var=var, ls=ls, noise=noise)

    def ucb_of_v(v: torch.Tensor) -> torch.Tensor:
        h = v / (torch.norm(v) + 1e-8) * np.sqrt(n_h)
        mu, var_post = gp_predict_one(h, X, L, alpha, var=var, ls=ls)
        return mu + beta * torch.sqrt(var_post)

    best_val = -1e30
    best_h = None

    for _ in range(restarts):
        v = torch.randn((n_h,), device=dev, dtype=dtype, requires_grad=True)
        opt = torch.optim.Adam([v], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            obj = ucb_of_v(v)
            loss = -obj
            loss.backward()
            opt.step()

        with torch.no_grad():
            obj = ucb_of_v(v).item()
            if obj > best_val:
                best_val = obj
                h = v / (torch.norm(v) + 1e-8) * np.sqrt(n_h)
                best_h = h.detach().cpu().numpy().astype(np.float32)

    return best_h


# -------------------------
# Main
# -------------------------
def main():
    rng = np.random.default_rng(0)

    # 1) generator / edge
    gen = SD35ControlNetGenerator(BASE_DIR, CN_DIR, torch_dtype=torch.float16)

    ref = Image.open(REF_IMG_PATH).convert("RGB").resize((W, H))
    edge = make_canny_edge(ref, low=100, high=200).convert("RGB")
    # edge.save(os.path.join(OUT_DIR, "edge_512.png"))

    # 2) CLIP scorer (simulate user rating)
    scorer = CLIPScorer(model_name=CLIP_MODEL, device=CLIP_DEVICE)

    # 3) infer diffusion latent shape
    pipe = gen.pipe
    vae_scale = getattr(pipe, "vae_scale_factor", None) or 8
    h_lat = H // vae_scale
    w_lat = W // vae_scale

    C_lat = None
    if getattr(pipe, "transformer", None) is not None and getattr(pipe.transformer, "config", None) is not None:
        C_lat = getattr(pipe.transformer.config, "in_channels", None)
    if C_lat is None:
        raise RuntimeError("Cannot infer latent channels. Please check pipe.transformer.config.in_channels.")

    n_z = int(C_lat * h_lat * w_lat)
    print(f"[INFO] latent shape=(1,{C_lat},{h_lat},{w_lat}) n_z={n_z}, n_h={N_H}")

    # 4) offline A
    A = make_A_constrained(n_z=n_z, n_h=N_H, rng=rng)
    np.save(os.path.join(OUT_DIR, "A.npy"), A)

    def h_to_latents(h: np.ndarray) -> torch.Tensor:
        z = A @ h.astype(np.float32)  # (n_z,)
        lat = torch.from_numpy(z).reshape(1, C_lat, h_lat, w_lat)
        return lat

    # 5) state
    X = []  # list of h
    y = []  # list of scores
    log = []
    best = {"score": -1e9, "img_path": None, "tag": None}

    # GP hyperparams state (warm start)
    gp_var = 1.0
    gp_ls = 1.0
    gp_noise = 1e-3

    def run_one(h: np.ndarray, step: int, tag: str) -> float:
        nonlocal best
        latents = h_to_latents(h)

        img = gen.generate(
            prompt=PROMPT,
            negative_prompt=NEG,
            control_image=edge,
            latents=latents,
            height=H,
            width=W,
            num_inference_steps=24,
            guidance_scale=5.0,
            controlnet_conditioning_scale=CONTROL_SCALE,
        )

        img_path = os.path.join(OUT_DIR, f"{step:03d}_{tag}.png")
        img.save(img_path)

        # CLIP score: image-text cosine similarity
        score = float(scorer(img, PROMPT))

        rec = {
            "step": step,
            "tag": tag,
            "score": score,
            "img_path": img_path,
            "h": h.tolist(),
            "gp_var": gp_var,
            "gp_ls": gp_ls,
            "gp_noise": gp_noise,
        }
        log.append(rec)

        if score > best["score"]:
            best = {"score": score, "img_path": img_path, "tag": tag}

        print(f"[{step:03d}] {tag} score={score:.6f} best={best['score']:.6f} | var={gp_var:.3g} ls={gp_ls:.3g} noise={gp_noise:.3g}")
        return score

    # -------------------------
    # Init (offline random points)
    # -------------------------
    step = 0
    for _ in range(INIT_POINTS):
        h0 = sample_h_on_sphere(N_H, rng)
        s0 = run_one(h0, step, "init")
        X.append(h0)
        y.append(s0)
        step += 1

    # -------------------------
    # Online loop: update hyperparams + continuous maximize UCB
    # -------------------------
    for t in range(ROUNDS):
        X_np = np.stack(X, axis=0).astype(np.float64)
        y_np = np.array(y, dtype=np.float64)

        # (A) update kernel hyperparams by maximizing GP marginal likelihood
        if (t % HP_UPDATE_EVERY) == 0:
            # 后续多轮次更新一次参数
            gp_var, gp_ls, gp_noise = fit_gp_hyperparams(
                X_np, y_np,
                init_var=gp_var,
                init_ls=gp_ls,
                init_noise=gp_noise,
                steps=HP_OPT_STEPS,
                lr=HP_LR,
                device="cpu",
            )

        # (B) maximize UCB in continuous h-space
        h_next = maximize_ucb_continuous(
            X_np, y_np,
            beta=BETA,
            var=gp_var,
            ls=gp_ls,
            noise=gp_noise,
            n_h=N_H,
            restarts=16,
            steps=90,
            lr=0.05,
            device="cpu",
        )

        s_next = run_one(h_next, step, "bo")
        X.append(h_next)
        y.append(s_next)
        step += 1

        with open(os.path.join(OUT_DIR, "log.json"), "w") as f:
            json.dump({"log": log, "best": best}, f, indent=2)

    print("\nDONE. Best:", best)


if __name__ == "__main__":
    main()
