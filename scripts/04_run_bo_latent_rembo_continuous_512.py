import os
import json
import numpy as np
import torch
from PIL import Image

from src.sd35_controlnet_generator import SD35ControlNetGenerator
from src.edge_utils import make_canny_edge
from src.scorer import ToyRedScorer, CLIPScorer


# -------------------------
# Paths / Config
# -------------------------
BASE_DIR = "/home/wan/guanting's/diffusion-customer/model/stabilityai/stable-diffusion-3.5-large"
CN_DIR   = "/home/wan/guanting's/diffusion-customer/model/controlnets/sd35_large_controlnet_canny"
REF_IMG_PATH = "/home/wan/guanting's/diffusion-customer/assets/ref.png"
OUT_DIR  = "/home/wan/guanting's/diffusion-customer/outputs/bo_latent_rembo_continuous_512"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Fixed task definition
# -------------------------
PROMPT = "product photo of a modern sneaker, studio lighting, white background, high detail"
NEG    = "lowres, blurry, worst quality, artifacts"
H = W = 512
CONTROL_SCALE = 0.3  # 你验证过 512 下 1.0 会灰，这里固定安全值

# BO params
N_H = 16
INIT_POINTS = 4
ROUNDS = 8
BETA = 1.5

# GP kernel (fixed, no hyperparam learning)
K_VAR = 1.0          # signal variance
K_LS  = 1.0          # lengthscale
NOISE = 1e-4         # observation noise


# -------------------------
# REMBO constrained A
# -------------------------
def make_A_constrained(n_z: int, n_h: int, rng: np.random.Generator) -> np.ndarray:
    """Construct A so that A^T A ≈ (n_z/n_h) I using QR + scaling."""
    M = rng.standard_normal((n_z, n_h)).astype(np.float64)
    Q, _ = np.linalg.qr(M)  # Q has orthonormal columns
    A = Q * np.sqrt(n_z / n_h)
    return A.astype(np.float32)


def sample_h_on_sphere(n_h: int, rng: np.random.Generator) -> np.ndarray:
    """Sample h then normalize to ||h|| = sqrt(n_h) (typical-set friendly)."""
    v = rng.standard_normal((n_h,)).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v * np.sqrt(n_h)


# -------------------------
# GP posterior (RBF) in torch (so we can take gradients for continuous UCB maximization)
# -------------------------
def rbf_kernel_torch(X1: torch.Tensor, X2: torch.Tensor, var: float, ls: float) -> torch.Tensor:
    """
    X1: (n1, d), X2: (n2, d)
    return: (n1, n2)
    """
    # ||x - x'||^2 = x^2 + x'^2 - 2xx'
    x1_sq = (X1 ** 2).sum(dim=1, keepdim=True)  # (n1,1)
    x2_sq = (X2 ** 2).sum(dim=1, keepdim=True).T  # (1,n2)
    dist2 = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)     # (n1,n2)
    return var * torch.exp(-0.5 * dist2 / (ls ** 2))


def gp_precompute(X_train: torch.Tensor, y_train: torch.Tensor, var: float, ls: float, noise: float):
    """
    Precompute K^{-1} and alpha = K^{-1} y for GP regression.
    X_train: (n,d), y_train: (n,)
    """
    K = rbf_kernel_torch(X_train, X_train, var, ls)
    n = K.shape[0]
    K = K + (noise + 1e-6) * torch.eye(n, device=K.device, dtype=K.dtype)  # jitter
    # Solve K * alpha = y
    alpha = torch.linalg.solve(K, y_train)
    # Also store K^{-1} (needed for predictive variance)
    K_inv = torch.linalg.inv(K)
    return K_inv, alpha


def gp_predict_one(x: torch.Tensor, X_train: torch.Tensor, K_inv: torch.Tensor, alpha: torch.Tensor,
                   var: float, ls: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predict mean/variance for a single x.
    x: (d,)  -> treated as (1,d)
    Return: mu (scalar), var (scalar, >=0)
    """
    x1 = x.unsqueeze(0)  # (1,d)
    k = rbf_kernel_torch(X_train, x1, var, ls).squeeze(1)  # (n,)
    mu = (k * alpha).sum()  # scalar

    k_xx = torch.tensor(var, device=x.device, dtype=x.dtype)  # RBF kernel at zero distance = var
    # var = k_xx - k^T K^{-1} k
    v = K_inv @ k
    var_post = k_xx - (k * v).sum()
    var_post = torch.clamp(var_post, min=1e-12)
    return mu, var_post


def ucb_objective(v: torch.Tensor, X_train: torch.Tensor, K_inv: torch.Tensor, alpha: torch.Tensor,
                  var: float, ls: float, beta: float, n_h: int) -> torch.Tensor:
    """
    Unconstrained variable v -> project to sphere h -> compute UCB(h).
    """
    # project to sphere: h = sqrt(n_h) * v / ||v||
    h = v / (torch.norm(v) + 1e-8) * np.sqrt(n_h)
    mu, var_post = gp_predict_one(h, X_train, K_inv, alpha, var, ls)
    std = torch.sqrt(var_post)
    return mu + beta * std


def maximize_ucb_continuous(X_np: np.ndarray, y_np: np.ndarray, beta: float,
                            var: float, ls: float, noise: float,
                            n_h: int,
                            restarts: int = 16,
                            steps: int = 80,
                            lr: float = 0.05,
                            device: str = "cpu") -> np.ndarray:
    """
    Continuous optimization: multi-start Adam on unconstrained v, with projection to sphere.
    Returns best h (np.float32, shape (n_h,))
    """
    # Use CPU for GP math (cheap) and to avoid GPU memory contention with diffusion
    dev = torch.device(device)
    dtype = torch.float64

    X_train = torch.from_numpy(X_np).to(dev, dtype=dtype)
    y_train = torch.from_numpy(y_np).to(dev, dtype=dtype)

    K_inv, alpha = gp_precompute(X_train, y_train, var, ls, noise)

    best_val = -1e30
    best_h = None

    for r in range(restarts):
        # random init v
        v = torch.randn((n_h,), device=dev, dtype=dtype, requires_grad=True)
        opt = torch.optim.Adam([v], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            obj = ucb_objective(v, X_train, K_inv, alpha, var, ls, beta, n_h)
            loss = -obj  # maximize obj
            loss.backward()
            opt.step()

        with torch.no_grad():
            obj = ucb_objective(v, X_train, K_inv, alpha, var, ls, beta, n_h)
            if obj.item() > best_val:
                best_val = obj.item()
                h = v / (torch.norm(v) + 1e-8) * np.sqrt(n_h)
                best_h = h.detach().cpu().numpy().astype(np.float32)

    return best_h


# -------------------------
# Main loop
# -------------------------
def main():
    rng = np.random.default_rng(0)

    # 1) generator / edge / scorer
    gen = SD35ControlNetGenerator(BASE_DIR, CN_DIR, torch_dtype=torch.float16)
    ref = Image.open(REF_IMG_PATH).convert("RGB").resize((W, H))
    edge = make_canny_edge(ref, low=100, high=200).convert("RGB")
    edge.save(os.path.join(OUT_DIR, "edge_512.png"))

    # scorer = ToyRedScorer()
    scorer = CLIPScorer(model_name="openai/clip-vit-base-patch32", device="cuda")

    # 2) infer diffusion latent shape
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
    print(f"[INFO] latent shape=(1,{C_lat},{h_lat},{w_lat}), n_z={n_z}, n_h={N_H}")

    # 3) offline prep: build A once
    A = make_A_constrained(n_z=n_z, n_h=N_H, rng=rng)
    np.save(os.path.join(OUT_DIR, "A.npy"), A)

    def h_to_latents(h: np.ndarray) -> torch.Tensor:
        z = A @ h.astype(np.float32)  # (n_z,)
        lat = torch.from_numpy(z).reshape(1, C_lat, h_lat, w_lat)
        return lat

    log = []
    best = {"score": -1e9, "img_path": None, "tag": None}

    X = []  # list of h
    y = []  # list of scores

    def run_one(h: np.ndarray, step: int, tag: str) -> float:
        nonlocal best
        latents = h_to_latents(h)
        img = gen.generate(
            prompt=PROMPT,
            control_image=edge,
            latents=latents,
            height=H,
            width=W,
            num_inference_steps=24,
            guidance_scale=5.0,
            controlnet_conditioning_scale=CONTROL_SCALE,
            negative_prompt=NEG,
        )
        img_path = os.path.join(OUT_DIR, f"{step:03d}_{tag}.png")
        img.save(img_path)

        score = float(scorer(img, PROMPT))  # 真实系统里这里换成“用户打分”
        rec = {"step": step, "tag": tag, "score": score, "img_path": img_path, "h": h.tolist()}
        log.append(rec)

        if score > best["score"]:
            best = {"score": score, "img_path": img_path, "tag": tag}

        print(f"[{step:03d}] {tag} score={score:.6f} best={best['score']:.6f}")
        return score

    # -------------------------
    # Offline init: a few random points
    # -------------------------
    step = 0
    for _ in range(INIT_POINTS):
        h0 = sample_h_on_sphere(N_H, rng)
        s0 = run_one(h0, step, "init")
        X.append(h0)
        y.append(s0)
        step += 1

    # -------------------------
    # Online loop: continuous maximize UCB each round
    # -------------------------
    for t in range(ROUNDS):
        X_np = np.stack(X, axis=0).astype(np.float64)  # (n, n_h)
        y_np = np.array(y, dtype=np.float64)           # (n,)

        # Continuous optimization of acquisition over h (instead of candidate pool)
        h_next = maximize_ucb_continuous(
            X_np=X_np,
            y_np=y_np,
            beta=BETA,
            var=K_VAR,
            ls=K_LS,
            noise=NOISE,
            n_h=N_H,
            restarts=16,
            steps=80,
            lr=0.05,
            device="cpu",  # CPU is enough
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
