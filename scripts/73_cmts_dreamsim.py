"""
Algorithm 2 driver: CM-TS with real SD3.5 + DreamSim oracle.

PDF "Going Continuous" §5 Task 2 实现：把 cmts_sim 的 synthetic Bernoulli 标签
换成"真 SD3.5 生成 + DreamSim 比较"。Algorithm 1' 直接复用 cmts_sim.argmax_over_M。

Per-round flow:
  1. β̃_t ~ N(β̂_{t-1}, v²·H_{t-1}^{-1})           (Laplace + variance inflation)
  2. z_t = argmax_over_M(β̃_t, Z, k=10, τ_d)        (Algorithm 1', cmts_sim)
  3. ẑ_t = U_d^T z_t + μ̄                          (PCA inverse_transform 回 ℝ^4096)
  4. image_t = SD3.5(sandwich-inject ẑ_t, ref_seed) (encode_batch_insert)
  5. y_t ~ Bernoulli(σ(α(D_B − DreamSim(image_t,R))))  (SOFT-Bernoulli label = simulate
        process, matches cmts_sim.py; y_hard=1[ds<D_B] kept only for plots/metrics)
  6. (β̂_t, H_t) = Laplace_MAP(Φ ∪ (z_t - a, y_t))   (refit; Euclidean clip ||β|| ≤ S)

Range-slice on replications: --seed_start/--seed_end lets you split NSEED 条 trajectory
across 4 GPU (each GPU 串行跑自己份额的 sim seeds，主循环内不能跨轮并行).

Outputs (per trajectory):
  sim<sss>/trajectory.csv     n0 warm + T main 行
  sim<sss>/summary.json
  sim<sss>/images/            n0+T 张 SD3.5 PNG
  sim<sss>/posterior.npz      最终 β̂, H, Φ, y
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator  # noqa: E402
from src.scorer import DreamSimScorer  # noqa: E402
from src.cmts_sim import (  # noqa: E402
    argmax_over_M, random_valid_design, calibrate_tau,
    laplace_map, project_norm, sigma,
)


def render_batch(z_4096_arr, gen, ref_seed, batch_size):
    """Sandwich-inject (n, 4096) anchor lifts; return PIL list."""
    images = []
    for s in range(0, len(z_4096_arr), batch_size):
        chunk = np.asarray(z_4096_arr[s:s + batch_size], dtype=np.float32)
        Z = torch.tensor(chunk, dtype=torch.float16, device=gen.device)
        embeds = gen.encode_batch_insert("", Z)
        imgs = gen.generate_batch(embeds, [ref_seed] * len(chunk))
        images.extend(imgs)
    return images


def dreamsim_to_ref(images, scorer, R_tensor):
    return np.array([float(scorer.model(R_tensor, scorer.preprocess(im)).item())
                     for im in images])


def _save_ckpt(sim_dir, state):
    """Atomic round-level checkpoint (write tmp then rename, so a kill mid-write
    never corrupts the resume file)."""
    p = os.path.join(sim_dir, "_ckpt.pkl")
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


def _load_ckpt(sim_dir):
    p = os.path.join(sim_dir, "_ckpt.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:                       # corrupt ckpt -> restart traj fresh
        print(f"  WARN: bad checkpoint ({e}); restarting trajectory", flush=True)
        return None


def _finalize(sim_dir, sim_seed, rows, Phi, y, z_comp, beta_hat, H,
              warm_hit, T, n0, B, d, v, S, D_B, tau):
    """Write trajectory.csv / summary.json / posterior.npz from accumulated rows."""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sim_dir, "trajectory.csv"), index=False)
    main_df = df[df["phase"] == "main"]
    round_wr = main_df.groupby("t")["y_hard"].mean()        # HARD winrate for plot/summary
    round_soft = (main_df.groupby("t")["true_p_soft"].mean()
                  if "true_p_soft" in main_df else round_wr * float("nan"))
    summary = {
        "sim_seed": int(sim_seed), "T": int(T), "n0": int(n0), "B": int(B),
        "d": int(d), "v": float(v), "S": float(S), "D_B": float(D_B), "tau_d": float(tau),
        "median_predicted_p": float(main_df["predicted_p"].median())
        if "predicted_p" in main_df else float("nan"),
        "mean_cos_beta_prev": float(main_df["cos_beta_prev"].mean())
        if "cos_beta_prev" in main_df else float("nan"),
        "best_ds": float(main_df["ds_to_R"].min()),
        "warm_hit_rate": float(warm_hit),
        "main_hit_rate": float(main_df["y_hard"].mean()),
        "winrate_first10": float(round_wr.iloc[:10].mean()),
        "winrate_last10": float(round_wr.iloc[-10:].mean()),
        "softrate_first10": float(round_soft.iloc[:10].mean()),
        "softrate_last10": float(round_soft.iloc[-10:].mean()),
        "mean_cov_eig_max": float(main_df["cov_eig_max"].mean())
        if "cov_eig_max" in main_df else float("nan"),
        "mean_cov_eig_min": float(main_df["cov_eig_min"].mean())
        if "cov_eig_min" in main_df else float("nan"),
        "final_beta_norm": float(np.linalg.norm(beta_hat)),
    }
    with open(os.path.join(sim_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(sim_dir, "posterior.npz"),
             beta_hat=beta_hat, H=H, Phi=Phi, y=y, z_comp=z_comp)
    return summary


def run_one_trajectory(sim_seed, parent_dir, gen, scorer, R_tensor,
                       pca, Z, z_comp, tau, k, n0, T, B, v, S, D_B,
                       ref_seed, batch_size, nn_idx, d, save_img_every=1, lam=None,
                       alpha=30.0):
    sim_dir = os.path.join(parent_dir, f"sim{sim_seed:03d}")
    img_dir = os.path.join(sim_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(sim_dir, "trajectory.csv")
    lam = float(d) if lam is None else float(lam)            # prior precision (default d)
    print(f"\n=== sim_seed={sim_seed}  (Z={Z.shape}, τ_d={tau:.4f}, "
          f"D_B={D_B:.4f}, B={B}) ===", flush=True)

    # ============ RESUME PATH ============
    ckpt = _load_ckpt(sim_dir)
    if ckpt is not None:
        t_start = int(ckpt["t_done"])
        rows = ckpt["rows"]
        Phi, y = ckpt["Phi"], ckpt["y"]
        beta_hat, H = ckpt["beta_hat"], ckpt["H"]
        warm_hit = ckpt["warm_hit"]
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng_state"]
        if t_start >= T:                          # already complete -> just finalize
            print(f"  sim{sim_seed:03d} already complete (t={t_start}); skip.",
                  flush=True)
            return _finalize(sim_dir, sim_seed, rows, Phi, y, z_comp, beta_hat, H,
                             warm_hit, T, n0, B, d, v, S, D_B, tau)
        print(f"  RESUME from round t={t_start}/{T} "
              f"(||β||={np.linalg.norm(beta_hat):.2f})", flush=True)
    # ============ FRESH PATH ============
    else:
        rng = np.random.default_rng(sim_seed * 1000 + 7)
        print(f"  Warm-start n0={n0} designs ...", flush=True)
        warm_z = np.stack([random_valid_design(Z, k, tau, rng, nn_idx)
                           for _ in range(n0)])
        warm_z_4096 = pca.inverse_transform(warm_z)
        warm_imgs = render_batch(warm_z_4096, gen, ref_seed, batch_size)
        warm_ds = dreamsim_to_ref(warm_imgs, scorer, R_tensor)
        warm_y_hard = (warm_ds < D_B).astype(int)                  # hard indicator (plots/metrics)
        warm_y = rng.binomial(1, sigma(alpha * (D_B - warm_ds))).astype(int)  # SOFT-Bernoulli training label
        for i, im in enumerate(warm_imgs):
            im.save(os.path.join(img_dir, f"w{i:02d}.png")); im.close()
        warm_hit = float(warm_y_hard.mean())
        print(f"    warm hit-rate: {warm_hit:.3f}  ds mean={warm_ds.mean():.4f}",
              flush=True)

        Phi = warm_z - z_comp
        y = warm_y.astype(float)
        beta_hat, H = laplace_map(Phi, y, lam, d)
        beta_hat = project_norm(beta_hat, S)
        rows = [{
            "t": i - n0, "phase": "warm", "b": 0, "y": int(warm_y[i]),
            "y_hard": int(warm_y_hard[i]),
            "ds_to_R": float(warm_ds[i]),
            "z_norm": float(np.linalg.norm(warm_z[i])),
            "beta_norm": float("nan"), "round_winrate": float("nan"),
            "predicted_p": float("nan"),
            "true_p_soft": float(sigma(alpha * (D_B - warm_ds[i]))),
            "cov_eig_max": float("nan"), "cov_eig_min": float("nan"),
        } for i in range(n0)]
        t_start = 0
        _save_ckpt(sim_dir, dict(t_done=0, rows=rows, Phi=Phi, y=y,
                                 beta_hat=beta_hat, H=H, warm_hit=warm_hit,
                                 rng_state=rng.bit_generator.state))

    # ---- Main loop (batch Thompson: B independent draws / round, ONE refit) ----
    print(f"  Main loop T={T}, B={B} (theta-batch Thompson) from t={t_start} ...",
          flush=True)
    for t in range(t_start, T):
        center = beta_hat.copy()                # β̂_{t-1}: the Thompson draw centre
        cn = np.linalg.norm(center) + 1e-12
        # B variance-inflated Thompson draws from the SAME current posterior
        try:
            Hinv = np.linalg.inv(H)
            cov = v * v * 0.5 * (Hinv + Hinv.T)
            betas = rng.multivariate_normal(beta_hat, cov, size=B)      # (B, d)
        except np.linalg.LinAlgError:
            print(f"    t={t}: H not invertible, fall back to diag cov", flush=True)
            d2 = np.clip(np.diag(np.linalg.pinv(H)), 1e-8, None)
            cov = v * v * np.diag(d2)
            betas = rng.multivariate_normal(beta_hat, cov, size=B)
        # Thompson covariance spectrum: max/min eigenvalue of THIS round's draw cov.
        # lam large -> H~=lam*I -> cov ~= (v^2/lam) I -> both eigvals small AND
        # eig_max/eig_min -> 1 (isotropic): exploration direction becomes near-random.
        cov_eigs = np.linalg.eigvalsh(cov)                              # ascending
        cov_eig_min = float(cov_eigs[0]); cov_eig_max = float(cov_eigs[-1])
        # exploration angular breadth: mean cos(draw, centre); ->1 greedy, ->0 wide
        ts_cos_mean = float(np.mean(
            (betas @ center) / (np.linalg.norm(betas, axis=1) + 1e-12) / cn))

        # Algorithm 1' for each draw -> B distinct designs (diversity from posterior)
        Z_batch = np.stack([argmax_over_M(b_, Z, k, tau) for b_ in betas])   # (B, d)
        Z_batch_4096 = pca.inverse_transform(Z_batch)
        imgs = render_batch(Z_batch_4096, gen, ref_seed, batch_size=B)
        ds_batch = dreamsim_to_ref(imgs, scorer, R_tensor)                   # (B,)
        y_hard_batch = (ds_batch < D_B).astype(int)                          # hard indicator (plots)
        y_batch = rng.binomial(1, sigma(alpha * (D_B - ds_batch))).astype(int)  # SOFT-Bernoulli training label
        # save the round's B images only every `save_img_every` rounds (disk thrift);
        # always keep the final round. close() regardless to free PIL buffers.
        save_this = (save_img_every <= 1) or ((t + 1) % save_img_every == 0) or (t == T - 1)
        for b_, im in enumerate(imgs):
            if save_this:
                im.save(os.path.join(img_dir, f"t{t:03d}_b{b_}.png"))
            im.close()

        # ONE Laplace refit on all B observations of this round
        Phi_batch = Z_batch - z_comp
        Phi = np.vstack([Phi, Phi_batch]); y = np.append(y, y_batch.astype(float))
        beta_hat, H = laplace_map(Phi, y, lam, d, beta0=beta_hat)
        beta_hat = project_norm(beta_hat, S)
        # direction adaptation: cos(β̂_t, β̂_{t-1}); ->1 frozen, <1 still rotating
        cos_beta_prev = float((beta_hat @ center) / ((np.linalg.norm(beta_hat) + 1e-12) * cn))

        round_winrate = float(y_hard_batch.mean())          # HARD winrate (for plots)
        bnorm = float(np.linalg.norm(beta_hat))
        for b_ in range(B):
            rows.append({
                "t": t, "phase": "main", "b": b_, "y": int(y_batch[b_]),
                "y_hard": int(y_hard_batch[b_]),
                "ds_to_R": float(ds_batch[b_]),
                "z_norm": float(np.linalg.norm(Z_batch[b_])),
                "beta_norm": bnorm, "round_winrate": round_winrate,
                "predicted_p": float(sigma(np.dot(Phi_batch[b_], beta_hat))),
                "true_p_soft": float(sigma(alpha * (D_B - ds_batch[b_]))),
                "cov_eig_max": cov_eig_max, "cov_eig_min": cov_eig_min,
                "cos_beta_prev": cos_beta_prev, "ts_cos_mean": ts_cos_mean,
            })

        # round-level resume checkpoint (after this round is fully committed)
        _save_ckpt(sim_dir, dict(t_done=t + 1, rows=rows, Phi=Phi, y=y,
                                 beta_hat=beta_hat, H=H, warm_hit=warm_hit,
                                 rng_state=rng.bit_generator.state))
        if (t + 1) % 10 == 0 or t == T - 1:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            best_so_far = min(r["ds_to_R"] for r in rows if r["phase"] == "main")
            print(f"    t={t+1:3d}/{T}  winrate={round_winrate:.2f}  "
                  f"best={best_so_far:.4f}  bestin={ds_batch.min():.4f}  "
                  f"||β||={bnorm:.2f}", flush=True)

    summary = _finalize(sim_dir, sim_seed, rows, Phi, y, z_comp, beta_hat, H,
                        warm_hit, T, n0, B, d, v, S, D_B, tau)
    print(f"  sim{sim_seed:03d} done: best_ds={summary['best_ds']:.4f} "
          f"winrate {summary['winrate_first10']:.3f}->{summary['winrate_last10']:.3f} "
          f"(first10->last10)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool_dir", default="outputs/strict_pool_s228_0429_0119",
                    help="contains embeddings.npz + reference.png")
    ap.add_argument("--dreams_npz",
                    default="outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz",
                    help="for D_B = dreams[canvas_idx, B_seed]")
    ap.add_argument("--B_word", default="canvas")
    ap.add_argument("--B_seed", type=int, default=34)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seed_end", type=int, default=1)
    ap.add_argument("--dim", type=int, default=16, dest="d",
                    help="PCA dim (renamed from --d to avoid conda-run arg clash)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--B", type=int, default=8,
                    help="theta-batch size: B independent TS draws per round, "
                         "ONE refit (parallel Thompson, like discrete theta_batch)")
    ap.add_argument("--n0", type=int, default=24)
    ap.add_argument("--v", type=float, default=0.4)
    ap.add_argument("--S", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="soft-oracle sharpness. Now drives BOTH (a) the SOFT-Bernoulli "
                         "TRAINING label  y ~ Bernoulli(sigma(alpha*(D_B-ds)))  [the "
                         "simulate process, matching cmts_sim.py's Bernoulli-of-sigmoid "
                         "feedback] AND (b) the true_p_soft diagnostic. Smaller alpha = "
                         "graded win-prob: barely beating B only wins ~50-68%%, so there is "
                         "real pressure to get much closer to R (kills the 'camp just past "
                         "B' pathology of the old HARD 1[ds<D_B] label). Plots still use the "
                         "HARD indicator y_hard=1[ds<D_B].")
    ap.add_argument("--lam", type=float, default=None,
                    help="logistic prior precision (ridge). Default None -> d "
                         "(cmts_sim convention). Larger lam keeps ||beta|| small / "
                         "sigma in the soft band so the fit DIRECTION keeps adapting "
                         "(de-saturates W=p(1-p)); this is the sweep axis, not S.")
    ap.add_argument("--ref_seed", type=int, default=1810772,
                    help="fixed SD3.5 generation seed for all rounds (PDF Pitfalls)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="only used for warm-start batching")
    ap.add_argument("--save_img_every", type=int, default=1,
                    help="save the round's B images only every N rounds "
                         "(1=every round; 10=disk-thrifty for large M). "
                         "Warm-start images and the final round are always saved.")
    ap.add_argument("--partial_id", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out_root", default=None,
                    help="if set, all partials share this root; else new timestamp")
    args = ap.parse_args()

    # === Load anchors + competitor + ref ===
    print(f"Loading pool: {args.pool_dir}")
    pool = np.load(os.path.join(args.pool_dir, "embeddings.npz"), allow_pickle=True)
    embs = pool["embs"].astype(np.float32)              # (228, 4096)
    words = [str(w) for w in pool["words"]]
    if args.B_word not in words:
        raise ValueError(f"competitor '{args.B_word}' not in pool")
    b_idx = words.index(args.B_word)
    print(f"  M={len(words)}, competitor='{args.B_word}' at idx {b_idx}")

    dm = np.load(args.dreams_npz, allow_pickle=True)
    dm_words = [str(w) for w in dm["words"]]
    D_B = float(dm["dreams"][dm_words.index(args.B_word), args.B_seed])
    print(f"  D_B = dreams[{args.B_word}, seed={args.B_seed}] = {D_B:.4f}")

    # Drop competitor from anchor set (canonical 227 setup)
    keep = np.ones(len(words), dtype=bool); keep[b_idx] = False
    embs_kept = embs[keep]                              # (227, 4096)
    words_kept = [w for w, k_ in zip(words, keep) if k_]
    embs_comp = embs[b_idx]                             # (4096,) competitor embedding

    # PCA(d) fit on 227 kept; project competitor to get algorithm anchor a
    pca = PCA(n_components=args.d, random_state=0)
    Z = pca.fit_transform(embs_kept).astype(np.float64)      # (227, d)
    z_comp = pca.transform(embs_comp.reshape(1, -1)).squeeze().astype(np.float64)
    print(f"  PCA({args.d}) fit, explained var ratio sum = "
          f"{pca.explained_variance_ratio_.sum():.4f}")

    # τ_d (95pct LOO kNN), nn_idx for random_valid_design
    tau = calibrate_tau(Z, args.k, q=0.95)
    print(f"  τ_d = {tau:.4f}")
    D2 = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    nn_idx = np.argsort(D2, axis=1)[:, 1:11]            # exclude self, top-10

    # === Load SD3.5 + DreamSim + ref image ===
    print(f"Loading SD3.5 on {args.device} ...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=args.device)
    print(f"Loading DreamSim ...")
    scorer = DreamSimScorer(device=args.device)
    R_img = Image.open(os.path.join(args.pool_dir, "reference.png")).convert("RGB")
    R_tensor = scorer.preprocess(R_img)
    print(f"  ref image loaded from {args.pool_dir}/reference.png")

    # === Output dir ===
    if args.out_root:
        run_dir = args.out_root
    else:
        stamp = datetime.now().strftime("%m%d_%H%M")
        run_dir = (f"outputs/cmts_{args.tag}_d{args.d}_T{args.T}_n0{args.n0}_"
                   f"v{args.v}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    # Each partial writes to the same parent (different sim subdir names)
    print(f"\nWriting trajectories to {run_dir}/")

    # Save run config once (per-partial)
    cfg = vars(args).copy()
    cfg.update({"D_B": D_B, "tau_d": float(tau), "M_kept": int(keep.sum()),
                "expl_var_ratio_sum": float(pca.explained_variance_ratio_.sum())})
    with open(os.path.join(run_dir, f"config_partial{args.partial_id}.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # === Trajectories ===
    all_summaries = []
    for sim_seed in range(args.seed_start, args.seed_end):
        s = run_one_trajectory(
            sim_seed, run_dir, gen, scorer, R_tensor,
            pca, Z, z_comp, tau, args.k, args.n0, args.T, args.B, args.v, args.S, D_B,
            args.ref_seed, args.batch_size, nn_idx, args.d, args.save_img_every, args.lam,
            args.alpha,
        )
        all_summaries.append(s)

    with open(os.path.join(run_dir, f"summaries_partial{args.partial_id}.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nDone partial{args.partial_id}: "
          f"{len(all_summaries)} trajectories in {run_dir}/")


if __name__ == "__main__":
    main()
