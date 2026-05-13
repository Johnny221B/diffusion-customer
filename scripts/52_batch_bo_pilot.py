"""End-to-end batch-BO pilot with real SD3.5 generation.

Uses pre-computed PCA basis (W, mean_emb) from existing d=16 data. Runs a
closed-loop BO:
  - Warmup with M iid word queries
  - Iterations: batch-Thompson propose K new z's via interpolation-anchored
    sampling -> generate images -> compute DreamSim -> binary label -> refit
  - Oracle = DreamSim (simulates perfect human for validation)

Tracked per iteration:
  - Fraction of batch beating competitor
  - Best dreamsim in batch / so far
  - Held-out AUC (using remaining unused words)
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


def get_word_emb(pipe, word):
    with torch.no_grad():
        out = pipe.encode_prompt(prompt=word, prompt_2=word, prompt_3=word, negative_prompt="")
        out_empty = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", negative_prompt="")
    pe, ee = out[0], out_empty[0]
    L_w, L_e = pe.shape[1], ee.shape[1]
    if L_w > L_e:
        n = L_w - L_e
        return pe[0, :n, :].mean(dim=0).detach()
    ml = min(L_w, L_e)
    diffs = (pe[0, :ml] - ee[0, :ml]).norm(dim=1)
    return pe[0, diffs.argmax().item(), :].detach()


def fit_gp(Z, y):
    """Fit GP classifier on (Z, y) with normalized inputs. Returns (gp, mu, sd) or None."""
    if len(set(y)) < 2:
        return None
    mu = Z.mean(0); sd = Z.std(0) + 1e-9
    Zn = (Z - mu) / sd
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=2, random_state=0)
    gp.fit(Zn, y)
    return gp, mu, sd


def gp_predict(gp_tup, Z):
    if gp_tup is None:
        return np.full(len(Z), 0.5)
    gp, mu, sd = gp_tup
    return gp.predict_proba((Z - mu) / sd)[:, 1]


def propose_batch(Z_known, y_known, gp_tup, K, rng, sigma=0.3, top_m=20, mix_alpha=None):
    """Interpolation-anchored batch proposal.
    For each batch member: pick two anchors from top-m words weighted by posterior prob,
    interpolate with alpha ~ U(0,1), add Gaussian noise in PCA space.
    """
    if gp_tup is not None:
        p = gp_predict(gp_tup, Z_known)
    else:
        p = np.full(len(Z_known), 0.5)
    # top-m by posterior prob
    order = np.argsort(p)[::-1]
    top_idx = order[:top_m]
    w = p[top_idx] + 1e-3
    w = w / w.sum()

    d = Z_known.shape[1]
    batch = np.zeros((K, d))
    for k in range(K):
        a, b = rng.choice(top_idx, size=2, replace=True, p=w)
        alpha = rng.uniform() if mix_alpha is None else mix_alpha
        batch[k] = alpha * Z_known[a] + (1 - alpha) * Z_known[b]
        batch[k] += sigma * rng.randn(d)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_npz", type=str,
                        default="outputs/scaling_red_vs_green_d16_0416_1723/data.npz",
                        help="Provides W, mean_emb, Z, embs, dreams, words to seed warmup")
    parser.add_argument("--raw_csv", type=str,
                        default="outputs/scaling_red_vs_green_d16_0416_1723/raw_data.csv")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--ref_word", type=str, default="red")
    parser.add_argument("--comp_word", type=str, default="indoor",
                        help="word whose generated image is the competitor B")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--batch_k", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="perturbation magnitude in PCA z-space")
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--run_tag", type=str, default="")
    args = parser.parse_args()

    device = args.device
    stamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = f"outputs/pilot_bo_{args.ref_word}_{args.comp_word}_{args.run_tag}_{stamp}"
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # === Load precomputed PCA basis & word bank ===
    print(f"Loading precomputed data from {args.data_npz}")
    data = np.load(args.data_npz)
    W = data["W"]                  # (4096, d)
    mean_emb = data["mean_emb"]    # (4096,)
    Z_vocab = data["Z"]            # (172, d)
    embs_vocab = data["embs"]      # (172, 4096)
    dreams_vocab = data["dreams"]  # (172,) dreamsim to ref=red_img
    df_raw = pd.read_csv(args.raw_csv)
    words_vocab = df_raw["word"].tolist()
    d = Z_vocab.shape[1]
    print(f"  d={d}, vocab_size={len(words_vocab)}")

    # === Load models ===
    print("Loading SD3.5 ...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim ...")
    scorer = DreamSimScorer(device=device)

    prompt = ("Product photo of a single shoe, full shoe visible, side profile, "
              "centered on a plain white background")

    def gen_image(emb_4096):
        emb_t = torch.from_numpy(emb_4096.astype(np.float32)).to(device, dtype=torch.float16).unsqueeze(0)
        embeds = gen.encode_batch_insert(prompt, emb_t)
        return gen.generate_batch(embeds, [args.seed])[0]

    def emb_from_z(z_d):
        """Project d-dim PCA z back to 4096 token embedding."""
        return mean_emb + W @ z_d

    # === Reference and competitor ===
    print(f"Generating reference '{args.ref_word}' and competitor '{args.comp_word}' ...")
    ref_emb = get_word_emb(gen.pipe, args.ref_word).float().cpu().numpy()
    comp_emb = get_word_emb(gen.pipe, args.comp_word).float().cpu().numpy()

    ref_img = gen_image(ref_emb)
    ref_img.save(os.path.join(run_dir, "reference.png"))
    ref_tensor = scorer.preprocess(ref_img)

    comp_img = gen_image(comp_emb)
    comp_img.save(os.path.join(run_dir, "competitor.png"))
    d_B = scorer.model(ref_tensor, scorer.preprocess(comp_img)).item()
    print(f"  d_B (competitor vs ref) = {d_B:.4f}")

    # === Warmup: M random iid words (REUSE existing dreamsim measurements) ===
    rng = np.random.RandomState(42)
    # Pick words whose dreamsim != d_B (drop comp if present)
    pool = [i for i, w in enumerate(words_vocab) if w != args.comp_word]
    warm_idx = rng.permutation(pool)[:args.warmup]

    Z_obs = Z_vocab[warm_idx].copy()
    d_obs = dreams_vocab[warm_idx].copy()
    y_obs = (d_obs < d_B).astype(int)
    words_used = [words_vocab[i] for i in warm_idx]
    print(f"\nWarmup: {args.warmup} words -> positives={y_obs.sum()}/{len(y_obs)}")

    # Track held-out pool for AUC
    heldout_mask = np.ones(len(words_vocab), dtype=bool)
    heldout_mask[warm_idx] = False
    heldout_mask[words_vocab.index(args.comp_word)] = False

    iter_log = []
    per_gen_log = []

    # Warmup aggregate metrics
    iter_log.append({
        "iter": 0, "n_obs": len(y_obs),
        "batch_pos": np.nan, "batch_best_d": np.nan, "batch_mean_d": np.nan,
        "overall_best_d": d_obs.min(), "overall_pos_rate": y_obs.mean(),
        "heldout_auc": np.nan,
    })

    # === BO iterations ===
    for it in range(1, args.n_iter + 1):
        print(f"\n=== Iteration {it}/{args.n_iter} ===")
        gp_tup = fit_gp(Z_obs, y_obs)

        # Held-out AUC (on vocab words never queried)
        if gp_tup is not None and heldout_mask.sum() > 5:
            p_held = gp_predict(gp_tup, Z_vocab[heldout_mask])
            y_held = (dreams_vocab[heldout_mask] < d_B).astype(int)
            if len(set(y_held)) == 2:
                from sklearn.metrics import roc_auc_score
                heldout_auc = roc_auc_score(y_held, p_held)
            else:
                heldout_auc = np.nan
        else:
            heldout_auc = np.nan

        # Propose batch via interpolation anchored in top-m known words
        Z_new = propose_batch(Z_obs, y_obs, gp_tup, args.batch_k, rng,
                              sigma=args.sigma, top_m=args.top_m)

        # Generate and label each
        batch_d = []
        batch_y = []
        for k in range(args.batch_k):
            z_k = Z_new[k]
            token_emb = emb_from_z(z_k)
            img = gen_image(token_emb)
            img_path = os.path.join(run_dir, f"iter{it:02d}_k{k}.png")
            img.save(img_path)
            d_k = scorer.model(ref_tensor, scorer.preprocess(img)).item()
            y_k = int(d_k < d_B)
            batch_d.append(d_k); batch_y.append(y_k)
            per_gen_log.append({
                "iter": it, "k": k, "dreamsim": d_k, "label": y_k, "path": img_path,
                "z": z_k.tolist(),
            })
            img.close()
            print(f"  [k={k}] d={d_k:.4f}  y={y_k}")

        batch_d = np.array(batch_d); batch_y = np.array(batch_y)
        # Append to observed
        Z_obs = np.concatenate([Z_obs, Z_new], axis=0)
        d_obs = np.concatenate([d_obs, batch_d])
        y_obs = np.concatenate([y_obs, batch_y])

        iter_log.append({
            "iter": it, "n_obs": len(y_obs),
            "batch_pos": batch_y.mean(),
            "batch_best_d": batch_d.min(),
            "batch_mean_d": batch_d.mean(),
            "overall_best_d": d_obs.min(),
            "overall_pos_rate": y_obs.mean(),
            "heldout_auc": heldout_auc,
        })
        print(f"  batch_pos={batch_y.mean():.2f}  batch_best_d={batch_d.min():.4f}  "
              f"overall_best_d={d_obs.min():.4f}  heldout_auc={heldout_auc:.3f}"
              if not np.isnan(heldout_auc) else
              f"  batch_pos={batch_y.mean():.2f}  batch_best_d={batch_d.min():.4f}  "
              f"overall_best_d={d_obs.min():.4f}")

    # === Save logs ===
    df_iter = pd.DataFrame(iter_log)
    df_iter.to_csv(os.path.join(run_dir, "iterations.csv"), index=False)
    pd.DataFrame(per_gen_log).to_csv(os.path.join(run_dir, "per_generation.csv"), index=False)

    # === Plot ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df = df_iter[df_iter["iter"] > 0]
    axes[0].plot(df["iter"], df["batch_pos"], marker="o", label="batch pos rate")
    axes[0].plot(df["iter"], df["overall_pos_rate"], marker="s", label="overall pos rate", alpha=0.5)
    axes[0].axhline(0.5, color="k", alpha=0.3)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("fraction beating competitor")
    axes[0].set_title("Pos rate"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["iter"], df["batch_best_d"], marker="o", label="batch best")
    axes[1].plot(df["iter"], df["batch_mean_d"], marker="s", label="batch mean")
    axes[1].plot(df["iter"], df["overall_best_d"], marker="^", label="overall best", color="k")
    axes[1].axhline(d_B, color="red", linestyle="--", alpha=0.5, label=f"competitor={d_B:.3f}")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("dreamsim to ref (lower=better)")
    axes[1].set_title("DreamSim trajectory"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["iter"], df["heldout_auc"], marker="o", color="tab:green")
    axes[2].axhline(0.5, color="k", alpha=0.3)
    axes[2].set_xlabel("iteration"); axes[2].set_ylabel("held-out AUC")
    axes[2].set_title("GP classifier AUC on unused vocab"); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Batch-BO pilot  ref={args.ref_word} comp={args.comp_word} "
                 f"batch={args.batch_k} sigma={args.sigma} d={d}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "trajectory.png"), dpi=150)
    print(f"\nSaved {run_dir}")


if __name__ == "__main__":
    main()
