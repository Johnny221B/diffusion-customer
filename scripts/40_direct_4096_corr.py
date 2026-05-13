"""
Experiment: Skip W projection, sample directly in 4096-dim.
Test if z-space distance correlates with DreamSim when we work
in the native embedding space.

Use a real word embedding as reference direction (since we know
word embeddings live in a meaningful region of 4096-dim space).

X-axis: ||z - z_ref|| or cosine similarity
Y-axis: DreamSim distance
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from scipy import stats

from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
from src.scorer import DreamSimScorer


def get_word_token_embedding(pipe, word):
    """Extract single token embedding for a word."""
    out = pipe.encode_prompt(prompt=word, prompt_2=word, prompt_3=word, negative_prompt="")
    out_empty = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", negative_prompt="")

    prompt_embeds = out[0]
    empty_embeds = out_empty[0]

    L_word = prompt_embeds.shape[1]
    L_empty = empty_embeds.shape[1]

    if L_word > L_empty:
        n_word_tokens = L_word - L_empty
        word_emb = prompt_embeds[0, :n_word_tokens, :].mean(dim=0)
    else:
        min_len = min(L_word, L_empty)
        diffs = (prompt_embeds[0, :min_len] - empty_embeds[0, :min_len]).norm(dim=1)
        max_diff_idx = diffs.argmax().item()
        word_emb = prompt_embeds[0, max_diff_idx, :]

    return word_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1810772)
    parser.add_argument("--n_samples", type=int, default=25)
    parser.add_argument("--ref_word", type=str, default="leather")
    args = parser.parse_args()

    device = args.device
    run_dir = f"outputs/direct_4096_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    seeds = [945737, 1763690, 1082459, 1755275]

    # Load models
    print("Loading SD3.5...")
    gen = SD35BatchEmbeddingGenerator(args.model_path, device=device)
    print("Loading DreamSim...")
    scorer = DreamSimScorer(device=device)

    prompt = "Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"

    # --- Get reference embedding from a real word ---
    print(f"Extracting '{args.ref_word}' embedding as reference direction...")
    ref_emb = get_word_token_embedding(gen.pipe, args.ref_word)  # (4096,)
    ref_emb_np = ref_emb.detach().float().cpu().numpy()
    ref_norm = np.linalg.norm(ref_emb_np)
    ref_dir = ref_emb_np / ref_norm
    print(f"  ||ref_emb||={ref_norm:.4f}")

    # Use R values based on actual token norm (~40)
    R_values = [20, 40, 80, 120]

    # --- Sample random directions in 4096-dim ---
    rng = np.random.RandomState(0)
    directions_4096 = []
    for _ in range(args.n_samples):
        x = rng.randn(4096).astype(np.float32)
        x = x / np.linalg.norm(x)
        directions_4096.append(x)

    # --- Run ---
    import pandas as pd
    all_results = []

    for R in R_values:
        print(f"\n=== R={R} ===")

        # Reference image: ref_dir * R
        z_ref = ref_dir * R
        z_ref_t = torch.from_numpy(z_ref).to(device=device, dtype=torch.float16).unsqueeze(0)
        embeds_ref = gen.encode_batch_insert(prompt, z_ref_t)
        ref_imgs = gen.generate_batch(embeds_ref, [args.seed])
        ref_img = ref_imgs[0]
        ref_img.save(os.path.join(run_dir, f"reference_R{R}.png"))
        ref_tensor = scorer.preprocess(ref_img)

        for seed in seeds:
            print(f"  seed={seed}...")
            for z_idx, d in enumerate(directions_4096):
                z = d * R  # random direction in 4096-dim, norm R
                z_dist = float(np.linalg.norm(z - z_ref))
                cos_sim = float(np.dot(d, ref_dir))

                z_t = torch.from_numpy(z).to(device=device, dtype=torch.float16).unsqueeze(0)
                embeds = gen.encode_batch_insert(prompt, z_t)
                imgs = gen.generate_batch(embeds, [seed])
                img = imgs[0]

                ds_dist = scorer.model(ref_tensor, scorer.preprocess(img)).item()
                all_results.append({
                    "R": R, "seed": seed, "z_idx": z_idx,
                    "z_dist_to_ref": z_dist, "cos_sim_to_ref": cos_sim,
                    "dreamsim_dist": ds_dist,
                })
                img.close()

        ref_img.close()

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(run_dir, "data.csv"), index=False)

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors_plt = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    n_R = len(R_values)

    # L2 distance
    fig, axes = plt.subplots(1, n_R, figsize=(5 * n_R, 5), sharey=True)
    for ax_idx, R in enumerate(R_values):
        ax = axes[ax_idx]
        subset_R = df[df["R"] == R]
        for s_idx, seed in enumerate(seeds):
            subset = subset_R[subset_R["seed"] == seed]
            ax.scatter(subset["z_dist_to_ref"], subset["dreamsim_dist"],
                       c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")
        r_val, p_val = stats.pearsonr(subset_R["z_dist_to_ref"], subset_R["dreamsim_dist"])
        ax.set_title(f"R={R}\nr={r_val:.4f}, p={p_val:.4f}")
        ax.set_xlabel("||z - z_ref|| (L2, 4096-dim)")
        if ax_idx == 0:
            ax.set_ylabel("DreamSim Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Direct 4096-dim: L2 Distance vs DreamSim", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "l2_vs_dreamsim.png"), dpi=150)

    # Cosine similarity
    fig2, axes2 = plt.subplots(1, n_R, figsize=(5 * n_R, 5), sharey=True)
    for ax_idx, R in enumerate(R_values):
        ax = axes2[ax_idx]
        subset_R = df[df["R"] == R]
        for s_idx, seed in enumerate(seeds):
            subset = subset_R[subset_R["seed"] == seed]
            ax.scatter(subset["cos_sim_to_ref"], subset["dreamsim_dist"],
                       c=colors_plt[s_idx], alpha=0.6, s=40, label=f"seed={seed}")
        r_val, p_val = stats.pearsonr(subset_R["cos_sim_to_ref"], subset_R["dreamsim_dist"])
        ax.set_title(f"R={R}\nr={r_val:.4f}, p={p_val:.4f}")
        ax.set_xlabel("Cosine Similarity (z, z_ref)")
        if ax_idx == 0:
            ax.set_ylabel("DreamSim Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig2.suptitle("Direct 4096-dim: Cosine Similarity vs DreamSim", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(run_dir, "cos_vs_dreamsim.png"), dpi=150)

    # --- Summary ---
    print("\n=== Summary (L2) ===")
    print(f"{'R':>5s} | {'r':>8s} | {'p':>10s} | per-seed r")
    for R in R_values:
        subset = df[df["R"] == R]
        r_val, p_val = stats.pearsonr(subset["z_dist_to_ref"], subset["dreamsim_dist"])
        per = []
        for seed in seeds:
            s = subset[subset["seed"] == seed]
            rs, _ = stats.pearsonr(s["z_dist_to_ref"], s["dreamsim_dist"])
            per.append(f"{rs:.3f}")
        print(f"{R:5d} | {r_val:8.4f} | {p_val:10.6f} | {', '.join(per)}")

    print("\n=== Summary (Cosine) ===")
    print(f"{'R':>5s} | {'r':>8s} | {'p':>10s} | per-seed r")
    for R in R_values:
        subset = df[df["R"] == R]
        r_val, p_val = stats.pearsonr(subset["cos_sim_to_ref"], subset["dreamsim_dist"])
        per = []
        for seed in seeds:
            s = subset[subset["seed"] == seed]
            rs, _ = stats.pearsonr(s["cos_sim_to_ref"], s["dreamsim_dist"])
            per.append(f"{rs:.3f}")
        print(f"{R:5d} | {r_val:8.4f} | {p_val:10.6f} | {', '.join(per)}")


if __name__ == "__main__":
    main()
