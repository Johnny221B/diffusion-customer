"""Render BO trajectory image grids.

For each (sim_seed, model) we pull the picked_idx per step from the BO CSV,
sample 8 random seeds of that word's 40, and tile the images into a grid
where rows = selected time points and cols = [ref, 8 picked-word seeds].

Usage:
  python scripts/64_render_bo_trajectory.py \
      --run_dir outputs/bo_thompson_bcanvas_0512_1424 \
      --imgs_dir outputs/multiseed_s228_M40_0510_0241/imgs \
      --raw_csv outputs/strict_pool_s228_0429_0119/raw_data.csv \
      --sim_seeds 0,33,66 \
      --steps 1,40,80,120,160,200 \
      --B 8 \
      --models logistic_bayesian,logistic_l2,poly2_logistic,gp_rbf,random_forest
"""

import argparse
import json
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_image(imgs_dir, orig_idx, word, seed_k):
    fname = f"{orig_idx:03d}_{word}_seed{seed_k:02d}.png"
    p = os.path.join(imgs_dir, fname)
    if os.path.exists(p):
        return p
    # fallback: try without leading zeros in seed
    fname2 = f"{orig_idx:03d}_{word}_seed{seed_k}.png"
    p2 = os.path.join(imgs_dir, fname2)
    if os.path.exists(p2):
        return p2
    return None


def render_one_trajectory(df_one, words_kept, word_to_orig, imgs_dir, ref_path,
                          steps, B, rng, out_path,
                          sim_seed, model, ora):
    """df_one: rows for one (sim_seed, model), columns: step, picked_idx, p_oracle, etc."""
    df_one = df_one.set_index("step")
    n_rows = len(steps)
    n_cols = 1 + B  # ref + B images
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    ref_img = Image.open(ref_path)

    p_soft = ora.get("p_soft")  # may be None if not stored

    for r, t in enumerate(steps):
        if t not in df_one.index:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        row = df_one.loc[t]
        picked_idx = int(row["picked_idx"])
        word = words_kept[picked_idx]
        orig_idx = word_to_orig[word]
        p_show = float(row.get("p_soft", row.get("p_oracle", np.nan)))

        # ref column
        ax = axes[r, 0]
        ax.imshow(ref_img)
        ax.axis("off")
        ax.set_title(f"t={t}\nref (canvas)", fontsize=8)

        # B random seeds of this word
        seeds_k = rng.choice(40, size=B, replace=False)
        for c, sk in enumerate(seeds_k):
            ax = axes[r, c + 1]
            ip = find_image(imgs_dir, orig_idx, word, int(sk))
            if ip is None:
                ax.text(0.5, 0.5, f"missing\n{word}_s{sk}",
                        ha="center", va="center", fontsize=6)
                ax.axis("off")
                continue
            ax.imshow(Image.open(ip))
            ax.axis("off")
            if c == 0:
                ax.set_title(f"{word}  s{sk:02d}\np_oracle={p_show:.2f}",
                             fontsize=8)
            else:
                ax.set_title(f"s{sk:02d}", fontsize=8)

    fig.suptitle(f"BO trajectory — model={model}, sim_seed={sim_seed}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="BO run output dir containing bo_per_step*.csv + oracle.npz")
    p.add_argument("--imgs_dir", required=True)
    p.add_argument("--raw_csv", required=True,
                   help="raw_data.csv used to map word -> original 3-digit idx")
    p.add_argument("--sim_seeds", default="0,33,66")
    p.add_argument("--steps", default="1,40,80,120,160,200")
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--models", default="logistic_bayesian,logistic_l2,"
                                       "poly2_logistic,gp_rbf,random_forest")
    p.add_argument("--ref_seed", type=int, default=34)
    p.add_argument("--ref_word", type=str, default="canvas")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sims  = [int(x) for x in args.sim_seeds.split(",")]
    steps = [int(x) for x in args.steps.split(",")]
    models = args.models.split(",")

    # Load BO CSV (prefer softoracle if present)
    soft_csv = os.path.join(args.run_dir, "bo_per_step_softoracle.csv")
    csv_path = soft_csv if os.path.exists(soft_csv) else os.path.join(args.run_dir, "bo_per_step.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}  rows={len(df)}")

    # words_kept from oracle.npz
    ora = np.load(os.path.join(args.run_dir, "oracle.npz"))
    words_kept = [str(w) for w in ora["words"]]

    # Original 3-digit index per word from raw_data.csv
    raw = pd.read_csv(args.raw_csv)
    word_to_orig = {w: i for i, w in enumerate(raw["word"].astype(str).tolist())}
    for w in words_kept:
        assert w in word_to_orig, f"word {w!r} not in raw_data.csv"

    # Reference image path
    ref_orig = word_to_orig[args.ref_word]
    ref_path = os.path.join(
        args.imgs_dir, f"{ref_orig:03d}_{args.ref_word}_seed{args.ref_seed:02d}.png"
    )
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"ref image missing: {ref_path}")
    print(f"Reference: {ref_path}")

    out_dir = os.path.join(args.run_dir, "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "sims": sims, "steps": steps, "B": args.B, "models": models,
        "csv": csv_path, "ref": ref_path,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    p_soft_avail = "p_soft" in df.columns

    for sim_seed in sims:
        for model in models:
            sub = df[(df["sim_seed"] == sim_seed) & (df["model"] == model)]
            if sub.empty:
                print(f"  skip: sim_seed={sim_seed}, model={model} (no rows)")
                continue
            rng = np.random.RandomState(args.seed + sim_seed * 31)
            ora_meta = {"p_soft": p_soft_avail}
            out_path = os.path.join(
                out_dir, f"sim{sim_seed:03d}_{model}.png"
            )
            render_one_trajectory(
                sub, words_kept, word_to_orig, args.imgs_dir, ref_path,
                steps, args.B, rng, out_path, sim_seed, model, ora_meta,
            )
            print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
