"""Pilot: does quality-filtering reduce within-word DreamSim noise WITHOUT biasing it?

Motivation: within-word seed noise (std~0.073) >> across-word signal (std~0.032),
which caps how separable the words can be. If broken/low-quality images are
DreamSim outliers, dropping them should shrink the within-word variance (noise)
while leaving the per-word mean (signal) put. But if the quality score correlates
with the DreamSim-to-R distance, filtering LEAKS the label and biases the oracle.

For each word: generate N seeds, score each image's intrinsic quality with CLIP
(orthogonal to R), keep the top-`keep`, and check:
  1. corr(Q, D) over all images               -> must be ~0  (no global bias)
  2. per-word mean(D | top) vs mean(D | all)   -> must barely move (no per-word bias)
  3. per-word std(D | top) vs std(D | all)     -> should DROP (noise removed)
  4. compare (3) against random-`keep`         -> drop must beat random subsampling

Quality Q = CLIP(img, POS) - CLIP(img, NEG): purely "is this a clean shoe photo",
never similarity to the reference R.

Multi-GPU: deterministic word grid (evenly spaced over current per-word mean dist),
sharded by --shard_id/--n_shards. Each shard dumps pilot_partial_<id>.csv into a
shared --out_dir; then run once more with --analyze to merge + plot.

Usage:
  # 4-GPU sharded generation (one per GPU), shared out_dir
  for g in 0 1 2 3; do
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$g python scripts/67_quality_filter_pilot.py \
      --device cuda:0 --n_words 40 --n_shards 4 --shard_id $g \
      --out_dir outputs/qual_pilot_4gpu_<stamp> &
  done; wait
  # merge + analysis (CPU)
  python scripts/67_quality_filter_pilot.py --analyze --out_dir outputs/qual_pilot_4gpu_<stamp>
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

POS = "a clear product photo of a single shoe on a white background"
NEG = "a blurry distorted glitchy broken corrupted image"


def pick_words(pool_npz, dreams_npz, n_words):
    """Evenly-spaced global word indices across the current per-word mean distance."""
    pool = np.load(pool_npz, allow_pickle=True)
    words_all = list(pool["words"])
    dreams = np.load(dreams_npz, allow_pickle=True)["dreams"].astype(np.float64)
    valid = ~np.any(np.isnan(dreams), axis=1)
    mean_d = np.where(valid, dreams.mean(1), np.nan)
    valid_idx = np.where(valid)[0]
    sorted_global = valid_idx[np.argsort(mean_d[valid])]
    picks = sorted_global[np.linspace(0, len(sorted_global) - 1, n_words).astype(int)]
    return picks, words_all, mean_d, pool["embs"]


def generate_shard(args):
    import torch
    from PIL import Image
    from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
    from src.scorer import DreamSimScorer, CLIPScorer

    picks_all, words_all, mean_d, embs_all = pick_words(args.pool_npz, args.dreams_npz, args.n_words)
    picks = picks_all[args.shard_id::args.n_shards] if args.n_shards > 1 else picks_all

    img_dir = os.path.join(args.out_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    print(f"[shard {args.shard_id}/{args.n_shards}] words: "
          + ", ".join(f"{words_all[g]}({mean_d[g]:.3f})" for g in picks), flush=True)

    gen = SD35BatchEmbeddingGenerator(args.model_path, device=args.device)
    scorer = DreamSimScorer(device=args.device)
    clip = CLIPScorer(device=args.device)
    ref_tensor = scorer.preprocess(Image.open(args.ref_png).convert("RGB"))

    seeds = list(range(args.n_seeds))
    rows = []
    for gi in picks:
        w = str(words_all[gi])
        emb_t = torch.from_numpy(embs_all[gi].astype(np.float32)).to(
            args.device, dtype=torch.float16).unsqueeze(0)
        for cs in range(0, args.n_seeds, args.gen_batch):
            chunk = seeds[cs:cs + args.gen_batch]
            emb_chunk = emb_t.expand(len(chunk), -1).contiguous()
            embeds = gen.encode_batch_insert(prompt="", z_vectors_batch=emb_chunk)
            imgs = gen.generate_batch(embeds, chunk)
            for s, img in zip(chunk, imgs):
                D = scorer.model(ref_tensor, scorer.preprocess(img)).item()
                qp, qn = clip(img, POS), clip(img, NEG)
                rows.append({"gid": int(gi), "word": w, "seed": int(s), "D": float(D),
                             "clip_pos": qp, "clip_neg": qn, "Q": float(qp - qn),
                             "mean_d_old": float(mean_d[gi])})
                img.resize((128, 128), Image.LANCZOS).save(
                    os.path.join(img_dir, f"{gi:03d}_{w}_seed{s:02d}.png"))
                img.close()
        sub = pd.DataFrame([r for r in rows if r["gid"] == gi])
        print(f"[shard {args.shard_id}] '{w}': D_mean={sub.D.mean():.4f} D_std={sub.D.std():.4f} "
              f"Q[{sub.Q.min():.3f},{sub.Q.max():.3f}]", flush=True)

    pid = args.shard_id if args.n_shards > 1 else "single"
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, f"pilot_partial_{pid}.csv"), index=False)
    print(f"[shard {args.shard_id}] wrote pilot_partial_{pid}.csv ({len(df)} imgs)", flush=True)
    return df


def analyze(args):
    parts = sorted(glob.glob(os.path.join(args.out_dir, "pilot_partial_*.csv")))
    if not parts:
        raise SystemExit(f"no pilot_partial_*.csv in {args.out_dir}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["gid", "seed"])
    df.to_csv(os.path.join(args.out_dir, "pilot.csv"), index=False)
    img_dir = os.path.join(args.out_dir, "imgs")
    gids = sorted(df["gid"].unique(), key=lambda g: df[df.gid == g]["mean_d_old"].iloc[0])
    print(f"Analyzing {len(df)} images over {len(gids)} words from {len(parts)} shards.")

    rng = np.random.RandomState(0)
    r_pear, p_pear = stats.pearsonr(df["Q"], df["D"])
    r_spear, p_spear = stats.spearmanr(df["Q"], df["D"])
    r_pos, _ = stats.pearsonr(df["clip_pos"], df["D"])

    per_word = []
    for gi in gids:
        sub = df[df["gid"] == gi]
        d_all = sub["D"].values
        d_top = sub.nlargest(args.keep, "Q")["D"].values
        rm, rs = [], []
        for _ in range(args.n_rand):
            idx = rng.choice(len(d_all), size=min(args.keep, len(d_all)), replace=False)
            rm.append(d_all[idx].mean()); rs.append(d_all[idx].std())
        per_word.append({
            "word": sub["word"].iloc[0], "n": len(d_all),
            "mean_all": d_all.mean(), "mean_top": d_top.mean(),
            "mean_shift": d_top.mean() - d_all.mean(),
            "mean_shift_rand": float(np.mean(rm)) - d_all.mean(),
            "std_all": d_all.std(), "std_top": d_top.std(), "std_rand": float(np.mean(rs)),
            "std_drop_frac": 1 - d_top.std() / d_all.std(),
            "std_drop_frac_rand": 1 - float(np.mean(rs)) / d_all.std(),
        })
    pw = pd.DataFrame(per_word)
    pw.to_csv(os.path.join(args.out_dir, "per_word.csv"), index=False)

    summary = {
        "n_images": len(df), "n_words": len(gids), "keep": args.keep,
        "corr_Q_D_pearson": [float(r_pear), float(p_pear)],
        "corr_Q_D_spearman": [float(r_spear), float(p_spear)],
        "corr_clippos_D_pearson": float(r_pos),
        "mean_abs_shift_top": float(pw["mean_shift"].abs().mean()),
        "mean_abs_shift_rand": float(pw["mean_shift_rand"].abs().mean()),
        "avg_std_drop_top": float(pw["std_drop_frac"].mean()),
        "avg_std_drop_rand": float(pw["std_drop_frac_rand"].mean()),
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)

    print("\n=== PILOT VERDICT ===")
    print(f"corr(Q,D): pearson r={r_pear:.3f} (p={p_pear:.3g}), spearman={r_spear:.3f}  [want ~0]")
    print(f"per-word mean |shift| after top-{args.keep}: {summary['mean_abs_shift_top']:.4f} "
          f"(random {summary['mean_abs_shift_rand']:.4f})  [want filter ~ random]")
    print(f"within-word std drop: filter={summary['avg_std_drop_top']*100:.1f}% "
          f"vs random={summary['avg_std_drop_rand']*100:.1f}%  [want filter >> random]")
    print(pw.round(4).to_string(index=False))

    # plot 1: Q vs D scatter
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(df["Q"], df["D"], s=10, alpha=0.4, c=df["mean_d_old"], cmap="viridis")
    ax.set_xlabel("quality Q = CLIP_pos - CLIP_neg  (higher = cleaner shoe)")
    ax.set_ylabel("DreamSim distance D to reference R")
    ax.set_title(f"Orthogonality check ({len(df)} imgs): "
                 f"pearson={r_pear:.3f}, spearman={r_spear:.3f}\n(near 0 = filter does NOT leak the label)")
    fig.colorbar(sc, ax=ax, label="word mean dist")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "Q_vs_D_scatter.png"), dpi=140); plt.close(fig)

    # plot 2: per-word std all vs top vs random
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(pw)), 5))
    x = np.arange(len(pw))
    ax.bar(x - 0.25, pw["std_all"], 0.25, label="all seeds", color="tab:gray")
    ax.bar(x, pw["std_top"], 0.25, label=f"top-{args.keep} by Q", color="tab:green")
    ax.bar(x + 0.25, pw["std_rand"], 0.25, label=f"random-{args.keep}", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(pw["word"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("within-word std of D"); ax.legend()
    ax.set_title("Noise reduction: quality-filter vs random subsample")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "within_word_std.png"), dpi=140); plt.close(fig)

    # plot 3: montages (lowest-Q vs highest-Q) for first 4 words with thumbnails
    from PIL import Image, ImageDraw
    done = 0
    for gi in gids:
        sub = df[df["gid"] == gi].sort_values("Q")
        w = sub["word"].iloc[0]
        paths_lo = [os.path.join(img_dir, f"{gi:03d}_{w}_seed{s:02d}.png") for s in sub.head(6)["seed"]]
        paths_hi = [os.path.join(img_dir, f"{gi:03d}_{w}_seed{s:02d}.png") for s in sub.tail(6)["seed"]]
        if not all(os.path.exists(p) for p in paths_lo + paths_hi):
            continue
        canvas = Image.new("RGB", (6 * 130, 2 * 150 + 30), "white")
        ImageDraw.Draw(canvas).text((5, 2), f"{w}: TOP = lowest Q (filtered) | BOTTOM = highest Q", fill="black")
        for col, p in enumerate(paths_lo):
            canvas.paste(Image.open(p), (col * 130 + 1, 20))
        for col, p in enumerate(paths_hi):
            canvas.paste(Image.open(p), (col * 130 + 1, 170))
        canvas.save(os.path.join(args.out_dir, f"montage_{w}.png"))
        done += 1
        if done >= 4:
            break
    print(f"\nSaved -> {args.out_dir}/  (Q_vs_D_scatter.png, within_word_std.png, montage_*.png, per_word.csv, summary.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool_npz", default="outputs/strict_pool_s228_0429_0119/embeddings.npz")
    ap.add_argument("--ref_png", default="outputs/strict_pool_s228_0429_0119/reference.png")
    ap.add_argument("--dreams_npz", default="outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz")
    ap.add_argument("--n_words", type=int, default=8)
    ap.add_argument("--n_seeds", type=int, default=80)
    ap.add_argument("--keep", type=int, default=40)
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--n_rand", type=int, default=200)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    if not args.out_dir:
        args.out_dir = os.path.join("outputs", f"qual_pilot_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, f"config_{'analyze' if args.analyze else args.shard_id}.json"), "w"), indent=2)

    if args.analyze:
        analyze(args)
    else:
        generate_shard(args)
        if args.n_shards == 1:
            analyze(args)


if __name__ == "__main__":
    main()
