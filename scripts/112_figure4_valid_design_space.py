"""Figure 4: 2-D slice of the data-driven kNN validity region."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def kth_dist_batch(points, anchors, k, chunk=12000):
    out = []
    for start in range(0, len(points), chunk):
        distances = cdist(points[start:start + chunk], anchors)
        out.append(np.partition(distances, k - 1, axis=1)[:, k - 1])
    return np.concatenate(out)


def calibrate_tau(anchors, k, q):
    distances = cdist(anchors, anchors)
    np.fill_diagonal(distances, np.inf)
    loo = np.partition(distances, k - 1, axis=1)[:, k - 1]
    return float(np.quantile(loo, q)), loo


def choose_point(X, Y, dk, valid, target, ratio_range, tau):
    mask = valid & (dk / tau >= ratio_range[0]) & (dk / tau <= ratio_range[1])
    indices = np.argwhere(mask)
    if not len(indices):
        raise RuntimeError(f"No candidate for target={target}, ratios={ratio_range}")
    xy = np.column_stack([X[mask], Y[mask]])
    scale = np.array([X.max() - X.min(), Y.max() - Y.min()])
    score = np.linalg.norm((xy - np.asarray(target)) / scale, axis=1)
    row, col = indices[np.argmin(score)]
    return int(row), int(col)


def prepare(pool_path, output, k=10, q=.95, dim=16, resolution=360):
    pool = np.load(pool_path, allow_pickle=True)
    embs = pool["embs"].astype(np.float64)
    words = [str(x) for x in pool["words"]]
    pca = PCA(n_components=dim, random_state=0)
    Z = pca.fit_transform(embs)
    tau, loo = calibrate_tau(Z, k, q)

    # A wider view makes the contrast between supported and clearly off-support
    # designs visible while retaining the anchor cloud at readable scale.
    lo = np.array([-70.0, -70.0])
    hi = np.array([75.0, 70.0])
    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    grid = np.zeros((X.size, dim), dtype=np.float64)
    grid[:, 0], grid[:, 1] = X.ravel(), Y.ravel()
    dk = kth_dist_batch(grid, Z, k).reshape(X.shape)
    valid = dk <= tau

    # Fixed representative coordinates: two supported points and two farther
    # off-support points selected from a 12-direction screening sweep.
    targets = [
        ("valid_1", True, (-3.6382086361, 15.6243590729)),
        ("valid_2", True, (27.5976637970, -11.0270085283)),
        ("invalid_extreme_1", False, (65.65, 0.0)),
        ("invalid_extreme_2", False, (0.0, -64.45)),
    ]
    samples = []
    for name, is_valid, target in targets:
        z = np.zeros(dim)
        z[:2] = target
        point_dk = float(kth_dist_batch(z[None], Z, k)[0])
        samples.append({"name": name, "valid": is_valid, "pc1": z[0], "pc2": z[1],
                        "d_k": point_dk, "ratio": point_dk / tau,
                        "z": z})

    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "figure4_geometry.npz", Z=Z, words=np.array(words), X=X, Y=Y,
             dk=dk, valid=valid, tau=tau, loo=loo,
             pca_components=pca.components_, pca_mean=pca.mean_)
    with (output / "figure4_samples.json").open("w") as handle:
        json.dump([{kk: vv for kk, vv in s.items() if kk != "z"} for s in samples],
                  handle, indent=2)
    return pca, Z, X, Y, dk, valid, tau, samples


def render_missing(samples, pca, output, model_path, device, seed):
    image_dir = output / "callouts"
    image_dir.mkdir(exist_ok=True)
    missing = [s for s in samples if not (image_dir / f"{s['name']}.png").is_file()]
    if not missing:
        return
    import torch
    from src.sd35_batch_generator import SD35BatchEmbeddingGenerator
    gen = SD35BatchEmbeddingGenerator(model_path, device=device)
    lifted = pca.inverse_transform(np.stack([s["z"] for s in missing])).astype(np.float32)
    tensor = torch.tensor(lifted, dtype=torch.float16, device=device)
    encoded = gen.encode_batch_insert("", tensor)
    images = gen.generate_batch(encoded, [seed] * len(missing))
    for sample, image in zip(missing, images):
        image.save(image_dir / f"{sample['name']}.png")
        image.close()


def plot(Z, X, Y, valid, tau, samples, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(10.2, 5.2))
    ax = fig.add_axes([.07, .12, .55, .80])
    ax.contourf(X, Y, valid.astype(float), levels=[-.5, .5, 1.5],
                colors=["#FFFFFF", "#DDEEE8"], alpha=1.0)
    ax.contour(X, Y, valid.astype(float), levels=[.5], colors=["#4C8C76"],
               linewidths=1.2)
    ax.scatter(Z[:, 0], Z[:, 1], s=13, c="#333333", alpha=.62,
               linewidths=0, label="Word anchors ($M=228$)", zorder=3)

    marker = {True: ("o", "#007F5F", "Valid samples"),
              False: ("X", "#C43C39", "Invalid samples")}
    used = set()
    for sample in samples:
        m, color, label = marker[sample["valid"]]
        ax.scatter(sample["pc1"], sample["pc2"], marker=m, s=75, c=color,
                   edgecolors="white", linewidths=.8, zorder=6,
                   label=label if label not in used else None)
        used.add(label)
        short = {"valid_1": "V1", "valid_2": "V2",
                 "invalid_extreme_1": "I1", "invalid_extreme_2": "I2"}[sample["name"]]
        ax.annotate(short, (sample["pc1"], sample["pc2"]), xytext=(5, 5),
                    textcoords="offset points", color=color, weight="bold", fontsize=9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(r"kNN validity region $\mathcal{M}$ (PC1--PC2 slice)")
    ax.text(.98, .96, rf"$k=10$, $\tau_d={tau:.2f}$ (95th percentile LOO)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=2))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", frameon=True, framealpha=.92)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#DADADA", linewidth=.5, alpha=.5)
    ax.set_axisbelow(True)
    ax.set_xlim(X.min() - 2.0, X.max() + 2.0)
    ax.set_ylim(Y.min() - 2.0, Y.max() + 2.0)

    # Separate call-out panel: images never obscure the geometry or word anchors.
    positions = [(0.67, .56), (0.835, .56), (0.67, .12), (0.835, .12)]
    for sample, pos in zip(samples, positions):
        iax = fig.add_axes([pos[0], pos[1], .145, .29])
        image = np.asarray(Image.open(output / "callouts" / f"{sample['name']}.png").convert("RGB"))
        iax.imshow(image)
        iax.set_xticks([]); iax.set_yticks([])
        color = marker[sample["valid"]][1]
        for spine in iax.spines.values():
            spine.set_color(color); spine.set_linewidth(1.7)
        short = {"valid_1": "V1", "valid_2": "V2",
                 "invalid_extreme_1": "I1", "invalid_extreme_2": "I2"}[sample["name"]]
        state = "valid" if sample["valid"] else "invalid"
        iax.set_title(f"{short}: {state}  " +
                      rf"($d_{{10}}/\tau_d={sample['ratio']:.2f}$)",
                      fontsize=9, color=color, pad=4)
    fig.text(.75, .93, "Rendered samples", ha="center", va="bottom",
             fontsize=11, weight="bold")

    stem = output / "figure4_knn_valid_design_space"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_npz", default="outputs/strict_pool_s228_0429_0119/embeddings.npz")
    ap.add_argument("--model_path", default="models/stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--output_dir", default="outputs/paper_figures/figure4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1810772)
    args = ap.parse_args()
    output = Path(args.output_dir)
    pca, Z, X, Y, dk, valid, tau, samples = prepare(Path(args.pool_npz), output)
    render_missing(samples, pca, output, args.model_path, args.device, args.seed)
    plot(Z, X, Y, valid, tau, samples, output)
    print(f"M={len(Z)}, k=10, tau_d={tau:.6f}")
    for s in samples:
        print(s["name"], f"valid={s['valid']}", f"d_k/tau={s['ratio']:.3f}")
    print(output / "figure4_knn_valid_design_space.png")


if __name__ == "__main__":
    main()
