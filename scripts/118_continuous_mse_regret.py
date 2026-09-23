#!/usr/bin/env python3
"""Evaluate saved continuous runs; no new rendering or oracle optimality claim.

Replays the historical MAP updates exactly. Uses independent warm-start designs
as a common held-out test set. The approximate oracle is the best numerically
feasible evaluated design across specified saved stochastic-search trajectories.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmts_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.cmts_sim import laplace_map, project_norm, sigma

DEFAULT_RUN = PROJECT / "outputs/cmts_a8_v1_lam100_bbright_d16_B8_T1000_0629_0842"
DEFAULT_TEST = PROJECT / "outputs/cmts_lam50_v4.0_bbright_a15_d16_B8_T300_0617_1156"


def config(root):
    configs = [json.loads(p.read_text()) for p in sorted(root.glob("config*.json"))]
    if not configs:
        raise ValueError(f"No configuration in {root}")
    keys = ["d", "k", "B_word", "B_seed", "ref_seed", "D_B", "tau_d", "n0", "B",
            "alpha", "lam", "S", "v", "pool_dir", "model_path"]
    for other in configs[1:]:
        assert all(other.get(k) == configs[0].get(k) for k in keys), "Mixed configurations"
    return configs[0]


def load_run(path):
    with np.load(path / "posterior.npz") as data:
        state = {k: data[k].copy() for k in data.files}
    df = pd.read_csv(path / "trajectory.csv")
    assert len(df) == len(state["Phi"]) == len(state["y"])
    np.testing.assert_array_equal(df.y, state["y"])
    z = state["Phi"] + state["z_comp"]
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), df.z_norm, atol=1e-8)
    return state, df, z


def normalized(vector):
    centered = vector - vector.mean()
    return centered / (np.linalg.norm(centered) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--test-run", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--sims", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--test-sims", type=int, nargs="+", default=list(range(5, 10)))
    ap.add_argument("--output", type=Path, default=PROJECT / "outputs/continuous_diagnostics/continuous_mse_regret")
    ap.add_argument("--validity-tolerance", type=float, default=1e-3,
                    help="Absolute tolerance for refitting historical float32 PCA (radius about 20)")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert set(args.sims).isdisjoint(args.test_sims), "Test warm-start seeds must be independent"
    cfg, test_cfg = config(args.run), config(args.test_run)
    for key in ["d", "k", "B_word", "B_seed", "ref_seed", "D_B", "tau_d", "pool_dir", "model_path"]:
        assert cfg[key] == test_cfg[key], f"Test geometry/rendering mismatch: {key}"
    pool = np.load(PROJECT / cfg["pool_dir"] / "embeddings.npz", allow_pickle=True)
    keep = pool["words"].astype(str) != cfg["B_word"]
    pca = PCA(n_components=cfg["d"], random_state=0)
    anchors = pca.fit_transform(pool["embs"][keep].astype(np.float32)).astype(float)
    rebuilt_comp = pca.transform(pool["embs"][~keep].astype(np.float32))[0]
    neighbors = NearestNeighbors(n_neighbors=cfg["k"]).fit(anchors)

    runs, test_parts, audits, oracle_candidates = [], [], [], []
    for root, sims, is_test in [(args.run, args.sims, False), (args.test_run, args.test_sims, True)]:
        for seed in sims:
            path = root / f"sim{seed:03d}"
            state, df, z = load_run(path)
            np.testing.assert_allclose(state["z_comp"], rebuilt_comp, atol=args.validity_tolerance, rtol=0)
            excess = neighbors.kneighbors(z)[0][:, -1] - cfg["tau_d"]
            valid = excess <= args.validity_tolerance
            df["validity_excess"] = excess
            df["numerically_valid"] = valid
            df["source"] = str(path)
            df["stored_index"] = np.arange(len(df))
            feasible = df.loc[valid]
            assert not feasible.empty
            row = feasible.loc[feasible.ds_to_R.idxmin()].to_dict()
            row["z"] = z[int(row["stored_index"])].tolist()
            oracle_candidates.append(row)
            audits.append(dict(source=str(path), count=len(df), invalid_count=int((~valid).sum()),
                               max_excess=float(excess.max()),
                               pca_comp_max_difference=float(abs(state["z_comp"]-rebuilt_comp).max())))
            if is_test:
                mask = (df.phase == "warm") & valid
                test_parts.append((z[mask], df.loc[mask].copy()))
            else:
                np.testing.assert_allclose(df.true_p_soft, sigma(cfg["alpha"]*(cfg["D_B"]-df.ds_to_R)), atol=1e-10)
                runs.append((seed, state, df, z))
    test_z = np.vstack([part[0] for part in test_parts])
    test_df = pd.concat([part[1] for part in test_parts], ignore_index=True)
    test_p = sigma(cfg["alpha"]*(cfg["D_B"]-test_df.ds_to_R.to_numpy()))
    test_norm = normalized(test_p)
    for _, _, _, z in runs:
        assert NearestNeighbors(n_neighbors=1).fit(z).kneighbors(test_z)[0].min() > 1e-6, "Test/training overlap"
    oracle = min(oracle_candidates, key=lambda row: row["ds_to_R"])
    oracle_p = float(sigma(cfg["alpha"]*(cfg["D_B"]-oracle["ds_to_R"])))
    oracle["probability"] = oracle_p
    manifest = dict(config=cfg, run=str(args.run), test_run=str(args.test_run),
                    sims=args.sims, test_sims=args.test_sims, test_count=len(test_z),
                    test_distribution="Independent warm-start designs, common fixed test set",
                    normalized_mse="mean squared difference of separately centered, unit-L2 probability vectors",
                    oracle_definition="Retrospective best numerically feasible observed design across listed saved searches; no new oracle search",
                    cumulative_regret="sum over rounds AND all B designs of (fixed oracle probability - true probability); signed, not clipped",
                    validity_tolerance=args.validity_tolerance, feasibility_audit=audits, oracle=oracle)
    (args.output / "evaluation.json").write_text(json.dumps(manifest, indent=2))
    test_df["evaluation_probability"] = test_p
    test_df.to_csv(args.output / "test_set.csv", index=False)
    np.savez(args.output / "test_set.npz", z=test_z, true_probability=test_p)
    records = []
    for seed, state, df, _ in runs:
        X, y = state["Phi"], state["y"]
        warm = int((df.phase == "warm").sum())
        assert warm == cfg["n0"]
        beta, _ = laplace_map(X[:warm], y[:warm], cfg["lam"], cfg["d"])
        beta = project_norm(beta, cfg["S"])
        cumulative, count = 0., warm
        round_ids = sorted(df.loc[df.phase == "main", "t"].unique())
        assert round_ids == list(range(len(round_ids)))
        for t, batch in df[df.phase == "main"].groupby("t", sort=True):
            assert len(batch) == cfg["B"]
            np.testing.assert_array_equal(batch.index, np.arange(count, count+len(batch)))
            count += len(batch)
            beta, _ = laplace_map(X[:count], y[:count], cfg["lam"], cfg["d"], beta0=beta)
            beta = project_norm(beta, cfg["S"])
            np.testing.assert_allclose(np.linalg.norm(beta), batch.beta_norm.iloc[0], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(sigma(X[batch.index] @ beta), batch.predicted_p, atol=1e-6, rtol=1e-5)
            pred = sigma((test_z-state["z_comp"]) @ beta)
            cumulative += float((oracle_p-batch.true_p_soft).sum())
            records.append(dict(sim=seed, round=int(t)+1,
                                norm_mse=float(np.mean((normalized(pred)-test_norm)**2)),
                                probability_mse=float(np.mean((pred-test_p)**2)),
                                cumulative_regret=cumulative,
                                invalid_count=int((~batch.numerically_valid).sum())))
            if (t+1) % 250 == 0:
                print(f"sim{seed:03d} replay verified through round {t+1}", flush=True)
        np.testing.assert_allclose(beta, state["beta_hat"], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(cumulative, (oracle_p-df.loc[df.phase=="main", "true_p_soft"]).sum())
        pd.DataFrame(records).to_csv(args.output / "metrics.csv", index=False)
    result = pd.DataFrame(records)
    common_T = int(result.groupby("sim")["round"].max().min())
    result = result[result["round"] <= common_T]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    label = rf"CM-TS ($\alpha={cfg['alpha']:g}$, $v={cfg['v']:g}$, $\lambda={cfg['lam']:g}$)"
    for ax, metric, title in zip(axes, ["norm_mse", "cumulative_regret"],
                                 ["Normalized MSE", "Approximate Cumulative Regret"]):
        curves = result.pivot(index="round", columns="sim", values=metric)
        mean = curves.mean(axis=1)
        se = curves.std(axis=1)/np.sqrt(curves.shape[1])
        ax.plot(mean.index, mean, color="#0072B2", lw=2, label=label)
        if curves.shape[1] > 1:
            ax.fill_between(mean.index, mean-se, mean+se, color="#0072B2", alpha=.15)
        ax.set(title=title, ylabel=title, xlabel=f"Round ({cfg['B']} designs per round)")
        ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), frameon=False)
    fig.suptitle("Continuous CM-TS: Held-out Prediction and Approximate Regret", y=.99)
    fig.text(.5, .015, "Fixed independent warm-start test set; reference: best feasible observed design in saved searches.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.085, right=.98, bottom=.16, top=.77, wspace=.27)
    for extension in ["png", "pdf"]:
        fig.savefig(args.output / f"continuous_normalized_mse_regret.{extension}", dpi=250)
    plt.close(fig)
    print(f"Saved {args.output}; test N={len(test_z)}, approximate oracle p={oracle_p:.6f}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
