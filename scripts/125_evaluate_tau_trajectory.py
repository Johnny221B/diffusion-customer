#!/usr/bin/env python3
"""Evaluate one completed radius-sweep trajectory against a fixed test set."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmts_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.cmts_sim import laplace_map, project_norm, sigma


def normalized(p):
    centered = p-p.mean()
    return centered/(np.linalg.norm(centered)+1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=PROJECT / "outputs/cmts_tau_sweep_20260922_a8_v05_lam100/tau1.25")
    ap.add_argument("--sim", type=int, default=0)
    ap.add_argument("--reference-json", type=Path,
                    help="Freeze the oracle from an earlier evaluation for longitudinal comparison")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--from-checkpoint", action="store_true",
                    help="Evaluate an atomic snapshot of an actively running trajectory")
    args = ap.parse_args()
    cfg = json.loads((args.run / "config_partial0.json").read_text())
    geometry = np.load(args.run / "geometry_partial0.npz")
    if args.from_checkpoint:
        with (args.run / f"sim{args.sim:03d}/_ckpt.pkl").open("rb") as handle:
            saved = pickle.load(handle)
        state = {key: saved[key] for key in ["Phi", "y", "beta_hat", "H"]}
        state["z_comp"] = geometry["z_comp"]
        history = pd.DataFrame(saved["rows"])
        assert len(history) == cfg["n0"] + saved["t_done"]*cfg["B"]
    else:
        state = np.load(args.run / f"sim{args.sim:03d}/posterior.npz")
        history = pd.read_csv(args.run / f"sim{args.sim:03d}/trajectory.csv")
    np.testing.assert_array_equal(history.y, state["y"])
    np.testing.assert_array_equal(geometry["z_comp"], state["z_comp"])
    test_dir = PROJECT / "outputs/continuous_diagnostics/continuous_mse_regret_v05"
    old_test = np.load(test_dir / "test_set.npz")
    test_meta = json.loads((test_dir / "evaluation.json").read_text())
    for key in ["alpha", "D_B", "B_word", "B_seed", "ref_seed", "d", "k", "pool_dir", "model_path"]:
        assert cfg[key] == test_meta["config"][key], key
    # Old tests predate saved PCA bases. Reconstruct that basis, then map into
    # this run's exact saved basis. Explicitly quantify this numerical limitation.
    pool = np.load(PROJECT / cfg["pool_dir"] / "embeddings.npz", allow_pickle=True)
    keep = pool["words"].astype(str) != cfg["B_word"]
    old_pca = PCA(n_components=cfg["d"], random_state=0).fit(pool["embs"][keep].astype(np.float32))
    reconstructed_comp = old_pca.transform(pool["embs"][~keep].astype(np.float32))[0]
    test_sources = pd.read_csv(test_dir / "test_set.csv").source.unique()
    for path in test_sources:
        test_state = np.load(Path(path) / "posterior.npz")
        np.testing.assert_allclose(reconstructed_comp,test_state["z_comp"],atol=1e-3,rtol=0)
    mapped_test = ((old_pca.inverse_transform(old_test["z"])-geometry["pca_mean"])
                   @ geometry["pca_components"].T)
    delta = float(abs(mapped_test-old_test["z"]).max())
    assert delta < 1e-3, "Unexpected PCA coordinate shift"
    true_p = old_test["true_probability"]
    neighbors = NearestNeighbors(n_neighbors=cfg["k"]).fit(geometry["anchors"])
    assert np.all(neighbors.kneighbors(mapped_test)[0][:,-1] <= cfg["tau_d"])
    training_z = state["Phi"]+state["z_comp"]
    assert NearestNeighbors(n_neighbors=1).fit(training_z).kneighbors(mapped_test)[0].min() > 1e-3
    validity_excess = neighbors.kneighbors(training_z)[0][:,-1]-cfg["tau_d"]

    # Snapshot all currently saved searches in this same domain, including the
    # selected run. The reference stays fixed throughout this reported curve.
    oracle_candidates, snapshot = [], []
    for sim in sorted(args.run.glob("sim*")):
        ckpt = sim / "_ckpt.pkl"
        if not ckpt.exists():
            continue
        with ckpt.open("rb") as f:
            saved = pickle.load(f)
        df = pd.DataFrame(saved["rows"])
        z = saved["Phi"]+geometry["z_comp"]
        np.testing.assert_array_equal(df.y,saved["y"])
        excess = neighbors.kneighbors(z)[0][:,-1]-cfg["tau_d"]
        valid = excess <= 1e-8
        idx = int(np.flatnonzero(valid)[np.argmin(df.loc[valid,"ds_to_R"].to_numpy())])
        row = df.iloc[idx].to_dict()
        row.update(source=str(sim), index=idx, z=z[idx].tolist(), excess=float(excess[idx]),
                   probability=float(sigma(cfg["alpha"]*(cfg["D_B"]-row["ds_to_R"]))))
        oracle_candidates.append(row)
        snapshot.append(dict(source=str(sim),rounds=saved["t_done"],observations=len(df),invalid_count=int((~valid).sum())))
    oracle = max(oracle_candidates,key=lambda r:r["probability"])
    if args.reference_json:
        reference = json.loads(args.reference_json.read_text())
        for key in ["alpha", "D_B", "B_word", "B_seed", "ref_seed", "d", "k", "tau_d"]:
            assert cfg[key] == reference["config"][key], f"Frozen reference mismatch: {key}"
        oracle = reference["oracle"]
        assert neighbors.kneighbors(np.asarray(oracle["z"])[None])[0][0,-1] <= cfg["tau_d"]+1e-8
    X, y = state["Phi"],state["y"]
    n0,B = cfg["n0"],cfg["B"]
    T = (len(y)-n0)//B
    beta = None
    cumulative = 0.
    records = []
    for t in range(T+1):
        end = n0+t*B
        beta,_ = laplace_map(X[:end],y[:end],cfg["lam"],cfg["d"],beta0=beta)
        beta = project_norm(beta,cfg["S"])
        pred = sigma((mapped_test-state["z_comp"]) @ beta)
        unmapped_pred = sigma((old_test["z"]-state["z_comp"]) @ beta)
        if t:
            batch = history.iloc[end-B:end]
            np.testing.assert_array_equal(batch.t,np.full(B,t-1))
            np.testing.assert_allclose(sigma(X[end-B:end] @ beta),batch.predicted_p,atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(batch.true_p_soft,sigma(cfg["alpha"]*(cfg["D_B"]-batch.ds_to_R)),atol=1e-10)
            cumulative += float((oracle["probability"]-batch.true_p_soft).sum())
        records.append(dict(round=t,norm_mse=float(np.mean((normalized(pred)-normalized(true_p))**2)),
                            probability_mse=float(np.mean((pred-true_p)**2)),
                            cumulative_regret=cumulative,
                            test_coordinate_sensitivity=float(np.max(abs(pred-unmapped_pred)))))
    np.testing.assert_allclose(beta,state["beta_hat"],atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(cumulative,(oracle["probability"]-history.loc[history.phase=="main","true_p_soft"]).sum())
    result = pd.DataFrame(records)
    out = args.output or PROJECT / "outputs/continuous_diagnostics" / f"continuous_tau{cfg['tau_scale']:g}_sim{args.sim:03d}"
    out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out / "metrics.csv",index=False)
    np.savez(out / "test_set.npz",z=mapped_test,true_probability=true_p)
    summary = dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),config=cfg,selected_sim=args.sim,
                   from_checkpoint=args.from_checkpoint,
                   frozen_reference_source=str(args.reference_json) if args.reference_json else None,
                   rounds=T,test_count=len(true_p),oracle=oracle,oracle_search_snapshot=snapshot,
                   test_basis_limitation="Old test PCA reconstructed then mapped to exact new PCA; reuses saved rendering probabilities, not new rendering",
                   max_test_coordinate_shift=delta,max_prediction_coordinate_sensitivity=float(result.test_coordinate_sensitivity.max()),
                   invalid_selected_designs=int((validity_excess[n0:]>1e-8).sum()),
                   initial=result.iloc[0].to_dict(),final=result.iloc[-1].to_dict(),
                   first20=result[result["round"].between(1,20)].mean().to_dict(),last20=result.tail(20).mean().to_dict(),
                   average_per_design_regret=cumulative/(T*B))
    (out / "evaluation.json").write_text(json.dumps(summary,indent=2))
    fig,axes = plt.subplots(1,2,figsize=(12,4.8))
    for ax,column,title in zip(axes,["norm_mse","cumulative_regret"],["Normalized MSE","Approximate Cumulative Regret"]):
        ax.plot(result["round"],result[column],color="#0072B2",lw=1.8)
        ax.set(title=title,ylabel=title,xlabel=f"Round ({B} designs per round)")
        ax.grid(alpha=.25)
    fig.suptitle(rf"Selected continuous trajectory: $\tau\times{cfg['tau_scale']:g}$, sim{args.sim:03d}",y=.97)
    fig.text(.5,.02,f"Fixed test set (N={len(true_p)}); observed-search reference p={oracle['probability']:.4f}; single selected run.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.085,right=.985,bottom=.16,top=.8,wspace=.27)
    for ext in ["png","pdf"]:fig.savefig(out/f"normalized_mse_regret.{ext}",dpi=250)
    plt.close(fig)
    print(json.dumps({k:summary[k] for k in ["rounds","invalid_selected_designs","max_test_coordinate_shift","max_prediction_coordinate_sensitivity","initial","final","first20","last20","average_per_design_regret"]},indent=2))
    print('Oracle:',oracle['probability'],'from',oracle['source'], 'stored round',oracle['t'])
    print(out)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
