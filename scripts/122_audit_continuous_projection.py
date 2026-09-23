#!/usr/bin/env python3
"""CPU replay of historical Thompson directions, tracing every acquisition step.

Refits use the original saved observations, never feedback from regenerated
designs. PCA is reconstructed; selected-point discrepancy is explicitly recorded.
No rendering, policy change, or new experimental trajectory is claimed.
"""
import argparse
import json
import os
import pickle
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmts_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.cmts_sim import argmax_over_M, random_valid_design, laplace_map, project_norm, kth_dist, sigma


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=PROJECT / "outputs/cmts_a8_v0.5_lam100_bbright_d16_B8_T1000_0629_0842")
    ap.add_argument("--sim", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--output", type=Path, default=PROJECT / "outputs/continuous_diagnostics/continuous_projection_audit")
    args = ap.parse_args()
    cfg = json.loads(sorted(args.run.glob("config*.json"))[0].read_text())
    path = args.run / f"sim{args.sim:03d}"
    state = np.load(path / "posterior.npz")
    history = pd.read_csv(path / "trajectory.csv")
    np.testing.assert_array_equal(state["y"], history.y)
    n0, B, d, k, tau = (cfg[key] for key in ["n0", "B", "d", "k", "tau_d"])
    T = (len(history)-n0)//B
    if args.rounds is not None:
        T = min(T, args.rounds)
    out = args.output / f"sim{args.sim:03d}"
    out.mkdir(parents=True, exist_ok=True)
    pool = np.load(PROJECT / cfg["pool_dir"] / "embeddings.npz", allow_pickle=True)
    keep = pool["words"].astype(str) != cfg["B_word"]
    anchors = PCA(n_components=d, random_state=0).fit_transform(pool["embs"][keep].astype(np.float32)).astype(float)
    nn = np.argsort(np.linalg.norm(anchors[:,None]-anchors[None], axis=2), axis=1)[:,1:11]
    rng = np.random.default_rng(args.sim*1000+7)
    warm = np.stack([random_valid_design(anchors, k, tau, rng, nn) for _ in range(n0)])
    warm_error = float(abs(warm-(state["Phi"][:n0]+state["z_comp"])).max())
    np.testing.assert_array_equal(rng.binomial(1, history.iloc[:n0].true_p_soft), state["y"][:n0])
    beta, H = laplace_map(state["Phi"][:n0], state["y"][:n0], cfg["lam"], d)
    beta = project_norm(beta, cfg["S"])
    acquisitions, steps, draws, cosine_errors = [], [], [], []
    for t in range(T):
        inv = np.linalg.inv(H)
        sampled = rng.multivariate_normal(beta, cfg["v"]**2*.5*(inv+inv.T), size=B)
        start, stop = n0+t*B, n0+(t+1)*B
        batch = history.iloc[start:stop]
        np.testing.assert_array_equal(batch.t, np.full(B,t))
        cosine = float(np.mean((sampled @ (beta/(np.linalg.norm(beta)+1e-12))) /
                               (np.linalg.norm(sampled,axis=1)+1e-12)))
        error = abs(cosine-float(batch.ts_cos_mean.iloc[0]))
        cosine_errors.append(error)
        for b, direction in enumerate(sampled):
            diagnostic = {}
            selected = argmax_over_M(direction, anchors, k, tau, diagnostics=diagnostic)
            if t == 0:
                np.testing.assert_array_equal(selected, argmax_over_M(direction, anchors, k, tau))
            original = state["Phi"][start+b]+state["z_comp"]
            for record in diagnostic.pop("steps"):
                steps.append(dict(round=t+1, b=b, direction_cosine_error=error, **record))
            acquisitions.append(dict(round=t+1,b=b,**diagnostic,
                                     direction_cosine_error=error,
                                     historical_final_excess=float(kth_dist(original,anchors,k)-tau),
                                     replay_l2_difference=float(np.linalg.norm(selected-original))))
            draws.append(direction)
        np.testing.assert_array_equal(rng.binomial(1,batch.true_p_soft), state["y"][start:stop])
        beta,H = laplace_map(state["Phi"][:stop],state["y"][:stop],cfg["lam"],d,beta0=beta)
        beta = project_norm(beta,cfg["S"])
        np.testing.assert_allclose(sigma(state["Phi"][start:stop] @ beta),
                                   batch.predicted_p,atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(beta),batch.beta_norm.iloc[0],atol=1e-6,rtol=1e-5)
        if (t+1)%50==0 or t+1==T:
            print(f"sim{args.sim:03d}: audited {t+1}/{T} rounds ({len(steps)} intermediate steps)",flush=True)
    replay_rng_matches = None
    if n0+T*B == len(history):
        np.testing.assert_allclose(beta,state["beta_hat"],atol=1e-6,rtol=1e-5)
        with (path / "_ckpt.pkl").open("rb") as handle:
            checkpoint = pickle.load(handle)
        replay_rng_matches = bool(rng.bit_generator.state == checkpoint["rng_state"])
        assert replay_rng_matches, "Final random-generator state differs from historical checkpoint"
    a, s = pd.DataFrame(acquisitions), pd.DataFrame(steps)
    a.to_csv(out / "acquisitions.csv", index=False)
    s.to_csv(out / "intermediate_steps.csv", index=False)
    np.savez_compressed(out / "replay_geometry_and_directions.npz", anchors=anchors,
                        sampled_beta=np.array(draws).reshape(T,B,d), tau=tau,k=k)
    eps = 1e-9
    projected = s.projection_called
    bad = s.result_excess > eps
    matched = a[(a.direction_cosine_error<=1e-7) & (a.replay_l2_difference<=1e-3)]
    matched_steps = s.merge(matched[["round","b"]],on=["round","b"])
    stats = dict(run=str(args.run),sim=args.sim,rounds=T,acquisitions=len(a),steps=len(s),
                 geometry="Historical float32 PCA refitted; no bitwise historical geometry claim",
                 warm_coordinate_max_difference=warm_error,
                 max_thompson_cosine_error=max(cosine_errors),final_rng_state_matches=replay_rng_matches,
                 rounds_with_thompson_cosine_error_above_1e_minus7=int((np.array(cosine_errors)>1e-7).sum()),
                 algorithm_geometry_tolerance=eps,historical_comparison_tolerance=1e-3,
                 outside_trial_count=int(projected.sum()),
                 outside_trial_fraction=float(projected.mean()),
                 projection_result_outside_count=int((projected & bad).sum()),
                 projection_result_outside_fraction=float(bad[projected].mean()),
                 invalid_centroid_count=int((projected & (s.centroid_excess > eps)).sum()),
                 accepted_step_count=int(s.accepted.sum()),
                 accepted_outside_step_count=int((s.accepted & bad).sum()),
                 infeasible_initial_point_count=int((a.initial_excess>eps).sum()),
                 infeasible_final_point_count=int((a.final_excess>eps).sum()),
                 historical_final_excess_above_001_count=int((a.historical_final_excess>1e-3).sum()),
                 replay_final_difference_above_001_count=int((a.replay_l2_difference>1e-3).sum()),
                 max_replay_final_difference=float(a.replay_l2_difference.max()),
                 max_projection_result_excess=float(s.result_excess.max()),
                 max_final_excess=float(a.final_excess.max()),
                 historically_matched_acquisitions=len(matched),
                 historically_matched_steps=len(matched_steps),
                 historically_matched_outside_trials=int(matched_steps.projection_called.sum()),
                 historically_matched_infeasible_projection_results=int((matched_steps.result_excess>eps).sum()))
    (out / "summary.json").write_text(json.dumps(stats,indent=2))
    s.loc[bad | (s.start_excess>eps)].to_csv(out / "invalid_intermediate_steps.csv",index=False)
    a.loc[(a.final_excess>eps) | (a.historical_final_excess>1e-3)].to_csv(out / "invalid_final_points.csv",index=False)
    s.merge(a.loc[a.final_excess>eps,["round","b"]],on=["round","b"]).to_csv(
        out / "steps_for_invalid_returns.csv",index=False)
    fig, axes=plt.subplots(1,2,figsize=(12,4.5))
    rate=s.groupby("round").projection_called.mean()
    axes[0].plot(rate.index,rate,color="#0072B2",lw=1)
    axes[0].set(title="Fraction of trial steps requiring projection",xlabel="Round",ylabel="Fraction",ylim=(0,1.03))
    axes[1].plot(a["round"],a.final_excess,".",ms=2,label="Returned point")
    axes[1].axhline(0,color="black",lw=1,ls="--")
    axes[1].set(title="Feasibility of returned designs",xlabel="Round",ylabel=r"$d_k(z)-\tau$")
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle(f"Continuous solver intermediate-step audit: sim{args.sim:03d}")
    fig.tight_layout()
    for ext in ["png","pdf"]: fig.savefig(out/f"projection_audit.{ext}",dpi=220)
    plt.close(fig)
    print(json.dumps(stats,indent=2),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
