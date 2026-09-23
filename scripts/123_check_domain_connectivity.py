#!/usr/bin/env python3
"""Check straight paths to a common center via exact ball/segment intervals.

This tests connectivity of observed points, not of the entire continuous domain.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]


def segment_valid(x, target, anchors, radius, k):
    """Is every point on this segment covered by at least k closed balls?"""
    v = target-x
    a = v @ v
    if a < 1e-20:
        return np.sum(np.linalg.norm(anchors-x, axis=1) <= radius) >= k
    delta = x-anchors
    b = 2*(delta @ v)
    c = np.sum(delta*delta,axis=1)-radius**2
    disc = b*b-4*a*c
    real = disc >= 0
    roots = np.sqrt(disc[real])
    starts = np.maximum(0,(-b[real]-roots)/(2*a))
    ends = np.minimum(1,(-b[real]+roots)/(2*a))
    keep = ends >= starts
    starts, ends = starts[keep], ends[keep]
    cuts = np.unique(np.r_[0,1,starts,ends])
    mid = (cuts[:-1]+cuts[1:])/2
    coverage = (np.searchsorted(np.sort(starts),mid,side="right")
                - np.searchsorted(np.sort(ends),mid,side="left"))
    return bool(coverage.min() >= k)


def main():
    geometry = np.load(PROJECT / "outputs/continuous_diagnostics/continuous_projection_audit/sim001/replay_geometry_and_directions.npz")
    Z, tau, k = geometry["anchors"], float(geometry["tau"]), int(geometry["k"])
    run = PROJECT / "outputs/cmts_a8_v0.5_lam100_bbright_d16_B8_T1000_0629_0842/sim001"
    state = np.load(run / "posterior.npz")
    history = pd.read_csv(run / "trajectory.csv")
    points = (state["Phi"]+state["z_comp"])[history.phase=="main"]
    center = Z.mean(0)
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    distances = nn.kneighbors(points)[0][:,-1]
    anchor_distances = nn.kneighbors(Z)[0][:,-1]
    rows = []
    for scale in [1.,1.1,1.25,1.5]:
        radius = tau*scale
        for tolerance in [0.,1e-3]:
            valid = distances <= radius+tolerance
            connected = sum(segment_valid(z,center,Z,radius+tolerance,k) for z in points[valid])
            rows.append(dict(tau_scale=scale,radius=radius,tolerance=tolerance,
                             valid_anchors=int((anchor_distances<=radius+tolerance).sum()),
                             valid_observed_points=int(valid.sum()),
                             valid_straight_paths_to_center=int(connected)))
    nearest = nn.kneighbors(points, n_neighbors=1, return_distance=False)[:,0]
    info = dict(source=str(run),center=center.tolist(),
                center_kth_distance=float(nn.kneighbors(center[None])[0][0,-1]),
                distinct_nearest_anchors_first100rounds=int(len(np.unique(nearest[:800]))),
                distinct_nearest_anchors_last100rounds=int(len(np.unique(nearest[-800:]))),
                limitation="Connectivity of observed designs via the anchor mean only; not a proof of full-domain connectivity. PCA reconstructed; tolerances reported separately.")
    out = PROJECT / "outputs/continuous_diagnostics/continuous_domain_geometry"
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "radius_connectivity.csv",index=False)
    (out / "summary.json").write_text(json.dumps(info,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(info,indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
