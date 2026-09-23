#!/usr/bin/env python3
"""Distinguish numerical solver initialization from an initial model belief.

Uses saved warm-start data only; does not change the prior, acquisition policy,
or claim to run a new online trajectory.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.cmts_sim import laplace_map


def main():
    evaluation = PROJECT / "outputs/continuous_diagnostics/continuous_mse_regret_v05"
    meta = json.loads((evaluation / "evaluation.json").read_text())
    root = Path(meta["run"])
    if not root.is_absolute():
        root = PROJECT / root
    data = np.load(root / "sim001/posterior.npz")
    cfg = meta["config"]
    X, y = data["Phi"][:cfg["n0"]], data["y"][:cfg["n0"]]
    lam, dim = cfg["lam"], cfg["d"]
    test = np.load(evaluation / "test_set.npz")
    X_test = test["z"] - data["z_comp"]
    truth = test["true_probability"]
    center_unit = lambda a: (a-a.mean()) / (np.linalg.norm(a-a.mean())+1e-12)
    objective = lambda b: float(np.logaddexp(0, X @ b).sum()-y @ (X @ b)+lam/2*(b @ b))
    gradient = lambda b: X.T @ (expit(X @ b)-y)+lam*b
    reference, _ = laplace_map(X, y, lam, dim)
    assert np.linalg.norm(gradient(reference)) < 1e-7
    direction = np.random.default_rng(20260920).normal(size=dim)
    direction /= np.linalg.norm(direction)
    records = []
    for scale in [0., .5, 2., 8.]:
        initial = scale*direction
        legacy, _ = laplace_map(X, y, lam, dim, beta0=initial)
        robust = minimize(objective, initial, jac=gradient, method="L-BFGS-B",
                          options={"gtol": 1e-9, "ftol": 1e-14, "maxiter": 2000})
        assert robust.success
        np.testing.assert_allclose(robust.x, reference, atol=1e-6, rtol=0)
        for name, beta in [("existing_undamped_Newton", legacy), ("L-BFGS-B_check", robust.x)]:
            prediction = expit(X_test @ beta)
            records.append(dict(solver=name, initial_norm=scale,
                                fitted_norm=float(np.linalg.norm(beta)),
                                distance_to_zero_init_MAP=float(np.linalg.norm(beta-reference)),
                                objective=objective(beta), gradient_norm=float(np.linalg.norm(gradient(beta))),
                                normalized_mse=float(np.mean((center_unit(prediction)-center_unit(truth))**2)),
                                probability_mse=float(np.mean((prediction-truth)**2))))
    prediction = expit(X_test @ reference)
    report = dict(source=str(root / "sim001"), warm_count=cfg["n0"],
                  solver_default_initial_beta="zero", prior_mean="zero", ridge_lambda=lam,
                  fitted_warm_MAP_norm=float(np.linalg.norm(reference)),
                  warm_normalized_mse=float(np.mean((center_unit(prediction)-center_unit(truth))**2)),
                  warm_probability_mse=float(np.mean((prediction-truth)**2)),
                  warm_centered_probability_cosine=float(center_unit(prediction) @ center_unit(truth)),
                  random_direction_seed=20260920,
                  interpretation="Numerical initialization only; all converged solutions agree. Large starts expose existing Newton nonconvergence, not a different posterior.")
    out = evaluation / "sim001/beta_initialization_check"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "solver_comparison.csv", index=False)
    (out / "initial_model.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
