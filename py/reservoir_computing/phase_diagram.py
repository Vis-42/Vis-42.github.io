"""Phase diagram sweep for ESN hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from chaotic_systems import generate_trajectory, lyapunov_time, effective_dt
from esn import ESNConfig, EchoStateNetwork

Array = np.ndarray


@dataclass
class PhasePoint:
    sparsity: float
    spectral_radius: float
    prediction_horizon: float
    rmse: float


def _prediction_horizon(true_traj: Array, pred_traj: Array, tol: float, dt: float, lya: float) -> float:
    n = min(len(true_traj), len(pred_traj))
    ns = float(np.sqrt(np.mean(true_traj**2))) + 1e-12
    for t in range(n):
        if np.linalg.norm(true_traj[t] - pred_traj[t]) / ns > tol:
            return t * dt / lya
    return n * dt / lya


def sweep_phase_diagram(
    system: str,
    sparsity_range,
    spectral_range,
    N_reservoir: int = 50,
    T_train: float = 40.0,
    T_test: float = 8.0,
    dt: float = 0.02,
    washout: int = 100,
    tol: float = 0.5,
) -> List[PhasePoint]:
    # Lyapunov time computed directly from the dynamics (self-consistent),
    # not a hardcoded literature constant.
    lya = lyapunov_time(system, {}, dt=dt)
    dt_eff = effective_dt(system, dt)
    _, traj_full = generate_trajectory(system, {}, T_train + T_test + dt*washout, dt)
    train_len = round(T_train / dt)
    test_len  = round(T_test  / dt)
    train_data = traj_full[:train_len + washout]
    true_test  = traj_full[train_len + washout:train_len + washout + test_len]
    n_in = traj_full.shape[1]

    results = []
    for sp in sparsity_range:
        for sr in spectral_range:
            cfg = ESNConfig(N_reservoir=N_reservoir, sparsity=float(sp),
                            spectral_radius=float(sr), seed=42)
            esn = EchoStateNetwork(cfg, n_input=n_in)
            esn.train(train_data, washout)
            pred = esn.predict(test_len)
            n_cmp = min(len(pred), len(true_test))
            rmse = float(np.sqrt(np.mean((pred[:n_cmp] - true_test[:n_cmp])**2)))
            ph   = _prediction_horizon(true_test[:n_cmp], pred[:n_cmp], tol, dt_eff, lya)
            results.append(PhasePoint(float(sp), float(sr), ph, rmse))
    return results
