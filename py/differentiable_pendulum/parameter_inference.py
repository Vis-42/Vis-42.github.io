"""Gradient descent parameter inference for the double pendulum."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pendulum_sim import PendulumParams
from gradient_calc import trajectory_loss, finite_diff_gradient

Array = np.ndarray


@dataclass
class InferenceConfig:
    lr: float = 0.01; n_steps: int = 100
    eps_fd: float = 1e-5; momentum: float = 0.9


@dataclass
class InferenceResult:
    inferred_params: PendulumParams
    loss_history: list[float]
    param_history: Array       # n_steps × 4
    param_error_pct: Array     # 4-vector


def infer_parameters(observed: Array, u0: Array, true_params: PendulumParams,
                     tspan: tuple, dt: float,
                     cfg: InferenceConfig = None,
                     init_params: Array = None) -> InferenceResult:
    cfg = cfg or InferenceConfig()
    if init_params is None:
        rng = np.random.default_rng(7)
        tv = np.array([true_params.m1, true_params.m2, true_params.L1, true_params.L2])
        init_params = np.clip(tv * (1 + 0.3 * rng.standard_normal(4)), 0.1, 10.0)

    pv  = init_params.copy()
    vel = np.zeros(4)
    history: list[float] = []
    ph = np.zeros((cfg.n_steps, 4))

    for step in range(cfg.n_steps):
        loss = trajectory_loss(pv, observed, u0, tspan, dt)
        grad = finite_diff_gradient(pv, observed, u0, tspan, dt, eps=cfg.eps_fd)
        history.append(loss)
        ph[step] = pv
        vel = cfg.momentum * vel + (1 - cfg.momentum) * grad
        pv = np.clip(pv - cfg.lr * vel, 0.05, 20.0)

    tv = np.array([true_params.m1, true_params.m2, true_params.L1, true_params.L2])
    err = np.abs(pv - tv) / (np.abs(tv) + 1e-12) * 100

    return InferenceResult(
        PendulumParams(m1=pv[0], m2=pv[1], L1=pv[2], L2=pv[3]),
        history, ph, err
    )
