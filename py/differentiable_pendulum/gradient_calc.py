"""Finite-difference gradient computation for pendulum parameter inference."""
import numpy as np
from pendulum_sim import PendulumParams, simulate

Array = np.ndarray


def trajectory_loss(params_vec: Array, observed: Array, u0: Array,
                    tspan: tuple, dt: float) -> float:
    m1, m2, L1, L2 = np.clip(params_vec, 0.05, 20.0)
    try:
        p = PendulumParams(m1=m1, m2=m2, L1=L1, L2=L2)
        _, sim = simulate(p, u0, tspan, dt)
        n = min(sim.shape[0], observed.shape[0])
        return float(np.mean((sim[:n] - observed[:n]) ** 2))
    except Exception:
        return 1e10


def finite_diff_gradient(params_vec: Array, observed: Array, u0: Array,
                          tspan: tuple, dt: float, eps: float = 5e-4) -> Array:
    grad = np.zeros_like(params_vec)
    for i in range(len(params_vec)):
        pv_hi = params_vec.copy(); pv_hi[i] += eps
        pv_lo = params_vec.copy(); pv_lo[i] -= eps
        grad[i] = (trajectory_loss(pv_hi, observed, u0, tspan, dt) -
                   trajectory_loss(pv_lo, observed, u0, tspan, dt)) / (2 * eps)
    return grad
