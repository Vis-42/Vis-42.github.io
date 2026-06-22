"""Chaotic systems: continuous flows (RK4) and discrete maps (iterated).

Continuous flows return a derivative du/dt and are integrated with RK4.
Discrete maps (Hénon) return the next state u_{n+1} directly and are iterated —
integrating a map with RK4 is meaningless.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

Array = np.ndarray


def _rk4_step(f, u: Array, t: float, dt: float, p: Dict) -> Array:
    k1 = f(u, t, p)
    k2 = f(u + 0.5*dt*k1, t+0.5*dt, p)
    k3 = f(u + 0.5*dt*k2, t+0.5*dt, p)
    k4 = f(u + dt*k3, t+dt, p)
    return u + (dt/6)*(k1+2*k2+2*k3+k4)


def lorenz63(u: Array, t: float, p: Dict) -> Array:
    x, y, z = u
    return np.array([p.get("sigma", 10)*(y-x),
                     x*(p.get("rho", 28)-z)-y,
                     x*y-p.get("beta", 8/3)*z])


def rossler(u: Array, t: float, p: Dict) -> Array:
    x, y, z = u
    a = p.get("a", 0.2); b = p.get("b", 0.2); c = p.get("c", 5.7)
    return np.array([-y-z, x+a*y, b+z*(x-c)])


def henon(u: Array, t: float, p: Dict) -> Array:
    """Hénon map — returns the NEXT state, not a derivative (discrete-time)."""
    x, y = u
    a = p.get("a", 1.4); b = p.get("b", 0.3)
    return np.array([1-a*x**2+y, b*x])


SYSTEMS = {"lorenz63": lorenz63, "rossler": rossler, "henon": henon}
DEFAULT_U0 = {"lorenz63": [1.0, 0.0, 0.0], "rossler": [1.0, 0.0, 0.0], "henon": [0.1, 0.0]}
DISCRETE_SYSTEMS = {"henon"}


def is_discrete(system: str) -> bool:
    return system in DISCRETE_SYSTEMS


def effective_dt(system: str, dt: float) -> float:
    """Per-step time: dt for flows, 1 iteration for maps."""
    return 1.0 if is_discrete(system) else dt


def _step_system(system: str, f, u: Array, t: float, dt: float, p: Dict) -> Array:
    """One step: RK4 for flows, direct iteration for maps."""
    return f(u, t, p) if is_discrete(system) else _rk4_step(f, u, t, dt, p)


def generate_trajectory(system: str, params: Dict, T: float, dt: float,
                        u0: Array | None = None) -> Tuple[Array, Array]:
    f = SYSTEMS[system]
    n = round(T/dt) + 1
    u0_ = np.array(u0 if u0 is not None else DEFAULT_U0[system], dtype=float)
    traj = np.zeros((n, len(u0_)))
    traj[0] = u0_
    t = 0.0
    for i in range(1, n):
        traj[i] = _step_system(system, f, traj[i-1], t, dt, params)
        t += dt
    dt_eff = effective_dt(system, dt)
    return np.arange(n) * dt_eff, traj


def largest_lyapunov_exponent(system: str, params: Dict | None = None,
                              dt: float = 0.01, n_steps: int = 100_000,
                              transient: int = 1_000, d0: float = 1e-9,
                              u0: Array | None = None) -> float:
    """Largest Lyapunov exponent via the Benettin two-trajectory method.

    Evolve a reference and a shadow trajectory separated by d0; after each step
    accumulate log(d/d0) and renormalise the shadow back to distance d0.
    lambda_max = (1/total_time) sum log(d_n/d0).  Uniform for flows and maps.

    Reproduces literature values: Lorenz ~0.906, Rössler ~0.071,
    Hénon ~0.419 (per iteration).
    """
    params = params or {}
    f = SYSTEMS[system]
    dt_eff = effective_dt(system, dt)
    u = np.array(u0 if u0 is not None else DEFAULT_U0[system], dtype=float)

    t = 0.0
    for _ in range(transient):
        u = _step_system(system, f, u, t, dt, params); t += dt

    v = u.copy(); v[0] += d0
    accum = 0.0
    for _ in range(n_steps):
        u = _step_system(system, f, u, t, dt, params)
        v = _step_system(system, f, v, t, dt, params)
        t += dt
        d = float(np.linalg.norm(v - u))
        if d < 1e-300:
            continue
        accum += np.log(d / d0)
        v = u + (d0 / d) * (v - u)
    return accum / (n_steps * dt_eff)


def lyapunov_time(system: str, params: Dict | None = None, **kwargs) -> float:
    """Lyapunov time = 1 / lambda_max, computed directly from the dynamics."""
    return 1.0 / largest_lyapunov_exponent(system, params, **kwargs)
