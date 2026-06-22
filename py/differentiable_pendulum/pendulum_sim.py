"""Double pendulum simulator using RK4."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray


@dataclass
class PendulumParams:
    m1: float = 1.0; m2: float = 1.0
    L1: float = 1.0; L2: float = 1.0
    g: float = 9.81


def pendulum_ode(u: Array, p: PendulumParams) -> Array:
    θ1, ω1, θ2, ω2 = u
    Δ = θ2 - θ1; sΔ = np.sin(Δ); cΔ = np.cos(Δ)
    d1 = (2*p.m1 + p.m2 - p.m2*np.cos(2*Δ)) * p.L1
    d2 = p.L2 / p.L1 * d1
    # Sign: Δ = θ2-θ1, so sin(θ1-θ2) = -sΔ; the standard formula has -2sin(θ1-θ2)·m2·(...)
    # which becomes +2sΔ·m2·(...) after substitution (and -2sΔ·(...) in α2).
    α1 = (-p.g*(2*p.m1+p.m2)*np.sin(θ1) - p.m2*p.g*np.sin(θ1-2*θ2)
           + 2*sΔ*p.m2*(ω2**2*p.L2 + ω1**2*p.L1*cΔ)) / d1
    α2 = (-2*sΔ*(ω1**2*p.L1*(p.m1+p.m2) + p.g*(p.m1+p.m2)*np.cos(θ1)
           + ω2**2*p.L2*p.m2*cΔ)) / d2
    return np.array([ω1, α1, ω2, α2])


def _rk4_step(u: Array, p: PendulumParams, dt: float) -> Array:
    k1 = pendulum_ode(u, p)
    k2 = pendulum_ode(u + dt/2 * k1, p)
    k3 = pendulum_ode(u + dt/2 * k2, p)
    k4 = pendulum_ode(u + dt   * k3, p)
    return u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def simulate(params: PendulumParams, u0: Array,
             tspan: tuple, dt: float) -> tuple[Array, Array]:
    t0, t1 = tspan
    n = max(2, int((t1 - t0) / dt) + 1)
    times = np.linspace(t0, t1, n)
    traj  = np.zeros((n, 4))
    traj[0] = u0
    for i in range(1, n):
        traj[i] = _rk4_step(traj[i-1], params, dt)
    return times, traj


def simulate_noisy(params: PendulumParams, u0: Array,
                   tspan: tuple, dt: float,
                   sigma: float = 0.05, seed: int = 42) -> tuple[Array, Array, Array]:
    times, traj = simulate(params, u0, tspan, dt)
    rng = np.random.default_rng(seed)
    return times, traj + sigma * rng.standard_normal(traj.shape), traj
