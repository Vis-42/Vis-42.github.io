"""Monte Carlo simulation for Gō model protein folding."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from go_model import ProteinConfig, go_energy, fraction_native_contacts

Array = np.ndarray


@dataclass
class ReplicaResult:
    T: float
    Q_traj: List[float]
    E_traj: List[float]
    coords_final: Array


def mc_step(coords: Array, cfg: ProteinConfig, T: float,
            rng: np.random.Generator, step_size: float = 0.05) -> bool:
    i = rng.integers(cfg.n_residues)
    delta = (rng.random(3) - 0.5) * 2 * step_size
    E_old = go_energy(coords, cfg)
    coords[i] += delta
    E_new = go_energy(coords, cfg)
    dE = E_new - E_old
    if dE <= 0 or rng.random() < np.exp(-dE / max(T, 1e-6)):
        return True
    coords[i] -= delta
    return False


def replica_exchange_mc(cfg: ProteinConfig, T_ladder: List[float],
                        n_steps: int = 5000, swap_freq: int = 200,
                        seed: int = 42) -> List[ReplicaResult]:
    rng = np.random.default_rng(seed)
    n_rep = len(T_ladder)
    replicas = [cfg.native_coords + 0.05 * rng.standard_normal(cfg.native_coords.shape)
                for _ in range(n_rep)]

    sample_freq = max(1, n_steps // 200)
    Q_trajs = [[] for _ in range(n_rep)]
    E_trajs = [[] for _ in range(n_rep)]

    for step in range(1, n_steps + 1):
        for r in range(n_rep):
            mc_step(replicas[r], cfg, T_ladder[r], rng)

        if step % sample_freq == 0:
            for r in range(n_rep):
                Q_trajs[r].append(fraction_native_contacts(replicas[r], cfg))
                E_trajs[r].append(go_energy(replicas[r], cfg))

        if step % swap_freq == 0:
            for r in range(n_rep - 1):
                E1 = go_energy(replicas[r], cfg)
                E2 = go_energy(replicas[r+1], cfg)
                beta1 = 1.0 / T_ladder[r]; beta2 = 1.0 / T_ladder[r+1]
                # Parallel-tempering swap acceptance: delta = (b1-b2)(E1-E2);
                # accept if delta >= 0 or rand < exp(delta). (Was (b1-b2)(E2-E1)
                # = -delta, which inverted the criterion and swapped almost every
                # attempt, destroying the temperature-stratified ensemble.)
                delta = (beta1 - beta2) * (E1 - E2)
                if delta >= 0 or rng.random() < np.exp(delta):
                    replicas[r], replicas[r+1] = replicas[r+1], replicas[r]

    return [ReplicaResult(T_ladder[r], Q_trajs[r], E_trajs[r], replicas[r].copy())
            for r in range(n_rep)]
