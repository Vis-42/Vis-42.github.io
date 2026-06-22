"""Free energy profile from MC trajectories."""
import numpy as np
from typing import List, Tuple

Array = np.ndarray


def free_energy_profile(Q_traj: List[float], T: float,
                        n_bins: int = 30) -> Tuple[Array, Array]:
    if not Q_traj:
        return np.linspace(0, 1, n_bins), np.zeros(n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(Q_traj, bins=edges)
    counts = counts.astype(float) + 0.5  # pseudocount
    P = counts / counts.sum()
    F = -T * np.log(P)
    F -= F.min()
    return centers, F


def wham_simple(results, n_bins: int = 30) -> Tuple[Array, Array]:
    best = min(results, key=lambda r: r.T)
    return free_energy_profile(best.Q_traj, best.T, n_bins=n_bins)
