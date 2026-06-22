"""Phase diagram computation for percolation."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .simplicial_complex import SimplicialComplex
from .percolation import giant_component_size, susceptibility


@dataclass
class PercolationResult:
    p_range: list[float]
    S: list[float]
    chi: list[float]
    dimension: int
    N: int

    @property
    def p_c(self) -> float:
        return self.p_range[int(np.argmax(self.chi))]


def sweep_percolation(
    cx: SimplicialComplex,
    p_range,
    dimension: int = 1,
    n_samples: int = 5,
) -> PercolationResult:
    p_vals = list(p_range)
    S   = [giant_component_size(cx, p, dimension=dimension, n_samples=n_samples) for p in p_vals]
    chi = susceptibility(cx, p_vals, dimension=dimension, n_samples=n_samples)
    return PercolationResult(p_range=p_vals, S=S, chi=chi, dimension=dimension, N=cx.N)
