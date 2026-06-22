"""Hessian spectrum analysis."""

from __future__ import annotations
import numpy as np


def eigenvalue_spectrum(diag_H: np.ndarray) -> np.ndarray:
    """Sorted diagonal Hessian curvatures Hkk (not true eigenvalues)."""
    return np.sort(diag_H)


def ipr(eigenvalues: np.ndarray) -> float:
    """Inverse Participation Ratio: IPR = sum(|Hkk| / sum|Hjj|)^2.

    1/N = uniform curvature (glassy); → 1 = few sharp axes (crystallised/ordered).
    """
    vals = np.abs(eigenvalues)
    s = vals.sum()
    if s < 1e-12:
        return 1.0
    return float(((vals / s) ** 2).sum())


def participation_ratio(eigenvalues: np.ndarray) -> float:
    i = ipr(eigenvalues)
    return float(len(eigenvalues)) if i < 1e-12 else 1.0 / i
