"""Synthetic classification datasets."""

from __future__ import annotations
import numpy as np


def make_spiral(n_per_class: int, noise: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = n_per_class * 2
    X, y = np.zeros((n, 2)), np.zeros(n, dtype=int)
    for c in range(2):
        ix = slice(c * n_per_class, (c + 1) * n_per_class)
        r = np.linspace(0, 1, n_per_class)
        t = np.linspace(c * 4, (c + 2) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X[ix, 0] = r * np.sin(t)
        X[ix, 1] = r * np.cos(t)
        y[ix] = c
    return X, y


def make_xor(N: int, noise: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = 2 * rng.random((N, 2)) - 1 + rng.normal(0, noise, (N, 2))
    y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(int)
    return X, y


def make_checkerboard(N: int, noise: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.5, (N, 2))
    y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])).astype(int) % 2 == 0).astype(int)
    return X, y
