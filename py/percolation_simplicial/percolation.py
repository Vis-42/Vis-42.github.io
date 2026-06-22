"""Bond percolation on simplicial complexes."""

from __future__ import annotations
import numpy as np
from .simplicial_complex import SimplicialComplex, connected_components


def percolate(cx: SimplicialComplex, p: float, dimension: int = 1, seed: int = 42) -> float:
    """Single realisation of bond percolation on k-simplices."""
    rng = np.random.default_rng(seed)
    N = cx.N
    adj = np.zeros((N, N), dtype=bool)

    if dimension == 1:
        for i, j in cx.edges:
            if rng.random() < p:
                adj[i, j] = adj[j, i] = True
    elif dimension == 2:
        for i, j, k in cx.triangles:
            if rng.random() < p:
                adj[i,j]=adj[j,i]=adj[i,k]=adj[k,i]=adj[j,k]=adj[k,j]=True
    elif dimension == 3:
        for i, j, k, l in cx.tetrahedra:
            if rng.random() < p:
                for a, b in [(i,j),(i,k),(i,l),(j,k),(j,l),(k,l)]:
                    adj[a,b]=adj[b,a]=True

    comp, n_comp = connected_components(adj)
    counts = np.bincount(comp[comp > 0], minlength=n_comp + 1)[1:]
    return float(counts.max() / N) if len(counts) > 0 else 0.0


def giant_component_size(cx: SimplicialComplex, p: float,
                          dimension: int = 1, n_samples: int = 5) -> float:
    return float(np.mean([percolate(cx, p, dimension=dimension, seed=s) for s in range(n_samples)]))


def susceptibility(cx: SimplicialComplex, p_range, dimension: int = 1, n_samples: int = 5):
    N = cx.N
    chi = []
    for p in p_range:
        samples = np.array([percolate(cx, p, dimension=dimension, seed=s) for s in range(n_samples)])
        ms, ms2 = samples.mean(), (samples**2).mean()
        chi.append(max(0.0, N * (ms2 - ms**2) / (ms + 1e-12)))
    return chi
