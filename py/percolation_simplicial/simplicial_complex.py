"""Simplicial complex construction."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimplicialComplex:
    N: int
    adj: np.ndarray
    edges: list[tuple[int, int]] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    tetrahedra: list[tuple[int, int, int, int]] = field(default_factory=list)


def generate_clique_complex(N: int, p_edge: float, dim_max: int = 3, seed: int = 42) -> SimplicialComplex:
    """Generate an Erdős-Rényi clique complex on N vertices."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((N, N), dtype=bool)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p_edge:
                adj[i, j] = adj[j, i] = True
                edges.append((i, j))

    triangles = []
    if dim_max >= 2:
        for idx, (i, j) in enumerate(edges):
            for k in range(j + 1, N):
                if adj[i, k] and adj[j, k]:
                    triangles.append((i, j, k))

    tetrahedra = []
    if dim_max >= 3:
        for i, j, k in triangles:
            for l in range(k + 1, N):
                if adj[i, l] and adj[j, l] and adj[k, l]:
                    tetrahedra.append((i, j, k, l))

    return SimplicialComplex(N=N, adj=adj, edges=edges,
                             triangles=triangles, tetrahedra=tetrahedra)


def connected_components(adj: np.ndarray) -> tuple[np.ndarray, int]:
    """BFS connected components of adjacency matrix."""
    N = adj.shape[0]
    comp = np.zeros(N, dtype=int)
    c = 0
    for start in range(N):
        if comp[start] != 0:
            continue
        c += 1
        queue = [start]
        comp[start] = c
        while queue:
            v = queue.pop(0)
            for u in np.where(adj[v])[0]:
                if comp[u] == 0:
                    comp[u] = c
                    queue.append(u)
    return comp, c
