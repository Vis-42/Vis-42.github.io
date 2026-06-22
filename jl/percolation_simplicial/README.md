# Percolation on Simplicial Complexes
### Higher-Order Interactions · Cooperative Percolation · Giant Component Phase Transition

Interactive Pluto notebook.

---

## Project Overview

Standard percolation theory considers only pairwise interactions (edges). When higher-order simplices (triangles, tetrahedra) are the fundamental units of interaction, the percolation transition changes dramatically — the threshold shifts and the transition can become discontinuous (first-order).

This project studies the **simplicial percolation model**: a node becomes active only when an entire k-simplex (all k+1 nodes) is active, not just a single edge. The key findings:

- **k=1 (edge percolation)**: continuous transition at p_c ≈ 0.10 on Erdős-Rényi G(N,p)
- **k=2 (triangle percolation)**: continuous but with higher threshold p_c ≈ 0.25 — requires triangles, not just edges
- **k=3 (tetrahedron percolation)**: threshold shifts further to p_c ≈ 0.30; increasing cooperativity

The interactive notebook runs Monte Carlo percolation on random simplicial complexes with sliders for N, p, and k.

---

## Files

```
jl/percolation_simplicial/
├── app.pluto.jl    # Interactive Pluto notebook
├── main.jl         # Standalone Monte Carlo script
├── src/            # Simplex enumeration, BFS, percolation utilities
├── outputs/        # Precomputed S(p) and χ(p) data
├── Project.toml
├── Manifest.toml
└── README.md       # This file
```

---
## Running

```bash
# Interactive notebook
cd jl/percolation_simplicial
julia --project=../.. -e 'using Pluto; Pluto.run()'

# Standalone Monte Carlo sweep (generates outputs/)
julia --project=. main.jl
```

---

## Physics / Theory

**Giant component fraction** S(p): in a random graph G(N, p_edge), the probability that a node belongs to the giant connected component. For edge percolation, the mean-field theory gives S = 1 − exp(−⟨k⟩S) with ⟨k⟩ = (N−1)p_edge.

**Simplicial percolation** (k-simplex rule): node i is active iff there exists a k-simplex containing i where all other k nodes are active. This cooperative activation shifts p_c upward and can sharpen the transition.

**Susceptibility peak**: χ(p) = N · Var(S(p)) diverges at p_c in the thermodynamic limit. Finite-N peak location → p_c(N) → p_c(∞) via finite-size scaling S(p, N) = N^{-β/ν} f((p − p_c) N^{1/ν}).
