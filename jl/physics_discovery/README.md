# Physics Discovery: SINDy, Conservation Laws, and Symbolic Regression
### STLSQ · RBF Kernel Eigenproblem · Symbolic GP

Interactive Pluto notebook + Keynote presentation with Julia-generated animations.

---

## Project Overview

Can an algorithm recover physical laws from trajectory data alone — without prior knowledge of the governing equations? This project implements and compares three complementary discovery strategies applied to the same systems (Van der Pol, pendulum, Kepler):

- **SINDy** (Sparse Identification of Nonlinear Dynamics): fit ẋ = Θ(x)ξ where Θ is a library of candidate terms (1, x, x², sin x, …) and ξ is found by sequentially thresholded least squares — promotes sparsity, returns interpretable equations
- **Kernel conservation-law discovery**: solve a generalized eigenproblem with RBF kernels to find functions h(x) with minimal temporal drift ⟨|∇h · ẋ|²⟩ — recovers conserved quantities without assuming a functional form
- **Symbolic regression via GP**: evolve expression trees (mutation + crossover + selection on MSE) to find compact closed-form expressions matching the data — most flexible but computationally expensive

---

## Files

```
jl/physics_discovery/
├── app.pluto.jl              # Interactive Pluto notebook
├── presentation.md           # Full slide content
├── README.md                 # This file
│
└── animations/
    ├── run_all.jl            # Render all 8 animations
    ├── Project.toml
    ├── Manifest.toml
    │
    ├── shared/
    │   ├── style.jl          # Dark theme, RES=(1280,720), FPS=30, color palette
    │   └── discovery.jl      # SINDy/STLSQ, kernel invariant, symbolic GP, data generators
    │
    ├── s01_motivation.jl      → s01_motivation.mp4
    ├── s02_sindy_library.jl   → s02_sindy_library.mp4
    ├── s03_sindy_reconstruction.jl → s03_sindy_reconstruction.mp4
    ├── s04_conservation_phase.jl → s04_conservation_phase.mp4
    ├── s05_kernel_invariant.jl → s05_kernel_invariant.mp4
    ├── s06_symbolic_gp.jl     → s06_symbolic_gp.mp4
    ├── s07_gp_convergence.jl  → s07_gp_convergence.mp4
    └── s08_comparison.jl      → s08_comparison.mp4
```

---

## Slide ↔ Animation Correspondence

| Animation | Slide topic | What it shows |
|-----------|-------------|---------------|
| `s01_motivation.mp4` | The inverse problem | Van der Pol trajectory + SINDy reconstruction + energy trace building progressively |
| `s02_sindy_library.mp4` | Library matrix Θ and sparsity | Sparsity vs λ threshold + RMSE vs λ; curves sweep as λ increases |
| `s03_sindy_reconstruction.mp4` | True vs SINDy phase portrait | Phase portrait and state traces: true (solid) vs reconstructed (dashed) filling up |
| `s04_conservation_phase.mp4` | Phase space + energy trace | Three systems (pendulum/Kepler/Van der Pol) with energy trace; trajectories grow |
| `s05_kernel_invariant.mp4` | Kernel conservation law | H_kernel vs H_true scatter + γ sweep correlation + temporal energy trace |
| `s06_symbolic_gp.mp4` | Symbolic GP expression trees | GP generations evolving; fitness vs generation convergence curves |
| `s07_gp_convergence.mp4` | GP R² on Feynman equations | R² convergence curves for 5 Feynman equations across generations |
| `s08_comparison.mp4` | All three methods on Kepler | Side-by-side: SINDy equation, kernel H(x), GP expression on same data |

---

## Running

```bash
cd jl/physics_discovery/animations
julia --project=. run_all.jl
```

All outputs land in `animations/output/`. Estimated render time: ~6–10 minutes (GP runs are expensive).

### Single animation

```bash
cd jl/physics_discovery/animations
julia --project=. s05_kernel_invariant.jl
```

### Interactive notebook

```bash
cd jl/physics_discovery
julia --project=../.. -e 'using Pluto; Pluto.run()'
```

---

## Physics / Theory

**SINDy / STLSQ** (Brunton et al. 2016):

$$\dot{X} = \Theta(X)\,\xi, \quad \Theta \in \mathbb{R}^{n \times p}$$

Iteratively: solve least squares, zero terms with |ξᵢⱼ| < λ, re-solve on surviving terms. Van der Pol ẋ₂ = μ(1−x₁²)x₂ − x₁ uses 3 terms from a ~20-term library.

**Kernel conservation-law discovery**: solve the generalized eigenproblem K_drift v = λ K v where K_drift encodes temporal drift of the RBF kernel feature map. Smallest eigenvector = maximally conserved quantity. Optimal bandwidth γ* ≈ median pairwise distance.

**Symbolic GP**: population of expression trees; fitness = 1/RMSE on held-out data. Operators: +, −, ×, ÷, sin, cos, exp, √. Selection = tournament; mutation = random subtree; crossover = subtree swap.

---

## Shared Infrastructure

**`shared/style.jl`**: dark theme, 1280×720, FPS=30. Colors: trajectory = teal, conserved = amber, theory = white, library terms = violet, active terms = green.

**`shared/discovery.jl`**: `sindy_stlsq()` sparse regression, `discover_invariant()` kernel eigenproblem, `symbolic_gp()` evolutionary optimizer, `pendulum_data()` / `kepler_data()` / `vanderpol_data()` trajectory generators.
