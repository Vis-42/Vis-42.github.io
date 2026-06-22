# Reservoir Computing: Echo State Networks for Chaotic System Prediction
### Echo State Property · Lorenz-63 · Phase Diagram (Sparsity × Spectral Radius)

Interactive Pluto notebook + Keynote presentation with Julia-generated animations.

---

## Project Overview

Echo State Networks (ESNs) are a minimal reservoir computing architecture for learning chaotic dynamical systems without backpropagating through time:

1. **Reservoir** (fixed): a sparse random recurrent network W_res with spectral radius ρ(W_res) < 1 — the Echo State Property ensures fading memory
2. **Input weights** (fixed): W_in maps the driving signal u(t) into the reservoir
3. **Readout** (trained): W_out = Y X⁺ fitted by ridge regression on reservoir states X and target outputs Y

This project applies ESNs to Lorenz-63 prediction, producing a systematic phase diagram of prediction horizon as a function of (sparsity, spectral radius). The interactive notebook runs live autonomous prediction.

---

## Files

```
jl/reservoir_computing/
├── app.pluto.jl              # Interactive Pluto notebook
├── presentation.md           # Full slide content
├── README.md                 # This file
│
└── animations/
    ├── run_all.jl            # Render all 11 animations
    ├── Project.toml
    ├── Manifest.toml
    │
    ├── shared/
    │   ├── style.jl          # Dark theme, RES=(1280,720), FPS=30, color palette
    │   └── lorenz.jl         # Lorenz RK4 integrator, ESN implementation, Lyapunov time
    │
    ├── s01_motivation.jl      → s01_motivation.mp4
    ├── s02_lyapunov.jl        → s02_lyapunov.mp4
    ├── s03_lorenz_attractor.jl → s03_lorenz_attractor.mp4
    ├── s04_echo_state.jl      → s04_echo_state.mp4
    ├── s05_spectral_radius.jl → s05_spectral_radius.mp4
    ├── s06_reservoir_init.jl  → s06_reservoir_init.mp4
    ├── s07_reservoir_dynamics.jl → s07_reservoir_dynamics.mp4
    ├── s08_ridge_regression.jl → s08_ridge_regression.mp4
    ├── s09_autonomous_pred.jl → s09_autonomous_pred.mp4
    ├── s10_phase_diagram.jl   → s10_phase_diagram.mp4
    └── s11_trajectory_recon.jl → s11_trajectory_recon.mp4
```

---

## Slide ↔ Animation Correspondence

| Animation | Slide topic | What it shows |
|-----------|-------------|---------------|
| `s01_motivation.mp4` | Lorenz sensitivity | Two nearby trajectories diverge exponentially on the butterfly attractor |
| `s02_lyapunov.mp4` | Lyapunov exponent | λ₁ convergence + prediction ceiling bar (max horizon = 1/λ₁) |
| `s03_lorenz_attractor.mp4` | 3D Lorenz butterfly | Attractor building up progressively in 3D |
| `s04_echo_state.mp4` | Echo State Property | Reservoir states from different initial conditions converging to same trajectory |
| `s05_spectral_radius.mp4` | Spectral radius and memory | Reservoir impulse response at ρ = 0.5, 0.9, 1.1: fading vs diverging memory |
| `s06_reservoir_init.mp4` | Reservoir weight matrix | W_res sparsity pattern + eigenvalue spectrum in complex plane |
| `s07_reservoir_dynamics.mp4` | Reservoir state traces | Lorenz x(t) input vs first 5 reservoir state traces |
| `s08_ridge_regression.mp4` | Ridge regression readout | Training: ||Y − W_out X||² + α||W_out||² vs ridge parameter α |
| `s09_autonomous_pred.mp4` | Autonomous prediction | ESN free-running on the attractor: predicted (solid) vs true (dashed) |
| `s10_phase_diagram.mp4` | (Sparsity, ρ) phase diagram | Heatmap of prediction horizon filling column-by-column |
| `s11_trajectory_recon.mp4` | 3D trajectory reconstruction | True vs predicted Lorenz trajectory in 3D; growing progressively |

---

## Running

```bash
cd jl/reservoir_computing/animations
julia --project=. run_all.jl
```

All outputs land in `animations/output/`. Estimated render time: ~5–8 minutes.

### Single animation

```bash
cd jl/reservoir_computing/animations
julia --project=. s09_autonomous_pred.jl
```

### Interactive notebook

```bash
cd jl/reservoir_computing
julia --project=../.. -e 'using Pluto; Pluto.run()'
```

---

## Physics / Theory

**Echo State Property**: reservoir has ESP iff for any two input sequences u, v with u(t) = v(t) for all t ≤ T, the reservoir states x_u(T) and x_v(T) converge as T → ∞. A sufficient condition: ρ(W_res) < 1 (spectral radius of fixed weight matrix).

**Lorenz-63** (chaotic attractor, σ=10, ρ=28, β=8/3):

$$\dot{x} = \sigma(y - x), \quad \dot{y} = x(\rho - z) - y, \quad \dot{z} = xy - \beta z$$

Lyapunov time T_λ = 1/λ₁ ≈ 0.9 (dimensionless units). Prediction horizon ≈ 3–8 Lyapunov times for a well-tuned ESN.

**Ridge readout**: W_out = Y Xᵀ (X Xᵀ + αI)⁻¹. α trades off fitting accuracy vs weight magnitude; optimal α found by cross-validation on the washout-excluded portion of the training run.

**Phase diagram**: prediction horizon H(sparsity p, spectral radius ρ) measured in Lyapunov times. Optimal region: p ≈ 0.1–0.2, ρ ≈ 0.9–1.1 (near the edge of chaos).

---

## Shared Infrastructure

**`shared/style.jl`**: dark theme, 1280×720, FPS=30. Colors: true trajectory = sky blue, predicted = amber, reservoir states = violet palette, attractor = viridis.

**`shared/lorenz.jl`**: `lorenz_rk4()` Lorenz integrator, `ESN` struct with `train!()` and `run!()` methods, `lyapunov_time()` estimation, `prediction_horizon()` measurement in Lyapunov times.
