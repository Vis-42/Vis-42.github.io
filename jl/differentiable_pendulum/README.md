# Differentiable Pendulum — Parameter Inference

Learn the masses and rod lengths of a double pendulum from noisy trajectory observations using gradient descent through a differentiable RK4 simulator.

## What it does

- Simulates a double pendulum with known parameters using RK4 integration
- Adds Gaussian observation noise to create "data"  
- Uses finite-difference gradients + momentum SGD to recover the parameters
- Tests recovery at multiple noise levels

## Files

```
jl/differentiable_pendulum/
├── app.pluto.jl    # Interactive Pluto notebook
├── main.jl         # Standalone inference script
├── src/            # RK4 double pendulum, FD gradient, momentum SGD
├── outputs/        # Precomputed inference runs
├── Project.toml
├── Manifest.toml
└── README.md       # This file
```

## Running

```bash
# Interactive notebook
cd jl/differentiable_pendulum
julia --project=../.. -e 'using Pluto; Pluto.run()'
# Open app.pluto.jl — animates true vs inferred pendulum live with the loss curve

# Standalone inference script
julia --project=. main.jl
```

## Physics

**Double pendulum** (2-DOF Lagrangian system, exact EOMs via Euler-Lagrange):

Parameters: masses m₁, m₂, rod lengths L₁, L₂. True values: m₁=1.0, m₂=0.8, L₁=1.2, L₂=0.9. Inference recovers L₁, L₂ from trajectory with known masses.

**Trajectory loss**: L(θ) = (1/T) Σ ||x_sim(t; θ) − x_obs(t)||² averaged over T timesteps.

**Gradient**: central-difference FD with ε = 10⁻⁵; momentum SGD with β=0.9.
