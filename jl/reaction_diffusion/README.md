# Reaction-Diffusion: Turing Patterns from the Gray-Scott Model
### Turing Instability · Gray-Scott Equations · Pearson Phase Diagram

Interactive Pluto notebook.

---

## Project Overview

Two colorless chemicals, mixed uniformly. Nothing distinguishes one location from another in the initial conditions. Yet the system spontaneously organizes into sharp, persistent spatial patterns: spots, stripes, labyrinthine worm-like structures, and replicating blobs — all emerging from a single set of coupled PDEs.

This is **Turing's 1952 insight**: reaction-diffusion systems can break spatial symmetry without any encoded spatial information. This project implements the Gray-Scott model:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + F(1-u)$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (F+k)v$$

where u is the substrate (feeds in at rate F, consumed in the reaction u+2v→3v), v is the autocatalytic activator (decays at rate F+k), and the differential diffusion Dᵤ/Dᵥ = 2 drives the Turing instability.

The interactive notebook runs live Gray-Scott integration on a 256×256 grid with sliders for F and k.

---

## Files

```
jl/reaction_diffusion/
├── app.pluto.jl    # Interactive Pluto notebook
└── README.md       # This file
```

---
## Running

```bash
cd jl/reaction_diffusion
julia --project=../.. -e 'using Pluto; Pluto.run()'
# Open app.pluto.jl — runs live Gray-Scott on a 256×256 grid with (F, k) sliders
```

---

## Physics

**Turing instability criterion**: a homogeneous steady state (u*, v*) is stable to homogeneous perturbations but unstable to spatially inhomogeneous ones when the Jacobian J satisfies:

$$\text{tr}(J) < 0, \quad \det(J) > 0, \quad \text{but} \quad D_v J_{11} + D_u J_{22} > 2\sqrt{D_u D_v \det(J)}$$

The last condition requires the inhibitor (u, fast diffuser) to have significantly larger diffusion than the activator (v). The Gray-Scott parameterization satisfies this when F and k are in the Turing regime.

**Numerics**: explicit Euler on a 256×256 periodic grid with Δt = 1.0 and a 5-point discrete Laplacian. The Fourier number D_u Δt/Δx² < 0.5 ensures stability.

**Pearson diagram**: the (F, k) parameter space partitions into ~12 distinct pattern types (spots, stripes, worms, replicating structures, chaos, uniform steady states). The notebook sweeps this space with a computed stability boundary.
