# Coupled Oscillators: Synchronization, Networks, and Chimera States
### Kuramoto Model · Ott-Antonsen Theory · Nonlocal Chimera States

Interactive Pluto notebook.

---

## Project Overview

The Kuramoto model is the minimal framework for coupled oscillators with a synchronization transition. This project covers:

- **Synchronization transition**: the order parameter r = |N⁻¹Σeⁱθᵢ| undergoes a continuous phase transition at critical coupling Kc = 2/πg(0), where g(ω) is the frequency distribution
- **Bifurcation diagram**: r vs K with mean-field Ott-Antonsen prediction and susceptibility peak χ = Var(r)
- **Network topology effects**: all-to-all vs Erdős-Rényi vs Barabási-Albert networks — heterogeneous degree shifts Kc
- **Chimera states**: identical oscillators with nonlocal coupling spontaneously split into synchronized and incoherent domains — a symmetry-breaking phenomenon with no external forcing

The interactive notebook runs live Kuramoto dynamics with sliders for K, η (noise), and N.

---

## Files

```
jl/coupled_oscillators/
├── app.pluto.jl    # Interactive Pluto notebook
└── README.md       # This file
```

---
## Running

```bash
cd jl/coupled_oscillators
julia --project=../.. -e 'using Pluto; Pluto.run()'
# Open app.pluto.jl in the Pluto UI
```

---

## Physics

**Kuramoto model** (continuous time, N oscillators):

$$\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i)$$

where ωᵢ ~ g(ω) (Lorentzian or Gaussian). **Ott-Antonsen** exact solution for the Lorentzian g(ω) = (γ/π)/((ω−ω₀)² + γ²):

$$r_\infty = \sqrt{1 - K_c/K}, \quad K_c = 2\gamma$$

**Chimera states** (nonlocal coupling, phase lag α ≈ π/2 − ε):

$$\dot{\theta}_i = \omega_0 - \frac{K}{2R+1} \sum_{|i-j| \leq R} \sin(\theta_i - \theta_j + \alpha)$$

The coexistence of synchronized and incoherent domains is stable and robust — it arises from the phase lag making coupling attractive within the synchronized domain and repulsive at the boundary.
