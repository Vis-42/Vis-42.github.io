# Spin Glass Neural Network: Hessian Spectrum and the Glass-to-Order Transition
### IPR · Hessian Diagonal Evolution · Decision Boundary Crystallization

Interactive Pluto notebook.

---

## Project Overview

During neural network training, the loss landscape undergoes a qualitative phase transition: early training resembles a spin glass (many nearly-equal local minima, broadly distributed Hessian curvatures), while late training crystallizes into an ordered phase (a few sharp curvature directions with localized Hessian diagonal). This project quantifies the transition using the **Inverse Participation Ratio (IPR)**:

$$\text{IPR} = \frac{\sum_k H_{kk}^4}{(\sum_k H_{kk}^2)^2}$$

- **IPR → 1/N**: delocalized, uniform curvature (glassy phase — early training)
- **IPR → 1**: localized, curvature concentrated in one direction (crystallized — late training)

This is the neural network analog of Anderson localization in disordered quantum systems.

---

## Files

```
jl/spin_glass_nn/
├── app.pluto.jl    # Interactive Pluto notebook
├── main.jl         # Standalone training script
├── src/            # MLP, Hessian diagonal, IPR utilities
├── outputs/        # Precomputed snapshots (used by notebook)
├── Project.toml
├── Manifest.toml
└── README.md       # This file
```

---
## Running

```bash
# Interactive notebook
cd jl/spin_glass_nn
julia --project=../.. -e 'using Pluto; Pluto.run()'

# Standalone training script (generates outputs/ snapshots)
julia --project=. main.jl
```

---

## Physics / Theory

**Hessian diagonal** (finite-difference approximation):

$$H_{kk} \approx \frac{\mathcal{L}(\theta + \epsilon e_k) - 2\mathcal{L}(\theta) + \mathcal{L}(\theta - \epsilon e_k)}{\epsilon^2}$$

**Participation Ratio** PR = 1/IPR: number of weight dimensions that carry significant curvature. PR ≫ 1 = glassy (spread-out), PR ≈ 1 = crystallized (one dominant direction).

**Anderson localization analogy**: the Hessian diagonal plays the role of the site energies in a disordered Hamiltonian. High IPR (localized Hessian) ↔ localized eigenstates in Anderson model.

**ε-sharpness** (Keskar et al.): max_{‖δ‖≤ε} [L(θ+δ) − L(θ)] measures sharpness of the minimum; flat minima generalize better.
