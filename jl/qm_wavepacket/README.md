# Quantum Wavepacket Dynamics
### Solving the Time-Dependent Schrödinger Equation via Crank-Nicolson

Interactive Pluto notebook.

---

## Project Overview

This project simulates a Gaussian wavepacket propagating through six physically distinct potential regimes in 1D, solved with the Crank-Nicolson (CN) method.

**Core physics:** The time-dependent Schrödinger equation (TDSE):

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \frac{\partial^2 \psi}{\partial x^2} + V(x)\psi$$

is evolved numerically. The initial state is a Gaussian wavepacket — the unique minimum-uncertainty state — visualized as a 3D Argand helix whose pitch encodes local de Broglie wavelength, whose radius encodes probability amplitude, and whose exponential collapse encodes evanescent (tunneling) character.

---

## Files

```
jl/qm_wavepacket/
├── app.pluto.jl    # Interactive Pluto notebook (the simulation)
├── Project.toml
└── README.md       # This file
```

---

## Running

```bash
cd jl/qm_wavepacket
julia --project=. -e 'using Pluto; Pluto.run()'
# Open app.pluto.jl in the Pluto UI
```

---

## Six Physics Regimes

| Potential | Key phenomenon | Signature |
|-----------|---------------|-----------|
| Free particle | Dispersion | Helix broadens; constant pitch; norm=1 |
| Rectangular barrier (A>1) | Quantum tunneling | Helix collapses inside; non-zero T |
| Rectangular barrier (A<1) | Impedance mismatch | T+R=1; two packets; pitch changes |
| Finite square well | Ramsauer-Townsend resonance | T→1 at resonant energies; tighter helix |
| Harmonic oscillator | Ehrenfest correspondence | ⟨x⟩ oscillates classically; width grows |
| Double barrier | Resonant tunneling (Fabry-Pérot) | T→1 at resonant k₀; cavity buildup |
| Potential step | Quantum reflection above barrier | R=((k₁-k₂)/(k₁+k₂))²; opposite-handed helix |

---

## The Argand Helix — Key Visual

The 3D curve `(x, Im ψ, Re ψ)` is the central visual of the entire project. It encodes:

- **Pitch** = local de Broglie wavelength λ_dB = 2π/k_local(x)
- **Radius** = probability amplitude |ψ(x)|
- **Exponential collapse** → evanescent wave (classically forbidden region, tunneling)
- **Handedness** → propagation direction (left-going reflected wave winds opposite)
- **Taper** → Gaussian envelope of the wavepacket

This single 3D object carries more information than any 2D |ψ|² plot.

---
