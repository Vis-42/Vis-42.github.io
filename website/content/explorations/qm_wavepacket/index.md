---
title: "Quantum Wavepacket — Schrödinger in Any Potential"
slug: "qm_wavepacket"
weight: 12
thumb: "/media/thumbs/qm_wavepacket.jpg"
filed: "Quantum mechanics · PDE numerics"
app: "/apps/qm_wavepacket/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/qm_wavepacket/qm_wavepacket.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/qm_wavepacket"

---

One solver for every one-dimensional quantum scattering problem in the textbook. A Gaussian wavepacket evolves under the time-dependent Schrodinger equation by a split-step method that stays unitary and conserves the norm, through any of six built-in potentials or one you type in yourself. The wavefunction is drawn as a 3D Argand helix: the real and imaginary parts live on perpendicular planes and the probability density sits on the floor, with the view tracking the packet as it moves and spreads. Below it, the live density reports the norm and the transmission and reflection coefficients.
