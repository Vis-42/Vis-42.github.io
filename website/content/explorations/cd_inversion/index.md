---
title: "Physics-Constrained CD Spectral Inversion"
slug: "cd_inversion"
weight: 6
thumb: "/media/thumbs/cd_inversion.jpg"
filed: "Biophysics · physics-informed ML"
app: "/apps/cd_inversion/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/cd_inversion/cd_inversion.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/cd_inversion"

---

Circular dichroism spectroscopy reads out a protein's secondary structure, but only forwards: given the helix, sheet, and coil fractions, you can predict the spectrum from reference curves. I wanted to go backwards and recover the composition from a measured, noisy spectrum. That inverse problem is ill-posed, so I pin it down with the physics that has to hold: the fractions are non-negative and sum to one. It comes straight out of CD experiments I ran in the lab, and watching that constraint rescue an otherwise unstable inversion is the satisfying part.
