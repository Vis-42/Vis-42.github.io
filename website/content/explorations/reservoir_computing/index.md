---
title: "Minimal Reservoir Computing"
slug: "reservoir_computing"
weight: 8
thumb: "/media/thumbs/reservoir_computing.jpg"
filed: "Chaos · reservoir computing"
app: "/apps/reservoir_computing/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/reservoir_computing/reservoir_computing.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/reservoir_computing"

---

A reservoir computer forecasts chaos by pushing a signal through a big fixed random network and training only a linear readout. The question I cared about: how small can the reservoir get before it stops working? I map prediction quality across the sparsity and spectral-radius plane, scoring each setup by how many Lyapunov times its forecast stays valid, with the Lyapunov exponent computed straight from the dynamics rather than looked up. I run it on the Lorenz and Rossler flows and the Henon map to find the smallest network that still tracks each attractor.
