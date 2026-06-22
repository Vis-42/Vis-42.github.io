---
title: "Spin Glass Analysis of Neural Network Training"
slug: "spin_glass_nn"
weight: 5
thumb: "/media/thumbs/spin_glass_nn.jpg"
filed: "Disordered systems · machine learning"
app: "/apps/spin_glass_nn/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/spin_glass_nn/spin_glass_nn.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/spin_glass_nn"

---

Training a neural network is really a walk downhill on a bumpy, high-dimensional surface, and that surface behaves a lot like a spin glass. I train a small network and watch the curvature of the loss landscape through the diagonal of the Hessian. The inverse participation ratio of that curvature spectrum tells me how the minimum is shaped: spread flat across many directions early on, the glassy phase, then concentrating into a few sharp directions as training settles. It is a small, hands-on way into the physics behind the 2024 Nobel.
