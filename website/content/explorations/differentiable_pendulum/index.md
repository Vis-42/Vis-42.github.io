---
title: "Differentiable Pendulum: Parameter Inference"
slug: "differentiable_pendulum"
weight: 9
thumb: "/media/thumbs/differentiable_pendulum.jpg"
filed: "Differentiable physics · inverse problems"
app: "/apps/differentiable_pendulum/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/differentiable_pendulum/differentiable_pendulum.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/differentiable_pendulum"

---

A double pendulum is chaotic, but its masses and arm lengths still leave a fingerprint in the motion. I treat the RK4 simulator as a differentiable forward model and run gradient descent on a loss that compares simulated and observed trajectories, recovering the physical parameters from noisy data to within a few percent. It is the mirror image of the chaos work: instead of predicting motion from parameters, I infer parameters from motion.
