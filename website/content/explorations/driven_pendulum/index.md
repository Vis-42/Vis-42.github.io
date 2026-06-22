---
title: "Driven Quadruple Pendulum"
slug: "driven_pendulum"
weight: 11
thumb: "/media/thumbs/pendulum.jpg"
filed: "Chaos · classical mechanics"
app: "/apps/driven_pendulum/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/driven_pendulum/driven_pendulum.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/driven_pendulum"

---

A driven, damped pendulum is about the simplest thing that goes chaotic, and this one has four coupled segments. I derived the equations of motion, wrote a custom RK4 integrator, and used phase portraits, Poincare sections, and Lyapunov divergence to watch the route into chaos as the drive turns up. Co-authored with Soham Bhar for a computational physics course.
