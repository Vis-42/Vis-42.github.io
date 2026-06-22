---
title: "Physics Discovery: SINDy + Conservation Laws + Symbolic Regression"
slug: "physics_discovery"
weight: 4
thumb: "/media/thumbs/physics_discovery.jpg"
filed: "Equation discovery · data-driven dynamics"
app: "/apps/physics_discovery/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/physics_discovery/physics_discovery.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/physics_discovery"

---

Three ways to pull physics back out of raw trajectory data. SINDy fits a sparse handful of terms from a candidate library and recovers the actual equations of motion for systems like the Van der Pol oscillator. A kernel method asks instead what stays constant along the motion, finding conserved quantities without being told their form. Symbolic regression evolves expression trees until one of them rediscovers a closed-form law, like the pendulum period. Same data, three different questions: what are the dynamics, what is invariant, and what is the law.
