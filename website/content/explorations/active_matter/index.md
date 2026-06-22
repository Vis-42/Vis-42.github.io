---
title: "Active Matter: Vicsek Model + Run-and-Tumble"
slug: "active_matter"
weight: 1
thumb: "/media/thumbs/active_matter.jpg"
filed: "Active matter · nonlinear dynamics"
app: "/apps/active_matter/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/active_matter/active_matter.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/active_matter"
---

Two faces of active matter in one place. The Vicsek model is the crowd: a few hundred self-propelled particles that each steer toward the average heading of their neighbours, with a little noise mixed in. Turn the noise up and the flock scatters; turn it down and the whole swarm commits to a single direction. That switch is a real phase transition, and the order parameter follows it as I sweep the noise. Run-and-tumble is the single cell: a swimmer that runs straight, reorients at random, and runs again. Its mean-squared displacement starts ballistic and turns diffusive right around the tumble time, and the effective diffusion constant I read off matches the analytic value.
