---
title: "Gray-Scott Reaction-Diffusion — Turing Patterns"
slug: "reaction_diffusion"
weight: 13
thumb: "/media/thumbs/reaction_diffusion.jpg"
filed: "Pattern formation · PDE numerics"
app: "/apps/reaction_diffusion/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/reaction_diffusion/reaction_diffusion.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/reaction_diffusion"

---

Two chemicals, one that activates and one that inhibits, diffusing at different rates. That is all it takes for a flat, featureless state to break into spots, stripes, and labyrinths on its own. This is Turing's idea made concrete with the Gray-Scott equations on a grid. Sweeping the feed and kill rates walks you across the whole Pearson diagram of patterns, and watching one grow out of random noise does not stop being a little surprising.
