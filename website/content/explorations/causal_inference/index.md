---
title: "Causal Inference: Transfer Entropy + CCM + Causal Emergence"
slug: "causal_inference"
weight: 3
thumb: "/media/thumbs/causal_inference.jpg"
filed: "Causal inference · information theory"
app: "/apps/causal_inference/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/causal_inference/causal_inference.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/causal_inference"

---

Telling which signal drives which, from data alone, is harder than it sounds, because correlation is symmetric and says nothing about direction. I built three tools that do say something. Transfer entropy measures how much one series' past cuts the uncertainty in another's future. Convergent cross-mapping handles the deterministic systems where transfer entropy struggles, rebuilding one variable's history from the other's attractor. Causal emergence asks a different question altogether: whether a coarse-grained view of a system can carry more causal weight than the fine-grained one underneath it. I run all three on shared benchmarks and watch where they agree and where they split.
