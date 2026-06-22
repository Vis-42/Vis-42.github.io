---
title: "Gō Model: Protein Folding Free Energy"
slug: "protein_landscape"
weight: 10
thumb: "/media/thumbs/protein_landscape.jpg"
filed: "Protein folding · Monte Carlo"
app: "/apps/protein_landscape/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/protein_landscape/protein_landscape.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/protein_landscape"

---

A simplified, off-lattice Go model of a small protein, folded with Monte Carlo. The payoff is the free energy landscape as a function of the fraction of native contacts: the folded basin, the unfolded one, the barrier between them, and the temperature where folding tips over. It is the coarse-grained, statistical-mechanics picture of folding, and it connects directly to the spectroscopy I did in the biophysics lab.
