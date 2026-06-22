---
title: "Percolation on Simplicial Complexes"
slug: "percolation_simplicial"
weight: 7
thumb: "/media/thumbs/percolation_simplicial.jpg"
filed: "Percolation · higher-order networks"
app: "/apps/percolation_simplicial/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/percolation_simplicial/percolation_simplicial.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/percolation_simplicial"

---

Ordinary percolation lives on a graph: add links until a giant connected cluster appears. I pushed it up a dimension, onto simplicial complexes where the pieces are triangles and tetrahedra, not just edges. The higher-order versions do not ease into their transition the way ordinary networks do; they snap, much closer to discontinuous. I track the giant component and the susceptibility across the occupation probability and use finite-size scaling to pin down where the jump sits.
