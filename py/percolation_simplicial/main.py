"""Percolation on simplicial complexes benchmark."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from percolation_simplicial.simplicial_complex import generate_clique_complex
from percolation_simplicial.phase_diagram import sweep_percolation
from percolation_simplicial.analysis import (
    plot_percolation_curve, plot_finite_size_scaling, save_critical_exponents
)


def main():
    outdir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print("Percolation on Simplicial Complexes")
    print("=" * 60)

    N = 200; p_edge = 0.05; n_samples = 8
    p_range = np.linspace(0, 1, 21)

    print(f"Generating clique complex (N={N}, p_edge={p_edge}) ...")
    cx = generate_clique_complex(N, p_edge, dim_max=3, seed=42)
    print(f"  Edges: {len(cx.edges)}, Triangles: {len(cx.triangles)}, Tets: {len(cx.tetrahedra)}")

    results = []
    for dim in [1, 2, 3]:
        simp_count = [len(cx.edges), len(cx.triangles), len(cx.tetrahedra)][dim - 1]
        if simp_count == 0:
            print(f"  dim={dim}: no simplices, skipping")
            continue
        print(f"  Sweeping dim={dim} ...")
        r = sweep_percolation(cx, p_range, dimension=dim, n_samples=n_samples)
        results.append(r)
        print(f"    p_c ≈ {r.p_c:.2f},  S_max = {max(r.S):.3f}")

    plot_percolation_curve(results, os.path.join(outdir, "ei_vs_rule.png"))
    print("Saved: outputs/ei_vs_rule.png")

    # Finite-size scaling for dim=1
    print("Finite-size scaling (dim=1) ...")
    fss = {}
    for N_fss in [60, 100, 120]:
        cx_fss = generate_clique_complex(N_fss, p_edge, dim_max=1, seed=42)
        fss[N_fss] = sweep_percolation(cx_fss, p_range, dimension=1, n_samples=n_samples)
    plot_finite_size_scaling(fss, os.path.join(outdir, "class_delta_ei.png"))
    print("Saved: outputs/class_delta_ei.png")

    save_critical_exponents(results, os.path.join(outdir, "rule_metrics.csv"))
    print("Saved: outputs/rule_metrics.csv")

    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(f"Percolation on Clique Complex (N={N}, p_edge={p_edge})\n")
        for r in results:
            f.write(f"  dim={r.dimension}: p_c={r.p_c:.3f}, S_max={max(r.S):.3f}\n")
    print("Saved: outputs/summary.txt\n\nDone.")


if __name__ == "__main__":
    main()
