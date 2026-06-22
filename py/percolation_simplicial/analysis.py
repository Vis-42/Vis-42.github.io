"""Analysis and plotting for percolation results."""

from __future__ import annotations
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .phase_diagram import PercolationResult


def plot_percolation_curve(results: list[PercolationResult], path: str) -> None:
    colors = ["#22c55e", "#a78bfa", "#f97316"]
    labels = ["k=1 (edges)", "k=2 (triangles)", "k=3 (tetrahedra)"]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0c0c0e")
    ax.set_facecolor("#111113")
    for r, c, l in zip(results, colors, labels):
        ax.plot(r.p_range, r.S, color=c, lw=2, label=f"{l}  p_c≈{r.p_c:.2f}")
    ax.set_xlabel("Bond probability p", color="#a1a1aa")
    ax.set_ylabel("Giant component S(p)", color="#a1a1aa")
    ax.set_title("Percolation on Simplicial Complex", color="#f4f4f5")
    ax.tick_params(colors="#71717a")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax.spines[s].set_color("#222226")
    ax.legend(framealpha=0.3, labelcolor="#a1a1aa")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_finite_size_scaling(results_by_N: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0c0c0e")
    ax.set_facecolor("#111113")
    colors = ["#22c55e", "#a78bfa", "#f97316", "#38bdf8"]
    for (N, r), c in zip(results_by_N.items(), colors):
        ax.plot(r.p_range, r.chi, color=c, lw=2, label=f"N={N}")
    ax.set_xlabel("p", color="#a1a1aa"); ax.set_ylabel("χ(p)", color="#a1a1aa")
    ax.set_title("Finite-Size Scaling of Susceptibility", color="#f4f4f5")
    ax.tick_params(colors="#71717a")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["bottom","left"]: ax.spines[s].set_color("#222226")
    ax.legend(framealpha=0.3, labelcolor="#a1a1aa")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_critical_exponents(results: list[PercolationResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "N", "p_c", "S_max", "chi_max"])
        for r in results:
            writer.writerow([r.dimension, r.N, r.p_c, max(r.S), max(r.chi)])
