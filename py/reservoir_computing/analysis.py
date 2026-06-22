"""Analysis and visualization for reservoir computing results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_diagram import PhasePoint


def plot_phase_diagram(results: List[PhasePoint], path: Path) -> None:
    sparsities = sorted(set(r.sparsity for r in results))
    spectral   = sorted(set(r.spectral_radius for r in results))
    grid = np.zeros((len(sparsities), len(spectral)))
    s_idx = {s: i for i, s in enumerate(sparsities)}
    sr_idx = {sr: j for j, sr in enumerate(spectral)}
    for r in results:
        grid[s_idx[r.sparsity], sr_idx[r.spectral_radius]] = r.prediction_horizon
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0c0c0e")
    ax.set_facecolor("#111113")
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[min(spectral), max(spectral), min(sparsities), max(sparsities)])
    plt.colorbar(im, ax=ax, label="Prediction horizon (Lyapunov times)")
    ax.set_xlabel("Spectral radius ρ", color="#a1a1aa")
    ax.set_ylabel("Sparsity", color="#a1a1aa")
    ax.set_title("ESN Phase Diagram — Lorenz-63 Prediction Horizon", color="#f4f4f5")
    ax.tick_params(colors="#52525b")
    for spine in ax.spines.values(): spine.set_edgecolor("#222226")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_prediction(true_traj: np.ndarray, pred_traj: np.ndarray, path: Path) -> None:
    n = min(len(true_traj), len(pred_traj))
    t = np.arange(n) * 0.02
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor="#0c0c0e")
    labels = ["x", "y", "z"]
    colors_true = "#1e90ff"
    colors_pred = "#22c55e"
    for i, ax in enumerate(axes):
        ax.set_facecolor("#111113")
        ax.plot(t, true_traj[:n, i], color=colors_true, linewidth=1.2, label="True")
        ax.plot(t, pred_traj[:n, i], color=colors_pred, linewidth=1.2, linestyle="--", label="ESN")
        ax.set_ylabel(labels[i], color="#a1a1aa")
        ax.tick_params(colors="#52525b")
        for spine in ax.spines.values(): spine.set_edgecolor("#222226")
        if i == 0:
            ax.legend(facecolor="#111113", labelcolor="#a1a1aa", edgecolor="#222226")
    axes[-1].set_xlabel("Time", color="#a1a1aa")
    axes[0].set_title("ESN Prediction vs True Lorenz-63", color="#f4f4f5")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_summary(results: List[PhasePoint], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sparsity", "spectral_radius", "prediction_horizon_lya", "rmse"])
        w.writeheader()
        for r in results:
            w.writerow({"sparsity": r.sparsity, "spectral_radius": r.spectral_radius,
                        "prediction_horizon_lya": r.prediction_horizon, "rmse": r.rmse})
