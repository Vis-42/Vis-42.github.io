"""Analysis and plotting for spin glass NN."""

from __future__ import annotations
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .hessian_spectrum import ipr, participation_ratio


def plot_spectrum_evolution(snapshots: list[dict], path: str) -> None:
    n = len(snapshots)
    cols = plt.cm.plasma(np.linspace(0.2, 0.9, n))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0c0c0e")
    ax.set_facecolor("#111113")
    for snap, c in zip(snapshots, cols):
        spec = np.sort(snap["diag"])
        ax.hist(spec, bins=30, density=True, alpha=0.5, color=c,
                label=f"epoch {snap['epoch']}")
    ax.set_xlabel("λ", color="#a1a1aa"); ax.set_ylabel("density", color="#a1a1aa")
    ax.set_title("Hessian Diagonal Spectrum Evolution", color="#f4f4f5")
    ax.tick_params(colors="#71717a")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax.spines[s].set_color("#222226")
    if n <= 6:
        ax.legend(framealpha=0.3, labelcolor="#a1a1aa", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curve(losses: list[float], epochs: list[int], path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0c0c0e")
    ax.set_facecolor("#111113")
    ax.plot(epochs, losses, color="#f97316", lw=2)
    ax.set_xlabel("Epoch", color="#a1a1aa"); ax.set_ylabel("Loss", color="#a1a1aa")
    ax.set_title("Training Loss", color="#f4f4f5")
    ax.tick_params(colors="#71717a")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["bottom","left"]: ax.spines[s].set_color("#222226")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ipr_evolution(snapshots: list[dict], path: str) -> None:
    epochs = [s["epoch"] for s in snapshots]
    iprs   = [ipr(s["diag"]) for s in snapshots]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0c0c0e")
    ax.set_facecolor("#111113")
    ax.plot(epochs, iprs, color="#a78bfa", lw=2)
    ax.set_xlabel("Epoch", color="#a1a1aa"); ax.set_ylabel("IPR", color="#a1a1aa")
    ax.set_title("Hessian IPR Evolution (glass → order)", color="#f4f4f5")
    ax.tick_params(colors="#71717a")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["bottom","left"]: ax.spines[s].set_color("#222226")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_summary(snapshots: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "ipr", "pr", "n_near_zero"])
        for s in snapshots:
            diag = s["diag"]
            writer.writerow([
                s["epoch"], s["loss"],
                ipr(diag), participation_ratio(diag),
                int(np.sum(np.abs(diag) < 0.05))
            ])
