"""Spin glass NN — Hessian spectrum benchmark."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from spin_glass_nn.network import MLP
from spin_glass_nn.data_generation import make_spiral
from spin_glass_nn.hessian_spectrum import ipr, participation_ratio
from spin_glass_nn.analysis import (
    plot_spectrum_evolution, plot_loss_curve, plot_ipr_evolution, save_summary
)


def main():
    outdir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print("Spin Glass NN — Hessian Spectrum During Training")
    print("=" * 60)

    X, y = make_spiral(80, noise=0.15, seed=42)
    print(f"Dataset: spiral, N={len(y)}")

    mlp = MLP(n_in=2, n_hidden=20, n_out=2, seed=7)

    n_epochs = 500
    snap_every = 50
    snapshots = []

    print("Training ...")
    for epoch in range(1, n_epochs + 1):
        current_loss = mlp.gradient_step(X, y, lr=0.02)
        if epoch % snap_every == 0:
            diag = mlp.hessian_diagonal(X, y, eps=5e-4)
            ip = ipr(diag)
            pr = participation_ratio(diag)
            n0 = int(np.sum(np.abs(diag) < 0.01))
            snapshots.append({"epoch": epoch, "loss": current_loss, "diag": diag})
            print(f"  epoch {epoch}: loss={current_loss:.4f}  IPR={ip:.4f}  PR={pr:.1f}  |Hkk|<0.01: {n0}")

    plot_spectrum_evolution(snapshots, os.path.join(outdir, "spectrum_evolution.png"))
    print("Saved: outputs/spectrum_evolution.png")

    plot_loss_curve([s["loss"] for s in snapshots],
                   [s["epoch"] for s in snapshots],
                   os.path.join(outdir, "loss_curve.png"))
    print("Saved: outputs/loss_curve.png")

    plot_ipr_evolution(snapshots, os.path.join(outdir, "ipr_evolution.png"))
    print("Saved: outputs/ipr_evolution.png")

    save_summary(snapshots, os.path.join(outdir, "summary.csv"))
    print("Saved: outputs/summary.csv")

    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write("Spin Glass NN — Spiral Dataset\n")
        f.write(f"Final loss: {snapshots[-1]['loss']:.4f}\n")
        f.write(f"Final IPR: {ipr(snapshots[-1]['diag']):.4f}\n")
        f.write(f"Initial IPR: {ipr(snapshots[0]['diag']):.4f}\n")
    print("Saved: outputs/summary.txt\n\nDone.")


if __name__ == "__main__":
    main()
