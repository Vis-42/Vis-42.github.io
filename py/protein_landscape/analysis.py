"""Analysis and plotting for protein landscape."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")


def plot_free_energy(Q_centers, F_Q, T, path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0c0c0e")
    ax.set_facecolor("#111113"); ax.tick_params(colors="#71717a")
    for sp in ax.spines.values(): sp.set_color("#222226")
    ax.plot(Q_centers, F_Q, color="#22c55e", linewidth=2.5)
    ax.fill_between(Q_centers, F_Q, alpha=0.15, color="#22c55e")
    ax.set_xlabel("Q (fraction native contacts)", color="#a1a1aa")
    ax.set_ylabel("F (kT units)", color="#a1a1aa")
    ax.set_title(f"Free Energy Profile F(Q)  T={T:.2f}", color="#f4f4f5")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0c0c0e"); plt.close()


def plot_folding_trajectory(Q_traj, E_traj, T, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), facecolor="#0c0c0e")
    for ax in axes:
        ax.set_facecolor("#111113"); ax.tick_params(colors="#71717a")
        for sp in ax.spines.values(): sp.set_color("#222226")
    axes[0].plot(Q_traj, color="#22c55e", linewidth=1.2, alpha=0.8)
    axes[0].set_ylabel("Q", color="#a1a1aa"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Folding trajectory T={T:.2f}", color="#f4f4f5")
    axes[1].plot(E_traj, color="#f97316", linewidth=1.2, alpha=0.8)
    axes[1].set_xlabel("MC sample", color="#a1a1aa"); axes[1].set_ylabel("Energy", color="#a1a1aa")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0c0c0e"); plt.close()


def save_summary(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("T,mean_Q,std_Q,mean_E,std_E,n_samples\n")
        for r in results:
            mQ = float(np.mean(r.Q_traj)) if r.Q_traj else 0.0
            sQ = float(np.std(r.Q_traj)) if len(r.Q_traj) > 1 else 0.0
            mE = float(np.mean(r.E_traj)) if r.E_traj else 0.0
            sE = float(np.std(r.E_traj)) if len(r.E_traj) > 1 else 0.0
            f.write(f"{r.T},{mQ:.4f},{sQ:.4f},{mE:.4f},{sE:.4f},{len(r.Q_traj)}\n")
