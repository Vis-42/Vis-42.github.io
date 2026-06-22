"""Analysis and plotting for CD inversion."""
import os, matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
from cd_forward_model import WAVELENGTHS


def plot_forward_model(path):
    from cd_forward_model import HELIX_BASIS, SHEET_BASIS, COIL_BASIS
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0c0c0e"); ax.set_facecolor("#111113")
    for sp in ax.spines.values(): sp.set_color("#222226"); ax.tick_params(colors="#71717a")
    ax.plot(WAVELENGTHS, HELIX_BASIS, color="#a78bfa", lw=2, label="α-Helix")
    ax.plot(WAVELENGTHS, SHEET_BASIS, color="#22c55e", lw=2, label="β-Sheet")
    ax.plot(WAVELENGTHS, COIL_BASIS, color="#f97316", lw=2, label="Coil")
    ax.set_xlabel("Wavelength (nm)", color="#a1a1aa"); ax.set_ylabel("Ellipticity (mdeg)", color="#a1a1aa")
    ax.set_title("CD Reference Spectra", color="#f4f4f5")
    ax.legend(facecolor="#111113", edgecolor="#222226", labelcolor="#a1a1aa")
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0c0c0e"); plt.close()


def plot_training_curve(losses_with, losses_without, path):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0c0c0e"); ax.set_facecolor("#111113")
    for sp in ax.spines.values(): sp.set_color("#222226"); ax.tick_params(colors="#71717a")
    ax.semilogy(losses_with, color="#22c55e", lw=2, label="With physics")
    ax.semilogy(losses_without, color="#a78bfa", lw=2, linestyle="--", label="Without physics")
    ax.set_xlabel("Epoch", color="#a1a1aa"); ax.set_ylabel("Loss", color="#a1a1aa")
    ax.set_title("Training Loss", color="#f4f4f5")
    ax.legend(facecolor="#111113", edgecolor="#222226", labelcolor="#a1a1aa")
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0c0c0e"); plt.close()


def save_summary(res_with, res_without, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("condition,test_rmse,spectral_r2\n")
        for label, res in [("with_physics", res_with), ("without_physics", res_without)]:
            f.write(f"{label},{res['rmse']:.4f},{res['r2']:.4f}\n")
