#!/usr/bin/env python3
"""
XRD FINAL CORRECTED ANALYSIS
- LiF: ALL 4 peaks with 5 reps each, all with error bars
- KBr: 2 individual plots (like LiF structure)
- Bremsstrahlung: quadratic fits with consistent derivatives
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Constants
H_ACCEPTED = 6.626e-34  # J·s
C = 2.998e8  # m/s
E = 1.602e-19  # C
LAMBDA_K_ALPHA = 1.5406  # Å
LAMBDA_K_BETA = 1.3922  # Å
LIF_THEORETICAL = 2.008  # Å
KBR_THEORETICAL = 3.290  # Å

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Determine paths relative to script location
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
data_dir = BASE_DIR / "data"
out_dir = BASE_DIR / "plots"
out_dir.mkdir(exist_ok=True)


def load_data(filename):
    filepath = data_dir / filename
    angles, intensities = [], []
    with open(filepath, "r", encoding="latin-1") as f:
        # One repeat (LiF_exp3.5) was exported against the crystal angle theta
        # rather than the detector angle 2*theta; put it on the 2*theta axis.
        crystal_axis = f.readline().startswith("Crystal angle")
        for line in f:
            line = line.strip()
            if not line or "Detector" in line or "2*theta" in line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    angles.append(float(parts[0]))
                    intensities.append(float(parts[1]))
                except ValueError:
                    continue
    angles = np.array(angles)
    return (2 * angles if crystal_axis else angles), np.array(intensities)


def load_multi_voltage(filename):
    filepath = data_dir / filename
    angles, i25, i30, i35 = [], [], [], []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or "Detector" in line or "2*theta" in line:
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                try:
                    angles.append(float(parts[0]))
                    i25.append(float(parts[1]))
                    i30.append(float(parts[2]))
                    i35.append(float(parts[3]))
                except ValueError:
                    continue
    return np.array(angles), np.array(i25), np.array(i30), np.array(i35)


def weighted_average(angles, intensities):
    return np.sum(angles * intensities) / np.sum(intensities)


print("\n" + "=" * 80)
print("XRD FINAL CORRECTED ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 1: LiF LATTICE CONSTANT - ALL 4 PEAKS WITH ERROR BARS
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: LiF LATTICE CONSTANT")
print("=" * 80)

# Peak 1 (K_β, n=1) around 39° - 5 reps
peak1_files = ["LiF_exp2", "LiF_exp2.2", "LiF_exp2.3", "LiF_exp2.4", "LiF_exp2.5"]
all_ang1, all_int1 = [], []
for f in peak1_files:
    ang, inten = load_data(f)
    mask = (ang >= 37) & (ang <= 42)
    all_ang1.append(ang[mask])
    all_int1.append(inten[mask])

ang1 = all_ang1[0]
int1_avg = np.mean(all_int1, axis=0)
int1_std = np.std(all_int1, axis=0, ddof=1)
peak1_pos = weighted_average(ang1, int1_avg)
peak1_std = np.std([weighted_average(all_ang1[i], all_int1[i]) for i in range(5)])

# Peak 2 (K_α, n=1) around 44° - ONLY 1 file available
ang2, int2 = load_data("LiF_exp4")
mask2 = (ang2 >= 42) & (ang2 <= 46)
ang2 = ang2[mask2]
int2 = int2[mask2]
peak2_pos = weighted_average(ang2, int2)
peak2_std = 0.2  # Conservative estimate for single measurement

# Peaks 3 & 4 (n=2) - 5 reps
peak34_files = ["LiF_exp3", "LiF_exp3.2", "LiF_exp3.3", "LiF_exp3.4", "LiF_exp3.5"]
all_ang34, all_int34 = [], []
for f in peak34_files:
    ang, inten = load_data(f)
    all_ang34.append(ang)
    all_int34.append(inten)

ang34 = all_ang34[0]
int34_avg = np.mean(all_int34, axis=0)
int34_std = np.std(all_int34, axis=0, ddof=1)

# Peak 3 (K_β, n=2) around 86°
mask3 = (ang34 >= 84) & (ang34 <= 89)
peak3_pos = weighted_average(ang34[mask3], int34_avg[mask3])
peak3_std = np.std(
    [weighted_average(all_ang34[i][mask3], all_int34[i][mask3]) for i in range(5)]
)

# Peak 4 (K_α, n=2) around 99°
mask4 = (ang34 >= 97) & (ang34 <= 101)
peak4_pos = weighted_average(ang34[mask4], int34_avg[mask4])
peak4_std = np.std(
    [weighted_average(all_ang34[i][mask4], all_int34[i][mask4]) for i in range(5)]
)

# Calculate d-spacings
d1 = LAMBDA_K_BETA / (2 * np.sin(np.radians(peak1_pos / 2)))
d2 = LAMBDA_K_ALPHA / (2 * np.sin(np.radians(peak2_pos / 2)))
d3 = 2 * LAMBDA_K_BETA / (2 * np.sin(np.radians(peak3_pos / 2)))
d4 = 2 * LAMBDA_K_ALPHA / (2 * np.sin(np.radians(peak4_pos / 2)))

d_lif_avg = np.mean([d1, d2, d3, d4])
d_lif_std = np.std([d1, d2, d3, d4], ddof=1)

print(f"\nPeak 1 (K_β n=1): 2θ = {peak1_pos:.2f}±{peak1_std:.2f}°, d = {d1:.3f} Å")
print(f"Peak 2 (K_α n=1): 2θ = {peak2_pos:.2f}±{peak2_std:.2f}°, d = {d2:.3f} Å")
print(f"Peak 3 (K_β n=2): 2θ = {peak3_pos:.2f}±{peak3_std:.2f}°, d = {d3:.3f} Å")
print(f"Peak 4 (K_α n=2): 2θ = {peak4_pos:.2f}±{peak4_std:.2f}°, d = {d4:.3f} Å")
print(f"\nAverage: d_LiF = {d_lif_avg:.3f} ± {d_lif_std:.3f} Å")
print(f"Theory:  d_LiF = {LIF_THEORETICAL:.3f} Å")
print(f"Discrepancy: {abs(d_lif_avg - LIF_THEORETICAL) / LIF_THEORETICAL * 100:.2f}%")

# Generate 4 INDIVIDUAL LiF peak plots (all with error bars)
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.errorbar(
    ang1,
    int1_avg,
    yerr=int1_std,
    fmt="o-",
    markersize=4,
    capsize=3,
    linewidth=1.2,
    color="darkblue",
    label="Data",
)
ax1.axvline(
    peak1_pos, color="red", linestyle="--", linewidth=2, label=f"Peak: {peak1_pos:.2f}°"
)
ax1.set_xlabel("2θ (degrees)", fontsize=11)
ax1.set_ylabel("Intensity (counts/s)", fontsize=11)
ax1.set_title("LiF Peak 1: K_β, n=1", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "lif_peak1_n1_kbeta.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {out_dir / 'lif_peak1_n1_kbeta.png'}")

fig2, ax2 = plt.subplots(figsize=(6, 5))
# Use counting statistics for error bars on intensity (σ_I ≈ √I)
yerr2 = np.sqrt(np.clip(int2, 0, None))
ax2.errorbar(
    ang2,
    int2,
    yerr=yerr2,
    fmt="o-",
    markersize=4,
    capsize=3,
    linewidth=1.2,
    color="darkgreen",
    label="Data",
)
ax2.axvline(
    peak2_pos, color="red", linestyle="--", linewidth=2, label=f"Peak: {peak2_pos:.2f}°"
)
ax2.set_xlabel("2θ (degrees)", fontsize=11)
ax2.set_ylabel("Intensity (counts/s)", fontsize=11)
ax2.set_title("LiF Peak 2: K_α, n=1", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "lif_peak2_n1_kalpha.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {out_dir / 'lif_peak2_n1_kalpha.png'}")

fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.errorbar(
    ang34[mask3],
    int34_avg[mask3],
    yerr=int34_std[mask3],
    fmt="o-",
    markersize=4,
    capsize=3,
    linewidth=1.2,
    color="darkorange",
    label="Data",
)
ax3.axvline(
    peak3_pos, color="red", linestyle="--", linewidth=2, label=f"Peak: {peak3_pos:.2f}°"
)
ax3.set_xlabel("2θ (degrees)", fontsize=11)
ax3.set_ylabel("Intensity (counts/s)", fontsize=11)
ax3.set_title("LiF Peak 3: K_β, n=2", fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "lif_peak3_n2_kbeta.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {out_dir / 'lif_peak3_n2_kbeta.png'}")

fig4, ax4 = plt.subplots(figsize=(6, 5))
ax4.errorbar(
    ang34[mask4],
    int34_avg[mask4],
    yerr=int34_std[mask4],
    fmt="o-",
    markersize=4,
    capsize=3,
    linewidth=1.2,
    color="darkviolet",
    label="Data",
)
ax4.axvline(
    peak4_pos, color="red", linestyle="--", linewidth=2, label=f"Peak: {peak4_pos:.2f}°"
)
ax4.set_xlabel("2θ (degrees)", fontsize=11)
ax4.set_ylabel("Intensity (counts/s)", fontsize=11)
ax4.set_title("LiF Peak 4: K_α, n=2", fontweight="bold")
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "lif_peak4_n2_kalpha.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {out_dir / 'lif_peak4_n2_kalpha.png'}")

# LiF Wide Scan
ang_wide_lif, int_wide_lif = load_data("LiF_exp1")
fig_lif_wide, ax_lif_wide = plt.subplots(figsize=(12, 4))
ax_lif_wide.plot(
    ang_wide_lif,
    int_wide_lif,
    "o-",
    markersize=3,
    linewidth=1.2,
    color="black",
    label="LiF",
)
ax_lif_wide.set_xlabel("2θ (degrees)", fontsize=12)
ax_lif_wide.set_ylabel("Intensity (counts/s)", fontsize=12)
ax_lif_wide.set_title(
    "LiF X-ray Diffraction Pattern (Wide Scan)", fontweight="bold", fontsize=13
)
ax_lif_wide.legend()
ax_lif_wide.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "lif_wide_scan.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {out_dir / 'lif_wide_scan.png'}")

# ============================================================================
# PART 2: PLANCK'S CONSTANT - CONSISTENT DERIVATIVE QUADRATIC FITS
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: PLANCK'S CONSTANT")
print("=" * 80)

# Load bremsstrahlung data
brems_files = ["LiF_exp5", "LiF_exp5.2", "LiF_exp5.3", "LiF_exp5.4", "LiF_exp5.5"]
all_ang_b, all_i25, all_i30, all_i35 = [], [], [], []

for f in brems_files:
    ang, i25, i30, i35 = load_multi_voltage(f)
    all_ang_b.append(ang)
    all_i25.append(i25)
    all_i30.append(i30)
    all_i35.append(i35)

ang_brems = all_ang_b[0]
i25_avg = np.mean(all_i25, axis=0)
i30_avg = np.mean(all_i30, axis=0)
i35_avg = np.mean(all_i35, axis=0)

print("\n1. Fitting quadratics with consistent derivatives...")


def fit_bremsstrahlung_consistent(
    angles, intensities, target_deriv_at_10=None, fit_range=(4, 25)
):
    """
    Fit quadratic with constraint on derivative to ensure consistency
    """
    mask = (angles >= fit_range[0]) & (angles <= fit_range[1])
    ang_fit = angles[mask]
    int_fit = intensities[mask]

    if len(ang_fit) < 5:
        return np.nan, None

    # If no target derivative specified, fit freely
    if target_deriv_at_10 is None:
        coeffs = np.polyfit(ang_fit, int_fit, 2)
    else:
        # Constrained fit: force derivative at θ=10° to be target_deriv
        # I(θ) = a*θ² + b*θ + c
        # I'(θ) = 2*a*θ + b
        # At θ=10: I'(10) = 20a + b = target_deriv
        # So b = target_deriv_at_10 - 20a

        # Use least squares with this constraint
        # Minimize: Σ(a*θ² + b*θ + c - I)²
        # Subject to: b = target_deriv_at_10 - 20a

        # Substituting: Σ(a*θ² + (target_deriv - 20a)*θ + c - I)²
        # = Σ(a*(θ² - 20θ) + target_deriv*θ + c - I)²

        A = np.column_stack([ang_fit**2 - 20 * ang_fit, np.ones_like(ang_fit)])
        y = int_fit - target_deriv_at_10 * ang_fit

        result = np.linalg.lstsq(A, y, rcond=None)[0]
        a = result[0]
        c = result[1]
        b = target_deriv_at_10 - 20 * a
        coeffs = np.array([a, b, c])

    a, b, c = coeffs

    # Find roots
    disc = b**2 - 4 * a * c
    if disc < 0 or a >= 0:
        return np.nan, coeffs

    root1 = (-b + np.sqrt(disc)) / (2 * a)
    root2 = (-b - np.sqrt(disc)) / (2 * a)

    valid_roots = [r for r in [root1, root2] if 0 < r < 15]
    if len(valid_roots) == 0:
        return np.nan, coeffs

    return min(valid_roots), coeffs


# First, fit 25 kV freely to establish reference derivative (use 13–22° window)
theta_min_25_all, coeffs_25_all = [], []
for rep in range(5):
    th, cf = fit_bremsstrahlung_consistent(
        ang_brems, all_i25[rep], target_deriv_at_10=None, fit_range=(13, 22)
    )
    if not np.isnan(th):
        theta_min_25_all.append(th)
        coeffs_25_all.append(cf)

# Get average derivative at θ=10° from 25 kV fits
avg_coeffs_25 = np.mean(coeffs_25_all, axis=0)
ref_deriv = 2 * avg_coeffs_25[0] * 10 + avg_coeffs_25[1]

print(f"Reference derivative at θ=10°: {ref_deriv:.2f} counts/s/deg")

# Now fit all voltages with similar derivative (use 13–22° window)
theta_min_25_list, theta_min_30_list, theta_min_35_list = [], [], []
coeffs_25_list, coeffs_30_list, coeffs_35_list = [], [], []

for rep in range(5):
    th25, cf25 = fit_bremsstrahlung_consistent(
        ang_brems, all_i25[rep], ref_deriv, fit_range=(13, 22)
    )
    if not np.isnan(th25):
        theta_min_25_list.append(th25)
        coeffs_25_list.append(cf25)

    th30, cf30 = fit_bremsstrahlung_consistent(
        ang_brems, all_i30[rep], ref_deriv, fit_range=(13, 22)
    )
    if not np.isnan(th30):
        theta_min_30_list.append(th30)
        coeffs_30_list.append(cf30)

    th35, cf35 = fit_bremsstrahlung_consistent(
        ang_brems, all_i35[rep], ref_deriv, fit_range=(13, 22)
    )
    if not np.isnan(th35):
        theta_min_35_list.append(th35)
        coeffs_35_list.append(cf35)

theta_min_25 = np.mean(theta_min_25_list)
theta_min_25_err = (
    np.std(theta_min_25_list, ddof=1) if len(theta_min_25_list) > 1 else 0.5
)

theta_min_30 = np.mean(theta_min_30_list)
theta_min_30_err = (
    np.std(theta_min_30_list, ddof=1) if len(theta_min_30_list) > 1 else 0.5
)

theta_min_35 = np.mean(theta_min_35_list)
theta_min_35_err = (
    np.std(theta_min_35_list, ddof=1) if len(theta_min_35_list) > 1 else 0.5
)

coeffs_25_avg = np.mean(coeffs_25_list, axis=0)
coeffs_30_avg = np.mean(coeffs_30_list, axis=0)
coeffs_35_avg = np.mean(coeffs_35_list, axis=0)

print(f"\nθ_min from consistent quadratic fits:")
print(
    f"  25 kV: θ_min = {theta_min_25:.2f} ± {theta_min_25_err:.2f}° ({len(theta_min_25_list)}/5 valid)"
)
print(
    f"  30 kV: θ_min = {theta_min_30:.2f} ± {theta_min_30_err:.2f}° ({len(theta_min_30_list)}/5 valid)"
)
print(
    f"  35 kV: θ_min = {theta_min_35:.2f} ± {theta_min_35_err:.2f}° ({len(theta_min_35_list)}/5 valid)"
)

# Calculate λ_min (NOTE: data angles are detector 2θ; Bragg uses θ = (2θ)/2)
print(
    f"\n2. Calculating λ_min = 2 d sin(θ) using θ = (2θ_det)/2 and d = {d_lif_avg:.4f} Å..."
)

theta25 = theta_min_25 / 2.0
theta30 = theta_min_30 / 2.0
theta35 = theta_min_35 / 2.0

theta25_err = theta_min_25_err / 2.0
theta30_err = theta_min_30_err / 2.0
theta35_err = theta_min_35_err / 2.0

lambda_min_25 = 2 * d_lif_avg * np.sin(np.radians(theta25))
lambda_min_30 = 2 * d_lif_avg * np.sin(np.radians(theta30))
lambda_min_35 = 2 * d_lif_avg * np.sin(np.radians(theta35))

# Uncertainty propagation: σ_λ ≈ λ |cot θ| σ_θ (θ in radians, θ is Bragg angle)
lambda_err_25 = (
    lambda_min_25 * abs(1 / np.tan(np.radians(theta25))) * np.radians(theta25_err)
)
lambda_err_30 = (
    lambda_min_30 * abs(1 / np.tan(np.radians(theta30))) * np.radians(theta30_err)
)
lambda_err_35 = (
    lambda_min_35 * abs(1 / np.tan(np.radians(theta35))) * np.radians(theta35_err)
)

print(f"\nλ_min values:")
print(f"  25 kV: λ_min = {lambda_min_25:.4f} ± {lambda_err_25:.4f} Å")
print(f"  30 kV: λ_min = {lambda_min_30:.4f} ± {lambda_err_30:.4f} Å")
print(f"  35 kV: λ_min = {lambda_min_35:.4f} ± {lambda_err_35:.4f} Å")

# Results table
print("\n" + "=" * 60)
print("TABLE: Bremsstrahlung Cutoff Analysis")
print("=" * 60)
print(f"{'Voltage (kV)':<15} {'θ_min (°)':<20} {'λ_min (Å)':<20}")
print("-" * 60)
print(
    f"{25:<15} {theta_min_25:.2f} ± {theta_min_25_err:.2f}{' ' * 10} {lambda_min_25:.4f} ± {lambda_err_25:.4f}"
)
print(
    f"{30:<15} {theta_min_30:.2f} ± {theta_min_30_err:.2f}{' ' * 10} {lambda_min_30:.4f} ± {lambda_err_30:.4f}"
)
print(
    f"{35:<15} {theta_min_35:.2f} ± {theta_min_35_err:.2f}{' ' * 10} {lambda_min_35:.4f} ± {lambda_err_35:.4f}"
)
print("=" * 60)

# Linear regression
print("\n3. Linear regression: λ_min vs 1/V to determine h...")

voltages = np.array([25, 30, 35]) * 1000
inv_V = 1 / voltages
lambda_values = np.array([lambda_min_25, lambda_min_30, lambda_min_35]) * 1e-10
lambda_errors = np.array([lambda_err_25, lambda_err_30, lambda_err_35]) * 1e-10

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_V, lambda_values)
h_calculated = (slope * E) / C

print(f"\nLinear fit: λ_min = {slope:.4e} × (1/V) + {intercept:.4e}")
print(f"R² = {r_value**2:.4f}")
print(f"\nPlanck's constant:")
print(f"  h_experimental = {h_calculated:.3e} J·s")
print(f"  h_accepted = {H_ACCEPTED:.3e} J·s")
print(f"  Error: {abs(h_calculated - H_ACCEPTED) / H_ACCEPTED * 100:.1f}%")

# Bremsstrahlung plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = {25: "#C74440", 30: "#388C46", 35: "#2D6FB1"}

# Full spectrum
for v, intensity in [(25, i25_avg), (30, i30_avg), (35, i35_avg)]:
    ax1.plot(ang_brems, intensity, "-", linewidth=1.5, color=colors[v], label=f"{v} kV")

ax1.set_xlabel("2θ (degrees)", fontsize=11)
ax1.set_ylabel("Intensity (counts/s)", fontsize=11)
ax1.set_title("Bremsstrahlung Continuous Spectrum", fontweight="bold")
ax1.legend()
ax1.set_xlim(0, 40)
ax1.grid(True, alpha=0.3)

# Zoom with consistent quadratic fits
zoom_mask = (ang_brems >= 0) & (ang_brems <= 30)

for v, coeffs, theta_min in [
    (25, coeffs_25_avg, theta_min_25),
    (30, coeffs_30_avg, theta_min_30),
    (35, coeffs_35_avg, theta_min_35),
]:
    intensity = {"25": i25_avg, "30": i30_avg, "35": i35_avg}[str(v)]

    ax2.plot(
        ang_brems[zoom_mask],
        intensity[zoom_mask],
        "o",
        markersize=4,
        color=colors[v],
        alpha=0.5,
    )

    fit_angles = np.linspace(0, 30, 300)
    fit_curve = np.polyval(coeffs, fit_angles)
    ax2.plot(
        fit_angles, fit_curve, "--", linewidth=2, color=colors[v], label=f"{v} kV fit"
    )

    if not np.isnan(theta_min):
        ax2.axvline(theta_min, color=colors[v], linestyle=":", linewidth=1.5, alpha=0.7)

ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
ax2.set_xlabel("2θ (degrees)", fontsize=11)
ax2.set_ylabel("Intensity (counts/s)", fontsize=11)
ax2.set_title("Quadratic Fits", fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xlim(5, 23)
ax2.set_ylim(bottom=-1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "bremsstrahlung_analysis.png", dpi=300, bbox_inches="tight")
print(f"\n✓ Saved: {out_dir / 'bremsstrahlung_analysis.png'}")

# Linear regression plot
fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(
    inv_V * 1000,
    lambda_values * 1e10,
    yerr=lambda_errors * 1e10,
    fmt="o",
    markersize=8,
    capsize=5,
    capthick=2,
    color="darkblue",
    ecolor="darkblue",
    label="Experimental data",
)

fit_line = slope * inv_V + intercept
ax.plot(
    inv_V * 1000,
    fit_line * 1e10,
    "r--",
    linewidth=2,
    label=f"Linear fit\nR² = {r_value**2:.4f}\nh = {h_calculated:.3e} J·s",
)

ax.set_xlabel("1/V (kV⁻¹)", fontsize=12)
ax.set_ylabel("λ_min (Å)", fontsize=12)
ax.set_title("Planck's Constant Determination", fontweight="bold", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "planck_linear_regression.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {out_dir / 'planck_linear_regression.png'}")

# ============================================================================
# PART 3: KBr - FOUR PEAKS (Kβ/Kα doublets for n=1 and n=2)
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: KBr IDENTIFICATION - FOUR PEAKS (DOUBLETS)")
print("=" * 80)

# Load KBr narrow scans (5 reps each)
kbr_low_files = ["KBr_exp7", "KBr_exp7.2", "KBr_exp7.3", "KBr_exp7.4", "KBr_exp7.5"]
kbr_high_files = ["KBr_exp8", "KBr_exp8.2", "KBr_exp8.3", "KBr_exp8.4", "KBr_exp8.5"]

all_ang_low, all_int_low = [], []
for f in kbr_low_files:
    ang, inten = load_data(f)
    mask = (ang >= 22) & (ang <= 30)
    all_ang_low.append(ang[mask])
    all_int_low.append(inten[mask])

all_ang_high, all_int_high = [], []
for f in kbr_high_files:
    ang, inten = load_data(f)
    mask = (ang >= 48) & (ang <= 60)
    all_ang_high.append(ang[mask])
    all_int_high.append(inten[mask])

# Establish common angle axes from first rep for plotting
ang_low = all_ang_low[0]
ang_high = all_ang_high[0]

# Split each order window into two sub-peaks using sub-windows
low_sub1_mask = (ang_low >= 23.8) & (ang_low <= 25.2)  # ~24° (n=1, Kβ)
low_sub2_mask = (ang_low >= 26.4) & (ang_low <= 28.2)  # ~27° (n=1, Kα)

high_sub1_mask = (ang_high >= 49.0) & (ang_high <= 51.5)  # ~50° (n=2, Kβ)
high_sub2_mask = (ang_high >= 55.0) & (ang_high <= 56.8)  # ~56° (n=2, Kα)


def per_rep_weighted_positions(all_angles, all_intensities, mask):
    positions = []
    for i in range(len(all_angles)):
        ang_i = all_angles[i]
        int_i = all_intensities[i]
        values = int_i[mask]
        if values.size == 0 or np.sum(values) == 0:
            continue
        positions.append(weighted_average(ang_i[mask], int_i[mask]))
    return np.array(positions)


# Compute per-rep positions and stats
pos_low1 = per_rep_weighted_positions(all_ang_low, all_int_low, low_sub1_mask)
pos_low2 = per_rep_weighted_positions(all_ang_low, all_int_low, low_sub2_mask)
pos_high1 = per_rep_weighted_positions(all_ang_high, all_int_high, high_sub1_mask)
pos_high2 = per_rep_weighted_positions(all_ang_high, all_int_high, high_sub2_mask)

theta_low1 = np.mean(pos_low1)
theta_low1_err = np.std(pos_low1, ddof=1) if pos_low1.size > 1 else 0.2

theta_low2 = np.mean(pos_low2)
theta_low2_err = np.std(pos_low2, ddof=1) if pos_low2.size > 1 else 0.2

theta_high1 = np.mean(pos_high1)
theta_high1_err = np.std(pos_high1, ddof=1) if pos_high1.size > 1 else 0.3

theta_high2 = np.mean(pos_high2)
theta_high2_err = np.std(pos_high2, ddof=1) if pos_high2.size > 1 else 0.3


def avg_std_in_window(all_intensities, mask):
    vals = np.array([ints[mask] for ints in all_intensities])
    return np.mean(vals, axis=0), np.std(vals, axis=0, ddof=1)


int_low1_avg, int_low1_std = avg_std_in_window(all_int_low, low_sub1_mask)
int_low2_avg, int_low2_std = avg_std_in_window(all_int_low, low_sub2_mask)
int_high1_avg, int_high1_std = avg_std_in_window(all_int_high, high_sub1_mask)
int_high2_avg, int_high2_std = avg_std_in_window(all_int_high, high_sub2_mask)

# Compute d for each identified sub-peak with correct wavelength/order
d_low1 = LAMBDA_K_BETA / (2 * np.sin(np.radians(theta_low1 / 2)))
d_low2 = LAMBDA_K_ALPHA / (2 * np.sin(np.radians(theta_low2 / 2)))
d_high1 = 2 * LAMBDA_K_BETA / (2 * np.sin(np.radians(theta_high1 / 2)))
d_high2 = 2 * LAMBDA_K_ALPHA / (2 * np.sin(np.radians(theta_high2 / 2)))


# Uncertainty propagation for d from 2θ uncertainty
def d_uncertainty_from_2theta(
    d_val: float, two_theta_deg: float, two_theta_err_deg: float
) -> float:
    theta_rad = np.radians(two_theta_deg / 2.0)
    theta_err_rad = np.radians(two_theta_err_deg / 2.0)
    # σ_d ≈ d |cot(θ)| σ_θ
    return float(d_val * abs(1.0 / np.tan(theta_rad)) * theta_err_rad)


d_low1_err = d_uncertainty_from_2theta(d_low1, theta_low1, theta_low1_err)
d_low2_err = d_uncertainty_from_2theta(d_low2, theta_low2, theta_low2_err)
d_high1_err = d_uncertainty_from_2theta(d_high1, theta_high1, theta_high1_err)
d_high2_err = d_uncertainty_from_2theta(d_high2, theta_high2, theta_high2_err)

print(
    f"\nPeak 1 (n=1, Kβ): 2θ = {theta_low1:.2f}±{theta_low1_err:.2f}°, d = {d_low1:.3f}±{d_low1_err:.3f} Å"
)
print(
    f"Peak 2 (n=1, Kα): 2θ = {theta_low2:.2f}±{theta_low2_err:.2f}°, d = {d_low2:.3f}±{d_low2_err:.3f} Å"
)
print(
    f"Peak 3 (n=2, Kβ): 2θ = {theta_high1:.2f}±{theta_high1_err:.2f}°, d = {d_high1:.3f}±{d_high1_err:.3f} Å"
)
print(
    f"Peak 4 (n=2, Kα): 2θ = {theta_high2:.2f}±{theta_high2_err:.2f}°, d = {d_high2:.3f}±{d_high2_err:.3f} Å"
)


def plot_subpeak(ang_subset, int_avg, int_std, theta_peak, title, color, outfile):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(
        ang_subset,
        int_avg,
        yerr=int_std,
        fmt="o-",
        markersize=4,
        capsize=3,
        linewidth=1.2,
        color=color,
        label="Data",
    )
    ax.axvline(
        theta_peak,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Peak: {theta_peak:.2f}°",
    )
    ax.set_xlabel("2θ (degrees)", fontsize=11)
    ax.set_ylabel("Intensity (counts/s)", fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out_dir / outfile}")


# Generate four KBr peak plots
plot_subpeak(
    ang_low[low_sub1_mask],
    int_low1_avg,
    int_low1_std,
    theta_low1,
    f"KBr Peak 1: Kβ, n=1 (2θ = {theta_low1:.2f}°; d = {d_low1:.3f}±{d_low1_err:.3f} Å)",
    "#2D6FB1",
    "kbr_peak1_n1_kbeta.png",
)

plot_subpeak(
    ang_low[low_sub2_mask],
    int_low2_avg,
    int_low2_std,
    theta_low2,
    f"KBr Peak 2: Kα, n=1 (2θ = {theta_low2:.2f}°; d = {d_low2:.3f}±{d_low2_err:.3f} Å)",
    "#388C46",
    "kbr_peak2_n1_kalpha.png",
)

plot_subpeak(
    ang_high[high_sub1_mask],
    int_high1_avg,
    int_high1_std,
    theta_high1,
    f"KBr Peak 3: Kβ, n=2 (2θ = {theta_high1:.2f}°; d = {d_high1:.3f}±{d_high1_err:.3f} Å)",
    "#C27BA0",
    "kbr_peak3_n2_kbeta.png",
)

plot_subpeak(
    ang_high[high_sub2_mask],
    int_high2_avg,
    int_high2_std,
    theta_high2,
    f"KBr Peak 4: Kα, n=2 (2θ = {theta_high2:.2f}°; d = {d_high2:.3f}±{d_high2_err:.3f} Å)",
    "#8C3A2D",
    "kbr_peak4_n2_kalpha.png",
)

# KBr Wide Scan
ang_wide_kbr, int_wide_kbr = load_data("KBr_exp6")
fig_kbr_wide, ax_kbr_wide = plt.subplots(figsize=(12, 4))
ax_kbr_wide.plot(
    ang_wide_kbr,
    int_wide_kbr,
    "o-",
    markersize=3,
    linewidth=1.2,
    color="black",
    label="KBr",
)
ax_kbr_wide.set_xlabel("2θ (degrees)", fontsize=12)
ax_kbr_wide.set_ylabel("Intensity (counts/s)", fontsize=12)
ax_kbr_wide.set_title(
    "KBr X-ray Diffraction Pattern (Wide Scan)", fontweight="bold", fontsize=13
)
ax_kbr_wide.legend()
ax_kbr_wide.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "kbr_wide_scan.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {out_dir / 'kbr_wide_scan.png'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# ============================================================================
# REPORT FIGURES (printed size, shared plot style)
# ============================================================================
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))  # plotstyle.py lives next to this script
from plotstyle import *  # noqa: E402,F403

apply_style()
FIG = BASE_DIR / "report" / "figures"
plt.close("all")
LINE_COLORS = {25: GOLD, 30: RED, 35: BLUE}  # tube voltage, low to high


def wide_scan(ang, inten, peaks, name):
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    ax.plot(ang, inten, color=BLUE, lw=0.8)
    for x, lab in peaks:
        y = inten[np.argmin(abs(ang - x))]
        ax.annotate(lab, (x, y), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7.5, color=INK)
    ax.set_xlabel(r"Detector angle, $2\theta$ (°)")
    ax.set_ylabel("Count rate (s$^{-1}$)")
    ax.set_ylim(0, inten.max() * 1.18)
    save(fig, FIG, name)


def peak_grid(panels, name):
    """2x2 grid: top row first order, bottom row second order; error bars are the
    standard deviation over the repeated scans, dashed line the intensity-weighted centre."""
    fig, axs = plt.subplots(2, 2, figsize=(COL, 2.9))
    for ax, (x, y, dy, centre, title) in zip(axs.flat, panels):
        if dy is None:
            ax.plot(x, y, "o-", color=BLUE, ms=2, lw=0.7)
        else:
            ax.errorbar(x, y, yerr=dy, fmt="o-", color=BLUE, ms=2, lw=0.7, elinewidth=0.6)
        ax.axvline(centre, color=RED, ls="--", lw=0.8)
        ax.set_title(title + f"\n$2\\theta$ = {centre:.2f}°", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    for ax in axs[1]:
        ax.set_xlabel(r"$2\theta$ (°)", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel("Count rate (s$^{-1}$)", fontsize=8)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
    save(fig, FIG, name)


wide_scan(ang_wide_lif, int_wide_lif,
          [(peak1_pos, r"K$_\beta$"), (peak2_pos, r"K$_\alpha$"), (peak3_pos, r"K$_\beta$"), (peak4_pos, r"K$_\alpha$")],
          "lif_wide_scan")
peak_grid([
    (ang1, int1_avg, int1_std, peak1_pos, r"K$_\beta$, $n=1$"),
    (ang2, int2, None, peak2_pos, r"K$_\alpha$, $n=1$ (one scan)"),
    (ang34[mask3], int34_avg[mask3], int34_std[mask3], peak3_pos, r"K$_\beta$, $n=2$"),
    (ang34[mask4], int34_avg[mask4], int34_std[mask4], peak4_pos, r"K$_\alpha$, $n=2$"),
], "lif_peaks")
wide_scan(ang_wide_kbr, int_wide_kbr,
          [(theta_low1, r"K$_\beta$"), (theta_low2, r"K$_\alpha$"), (theta_high1, r"K$_\beta$"), (theta_high2, r"K$_\alpha$")],
          "kbr_wide_scan")
peak_grid([
    (ang_low[low_sub1_mask], int_low1_avg, int_low1_std, theta_low1, r"K$_\beta$, $n=1$"),
    (ang_low[low_sub2_mask], int_low2_avg, int_low2_std, theta_low2, r"K$_\alpha$, $n=1$"),
    (ang_high[high_sub1_mask], int_high1_avg, int_high1_std, theta_high1, r"K$_\beta$, $n=2$"),
    (ang_high[high_sub2_mask], int_high2_avg, int_high2_std, theta_high2, r"K$_\alpha$, $n=2$"),
], "kbr_peaks")

# Bremsstrahlung: (a) full spectra, (b) cutoff region with the constrained quadratic fits
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.3))
for v, inten in ((25, i25_avg), (30, i30_avg), (35, i35_avg)):
    a.plot(ang_brems, inten, color=LINE_COLORS[v], lw=0.9)
    a.text(ang_brems[np.argmax(inten)] + 0.8, inten.max(), f"{v} kV", color=LINE_COLORS[v], va="center")
a.set_xlim(0, 40)
a.set_xlabel(r"Detector angle, $2\theta$ (°)")
a.set_ylabel("Count rate (s$^{-1}$)")
label(a, "a")
zm = (ang_brems >= 5) & (ang_brems <= 23)
fx = np.linspace(5, 23, 300)
for v, inten, cf, tm in ((25, i25_avg, coeffs_25_avg, theta_min_25), (30, i30_avg, coeffs_30_avg, theta_min_30),
                         (35, i35_avg, coeffs_35_avg, theta_min_35)):
    b.plot(ang_brems[zm], inten[zm], "o", color=LINE_COLORS[v], ms=2.2)
    b.plot(fx, np.polyval(cf, fx), color=LINE_COLORS[v], lw=0.8, alpha=0.7)
    b.axvline(tm, color=LINE_COLORS[v], ls=":", lw=0.9)
    b.text(tm, -0.9, f"{tm:.1f}°", color=LINE_COLORS[v], ha="center", va="top", fontsize=7.5,
           path_effects=halo())
b.axhline(0, color=LIGHT, lw=0.7)
b.axvspan(13, 22, color=LIGHT, alpha=0.35, lw=0)
b.set_xlim(5, 23)
b.set_ylim(-3, 23)
b.text(17.5, 22.5, "fit window", ha="center", va="top", color=GREY, fontsize=7.5)
b.set_xlabel(r"Detector angle, $2\theta$ (°)")
b.set_ylabel("Count rate (s$^{-1}$)")
label(b, "b")
fig.tight_layout(w_pad=1.5)
save(fig, FIG, "bremsstrahlung_analysis")

# Planck regression
fig, ax = plt.subplots(figsize=(COL, 2.3))
xv = inv_V * 1e3
xx = np.linspace(xv.min() * 0.95, xv.max() * 1.05, 50)
ax.plot(xx, (slope * xx / 1e3 + intercept) * 1e10, color=GREY, lw=0.9)
ax.errorbar(xv, lambda_values * 1e10, yerr=lambda_errors * 1e10, fmt="o", color=BLUE, ms=3.5)
for x, y, v in zip(xv, lambda_values * 1e10, (25, 30, 35)):
    ax.annotate(f"{v} kV", (x, y), xytext=(6, -2), textcoords="offset points", fontsize=7.5)
ax.set_xlabel(r"Inverse tube voltage, $1/V$ (kV$^{-1}$)")
ax.set_ylabel(r"Cutoff wavelength, $\lambda_{\rm min}$ (Å)")
save(fig, FIG, "planck_linear_regression")
print(f"report figures written; h = {h_calculated:.3e}, R2 = {r_value**2:.4f}")
