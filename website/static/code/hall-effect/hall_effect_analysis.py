#!/usr/bin/env python3
"""
Hall Effect Experiment Data Analysis
PC2193 Experimental Physics & Data Analysis
Experiment D: Hall Effect in p-type and n-type Germanium

This program analyzes experimental data from Hall effect measurements on
germanium semiconductors to determine:
- Carrier type (electrons vs holes)
- Hall coefficient
- Carrier density
- Carrier mobility (via Hall effect and magnetoresistance)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Set publication-quality plot parameters
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"

# Physical constants
e = 1.602e-19  # Elementary charge (C)

# Sample dimensions (from experiment manual)
L = 20e-3  # Length (m)
w = 10e-3  # Width (m)
d = 1e-3  # Thickness (m)
A = w * d  # Cross-sectional area (m²)

# Instrument uncertainties
delta_I = 0.1e-3  # Current uncertainty (A)
delta_B = 2e-3  # Magnetic field uncertainty (T)
delta_UH = 0.5e-3  # Hall voltage uncertainty (V)
delta_Up = 0.5e-3  # Sample voltage uncertainty (V)
delta_d = 0.05e-3  # Thickness uncertainty (m)

print("=" * 70)
print("HALL EFFECT EXPERIMENT - DATA ANALYSIS")
print("=" * 70)
print(f"\nSample dimensions:")
print(f"  Length L = {L * 1e3:.1f} mm")
print(f"  Width w = {w * 1e3:.1f} mm")
print(f"  Thickness d = {d * 1e3:.1f} ± {delta_d * 1e3:.2f} mm")
print(f"  Cross-sectional area A = {A * 1e6:.2f} mm²")
print("\n" + "=" * 70 + "\n")


def load_data_2col(filename):
    """Load data file with 2 columns (skipping header)"""
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find where data starts (skip header lines)
    data_start = 0
    for i, line in enumerate(lines):
        if (
            line.strip()
            and not line.startswith("Sample")
            and not line.startswith("Flux")
        ):
            try:
                float(line.split()[0])
                data_start = i
                break
            except:
                continue

    data = []
    for line in lines[data_start:]:
        if line.strip():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    data.append([float(parts[0]), float(parts[1])])
                except:
                    continue

    return np.array(data)


def load_data_3col(filename):
    """Load data file with 3 columns (skipping header)"""
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find where data starts
    data_start = 0
    for i, line in enumerate(lines):
        if (
            line.strip()
            and not line.startswith("Flux")
            and not line.startswith("Sample")
        ):
            try:
                float(line.split()[0])
                data_start = i
                break
            except:
                continue

    data = []
    for line in lines[data_start:]:
        if line.strip():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except:
                    continue

    return np.array(data)


def linear_fit(x, y):
    """Perform linear regression and return slope, intercept, and statistics"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value**2, std_err


def quadratic_func(x, a, b, c):
    """Quadratic function for magnetoresistance fitting"""
    return a + b * x + c * x**2


def analyze_experiment1(sample_type, plots_dir):
    """
    Analyze Experiment 1: Hall Voltage vs Control Current
    Determine linearity and extract sensitivity dUH/dI
    """
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 1: Hall Voltage vs Control Current ({sample_type})")
    print(f"{'=' * 70}\n")

    # Load all three runs
    prefix = "nGE" if sample_type == "n-type" else "pGe"
    files = [
        f"{prefix} exp 1.1",
        f"{prefix} exp 1.2",
        f"{prefix.replace('nGE', 'nGe')} exp 1.3",
    ]

    all_currents = []
    all_voltages = []

    for filename in files:
        try:
            data = load_data_2col(filename)
            currents = data[:, 0]  # mA
            voltages = data[:, 1]  # mV
            all_currents.append(currents)
            all_voltages.append(voltages)
            print(f"  Loaded {filename}: {len(currents)} data points")
        except Exception as ex:
            print(f"  Warning: Could not load {filename}: {ex}")

    # Find common length (minimum across all runs)
    min_len = min(len(c) for c in all_currents)

    # Truncate all to same length and average
    all_currents = [c[:min_len] for c in all_currents]
    all_voltages = [v[:min_len] for v in all_voltages]

    # Average the three runs
    I_avg = np.mean(all_currents, axis=0)
    UH_avg = np.mean(all_voltages, axis=0)
    UH_std = np.std(all_voltages, axis=0)

    # Convert units: mA to A, mV to V
    I_A = I_avg * 1e-3
    UH_V = UH_avg * 1e-3
    UH_err = UH_std * 1e-3

    # Linear fit
    slope, intercept, r_sq, slope_err = linear_fit(I_A, UH_V)

    print(f"\nLinear fit: UH = m*I + b")
    print(f"  Slope m = {slope:.3f} ± {slope_err:.3f} V/A")
    print(f"  Intercept b = {intercept * 1e3:.3f} mV")
    print(f"  R² = {r_sq:.6f}")
    print(f"  Carrier type: {'Electrons (n-type)' if slope < 0 else 'Holes (p-type)'}")

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot experimental data with error bars
    ax.errorbar(
        I_avg,
        UH_avg,
        yerr=UH_std,
        fmt="o",
        markersize=5,
        label=f"{sample_type} Ge (experimental)",
        color="blue" if sample_type == "n-type" else "red",
        capsize=3,
        alpha=0.7,
    )

    # Plot fitted line
    I_fit = np.linspace(I_avg.min(), I_avg.max(), 100)
    UH_fit = (slope * I_fit * 1e-3 + intercept) * 1e3  # Convert back to mV
    ax.plot(
        I_fit,
        UH_fit,
        "-",
        linewidth=2,
        label=f"Linear fit: UH = {slope:.2f}I + {intercept * 1e3:.2f}",
        color="darkblue" if sample_type == "n-type" else "darkred",
    )

    ax.set_xlabel("Sample Current $I$ (mA)", fontsize=12)
    ax.set_ylabel("Hall Voltage $U_H$ (mV)", fontsize=12)
    ax.set_title(
        f"Hall Voltage vs Control Current ({sample_type} Ge)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Add R² value to plot
    ax.text(
        0.05,
        0.95,
        f"$R^2$ = {r_sq:.5f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    output_file = f"exp1_{sample_type.replace('-', '')}_UH_vs_I.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n  Plot saved: {output_file}")
    plt.close()

    # Return data for combined plotting
    return {
        "slope": slope,
        "slope_err": slope_err,
        "intercept": intercept,
        "r_sq": r_sq,
        "I_avg": I_avg,
        "UH_avg": UH_avg,
        "UH_std": UH_std,
        "sample_type": sample_type,
    }


def analyze_experiment2(sample_type, plots_dir):
    """
    Analyze Experiment 2: Hall Voltage and Sample Voltage vs Magnetic Field
    Determine Hall coefficient, carrier density, and mobility
    """
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 2: Hall Effect and Magnetoresistance ({sample_type})")
    print(f"{'=' * 70}\n")

    # Load all three runs
    prefix = "nGE" if sample_type == "n-type" else "pGe"
    files = [
        f"{prefix} exp 2.1",
        f"{prefix.replace('nGE', 'nGe')} exp 2.2",
        f"{prefix.replace('nGE', 'nGe')} exp 2.3",
    ]

    all_B = []
    all_UH = []
    all_Up = []

    for filename in files:
        try:
            data = load_data_3col(filename)
            B = data[:, 0]  # mT
            UH = data[:, 1]  # mV
            Up = data[:, 2]  # V
            all_B.append(B)
            all_UH.append(UH)
            all_Up.append(Up)
            print(f"  Loaded {filename}: {len(B)} data points")
        except Exception as ex:
            print(f"  Warning: Could not load {filename}: {ex}")

    # Average the three runs
    B_avg = np.mean(all_B, axis=0)
    UH_avg = np.mean(all_UH, axis=0)
    Up_avg = np.mean(all_Up, axis=0)

    UH_std = np.std(all_UH, axis=0)
    Up_std = np.std(all_Up, axis=0)

    # Convert units
    B_T = B_avg * 1e-3  # mT to T
    UH_V = UH_avg * 1e-3  # mV to V
    UH_err = UH_std * 1e-3
    Up_V = Up_avg
    Up_err = Up_std

    # ======== Part A: Hall Effect Analysis ========
    print("\n--- Part A: Hall Effect Analysis ---")

    # Linear fit: UH vs B
    slope_UH, intercept_UH, r_sq_UH, slope_err_UH = linear_fit(B_T, UH_V)

    # Calculate Hall coefficient: H = (UH/B) * (d/I)
    # Using I = 30 mA = 0.030 A (from experiment procedure)
    I = 0.030  # A
    H = slope_UH * (d / I)  # m³/C

    # Uncertainty in H
    delta_H = H * np.sqrt(
        (slope_err_UH / slope_UH) ** 2 + (delta_d / d) ** 2 + (delta_I / I) ** 2
    )

    print(f"\nLinear fit: UH = m*B + b")
    print(f"  Slope dUH/dB = {slope_UH * 1e3:.3f} ± {slope_err_UH * 1e3:.3f} mV/T")
    print(f"  Intercept = {intercept_UH * 1e3:.3f} mV")
    print(f"  R² = {r_sq_UH:.6f}")
    print(f"\nHall Coefficient:")
    print(f"  H = {H:.4e} ± {delta_H:.2e} m³/C")
    print(f"  Sign: {'Negative (electrons)' if H < 0 else 'Positive (holes)'}")

    # Calculate carrier density: n = 1/(e*|H|)
    n = 1 / (e * abs(H))  # m⁻³
    delta_n = n * (delta_H / abs(H))

    print(f"\nCarrier Density:")
    print(f"  n = {n:.3e} ± {delta_n:.2e} m⁻³")
    print(f"  n = {n * 1e-6:.3e} ± {delta_n * 1e-6:.2e} cm⁻³")

    # ======== Part B: Magnetoresistance Analysis ========
    print("\n--- Part B: Magnetoresistance Analysis ---")

    # Calculate resistance
    R = Up_V / I  # Ω
    R_err = Up_err / I

    # Find R0 (resistance at B=0)
    idx_zero = np.argmin(np.abs(B_T))
    R0 = R[idx_zero]
    R0_err = R_err[idx_zero]

    print(f"\nResistance at B=0:")
    print(f"  R₀ = {R0:.3f} ± {R0_err:.3f} Ω")

    # Calculate conductivity
    sigma_0 = L / (R0 * A)  # S/m
    delta_sigma = sigma_0 * np.sqrt((R0_err / R0) ** 2 + (delta_d / d) ** 2)

    print(f"\nConductivity:")
    print(f"  σ₀ = {sigma_0:.2f} ± {delta_sigma:.2f} S/m")

    # Fit magnetoresistance: R = R0 + R0*(mu*B)²
    # Fit directly to R vs B²: R = a + b*B²
    B_sq = B_T**2

    # Linear fit: R vs B²
    slope_R, intercept_R, r_sq_R, slope_err_R = linear_fit(B_sq, R)

    # Extract mobility from magnetoresistance
    # From R = R0 + R0*mu²*B², slope = R0*mu²
    # Therefore: mu = sqrt(slope / R0)
    if slope_R > 0:
        mu_MR = np.sqrt(slope_R / R0)
        delta_mu_MR = (
            mu_MR * 0.5 * np.sqrt((slope_err_R / slope_R) ** 2 + (R0_err / R0) ** 2)
        )
    else:
        mu_MR = 0
        delta_mu_MR = 0
        print(
            "  Warning: Negative magnetoresistance coefficient - mobility cannot be calculated"
        )

    print(f"\nMagnetoresistance fit: R = R₀ + k*B²")
    print(f"  Intercept R₀ = {intercept_R:.3f} Ω")
    print(f"  Coefficient k = {slope_R:.6f} ± {slope_err_R:.6f} Ω/T²")
    print(f"  R² = {r_sq_R:.6f}")
    if slope_R > 0:
        print(f"\nMobility (from magnetoresistance):")
        print(f"  μ_MR = {mu_MR:.4f} ± {delta_mu_MR:.4f} m²/(V·s)")
    else:
        print(
            f"\nMobility (from magnetoresistance): Cannot calculate (negative coefficient)"
        )

    # Calculate mobility from Hall effect: mu = H * sigma_0
    mu_Hall = abs(H) * sigma_0
    delta_mu_Hall = mu_Hall * np.sqrt(
        (delta_H / abs(H)) ** 2 + (delta_sigma / sigma_0) ** 2
    )

    print(f"\nMobility (from Hall effect):")
    print(f"  μ_Hall = {mu_Hall:.4f} ± {delta_mu_Hall:.4f} m²/(V·s)")

    # Compare the two mobility values
    discrepancy = abs(mu_MR - mu_Hall) / mu_Hall * 100
    print(f"\nMobility comparison:")
    print(f"  Discrepancy = {discrepancy:.1f}%")

    # ======== Plotting ========

    # Figure 1: Hall Voltage vs Magnetic Field
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        B_avg,
        UH_avg,
        yerr=UH_std,
        fmt="o",
        markersize=5,
        label=f"{sample_type} Ge (experimental)",
        color="blue" if sample_type == "n-type" else "red",
        capsize=3,
        alpha=0.7,
    )

    B_fit = np.linspace(B_avg.min(), B_avg.max(), 100)
    UH_fit = (slope_UH * B_fit * 1e-3 + intercept_UH) * 1e3
    ax.plot(
        B_fit,
        UH_fit,
        "-",
        linewidth=2,
        label=f"Linear fit (R² = {r_sq_UH:.5f})",
        color="darkblue" if sample_type == "n-type" else "darkred",
    )

    ax.set_xlabel("Magnetic Flux Density $B$ (mT)", fontsize=12)
    ax.set_ylabel("Hall Voltage $U_H$ (mV)", fontsize=12)
    ax.set_title(
        f"Hall Voltage vs Magnetic Field ({sample_type} Ge)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Add Hall coefficient to plot
    ax.text(
        0.05,
        0.95,
        f"$H$ = {H:.3e} m³/C",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    output_file = f"exp2a_{sample_type.replace('-', '')}_UH_vs_B.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n  Plot saved: {output_file}")
    plt.close()

    # Figure 2: Magnetoresistance
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        B_avg,
        R,
        yerr=R_err,
        fmt="o",
        markersize=5,
        label=f"{sample_type} Ge (experimental)",
        color="blue" if sample_type == "n-type" else "red",
        capsize=3,
        alpha=0.7,
    )

    B_fit_T = np.linspace(B_T.min(), B_T.max(), 100)
    R_fit = slope_R * (B_fit_T**2) + intercept_R
    ax.plot(
        B_fit_T * 1e3,
        R_fit,
        "-",
        linewidth=2,
        label=f"Fit: R = R₀ + k*B² (R²={r_sq_R:.4f})",
        color="darkblue" if sample_type == "n-type" else "darkred",
    )

    # Mark R0
    ax.axhline(
        y=R0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"R₀ = {R0:.2f} Ω",
    )

    ax.set_xlabel("Magnetic Flux Density $B$ (mT)", fontsize=12)
    ax.set_ylabel("Sample Resistance $R$ (Ω)", fontsize=12)
    ax.set_title(
        f"Magnetoresistance ({sample_type} Ge)", fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Add mobility to plot
    if mu_MR > 0:
        ax.text(
            0.05,
            0.95,
            f"$μ_{{MR}}$ = {mu_MR:.3f} m²/(V·s)",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    output_file = f"exp2b_{sample_type.replace('-', '')}_R_vs_B.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Plot saved: {output_file}")
    plt.close()

    # Return results for summary table
    results = {
        "H": H,
        "delta_H": delta_H,
        "n": n,
        "delta_n": delta_n,
        "R0": R0,
        "delta_R0": R0_err,
        "sigma_0": sigma_0,
        "delta_sigma": delta_sigma,
        "mu_Hall": mu_Hall,
        "delta_mu_Hall": delta_mu_Hall,
        "mu_MR": mu_MR,
        "delta_mu_MR": delta_mu_MR,
        "r_sq_UH": r_sq_UH,
        # Store data for combined plotting
        "B_avg": B_avg,
        "UH_avg": UH_avg,
        "UH_std": UH_std,
        "slope_UH": slope_UH,
        "intercept_UH": intercept_UH,
        "sample_type": sample_type,
    }

    return results


def analyze_experiment3(plots_dir):
    """
    Analyze Experiment 3: Hall Voltage vs Temperature (p-type only)
    """
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 3: Hall Voltage vs Temperature (p-type Ge)")
    print(f"{'=' * 70}\n")

    try:
        data = load_data_2col("pGe exp 3")
        T = data[:, 0]  # °C
        UH = data[:, 1]  # mV
        print(f"  Loaded pGe exp 3: {len(T)} data points")
    except Exception as ex:
        print(f"  Error loading data: {ex}")
        return

    print(f"\nTemperature range: {T.min():.1f}°C to {T.max():.1f}°C")
    print(f"Hall voltage range: {UH.min():.2f} mV to {UH.max():.2f} mV")

    # Find transition point (where UH crosses zero)
    sign_changes = np.where(np.diff(np.sign(UH)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        T_transition = (T[idx] + T[idx + 1]) / 2
        print(f"\nHall voltage sign reversal at T ≈ {T_transition:.1f}°C")
        print(f"  This indicates transition from extrinsic to intrinsic conduction")

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        T,
        UH,
        "o-",
        markersize=4,
        linewidth=1.5,
        color="red",
        label="p-type Ge",
        alpha=0.8,
    )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    if len(sign_changes) > 0:
        ax.axvline(
            x=T_transition,
            color="gray",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"Transition ≈ {T_transition:.0f}°C",
        )

    ax.set_xlabel("Sample Temperature $T$ (°C)", fontsize=12)
    ax.set_ylabel("Hall Voltage $U_H$ (mV)", fontsize=12)
    ax.set_title(
        "Hall Voltage vs Temperature (p-type Ge)", fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Add annotations for regions
    ax.text(
        0.15,
        0.85,
        "Extrinsic\n(p-doping dominant)",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        fontsize=9,
    )
    ax.text(
        0.75,
        0.25,
        "Intrinsic\n(thermal excitation)",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        fontsize=9,
    )

    plt.tight_layout()
    output_file = os.path.join(plots_dir, "exp3_ptype_UH_vs_T.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n  Plot saved: {output_file}")
    plt.close()


def plot_combined_experiment1(data_n, data_p, plots_dir):
    """Create combined plot for Experiment 1 (n-type and p-type together)"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot n-type data
    ax.errorbar(
        data_n["I_avg"],
        data_n["UH_avg"],
        yerr=data_n["UH_std"],
        fmt="o",
        markersize=5,
        label="n-type Ge (experimental)",
        color="blue",
        capsize=3,
        alpha=0.7,
    )

    # Plot n-type fit
    I_fit_n = np.linspace(data_n["I_avg"].min(), data_n["I_avg"].max(), 100)
    UH_fit_n = (data_n["slope"] * I_fit_n * 1e-3 + data_n["intercept"]) * 1e3
    ax.plot(
        I_fit_n,
        UH_fit_n,
        "-",
        linewidth=2.5,
        label=f"n-type fit: slope = {data_n['slope']:.3f} V/A, $R^2$ = {data_n['r_sq']:.5f}",
        color="darkblue",
    )

    # Plot p-type data
    ax.errorbar(
        data_p["I_avg"],
        data_p["UH_avg"],
        yerr=data_p["UH_std"],
        fmt="s",
        markersize=5,
        label="p-type Ge (experimental)",
        color="red",
        capsize=3,
        alpha=0.7,
    )

    # Plot p-type fit
    I_fit_p = np.linspace(data_p["I_avg"].min(), data_p["I_avg"].max(), 100)
    UH_fit_p = (data_p["slope"] * I_fit_p * 1e-3 + data_p["intercept"]) * 1e3
    ax.plot(
        I_fit_p,
        UH_fit_p,
        "-",
        linewidth=2.5,
        label=f"p-type fit: slope = {data_p['slope']:.3f} V/A, $R^2$ = {data_p['r_sq']:.5f}",
        color="darkred",
    )

    ax.set_xlabel("Sample Current $I$ (mA)", fontsize=13)
    ax.set_ylabel("Hall Voltage $U_H$ (mV)", fontsize=13)
    ax.set_title(
        "Hall Voltage vs Control Current (n-type and p-type Ge)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Add annotation about carrier types
    ax.text(
        0.02,
        0.98,
        "Negative slope → electrons (n-type)\nPositive slope → holes (p-type)",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=9,
    )

    plt.tight_layout()
    output_file = os.path.join(plots_dir, "exp1_combined_UH_vs_I.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n  Combined plot saved: {output_file}")
    plt.close()


def plot_combined_experiment2a(results_n, results_p, plots_dir):
    """Create combined plot for Experiment 2a (Hall voltage vs B for both types)"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot n-type data
    ax.errorbar(
        results_n["B_avg"],
        results_n["UH_avg"],
        yerr=results_n["UH_std"],
        fmt="o",
        markersize=5,
        label="n-type Ge (experimental)",
        color="blue",
        capsize=3,
        alpha=0.7,
    )

    # Plot n-type fit
    B_fit_n = np.linspace(results_n["B_avg"].min(), results_n["B_avg"].max(), 100)
    UH_fit_n = (
        results_n["slope_UH"] * B_fit_n * 1e-3 + results_n["intercept_UH"]
    ) * 1e3
    ax.plot(
        B_fit_n,
        UH_fit_n,
        "-",
        linewidth=2.5,
        label=f"n-type fit: $R^2$ = {results_n['r_sq_UH']:.5f}",
        color="darkblue",
    )

    # Plot p-type data
    ax.errorbar(
        results_p["B_avg"],
        results_p["UH_avg"],
        yerr=results_p["UH_std"],
        fmt="s",
        markersize=5,
        label="p-type Ge (experimental)",
        color="red",
        capsize=3,
        alpha=0.7,
    )

    # Plot p-type fit
    B_fit_p = np.linspace(results_p["B_avg"].min(), results_p["B_avg"].max(), 100)
    UH_fit_p = (
        results_p["slope_UH"] * B_fit_p * 1e-3 + results_p["intercept_UH"]
    ) * 1e3
    ax.plot(
        B_fit_p,
        UH_fit_p,
        "-",
        linewidth=2.5,
        label=f"p-type fit: $R^2$ = {results_p['r_sq_UH']:.5f}",
        color="darkred",
    )

    ax.set_xlabel("Magnetic Flux Density $B$ (mT)", fontsize=13)
    ax.set_ylabel("Hall Voltage $U_H$ (mV)", fontsize=13)
    ax.set_title(
        "Hall Voltage vs Magnetic Field (n-type and p-type Ge)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Add Hall coefficients
    ax.text(
        0.02,
        0.98,
        f"n-type: $H$ = {results_n['H']:.3e} m³/C\np-type: $H$ = {results_p['H']:.3e} m³/C",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=9,
    )

    plt.tight_layout()
    output_file = os.path.join(plots_dir, "exp2a_combined_UH_vs_B.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n  Combined plot saved: {output_file}")
    plt.close()


def main():
    """Main analysis routine"""

    # Change to data directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    plots_dir = os.path.join(os.path.dirname(script_dir), "plots")
    os.chdir(data_dir)

    # Analyze Experiment 1 for both sample types
    print("\n" + "█" * 70)
    print("PART 1: HALL VOLTAGE vs CONTROL CURRENT")
    print("█" * 70)

    data_n_exp1 = analyze_experiment1("n-type", plots_dir)
    data_p_exp1 = analyze_experiment1("p-type", plots_dir)

    # Create combined plot for Experiment 1
    plot_combined_experiment1(data_n_exp1, data_p_exp1, plots_dir)

    # Extract individual values for summary (backward compatibility)
    slope_n = data_n_exp1["slope"]
    slope_err_n = data_n_exp1["slope_err"]
    intercept_n = data_n_exp1["intercept"]
    r_sq_n = data_n_exp1["r_sq"]
    slope_p = data_p_exp1["slope"]
    slope_err_p = data_p_exp1["slope_err"]
    intercept_p = data_p_exp1["intercept"]
    r_sq_p = data_p_exp1["r_sq"]

    # Analyze Experiment 2 for both sample types
    print("\n" + "█" * 70)
    print("PART 2: HALL EFFECT AND MAGNETORESISTANCE")
    print("█" * 70)

    results_n = analyze_experiment2("n-type", plots_dir)
    results_p = analyze_experiment2("p-type", plots_dir)

    # Create combined plot for Experiment 2a (Hall voltage vs B)
    plot_combined_experiment2a(results_n, results_p, plots_dir)

    # Analyze Experiment 3 (p-type only)
    print("\n" + "█" * 70)
    print("PART 3: TEMPERATURE DEPENDENCE")
    print("█" * 70)

    analyze_experiment3(plots_dir)

    # Summary Table
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70 + "\n")

    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "HALL EFFECT EXPERIMENTAL RESULTS" + " " * 20 + "║")
    print("╠" + "═" * 68 + "╣")
    print("║ Parameter                │ n-type Ge           │ p-type Ge           ║")
    print("╠" + "═" * 68 + "╣")

    print(
        f"║ Hall coefficient H       │ {results_n['H']:+.3e} ± {results_n['delta_H']:.1e} │ {results_p['H']:+.3e} ± {results_p['delta_H']:.1e} ║"
    )
    print(f"║    (m³/C)                │                     │                     ║")
    print("╠" + "─" * 68 + "╣")

    print(f"║ Carrier type             │ Electrons (n-type)  │ Holes (p-type)      ║")
    print("╠" + "─" * 68 + "╣")

    print(
        f"║ Carrier density n        │ {results_n['n']:.3e}       │ {results_p['n']:.3e}       ║"
    )
    print(
        f"║    (m⁻³)                 │ ± {results_n['delta_n']:.2e}        │ ± {results_p['delta_n']:.2e}        ║"
    )
    print("╠" + "─" * 68 + "╣")

    print(
        f"║ Resistance R₀ (Ω)       │ {results_n['R0']:.3f} ± {results_n['delta_R0']:.3f}      │ {results_p['R0']:.3f} ± {results_p['delta_R0']:.3f}      ║"
    )
    print("╠" + "─" * 68 + "╣")

    print(
        f"║ Conductivity σ₀ (S/m)   │ {results_n['sigma_0']:.2f} ± {results_n['delta_sigma']:.2f}       │ {results_p['sigma_0']:.2f} ± {results_p['delta_sigma']:.2f}        ║"
    )
    print("╠" + "─" * 68 + "╣")

    print(
        f"║ Mobility μ_Hall          │ {results_n['mu_Hall']:.4f} ± {results_n['delta_mu_Hall']:.4f}   │ {results_p['mu_Hall']:.4f} ± {results_p['delta_mu_Hall']:.4f}   ║"
    )
    print(f"║    (m²/V·s)              │                     │                     ║")
    print("╠" + "─" * 68 + "╣")

    print(
        f"║ Mobility μ_MR            │ {results_n['mu_MR']:.4f} ± {results_n['delta_mu_MR']:.4f}   │ {results_p['mu_MR']:.4f} ± {results_p['delta_mu_MR']:.4f}   ║"
    )
    print(f"║    (m²/V·s)              │                     │                     ║")
    print("╠" + "─" * 68 + "╣")

    ratio_n = results_n["mu_MR"] / results_n["mu_Hall"]
    ratio_p = results_p["mu_MR"] / results_p["mu_Hall"]
    print(
        f"║ μ_MR / μ_Hall ratio      │ {ratio_n:.3f}               │ {ratio_p:.3f}               ║"
    )
    print("╚" + "═" * 68 + "╝")

    print("\n" + "=" * 70)
    print("Literature values for Ge at room temperature:")
    print("  Electron mobility: ~0.39 m²/(V·s) = 3900 cm²/(V·s)")
    print("  Hole mobility: ~0.19 m²/(V·s) = 1900 cm²/(V·s)")
    print("  Intrinsic carrier density: ~2.4×10¹⁹ m⁻³")
    print("=" * 70)

    print("\n✓ Analysis complete! All plots generated successfully.\n")


if __name__ == "__main__":
    main()
