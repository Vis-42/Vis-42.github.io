#!/usr/bin/env python3
"""
Magnetic Moment Experiment - Data Analysis and Plot Generation
PC2193 Experimental Physics & Data Analysis

This script analyzes torque measurements for current loops in a Helmholtz field.
It calibrates the Helmholtz coil constant c (B = c * I_H), cross-validates with
multiple experimental sweeps, and estimates the area of an unknown (star-shaped) loop.
It also generates publication-quality plots for inclusion in the Typst report.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Configure matplotlib to use a writable config dir and non-GUI backend
plt = None
try:
    import matplotlib

    _cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")
    os.makedirs(_cfg_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _cfg_dir
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # Plotting will be skipped if matplotlib is unavailable


# ----------------------------- Utilities -------------------------------------


@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float


def linear_fit(x: List[float], y: List[float]) -> FitResult:
    """
    Simple least squares linear regression (y = a x + b) without external deps.
    Returns slope, intercept, and R^2.
    """
    n = len(x)
    if n == 0:
        return FitResult(0.0, 0.0, 0.0)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(x[i] * y[i] for i in range(n))
    denom = n * sxx - sx * sx
    if denom == 0:
        return FitResult(0.0, 0.0, 0.0)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_mean = sy / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return FitResult(slope, intercept, r2)


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


# ----------------------------- Data Loading ----------------------------------

# Determine paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

DATA_CSV = os.path.join(DATA_DIR, "all_data_corrected.csv")


def load_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_float(s: str) -> float:
    if (
        s is None
        or s.strip() == ""
        or s.strip().upper() == "NA"
        or s.strip().lower() == "unknown"
    ):
        return float("nan")
    return float(s)


# ----------------------------- Physics Helpers --------------------------------

# Units:
# - Torque provided as T_mNcm (mN·cm) → convert to SI N·m via 1 mN·cm = 1e-5 N·m
MN_CM_TO_NM = 1e-5
PI = math.pi


def area_from_diameter_cm(d_cm: float) -> float:
    """Area of circular loop for diameter in cm; returns area in m^2."""
    d_m = d_cm / 100.0
    return PI * (d_m**2) / 4.0


# ----------------------------- Analysis Steps ---------------------------------


def analyze_exp1(
    rows: List[Dict[str, str]], A_single: float, n_loops: int, IL_A: float
) -> Tuple[FitResult, float]:
    """
    Experiment 1: Torque vs Helmholtz current IH at fixed n, IL, alpha=90deg.
    T = c * IH * n * IL * A
    Using slope s1 = dT/dIH, c = s1 / (n * IL * A)
    """
    xs, ys = [], []
    for row in rows:
        if row["Experiment"] == "1-Torque_vs_IH":
            ih = to_float(row["IH_A"])
            T = to_float(row["T_mNcm"]) * MN_CM_TO_NM
            xs.append(ih)
            ys.append(T)
    fit = linear_fit(xs, ys)
    c = fit.slope / (n_loops * IL_A * A_single)
    return fit, c


def analyze_exp2(
    rows: List[Dict[str, str]], A_single: float, IH_A: float, IL_A: float
) -> Tuple[FitResult, float]:
    """
    Experiment 2: Torque vs number of loops n at fixed IH, IL, alpha=90deg.
    T = c * IH * n * IL * A  => slope s2 = c * IH * IL * A  => c = s2 / (IH * IL * A)
    """
    xs, ys = [], []
    for row in rows:
        if row["Experiment"] == "2-Torque_vs_n":
            n = to_float(row["n_loops"])
            T = to_float(row["T_mNcm"]) * MN_CM_TO_NM
            xs.append(n)
            ys.append(T)
    fit = linear_fit(xs, ys)
    c = fit.slope / (IH_A * IL_A * A_single)
    return fit, c


def analyze_exp3(
    rows: List[Dict[str, str]], A_single: float, IH_A: float, IL_A: float, n_loops: int
) -> Tuple[FitResult, float]:
    """
    Experiment 3: Torque vs sin(alpha) at fixed IH, IL, n.
    T = c * IH * n * IL * A * sin(alpha) => slope s3 = c * IH * n * IL * A => c = s3 / (IH * n * IL * A)
    """
    xs, ys = [], []
    for row in rows:
        if row["Experiment"] == "3-Torque_vs_angle":
            s = to_float(row["sin_alpha"])
            T = to_float(row["T_mNcm"]) * MN_CM_TO_NM
            xs.append(s)
            ys.append(T)
    fit = linear_fit(xs, ys)
    c = fit.slope / (IH_A * n_loops * IL_A * A_single)
    return fit, c


def analyze_exp4(
    rows: List[Dict[str, str]], IH_A: float, IL_A: float
) -> Tuple[FitResult, float]:
    """
    Experiment 4: Torque vs loop diameter with 1 loop at alpha=90deg.
    T = c * IH * IL * (pi/4) * d^2
    Let x = d^2 (m^2), slope s4 = c * IH * IL * (pi/4)  => c = s4 / (IH * IL * (pi/4))
    """
    xs, ys = [], []
    for row in rows:
        if row["Experiment"] == "4-Torque_vs_diameter":
            d_cm = to_float(row["d_cm"])
            d_m2 = (d_cm / 100.0) ** 2
            T = to_float(row["T_mNcm"]) * MN_CM_TO_NM
            xs.append(d_m2)
            ys.append(T)
    fit = linear_fit(xs, ys)
    denom = IH_A * IL_A * (PI / 4.0)
    c = fit.slope / denom if denom != 0.0 else float("nan")
    return fit, c


def analyze_exp5(
    rows: List[Dict[str, str]], A_single: float, IH_A: float, n_loops: int
) -> Tuple[FitResult, float]:
    """
    Experiment 5: Torque vs loop current IL at fixed IH, n, alpha=90deg.
    T = c * IH * n * IL * A  => slope s5 = c * IH * n * A  => c = s5 / (IH * n * A)
    """
    xs, ys = [], []
    for row in rows:
        if row["Experiment"] == "5-Torque_vs_IL":
            il = to_float(row["IL_A"])
            T = to_float(row["T_mNcm"]) * MN_CM_TO_NM
            xs.append(il)
            ys.append(T)
    fit = linear_fit(xs, ys)
    c = fit.slope / (IH_A * n_loops * A_single)
    return fit, c


def estimate_unknown_area(rows: List[Dict[str, str]], c: float) -> Tuple[float, float]:
    """
    Experiment 6: Unknown loop area at alpha≈90deg, n≈1.
    T = c * IH * n * IL * A  => A = T / (c * IH * n * IL)
    Returns (A_unknown [m^2], diameter_equiv [m]) where diameter_equiv assumes circular area.
    """
    for row in rows:
        if row["Experiment"] == "6-Unknown_loop":
            T = to_float(row["T_mNcm"]) * MN_CM_TO_NM
            IH = to_float(row["IH_A"])
            IL = to_float(row["IL_A"])
            n = int(to_float(row["n_loops"])) if row["n_loops"] else 1
            denom = c * IH * max(n, 1) * IL
            A_unknown = T / denom if denom != 0 else float("nan")
            d_equiv = math.sqrt(4.0 * A_unknown / PI) if A_unknown > 0 else float("nan")
            return A_unknown, d_equiv
    return float("nan"), float("nan")


# ----------------------------- Plotting ---------------------------------------


def plot_xy_with_fit(
    x: List[float],
    y: List[float],
    fit: FitResult,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", markersize=5, label="Data", color="#1f77b4", alpha=0.8)
    x_min, x_max = min(x), max(x)
    x_fit = [x_min, x_max]
    y_fit = [fit.slope * v + fit.intercept for v in x_fit]
    ax.plot(
        x_fit,
        y_fit,
        "-",
        linewidth=2,
        color="#d62728",
        label=f"Fit: y = {fit.slope:.3e} x + {fit.intercept:.3e}",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.text(
        0.05,
        0.95,
        f"$R^2$ = {fit.r2:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Main -------------------------------------------


def main() -> None:
    ensure_dir(PLOT_DIR)
    rows = load_csv(DATA_CSV)

    # Known fixed conditions from data headers
    # From the master file: IL = 5.00 A (exp 1-4), IH = 2.50 A (exp 2-5), n = 3 (exp 1,3,5)
    d_ref_cm = 11.98
    A_single = area_from_diameter_cm(d_ref_cm)  # m^2

    # Extract typical fixed values for each exp from the CSV (robust against future edits)
    def first_value(experiment: str, key: str) -> float:
        for row in rows:
            if row["Experiment"] == experiment:
                return to_float(row[key])
        return float("nan")

    IL_exp1 = first_value("1-Torque_vs_IH", "IL_A")
    n_exp1 = int(first_value("1-Torque_vs_IH", "n_loops"))
    IH_exp2 = first_value("2-Torque_vs_n", "IH_A")
    IL_exp2 = first_value("2-Torque_vs_n", "IL_A")
    IH_exp3 = first_value("3-Torque_vs_angle", "IH_A")
    IL_exp3 = first_value("3-Torque_vs_angle", "IL_A")
    n_exp3 = int(first_value("3-Torque_vs_angle", "n_loops"))
    IH_exp4 = first_value("4-Torque_vs_diameter", "IH_A")
    IL_exp4 = first_value("4-Torque_vs_diameter", "IL_A")
    IH_exp5 = first_value("5-Torque_vs_IL", "IH_A")
    n_exp5 = int(first_value("5-Torque_vs_IL", "n_loops"))

    # Perform analyses
    fit1, c1 = analyze_exp1(rows, A_single=A_single, n_loops=n_exp1, IL_A=IL_exp1)
    fit2, c2 = analyze_exp2(rows, A_single=A_single, IH_A=IH_exp2, IL_A=IL_exp2)
    fit3, c3 = analyze_exp3(
        rows, A_single=A_single, IH_A=IH_exp3, IL_A=IL_exp3, n_loops=n_exp3
    )
    fit4, c4 = analyze_exp4(rows, IH_A=IH_exp4, IL_A=IL_exp4)
    fit5, c5 = analyze_exp5(rows, A_single=A_single, IH_A=IH_exp5, n_loops=n_exp5)

    c_values = [c for c in (c1, c2, c3, c4, c5) if math.isfinite(c)]
    c_avg = sum(c_values) / len(c_values) if c_values else float("nan")
    c_std = (
        (sum((v - c_avg) ** 2 for v in c_values) / (len(c_values) - 1)) ** 0.5
        if len(c_values) >= 2
        else float("nan")
    )

    # Unknown loop area
    A_unknown, d_unknown = estimate_unknown_area(rows, c=c_avg)

    # Console summary
    print("=" * 72)
    print("MAGNETIC MOMENT - DATA ANALYSIS SUMMARY")
    print("=" * 72)
    print(
        f"Reference circular loop diameter d = {d_ref_cm:.2f} cm → A_single = {A_single:.6e} m^2"
    )
    print()
    print("Calibrated Helmholtz constant c (B = c · I_H):")
    print(f"  From Exp1 (T vs I_H):            c1 = {c1:.6e} T/A   (R^2={fit1.r2:.4f})")
    print(f"  From Exp2 (T vs n):              c2 = {c2:.6e} T/A   (R^2={fit2.r2:.4f})")
    print(f"  From Exp3 (T vs sin α):          c3 = {c3:.6e} T/A   (R^2={fit3.r2:.4f})")
    print(f"  From Exp4 (T vs d^2):            c4 = {c4:.6e} T/A   (R^2={fit4.r2:.4f})")
    print(f"  From Exp5 (T vs I_L):            c5 = {c5:.6e} T/A   (R^2={fit5.r2:.4f})")
    print(
        f"  Average ± std:                   c  = {c_avg:.6e} ± {c_std if math.isfinite(c_std) else float('nan'):.2e} T/A"
    )
    print()
    if math.isfinite(A_unknown):
        print(f"Unknown loop area A_unknown = {A_unknown:.6e} m^2")
        if math.isfinite(d_unknown):
            print(f"Equivalent circular diameter d_eq = {d_unknown * 100:.3f} cm")
    else:
        print("Unknown loop area could not be estimated (insufficient data).")
    print("=" * 72)

    # Generate plots
    if plt is not None:
        # Exp 1 plot: T vs IH
        x1, y1 = [], []
        for row in rows:
            if row["Experiment"] == "1-Torque_vs_IH":
                x1.append(to_float(row["IH_A"]))
                y1.append(to_float(row["T_mNcm"]) * MN_CM_TO_NM)
        plot_xy_with_fit(
            x1,
            y1,
            fit1,
            title="Torque vs Helmholtz Current (n=3, IL=5 A, α=90°)",
            xlabel="Helmholtz Current $I_H$ (A)",
            ylabel="Torque $T$ (N·m)",
            outpath=os.path.join(PLOT_DIR, "mm_exp1_torque_vs_IH.png"),
        )

        # Exp 2 plot: T vs n
        x2, y2 = [], []
        for row in rows:
            if row["Experiment"] == "2-Torque_vs_n":
                x2.append(to_float(row["n_loops"]))
                y2.append(to_float(row["T_mNcm"]) * MN_CM_TO_NM)
        plot_xy_with_fit(
            x2,
            y2,
            fit2,
            title="Torque vs Number of Loops (IH=2.5 A, IL=5 A, α=90°)",
            xlabel="Number of Loops $n$",
            ylabel="Torque $T$ (N·m)",
            outpath=os.path.join(PLOT_DIR, "mm_exp2_torque_vs_n.png"),
        )

        # Exp 3 plot: T vs sin(alpha)
        x3, y3 = [], []
        for row in rows:
            if row["Experiment"] == "3-Torque_vs_angle":
                x3.append(to_float(row["sin_alpha"]))
                y3.append(to_float(row["T_mNcm"]) * MN_CM_TO_NM)
        plot_xy_with_fit(
            x3,
            y3,
            fit3,
            title="Torque vs sin(α) (IH=2.5 A, IL=5 A, n=3)",
            xlabel="$\\sin(\\alpha)$",
            ylabel="Torque $T$ (N·m)",
            outpath=os.path.join(PLOT_DIR, "mm_exp3_torque_vs_sin_alpha.png"),
        )

        # Exp 4 plot: T vs d^2
        x4, y4 = [], []
        for row in rows:
            if row["Experiment"] == "4-Torque_vs_diameter":
                d_cm = to_float(row["d_cm"])
                x4.append((d_cm / 100.0) ** 2)
                y4.append(to_float(row["T_mNcm"]) * MN_CM_TO_NM)
        plot_xy_with_fit(
            x4,
            y4,
            fit4,
            title="Torque vs Loop Diameter Squared (IH=2.5 A, IL=5 A, n=1, α=90°)",
            xlabel="Diameter Squared $d^2$ (m²)",
            ylabel="Torque $T$ (N·m)",
            outpath=os.path.join(PLOT_DIR, "mm_exp4_torque_vs_d2.png"),
        )

        # Exp 5 plot: T vs IL
        x5, y5 = [], []
        for row in rows:
            if row["Experiment"] == "5-Torque_vs_IL":
                x5.append(to_float(row["IL_A"]))
                y5.append(to_float(row["T_mNcm"]) * MN_CM_TO_NM)
        plot_xy_with_fit(
            x5,
            y5,
            fit5,
            title="Torque vs Loop Current (IH=2.5 A, n=3, α=90°)",
            xlabel="Loop Current $I_L$ (A)",
            ylabel="Torque $T$ (N·m)",
            outpath=os.path.join(PLOT_DIR, "mm_exp5_torque_vs_IL.png"),
        )
    else:
        print("matplotlib not available; skipping plot generation.")


if __name__ == "__main__":
    main()
