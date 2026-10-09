#!/usr/bin/env python3
"""
Analyse Paschen curve data: generate plots, compute the second Townsend
coefficient gamma, and overlay theoretical Paschen curves.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # plotstyle.py lives next to this script
from plotstyle import COL, BLUE, GOLD, RED, GREY, INK, apply_style, save  # noqa: E402

apply_style()

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "paschen_data.json"
PLOT_DIR = ROOT / "plots"
FIG_DIR = ROOT / "report" / "figures"
RESULTS_FILE = ROOT / "analysis_results.json"

PLOT_DIR.mkdir(exist_ok=True)

A = 20.0      # 1/(Pa m)
B = 487.0     # 1/(Pa m)
E_EULER = np.e

PRESSURE_COLORS = {"8": BLUE, "6": GOLD, "4": RED}
PRESSURE_MARKERS = {"8": "o", "6": "s", "4": "^"}


def load_data():
    with open(DATA_FILE) as f:
        raw = json.load(f)

    processed = {}
    for p_str, trials_by_d in raw["data"].items():
        d_vals, u_avg, u_std = [], [], []
        for d_str, trials in trials_by_d.items():
            if trials is None:
                continue
            d_vals.append(float(d_str))
            arr = np.array(trials, dtype=float)
            u_avg.append(arr.mean())
            u_std.append(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        processed[p_str] = {
            "d_mm": np.array(d_vals),
            "U_avg": np.array(u_avg),
            "U_std": np.array(u_std),
            "p_hPa": float(p_str),
            "p_Pa": float(p_str) * 100.0,
        }
    return processed


def end_labels(ax, data, x_of):
    """Name each pressure series at its last point, in its own colour."""
    for p_str, d in data.items():
        x = x_of(d)
        i = np.argmax(x)
        ax.annotate(f"{p_str} hPa", (x[i], d["U_avg"][i]), xytext=(4, 0), textcoords="offset points",
                    color=PRESSURE_COLORS[p_str], va="center", fontsize=8)


def paschen_voltage(pd, C):
    """Paschen breakdown voltage: U_br = B * pd / ln(A * pd / C)"""
    arg = A * pd / C
    arg = np.where(arg > 1.0, arg, np.nan)
    return B * pd / np.log(arg)


def pd_min_from_C(C):
    return E_EULER * C / A


def U_min_from_C(C):
    return E_EULER * B * C / A


def plot_U_vs_d(data):
    """Plot 1: U_br vs d for each pressure."""
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for p_str, d in sorted(data.items(), key=lambda x: -x[1]["p_hPa"]):
        ax.errorbar(
            d["d_mm"], d["U_avg"], yerr=d["U_std"],
            fmt=PRESSURE_MARKERS[p_str] + "-",
            color=PRESSURE_COLORS[p_str],
            markersize=3.5, linewidth=0.9,
            label=f"{p_str} hPa"
        )
    ax.set_xlabel("Electrode gap, $d$ (mm)")
    ax.set_ylabel("Breakdown voltage, $U_{\\rm br}$ (V)")
    end_labels(ax, data, lambda d: d["d_mm"])
    ax.set_xlim(right=6.4)
    save(fig, FIG_DIR, "U_vs_d")


def plot_U_vs_pd(data):
    """Plot 2: U_br vs pd (should collapse onto single curve)."""
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for p_str, d in sorted(data.items(), key=lambda x: -x[1]["p_hPa"]):
        pd_vals = d["d_mm"] * 1e-3 * d["p_Pa"]  # Pa m
        ax.errorbar(
            pd_vals, d["U_avg"], yerr=d["U_std"],
            fmt=PRESSURE_MARKERS[p_str],
            color=PRESSURE_COLORS[p_str],
            markersize=3.5,
            label=f"{p_str} hPa"
        )
    ax.set_xlabel("Pressure × gap, $pd$ (Pa m)")
    ax.set_ylabel("Breakdown voltage, $U_{\\rm br}$ (V)")
    end_labels(ax, data, lambda d: d["d_mm"] * 1e-3 * d["p_Pa"])
    ax.set_xlim(right=5.1)
    save(fig, FIG_DIR, "U_vs_pd")


def plot_linear(data):
    """Plot 3: d/U_br vs ln(d) — all data uniformly, with theoretical Paschen
    lines.  The linearisation d/U = [ln(d) + ln(Ap/C)]/(Bp) is exact when the
    Townsend model holds; deviations reveal where it breaks down."""
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    fit_results = {}

    gamma_ref = 0.0017
    C_ref = np.log(1.0 / gamma_ref + 1)

    for p_str, d in sorted(data.items(), key=lambda x: -x[1]["p_hPa"]):
        mask = d["d_mm"] > 0
        d_mm = d["d_mm"][mask]
        d_m = d_mm * 1e-3
        U = d["U_avg"][mask]
        p_Pa = d["p_Pa"]

        ln_d = np.log(d_mm)
        y = d_mm / U  # mm/V

        ax.plot(
            ln_d, y,
            PRESSURE_MARKERS[p_str],
            color=PRESSURE_COLORS[p_str],
            markersize=3.5, label=f"{p_str} hPa",
        )

        # Theoretical straight line for this pressure (valid where Apd/C > 1)
        d_theory = np.linspace(d_mm.min(), d_mm.max(), 200)
        arg = A * p_Pa * d_theory * 1e-3 / C_ref
        valid = arg > 1
        y_th = np.where(valid, np.log(arg) / (B * p_Pa) * 1e3, np.nan)
        ax.plot(
            np.log(d_theory)[valid], y_th[valid],
            "-", color=PRESSURE_COLORS[p_str], linewidth=0.9, alpha=0.5,
        )

        # Compute fit parameters (not plotted)
        coeffs = np.polyfit(ln_d, y, 1)
        slope, intercept = coeffs
        B_est = 1e3 / (slope * p_Pa) if slope != 0 else np.nan
        C_direct = np.nan
        gamma_lin = np.nan
        if slope != 0:
            val = intercept / slope
            ln_C = np.log(A * p_Pa / 1e3) - val
            if not np.isnan(ln_C):
                C_direct = np.exp(ln_C)
                if C_direct > 0:
                    gamma_lin = 1.0 / (np.exp(C_direct) - 1)

        fit_results[p_str] = {
            "slope": slope,
            "intercept": intercept,
            "B_estimated": float(B_est),
            "n_points": int(len(ln_d)),
            "C_from_intercept": float(C_direct) if not np.isnan(C_direct) else None,
            "gamma_from_linear_fit": float(gamma_lin) if not np.isnan(gamma_lin) else None,
        }

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=GREY, linewidth=1.0))
    labels.append(f"Paschen theory (γ = {gamma_ref:.4f})")
    ax.legend(handles, labels)

    ax.set_xlabel("$\\ln(d / 1\\,{\\rm mm})$")
    ax.set_ylabel("$d/U_{\\rm br}$ (mm V$^{-1}$)")
    save(fig, FIG_DIR, "linear_plot")
    return fit_results


def compute_gamma_from_minimum(data):
    """Extract gamma from the Paschen minimum using Eq. 7-9."""
    results = {}
    for p_str, d in data.items():
        i_min = np.argmin(d["U_avg"])
        U_min = d["U_avg"][i_min]
        d_min_mm = d["d_mm"][i_min]
        d_min_m = d_min_mm * 1e-3
        pd_min = d["p_Pa"] * d_min_m

        C_from_Umin = U_min * A / (E_EULER * B)
        C_from_pdmin = A * pd_min / E_EULER

        gamma_from_Umin = 1.0 / (np.exp(C_from_Umin) - 1)
        gamma_from_pdmin = 1.0 / (np.exp(C_from_pdmin) - 1)

        results[p_str] = {
            "d_min_mm": float(d_min_mm),
            "U_min_V": float(U_min),
            "pd_min_Pa_m": float(pd_min),
            "C_from_Umin": float(C_from_Umin),
            "C_from_pdmin": float(C_from_pdmin),
            "gamma_from_Umin": float(gamma_from_Umin),
            "gamma_from_pdmin": float(gamma_from_pdmin),
        }
    return results


def plot_with_theory(data, gamma_results):
    """Plot 4: measured data with theoretical Paschen overlay."""
    fig, ax = plt.subplots(figsize=(COL, 2.4))

    r8 = gamma_results["8"]
    gamma_avg = 0.5 * (r8["gamma_from_Umin"] + r8["gamma_from_pdmin"])
    C_avg = np.log(1.0 / gamma_avg + 1)

    pd_theory = np.linspace(0.4, 5.0, 500)
    U_theory = paschen_voltage(pd_theory, C_avg)
    ax.plot(
        pd_theory, U_theory, "-", color=GREY, linewidth=1.0,
        label=f"Theory (γ = {gamma_avg:.4f})"
    )

    for p_str, d in sorted(data.items(), key=lambda x: -x[1]["p_hPa"]):
        pd_vals = d["d_mm"] * 1e-3 * d["p_Pa"]
        ax.errorbar(
            pd_vals, d["U_avg"], yerr=d["U_std"],
            fmt=PRESSURE_MARKERS[p_str],
            color=PRESSURE_COLORS[p_str],
            markersize=3.5,
            label=f"{p_str} hPa data"
        )

    ax.set_xlabel("Pressure × gap, $pd$ (Pa m)")
    ax.set_ylabel("Breakdown voltage, $U_{\\rm br}$ (V)")
    ax.set_ylim(bottom=350)
    end_labels(ax, data, lambda d: d["d_mm"] * 1e-3 * d["p_Pa"])
    ax.text(4.9, paschen_voltage(np.array([4.9]), C_avg)[0], f"theory\n($\\gamma$ = {gamma_avg:.4f})",
            color=GREY, fontsize=8, ha="right", va="bottom")
    ax.set_xlim(right=5.3)
    save(fig, FIG_DIR, "U_vs_pd_with_theory")
    return gamma_avg, C_avg


def main():
    data = load_data()

    plot_U_vs_d(data)
    print("Generated: U_vs_d.png")

    plot_U_vs_pd(data)
    print("Generated: U_vs_pd.png")

    linear_results = plot_linear(data)
    print("Generated: linear_plot.png")

    gamma_min_results = compute_gamma_from_minimum(data)

    gamma_avg, C_avg = plot_with_theory(data, gamma_min_results)
    print("Generated: U_vs_pd_with_theory.png")

    results = {
        "gamma_from_minimum": gamma_min_results,
        "gamma_from_linear_fit": linear_results,
        "gamma_average": float(gamma_avg),
        "C_average": float(C_avg),
        "U_min_theoretical": float(U_min_from_C(C_avg)),
        "pd_min_theoretical": float(pd_min_from_C(C_avg)),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)

    print(f"\nResults saved to {RESULTS_FILE}")

    print("\n--- Gamma from Paschen Minimum ---")
    for p, r in sorted(gamma_min_results.items()):
        print(f"  {p} hPa: d_min = {r['d_min_mm']:.2f} mm, "
              f"U_min = {r['U_min_V']:.0f} V, "
              f"γ(U_min) = {r['gamma_from_Umin']:.6f}, "
              f"γ(pd_min) = {r['gamma_from_pdmin']:.6f}")

    print(f"\n  Average γ = {gamma_avg:.6f}")
    print(f"  Theoretical U_min = {U_min_from_C(C_avg):.1f} V")
    print(f"  Theoretical (pd)_min = {pd_min_from_C(C_avg):.4f} Pa·m")


if __name__ == "__main__":
    main()
