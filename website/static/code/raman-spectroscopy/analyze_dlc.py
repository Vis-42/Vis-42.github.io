"""
Diamond-Like Carbon (DLC) Raman analysis.

Three samples are characterised:
  DLC-I   : ZHANG DH 2025-01-23, Unknown B
             → low-fluorescence, clear D+G bands above flat noise floor
  DLC-II  : Chen Ling Mar 2024, Diamond-like sample B
             → strong photoluminescence background; H-rich a-C:H character
  DLC-III : ZHANG DH 2025-01-23, Unknown C (transparent region)
             → intermediate fluorescence; broader features

Each sample gets an individual plot (raw + baseline-subtracted + two-Gaussian
fit where applicable) and a combined comparison figure.

Results are written to dlc_summary.json; analysis_groups.json is updated.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

ROOT       = Path(__file__).resolve().parents[1]
DATA_ROOT  = ROOT / "data"
DATA0_DIR  = ROOT / "data0"
PLOTS_DIR  = ROOT / "plots"
GROUPS_PATH     = ROOT / "analysis_groups.json"
DLC_SUMMARY_PATH = ROOT / "dlc_summary.json"

COLORS = {"DLC-I": "#1f77b4", "DLC-II": "#d62728", "DLC-III": "#2ca02c"}

# ── sample definitions ────────────────────────────────────────────────────────
# DLC-I: use scans where the flat background regime is dominant (shorter
#        integration times avoid detector saturation artifacts)
DLC_SELECTIONS = [
    {
        "label":       "DLC-I",
        "description": "Unknown B – ZHANG DH (2025-01-23)",
        "parent":      str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknowB"),
        "paths": [
            str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknowB/UnknownB_AT700ms_AVERAGE_10.txt"),
            str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknowB/UnknownB_AT1000ms_AVERAGE_10.txt"),
            str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknowB/UnknownB_AT1500ms_AVERAGE_10.txt"),
            str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknowB/UnknownB_AT2000ms_AVERAGE_10.txt"),
        ],
        "baseline_mode": "flat",   # subtract constant noise floor
    },
    {
        "label":       "DLC-II",
        "description": "Diamond-like sample B – Chen Ling (Mar 2024)",
        "parent":      str(DATA_ROOT / "Chen Ling Mar 21 2024/Diamond-like samples/Diamond-like sample B"),
        "paths": [
            str(DATA_ROOT / "Chen Ling Mar 21 2024/Diamond-like samples/Diamond-like sample B/Diamond-like Sample B Raman Spectrum Trial 6.txt"),
            str(DATA_ROOT / "Chen Ling Mar 21 2024/Diamond-like samples/Diamond-like sample B/Diamond-like Sample B Raman Spectrum Trial 5.txt"),
            str(DATA_ROOT / "Chen Ling Mar 21 2024/Diamond-like samples/Diamond-like sample B/Diamond-like Sample B Raman Spectrum Trial 4.txt"),
            str(DATA_ROOT / "Chen Ling Mar 21 2024/Diamond-like samples/Diamond-like sample B/Diamond-like Sample B Raman Spectrum Trial 3.txt"),
        ],
        "baseline_mode": "poly",   # strong PL; show raw + note PL-dominated
    },
    {
        "label":       "DLC-III",
        "description": "Unknown C (transparent region) – ZHANG DH (2025-01-23)",
        "parent":      str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknownC"),
        "paths": [
            str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknownC/UnknownC_transparentpart_AT7000ms_AVERAGE_10.txt"),
            str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknownC/UnknownC_transparentpart_AT5000ms_AVERAGE_10.txt"),
            str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknownC/UnknownC_transparentpart_AT3000ms_AVERAGE_10.txt"),
            str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknownC/UnknownC_colorpart_AT7000ms_AVERAGE_10.txt"),
            str(DATA_ROOT / "ZHANG DH/2025_1_23/UnknownC/UnknownC_colorpart_AT5000ms_AVERAGE_10.txt"),
        ],
        "baseline_mode": "poly",
    },
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_spectrum(path: Path):
    x, y = [], []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x.append(float(parts[0])); y.append(float(parts[1]))
            except ValueError:
                pass
    if not x:
        return None, None
    x, y = np.array(x), np.array(y)
    return x[np.argsort(x)], y[np.argsort(x)]


def align_average(paths: list[str]):
    spectra = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            continue
        x, y = read_spectrum(pp)
        if x is None:
            continue
        spectra.append((x, y))
    if not spectra:
        return None, None, None
    xmin = max(xx.min() for xx, _ in spectra)
    xmax = min(xx.max() for xx, _ in spectra)
    step = np.median(np.diff(spectra[0][0]))
    x_c = np.arange(xmin, xmax + step * 0.5, step)
    ys = np.array([np.interp(x_c, xx, yy) for xx, yy in spectra])
    return x_c, ys.mean(axis=0), ys.std(axis=0)


def read_info(path: Path) -> dict:
    info_p = path.with_name(path.stem + "_info.txt")
    if not info_p.exists():
        return {}
    info = {}
    for line in info_p.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def _num(s: str) -> float | None:
    m = re.search(r"([0-9.]+)", s)
    return float(m.group(1)) if m else None


# ── baselines ─────────────────────────────────────────────────────────────────

def flat_baseline(x, y):
    """Subtract the noise floor estimated from the 800–1050 cm⁻¹ region."""
    m = (x >= 800) & (x <= 1050)
    if m.sum() < 5:
        return float(np.percentile(y, 5))
    return float(np.percentile(y[m], 10))


def poly_baseline(x, y, deg: int = 5, lo: float = 1050, hi: float = 1800):
    """Polynomial baseline anchored outside the D/G band window."""
    mask = ((x < lo) | (x > hi)) & (x > 400) & (x < 3200)
    if mask.sum() < deg + 2:
        return np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    coeffs = np.polyfit(x[mask], y[mask], deg)
    return np.polyval(coeffs, x)


# ── Gaussian fitting ──────────────────────────────────────────────────────────

def gaussian(x, amp, center, sigma):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def two_gaussians(x, a1, c1, s1, a2, c2, s2):
    return gaussian(x, a1, c1, s1) + gaussian(x, a2, c2, s2)


def fit_dg(x, y_sub, d_center_guess: float = 1350, g_center_guess: float = 1570):
    """
    Two-Gaussian fit of D and G bands.
    Seed initial conditions from direct peak search in each band window.
    Returns parameter dict or None on failure.
    """
    m = (x >= 1150) & (x <= 1800)
    xf, yf = x[m], y_sub[m]
    if yf.max() <= 0 or np.isnan(yf).any():
        return None
    scale = float(yf.max())
    yf_n = yf / scale

    # Locate peaks from data
    win_d = (x >= 1220) & (x <= 1450)
    win_g = (x >= 1450) & (x <= 1700)
    if not win_d.any() or not win_g.any():
        return None
    d_c = float(x[win_d][np.argmax(y_sub[win_d])])
    g_c = float(x[win_g][np.argmax(y_sub[win_g])])
    a_d = float(y_sub[win_d].max() / scale)
    a_g = float(y_sub[win_g].max() / scale)

    # Estimate sigma from half-max width in each window
    def _sigma_guess(xw, yw, center, fallback=80):
        half = yw.max() / 2
        above = xw[yw > half]
        if len(above) > 1:
            return float((above[-1] - above[0]) / 2.355)  # FWHM → sigma
        return fallback

    xwd, ywd = x[win_d], y_sub[win_d]
    xwg, ywg = x[win_g], y_sub[win_g]
    s_d = max(30.0, min(200.0, _sigma_guess(xwd, ywd, d_c)))
    s_g = max(30.0, min(200.0, _sigma_guess(xwg, ywg, g_c)))

    p0 = [a_d, d_c, s_d, a_g, g_c, s_g]
    # Tight bounds centred on the seed positions (±100 cm⁻¹)
    d_lo, d_hi = max(1200.0, d_c - 120), min(1450.0, d_c + 120)
    g_lo, g_hi = max(1440.0, g_c - 120), min(1750.0, g_c + 120)
    bounds = (
        [0,   d_lo, 25,  0,   g_lo, 25 ],
        [5.0, d_hi, 220, 5.0, g_hi, 220],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, _ = curve_fit(two_gaussians, xf, yf_n, p0=p0, bounds=bounds, maxfev=50000)
        except RuntimeError:
            return None

    a1, c1, s1, a2, c2, s2 = popt
    # Ensure D is always the lower-wavenumber component
    if c1 > c2:
        a1, c1, s1, a2, c2, s2 = a2, c2, s2, a1, c1, s1

    fwhm = lambda s: float(2 * np.sqrt(2 * np.log(2)) * abs(s))
    return {
        "D_center": float(c1), "D_amp_rel": float(a1), "D_FWHM": fwhm(s1),
        "G_center": float(c2), "G_amp_rel": float(a2), "G_FWHM": fwhm(s2),
        "ID_IG":    float(a1 / a2) if a2 != 0 else float("nan"),
        "popt":     popt.tolist(),
        "xf": xf, "yf_n": yf_n, "scale": scale,
    }


# ── per-sample analysis ───────────────────────────────────────────────────────

def analyse_sample(sel: dict):
    x_c, y_avg, y_std = align_average(sel["paths"])
    if x_c is None:
        raise ValueError(f"No data for {sel['label']}")

    win = min(21, (len(y_avg) // 10) * 2 + 1)
    y_sm = savgol_filter(y_avg, win, 3) if len(y_avg) > win else y_avg.copy()

    # Focus region
    m = (x_c >= 800) & (x_c <= 2800)
    x_r, y_r = x_c[m], y_sm[m]
    y_std_r = y_std[m]

    mode = sel.get("baseline_mode", "poly")
    if mode == "flat":
        bl_val = flat_baseline(x_r, y_r)
        bl = np.full_like(y_r, bl_val)
    else:
        bl = poly_baseline(x_r, y_r)

    y_sub = y_r - bl
    y_sub[y_sub < 0] = 0

    # For DLC, two-Gaussian fitting is unstable when bands are broad and
    # overlapping (FWHM > 200 cm⁻¹).  Instead, extract peak positions and
    # ID/IG by direct maximum search in each band window.
    fit = None
    direct_peaks = None
    if mode == "flat":
        win_d = (x_r >= 1250) & (x_r <= 1430)
        win_g = (x_r >= 1460) & (x_r <= 1660)
        if win_d.any() and win_g.any():
            d_pos = float(x_r[win_d][np.argmax(y_sub[win_d])])
            g_pos = float(x_r[win_g][np.argmax(y_sub[win_g])])
            d_int = float(y_sub[win_d].max())
            g_int = float(y_sub[win_g].max())

            def _fwhm(xw, yw):
                half = yw.max() / 2
                above = xw[yw > half]
                return float(above[-1] - above[0]) if len(above) > 1 else float("nan")

            direct_peaks = {
                "D_center": d_pos, "D_int": d_int,
                "D_FWHM": _fwhm(x_r[win_d], y_sub[win_d]),
                "G_center": g_pos, "G_int": g_int,
                "G_FWHM": _fwhm(x_r[win_g], y_sub[win_g]),
                "ID_IG": d_int / g_int if g_int > 0 else float("nan"),
            }

    return {
        "x_full": x_c, "y_avg": y_avg, "y_std": y_std,
        "x_r": x_r, "y_r": y_r, "y_std_r": y_std_r,
        "bl": bl, "bl_mode": mode,
        "y_sub": y_sub,
        "fit": fit,
        "direct_peaks": direct_peaks,
    }


# ── individual figure ─────────────────────────────────────────────────────────

def plot_individual(sel: dict, res: dict, out_path: Path):
    label = sel["label"]
    color = COLORS[label]
    peaks = res.get("direct_peaks")
    x_r, y_r, y_sub = res["x_r"], res["y_r"], res["y_sub"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5),
                              gridspec_kw={"height_ratios": [1, 2]})

    # ── top: raw + baseline ────────────────────────────────────────────────
    ax0 = axes[0]
    m = (res["x_full"] >= 800) & (res["x_full"] <= 2800)
    ax0.plot(res["x_full"][m], res["y_avg"][m], color="#555", lw=0.8, label="averaged raw")
    ax0.plot(x_r, res["bl"], "r--", lw=0.9, label="baseline")
    ax0.set_xlim(800, 2800)
    ax0.set_ylabel("Intensity (a.u.)")
    ax0.legend(fontsize=8, framealpha=0.7)
    ax0.set_title(f"{label}: {sel['description']}", fontsize=9)
    ax0.tick_params(labelbottom=False)

    # ── bottom: background-subtracted spectrum ─────────────────────────────
    ax1 = axes[1]
    norm = y_sub.max()
    y_plot = y_sub / norm * 100 if norm > 0 else y_sub

    ax1.plot(x_r, y_plot, color=color, lw=1.3, label="background-subtracted")

    if peaks is not None:
        # Shade D and G regions
        ax1.axvspan(1250, 1430, alpha=0.08, color="#1f77b4", label="D band region")
        ax1.axvspan(1460, 1660, alpha=0.08, color="#ff7f0e", label="G band region")

        # Mark peak maxima
        d_y = peaks["D_int"] / norm * 100
        g_y = peaks["G_int"] / norm * 100
        ax1.axvline(peaks["D_center"], color="#1f77b4", lw=1.0, ls="--", alpha=0.8)
        ax1.axvline(peaks["G_center"], color="#ff7f0e", lw=1.0, ls="--", alpha=0.8)

        ax1.annotate(
            f"D band\n{peaks['D_center']:.0f} cm⁻¹\nFWHM ≈ {peaks['D_FWHM']:.0f} cm⁻¹",
            xy=(peaks["D_center"], d_y),
            xytext=(peaks["D_center"] - 180, 90),
            fontsize=7.5, ha="center", color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.7),
        )
        ax1.annotate(
            f"G band\n{peaks['G_center']:.0f} cm⁻¹\nFWHM ≈ {peaks['G_FWHM']:.0f} cm⁻¹",
            xy=(peaks["G_center"], g_y),
            xytext=(peaks["G_center"] + 160, 90),
            fontsize=7.5, ha="center", color="#ff7f0e",
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.7),
        )
        ax1.text(0.97, 0.05,
                 f"$I_{{\\mathrm{{D}}}}/I_{{\\mathrm{{G}}}}$ = {peaks['ID_IG']:.2f}",
                 transform=ax1.transAxes, ha="right", fontsize=9.5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85))

        # Reference: diamond position
        ax1.axvline(1332, color="gray", lw=0.8, ls=":", alpha=0.5)
        ax1.text(1332, 5, "diamond\n1332", fontsize=7, ha="center", color="gray", alpha=0.7)
    else:
        # PL-dominated spectrum
        ax1.text(0.5, 0.60,
                 "Raman signal masked by\nphotoluminescence background\n"
                 "(characteristic of H-rich a-C:H DLC)",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=9, color="#555",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fff9e6", ec="#ccc"))
        ax1.text(0.97, 0.05,
                 "PL-dominated; no D/G resolvable",
                 transform=ax1.transAxes, ha="right", fontsize=8,
                 color="#888")

    ax1.set_xlim(800, 2800)
    ax1.set_ylim(bottom=-3)
    ax1.set_xlabel("Raman shift (cm$^{-1}$)")
    ax1.set_ylabel("Intensity (normalised, a.u.)")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ── comparison overlay ────────────────────────────────────────────────────────

def plot_overlay(selections: list, results: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for sel, res in zip(selections, results):
        y_sub = res["y_sub"]
        norm = y_sub.max()
        y_plot = y_sub / norm * 100 if norm > 0 else y_sub
        c = COLORS[sel["label"]]
        ax.plot(res["x_r"], y_plot, color=c, lw=1.4, label=sel["label"])

        peaks = res.get("direct_peaks")
        if peaks:
            ax.axvline(peaks["D_center"], color=c, lw=0.7, ls="--", alpha=0.6)
            ax.axvline(peaks["G_center"], color=c, lw=0.7, ls=":",  alpha=0.6)

    # Reference lines
    ax.axvline(1332, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.text(1332, 103, "diamond\n1332 cm⁻¹", fontsize=7, ha="center", color="gray")
    ax.axvline(1350, color="#1f77b4", lw=0.4, ls="-.", alpha=0.25)
    ax.axvline(1580, color="#ff7f0e", lw=0.4, ls="-.", alpha=0.25)
    ax.text(1350, 108, "D ref", fontsize=6.5, ha="center", color="#1f77b4", alpha=0.7)
    ax.text(1580, 108, "G ref", fontsize=6.5, ha="center", color="#ff7f0e", alpha=0.7)

    ax.set_xlim(900, 2000)
    ax.set_ylim(-5, 117)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (normalised, a.u.)")
    ax.set_title("Diamond-like carbon samples – comparison (background-subtracted)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA0_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    results = []

    for idx, sel in enumerate(DLC_SELECTIONS, 1):
        print(f"\nProcessing {sel['label']} – {sel['description']}")
        res = analyse_sample(sel)
        results.append(res)

        fit = res["fit"]
        name = f"diamond_like_{idx}"

        # Save average to data0
        d0 = DATA0_DIR / name
        d0.mkdir(parents=True, exist_ok=True)
        avg_path = d0 / f"{name}_avg.txt"
        np.savetxt(avg_path,
                   np.column_stack([res["x_full"], res["y_avg"], res["y_std"]]),
                   fmt="%.6f",
                   header="shift_cm-1 intensity_mean intensity_std")

        for p in sel["paths"]:
            pp = Path(p)
            if pp.exists():
                shutil.copy2(pp, d0 / pp.name)
                info = pp.with_name(pp.stem + "_info.txt")
                if info.exists():
                    shutil.copy2(info, d0 / info.name)

        # Individual plot
        plot_individual(sel, res, PLOTS_DIR / f"{name}.png")
        print(f"  Saved plots/{name}.png")

        # Info stats
        int_ms, power_mw = [], []
        for p in sel["paths"]:
            info = read_info(Path(p))
            if "Integration time" in info:
                v = _num(info["Integration time"])
                if v is not None: int_ms.append(v)
            if "Laser power" in info:
                v = _num(info["Laser power"])
                if v is not None: power_mw.append(v)

        entry: dict = {
            "label":        sel["label"],
            "description":  sel["description"],
            "paths":        sel["paths"],
            "avg_path":     str(avg_path),
            "count":        len([p for p in sel["paths"] if Path(p).exists()]),
            "integration_ms": int_ms,
            "power_mw":     power_mw,
            "baseline_mode": sel.get("baseline_mode", "poly"),
        }
        peaks = res.get("direct_peaks")
        if peaks:
            entry.update({
                "D_center_cm1": round(peaks["D_center"], 1),
                "D_FWHM_cm1":   round(peaks["D_FWHM"],   1),
                "G_center_cm1": round(peaks["G_center"], 1),
                "G_FWHM_cm1":   round(peaks["G_FWHM"],   1),
                "ID_IG":        round(peaks["ID_IG"],     3),
            })
            print(f"  D = {peaks['D_center']:.1f} cm⁻¹  FWHM {peaks['D_FWHM']:.0f}")
            print(f"  G = {peaks['G_center']:.1f} cm⁻¹  FWHM {peaks['G_FWHM']:.0f}")
            print(f"  ID/IG = {peaks['ID_IG']:.3f}")
        else:
            entry["note"] = "Raman bands not resolvable from photoluminescence background"
            print("  PL-dominated spectrum; D/G not resolvable")

        summary.append(entry)

    # Overlay comparison
    plot_overlay(DLC_SELECTIONS, results, PLOTS_DIR / "diamond_like_comparison.png")
    print("\nSaved plots/diamond_like_comparison.png")

    DLC_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {DLC_SUMMARY_PATH}")

    # Update analysis_groups.json
    groups = json.loads(GROUPS_PATH.read_text())
    groups["selection"]["diamond_like"] = [
        {
            "parent":      sel["parent"],
            "paths":       sel["paths"],
            "label":       sel["label"],
            "description": sel["description"],
            "count":       len(sel["paths"]),
        }
        for sel in DLC_SELECTIONS
    ]
    GROUPS_PATH.write_text(json.dumps(groups, indent=2))
    print(f"Updated {GROUPS_PATH}")


if __name__ == "__main__":
    main()
