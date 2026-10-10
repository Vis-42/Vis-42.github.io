import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # local helper modules (plotstyle comes from ~/-uni/_shared/python)
from plotstyle import COL, BLUE, RED, apply_style, save  # noqa: E402

apply_style()


DATA_DIR = Path(__file__).resolve().parents[1] / "data "
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
OUT_DIR = Path(__file__).resolve().parents[1]

LABELS = {
    "sol1": "BSA + PBS",
    "sol2": "BSA + acid",
    "sol3": "Lysozyme + PBS",
    "sol4": "Lysozyme + acid",
}

# Colour identifies the protein, line style the solvent (solid: native in PBS, dashed: acid).
COLORS = {"sol1": BLUE, "sol2": BLUE, "sol3": RED, "sol4": RED}
STYLES = {"sol1": "-", "sol2": "--", "sol3": "-", "sol4": "--"}


def read_fluorescence_csv(path: Path):
    wavelengths = []
    intensities = []
    meta = {}
    in_data = False
    in_meta = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                if in_data:
                    in_data = False
                continue
            if stripped.startswith("Wavelength (nm)"):
                in_data = True
                continue
            if in_data:
                parts = stripped.split(",")
                try:
                    wavelengths.append(float(parts[0]))
                    intensities.append(float(parts[1]))
                except (ValueError, IndexError):
                    in_data = False
                    continue
            if ":" in stripped and not in_data:
                key_val = stripped.split(":", 1)
                if len(key_val) == 2:
                    meta[key_val[0].strip()] = key_val[1].strip()
    return np.array(wavelengths), np.array(intensities), meta


def smooth(y, window=5):
    if window < 3 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def find_peak(x, y, window=5):
    y_s = smooth(y, window)
    idx = np.argmax(y_s)
    return float(x[idx]), float(y[idx]), float(y_s[idx])


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    fig, ax = plt.subplots(figsize=(COL, 2.3))

    for stem in ["sol1", "sol2", "sol3", "sol4"]:
        path = DATA_DIR / f"{stem}.csv"
        x, y, meta = read_fluorescence_csv(path)
        label = LABELS.get(stem, stem)
        ax.plot(x, y, linewidth=1.0, ls=STYLES[stem], color=COLORS[stem])

        raw_peak_x, raw_peak_y, _ = find_peak(x, y, window=1)
        smooth_peak_x, _, smooth_peak_y = find_peak(x, y, window=7)
        ax.plot([smooth_peak_x], [smooth_peak_y], "o", ms=3, color=COLORS[stem])
        ax.lines[-2].set_label(f"{label}, peak {smooth_peak_x:.0f} nm")

        results.append({
            "file": path.name,
            "label": label,
            "x_range": [float(x.min()), float(x.max())],
            "peak_raw": {"wavelength_nm": raw_peak_x, "intensity": raw_peak_y},
            "peak_smoothed": {"wavelength_nm": smooth_peak_x, "intensity": round(smooth_peak_y, 2)},
            "meta": meta,
        })

    ax.set_xlabel("Emission wavelength (nm)")
    ax.set_ylabel("Fluorescence intensity (a.u.)")
    ax.set_xlim(300, 400)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.62)
    ax.legend(loc="upper right", fontsize=7.5, handlelength=2.2)
    plot_path = save(fig, Path(__file__).resolve().parents[1] / "report" / "figures", "fluorescence_emission")

    out_path = OUT_DIR / "fluorescence_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path} and {plot_path}")


if __name__ == "__main__":
    main()
