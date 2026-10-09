import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # plotstyle.py lives next to this script
from plotstyle import COL, FULL, BLUE, RED, GREY, INK, apply_style, save  # noqa: E402

apply_style()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
OUT_DIR = Path(__file__).resolve().parents[1]

INDEXED_PEAKS = {
    "Sample Run_11": [
        (28.43, "(111)"),
        (47.33, "(220)"),
        (56.15, "(311)"),
        (69.11, "(400)"),
        (76.35, "(331)"),
        (88.01, "(422)"),
        (94.93, "(511)"),
        (106.69, "(440)"),
        (114.07, "(531)"),
        (127.53, "(620)"),
        (136.89, "(533)"),
    ],
    "Sample Run_7": [
        (40.67, "(110)"),
        (58.81, "(200)"),
        (73.83, "(211)"),
        (116.07, "(222)"),
    ],
}


def read_xrc_csv(path: Path):
    angles = []
    intensities = []
    meta = {}
    in_scan = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip() == "[Scan points]":
                in_scan = True
                next(reader, None)
                continue
            if in_scan:
                try:
                    angles.append(float(row[0]))
                    intensities.append(float(row[1]))
                except (ValueError, IndexError):
                    continue
            else:
                if len(row) >= 2 and row[0] and row[1]:
                    meta[row[0].strip()] = row[1].strip()
    return np.array(angles), np.array(intensities), meta


def smooth(y, window=9):
    if window < 3:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def find_peaks(x, y, min_distance=0.15):
    y_s = smooth(y, 9)
    baseline = np.percentile(y_s, 20)
    noise = np.std(y_s[y_s < np.percentile(y_s, 50)])
    threshold = baseline + 4 * noise
    peaks = []
    for i in range(2, len(y_s) - 2):
        if y_s[i] > threshold and y_s[i] >= y_s[i - 1] and y_s[i] >= y_s[i + 1]:
            peaks.append((x[i], y[i]))
    peaks = sorted(peaks, key=lambda p: p[1], reverse=True)
    filtered = []
    for px, py in peaks:
        if all(abs(px - fx) >= min_distance for fx, _ in filtered):
            filtered.append((px, py))
        if len(filtered) >= 20:
            break
    return sorted(filtered, key=lambda p: p[0])


def bragg_d(two_theta_deg, wavelength):
    theta = np.radians(two_theta_deg / 2.0)
    return wavelength / (2 * np.sin(theta))


def plot_annotated(path: Path, x, y, indexed, sample_label):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=((FULL, 2.0) if len(indexed) > 6 else (COL, 2.1)))
    ax.plot(x, y, color=BLUE, linewidth=0.5)

    y_max = y.max()
    for two_theta_target, hkl_label in indexed:
        idx = np.argmin(np.abs(x - two_theta_target))
        search_lo = max(0, idx - 15)
        search_hi = min(len(y), idx + 15)
        peak_idx = search_lo + np.argmax(y[search_lo:search_hi])
        peak_x = x[peak_idx]
        peak_y = y[peak_idx]

        ax.annotate(
            hkl_label,
            xy=(peak_x, peak_y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK,
        )

    ax.set_xlabel(r"Diffraction angle, $2\theta$ (°)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0, top=y.max() * 1.12)
    out_path = save(fig, Path(__file__).resolve().parents[1] / "report" / "figures", f"{path.stem}_xrc")
    return out_path


def process_file(path: Path):
    x, y, meta = read_xrc_csv(path)
    if len(x) == 0:
        raise ValueError(f"No scan points found in {path}")
    wavelength = float(meta.get("K-Alpha1 wavelength", "1.5405980"))
    peaks = find_peaks(x, y)
    peak_rows = []
    for two_theta, intensity in peaks:
        d = bragg_d(two_theta, wavelength)
        peak_rows.append(
            {
                "two_theta_deg": round(two_theta, 3),
                "intensity": round(float(intensity), 1),
                "d_angstrom": round(float(d), 4),
            }
        )

    stem = path.stem
    indexed = INDEXED_PEAKS.get(stem, [])
    if stem == "Sample Run_11":
        label = "Unknown 1 (Sample Run 11) — Diamond-Cubic Si"
    elif stem == "Sample Run_7":
        label = "Unknown 2 (Sample Run 7) — BCC Mo"
    else:
        label = stem.replace("_", " ")

    plot_path = plot_annotated(path, x, y, indexed, label)

    return {
        "file": str(path.name),
        "wavelength": wavelength,
        "scan_range": [float(x.min()), float(x.max())],
        "step": float(np.median(np.diff(x))),
        "peaks": peak_rows,
        "plot": str(Path(plot_path).relative_to(OUT_DIR)),
    }


def main():
    results = []
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        results.append(process_file(csv_path))
    out_path = OUT_DIR / "xrc_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
