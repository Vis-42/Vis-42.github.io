from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import shutil

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
DATA0_DIR = ROOT / "data0"
GROUPS_PATH = ROOT / "analysis_groups.json"
SUMMARY_PATH = ROOT / "analysis_summary.json"
SELECTION_PATH = ROOT / "analysis_selection.json"


@dataclass
class InfoStats:
    integration_ms: list[float]
    averaging: list[int]
    laser_nm: list[float]
    power_mw: list[float]
    smoothing: list[str]


def read_spectrum(path: Path):
    data = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            return None
        data.append((x, y))
    if not data:
        return None
    arr = np.array(data)
    return arr[:, 0], arr[:, 1]


def read_info(path: Path) -> dict:
    info_path = path.with_name(path.stem + "_info.txt")
    if not info_path.exists():
        return {}
    info = {}
    for line in info_path.read_text().splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            info[key.strip()] = val.strip()
    return info


def parse_number(text: str) -> float | None:
    m = re.search(r"([0-9.]+)", text)
    return float(m.group(1)) if m else None


def gather_info_stats(paths: list[str]) -> InfoStats:
    integration_ms = []
    averaging = []
    laser_nm = []
    power_mw = []
    smoothing = []
    for p in paths:
        info = read_info(Path(p))
        if "Integration time" in info:
            val = parse_number(info["Integration time"])
            if val is not None:
                integration_ms.append(val)
        if "Averaging" in info:
            val = parse_number(info["Averaging"])
            if val is not None:
                averaging.append(int(val))
        if "Laser wavelength" in info:
            val = parse_number(info["Laser wavelength"])
            if val is not None:
                laser_nm.append(val)
        if "Laser power" in info:
            val = parse_number(info["Laser power"])
            if val is not None:
                power_mw.append(val)
        if "Smoothing" in info:
            smoothing.append(info["Smoothing"])
    return InfoStats(integration_ms, averaging, laser_nm, power_mw, smoothing)


def align_average(paths: list[str]):
    spectra = []
    for p in paths:
        spec = read_spectrum(Path(p))
        if spec is None:
            continue
        spectra.append(spec)
    if not spectra:
        return None
    xmin = max(x.min() for x, _ in spectra)
    xmax = min(x.max() for x, _ in spectra)
    if xmax <= xmin:
        return None
    x0 = spectra[0][0]
    step = np.median(np.diff(x0))
    if step <= 0:
        step = 1.0
    x_common = np.arange(xmin, xmax + step * 0.5, step)
    ys = []
    for x, y in spectra:
        ys.append(np.interp(x_common, x, y))
    y_stack = np.vstack(ys)
    return x_common, y_stack.mean(axis=0), y_stack.std(axis=0)


def peak_in_window(x, y, center, window=30):
    mask = (x >= center - window) & (x <= center + window)
    if mask.sum() == 0:
        return None
    idx = np.argmax(y[mask])
    return float(x[mask][idx]), float(y[mask][idx])


def band_ratio(x, y, c1, c2):
    p1 = peak_in_window(x, y, c1)
    p2 = peak_in_window(x, y, c2)
    if not p1 or not p2:
        return None
    return p1[1] / p2[1] if p2[1] != 0 else None


def save_plot(x, y, ystd, out_path, title, peak_labels):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=1)
    if ystd is not None:
        plt.fill_between(x, y - ystd, y + ystd, color="gray", alpha=0.2, linewidth=0)
    for label, cx in peak_labels:
        plt.axvline(cx, color="red", linewidth=0.6, alpha=0.6)
        plt.text(cx, y.max() * 0.9, label, rotation=90, va="top", ha="right", fontsize=8)
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def copy_group_to_data0(group_name: str, paths: list[str], avg_path: Path):
    dest_dir = DATA0_DIR / group_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        p = Path(p)
        shutil.copy2(p, dest_dir / p.name)
        info = p.with_name(p.stem + "_info.txt")
        if info.exists():
            shutil.copy2(info, dest_dir / info.name)
    dest_avg = dest_dir / avg_path.name
    if avg_path.resolve() != dest_avg.resolve():
        shutil.copy2(avg_path, dest_avg)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA0_DIR.mkdir(parents=True, exist_ok=True)

    groups = json.loads(GROUPS_PATH.read_text())
    selection = groups["selection"]

    summary = {}

    def process_group(name, group, peak_labels, peak_centers):
        paths = group["paths"]
        avg = align_average(paths)
        if avg is None:
            return
        x, y, ystd = avg

        avg_path = DATA0_DIR / name / f"{name}_avg.txt"
        avg_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            avg_path,
            np.column_stack([x, y, ystd]),
            fmt="%.6f",
            header="shift_cm-1 intensity_mean intensity_std",
        )

        peaks = {}
        for label, c in peak_centers.items():
            pk = peak_in_window(x, y, c)
            if pk:
                peaks[label] = pk[0]

        info_stats = gather_info_stats(paths)
        summary[name] = {
            "paths": paths,
            "avg_path": str(avg_path),
            "count": len(paths),
            "peaks": peaks,
            "integration_ms": info_stats.integration_ms,
            "averaging": info_stats.averaging,
            "laser_nm": info_stats.laser_nm,
            "power_mw": info_stats.power_mw,
            "smoothing": info_stats.smoothing,
        }

        plot_labels = []
        for label, c in peak_labels:
            pk = peak_in_window(x, y, c)
            plot_labels.append((label, pk[0] if pk else c))

        plot_path = PLOTS_DIR / f"{name}.png"
        save_plot(
            x,
            y,
            ystd,
            plot_path,
            name.replace("_", " ").title(),
            plot_labels,
        )
        copy_group_to_data0(name, paths, avg_path)

    process_group(
        "carbon_rod",
        selection["carbon_rod"],
        [("D", 1350), ("G", 1580), ("2D", 2700)],
        {"D": 1350, "G": 1580, "2D": 2700, "Dprime": 1620},
    )
    process_group(
        "hopg",
        selection["hopg"],
        [("D", 1350), ("G", 1580), ("2D", 2700)],
        {"D": 1350, "G": 1580, "2D": 2700, "Dprime": 1620},
    )
    process_group(
        "graphene",
        selection["graphene"],
        [("D", 1350), ("G", 1580), ("2D", 2700), ("Si", 520)],
        {"D": 1350, "G": 1580, "2D": 2700, "Dprime": 1620},
    )
    process_group(
        "cnt",
        selection["cnt"],
        [("D", 1350), ("G", 1580), ("2D", 2700)],
        {"D": 1350, "G": 1580, "2D": 2700, "Dprime": 1620},
    )
    process_group(
        "diamond",
        selection["diamond"],
        [("1332", 1332)],
        {"diamond": 1332},
    )
    process_group(
        "unknown_semiconductor",
        selection["unknown_semiconductor"],
        [("520", 520), ("~300", 300)],
        {"520": 520, "268": 268},
    )

    # diamond-like groups
    diamond_like = []
    for idx, group in enumerate(selection.get("diamond_like", []), 1):
        name = f"diamond_like_{idx}"
        paths = group["paths"]
        avg = align_average(paths)
        if avg is None:
            continue
        x, y, ystd = avg
        avg_path = DATA0_DIR / name / f"{name}_avg.txt"
        avg_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            avg_path,
            np.column_stack([x, y, ystd]),
            fmt="%.6f",
            header="shift_cm-1 intensity_mean intensity_std",
        )
        pk = peak_in_window(x, y, 1332)
        plot_path = PLOTS_DIR / f"{name}.png"
        save_plot(x, y, ystd, plot_path, f"Diamond-like {idx}", [("1332", 1332)])
        copy_group_to_data0(name, paths, avg_path)
        info_stats = gather_info_stats(paths)
        diamond_like.append(
            {
                "paths": paths,
                "avg_path": str(avg_path),
                "count": len(paths),
                "diamond_peak": pk[0] if pk else None,
                "integration_ms": info_stats.integration_ms,
                "averaging": info_stats.averaging,
                "laser_nm": info_stats.laser_nm,
                "power_mw": info_stats.power_mw,
                "smoothing": info_stats.smoothing,
            }
        )

    if diamond_like:
        summary["diamond_like"] = diamond_like

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    SELECTION_PATH.write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
