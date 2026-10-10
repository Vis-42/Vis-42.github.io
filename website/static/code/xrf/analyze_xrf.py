import csv
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # local helper modules (plotstyle comes from ~/-uni/_shared/python)
from plotstyle import COL, BLUE, RED, GOLD, PURPLE, TEAL, GREY, INK, LIGHT, apply_style, save  # noqa: E402

apply_style()


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
FIG_DIR = ROOT / "report" / "figures"  # report figures (SVG, printed size)
OUT_JSON = ROOT / "xrf_analysis.json"

DENSITIES = {
    "Cu": 8.96,
    "Zn": 7.14,
    "Pb": 11.34,
}

LINE_ENERGIES = {
    "Fe Kα": 6.403,
    "Fe Kβ": 7.057,
    "Zn Kα": 8.638,
    "Zn Kβ": 9.572,
    "Cu Kα": 8.044,
    "Cu Kβ": 8.905,
    "Ni Kα": 7.478,
    "Ni Kβ": 8.265,
    "Pb Lα": 10.551,
    "Pb Lβ": 12.614,
    "Pb Lγ": 14.764,
    "Co Kα": 6.930,
    "Co Kβ": 7.649,
    "Sm Lα": 5.637,
    "Sm Lβ": 6.370,
}

SAMPLE_CONFIG = {
    "calibration": {
        "csv": DATA_DIR / "exp 1" / "calibration" / "Calibration curve.csv",
        "title": "Calibration Target (Fe + Zn)",
        "labels": [("Fe Kα", 6.403), ("Fe Kβ", 7.057), ("Zn Kα", 8.638), ("Zn Kβ", 9.572)],
        "out": "calibration_spectrum.png",
    },
    "brass": {
        "csv": DATA_DIR / "exp 1" / "brass" / "brass.csv",
        "title": "Brass (Target 3)",
        "labels": [
            ("Cu Kα", 8.044), ("Zn Kα", 8.638), ("Cu Kβ", 8.905),
            ("Zn Kβ", 9.572), ("Pb Lα", 10.551), ("Pb Lβ", 12.614), ("Pb Lγ", 14.764),
        ],
        "out": "brass_spectrum.png",
    },
    "sample4": {
        "csv": DATA_DIR / "exp 1" / "sample 4" / "sample.csv",
        "title": "Magnetic Sample (Target 4)",
        "labels": [("Sm Lα", 5.637), ("Sm Lβ", 6.370), ("Co Kα", 6.930), ("Co Kβ", 7.649)],
        "out": "sample4_spectrum.png",
    },
    "old50c": {
        "csv": DATA_DIR / "exp 1" / "old 50 cents" / "old 50 cents.csv",
        "title": "Old 50 Cents Coin",
        "labels": [("Ni Kα", 7.478), ("Cu Kα", 8.044), ("Ni Kβ", 8.265), ("Cu Kβ", 8.905)],
        "out": "old50c_spectrum.png",
    },
    "new50c": {
        "csv": DATA_DIR / "exp 1" / "new 50 cents" / "new 50 cents.csv",
        "title": "New 50 Cents Coin",
        "labels": [("Ni Kα", 7.478), ("Cu Kα", 8.044), ("Ni Kβ", 8.265), ("Cu Kβ", 8.905)],
        "out": "new50c_spectrum.png",
    },
    "old1d": {
        "csv": DATA_DIR / "exp 1" / "old 1 dollar" / "old 1 dollar.csv",
        "title": "Old 1 Dollar Coin",
        "labels": [("Cu Kα", 8.044), ("Cu Kβ", 8.905)],
        "out": "old1d_spectrum.png",
    },
    "new1d": {
        "csv": DATA_DIR / "exp 1" / "new 1 dollar" / "old 1 dollar.csv",
        "title": "New 1 Dollar Coin",
        "labels": [("Ni Kα", 7.478), ("Cu Kα", 8.044), ("Ni Kβ", 8.265), ("Cu Kβ", 8.905)],
        "out": "new1d_spectrum.png",
    },
}


@dataclass
class Spectrum:
    energy: np.ndarray
    events: np.ndarray
    source_file: str
    segment_count: int
    selected_segment: int


def _parse_float(value: str):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def read_spectrum_csv(path: Path) -> Spectrum:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            events = _parse_float(row[1])
            channel = _parse_float(row[2])
            energy = _parse_float(row[3])
            if channel is None or energy is None:
                continue
            rows.append((int(round(channel)), energy, 0.0 if events is None else events))

    if not rows:
        raise ValueError(f"No spectrum rows found in {path}")

    segments: List[List[Tuple[int, float, float]]] = [[]]
    prev_channel = rows[0][0]
    for ch, en, ev in rows:
        if ch < prev_channel - 5:
            segments.append([])
        segments[-1].append((ch, en, ev))
        prev_channel = ch

    segment_sums = [sum(r[2] for r in seg) for seg in segments]
    idx = int(np.argmax(segment_sums))
    seg = segments[idx]
    energy = np.array([r[1] for r in seg], dtype=float)
    events = np.array([r[2] for r in seg], dtype=float)
    return Spectrum(
        energy=energy,
        events=events,
        source_file=str(path.relative_to(ROOT)),
        segment_count=len(segments),
        selected_segment=idx,
    )


def locate_peak(energy: np.ndarray, events: np.ndarray,
                expected_keV: float, window_keV: float = 0.16) -> Optional[Tuple[float, float]]:
    lo = expected_keV - window_keV
    hi = expected_keV + window_keV
    mask = (energy >= lo) & (energy <= hi)
    if not np.any(mask):
        return None
    idx_local = np.argmax(events[mask])
    e_local = energy[mask][idx_local]
    n_local = events[mask][idx_local]
    return float(e_local), float(n_local)


def _merge_and_place_labels(peaks: List[dict], ymax: float) -> List[dict]:
    """Merge peaks that resolve to the same channel and compute label positions."""
    if not peaks:
        return []

    MERGE_TOL = 0.06  # keV — merge if observed positions within this tolerance
    merged = []
    used = [False] * len(peaks)

    for i, p in enumerate(peaks):
        if used[i]:
            continue
        group = [p]
        used[i] = True
        for j in range(i + 1, len(peaks)):
            if used[j]:
                continue
            if abs(peaks[j]["observed_keV"] - p["observed_keV"]) < MERGE_TOL:
                group.append(peaks[j])
                used[j] = True
        if len(group) == 1:
            merged.append(group[0])
        else:
            combined_label = " + ".join(g["label"] for g in group)
            merged.append({
                "label": combined_label,
                "expected_keV": group[0]["expected_keV"],
                "observed_keV": group[0]["observed_keV"],
                "height": group[0]["height"],
                "merged_from": group,
            })

    # Labels sit in three staggered tiers above the spectrum, in order of energy,
    # so neighbouring lines never share a tier; a thin leader joins each to its peak.
    merged.sort(key=lambda pk: pk["observed_keV"])
    for i, pk in enumerate(merged):
        pk["y_text"] = ymax * (1.22 - 0.13 * (i % 3))

    return merged


def plot_spectrum(sample_key: str, spec: Spectrum,
                  labels: List[Tuple[str, float]], out_file: Path):
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    ax.plot(spec.energy, spec.events, color=BLUE, linewidth=0.8)

    raw_peaks = []
    ymax = max(1.0, float(np.max(spec.events)))
    for label, e0 in labels:
        hit = locate_peak(spec.energy, spec.events, e0, window_keV=0.20)
        if hit is None:
            continue
        ep, npk = hit
        raw_peaks.append({
            "label": label, "expected_keV": e0,
            "observed_keV": ep, "height": npk,
        })

    merged = _merge_and_place_labels(raw_peaks, ymax)

    drawn_vlines = set()
    for pk in merged:
        ep = pk["observed_keV"]
        vline_key = round(ep, 2)
        if vline_key not in drawn_vlines:
            ax.axvline(ep, color=GREY, ls="--", linewidth=0.6, zorder=1)
            drawn_vlines.add(vline_key)

        ax.annotate(
            pk["label"].replace(" + ", "\n+ "), xy=(ep, pk["height"]),
            xytext=(ep + 0.15, pk["y_text"]),
            fontsize=7, va="bottom", ha="left", rotation=0, color=INK,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.4),
        )

    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel("Counts per channel")
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(bottom=0, top=ymax * 1.38)
    save(fig, FIG_DIR, Path(out_file).stem)
    return raw_peaks


def parse_labx_egaussfits(path: Path):
    with zipfile.ZipFile(path, "r") as zf:
        xml_data = zf.read("data.xml")
    root = ET.fromstring(xml_data)
    fits = []
    for elem in root.iter("egaussfit"):
        attrs = {}
        for k, v in elem.attrib.items():
            try:
                attrs[k] = float(v)
            except ValueError:
                attrs[k] = v
        fits.append(attrs)
    return fits


def _collect_components(fits: List[Dict], positive_only: bool = False):
    components = []
    for fit in fits:
        n = int(fit.get("n", 0))
        for i in range(1, n + 1):
            a = fit.get(f"a{i}")
            m = fit.get(f"m{i}")
            w = fit.get(f"w{i}")
            if a is None or m is None or w is None:
                continue
            if positive_only and a < 0:
                continue
            components.append({"a": float(a), "m": float(m), "w": float(w)})
    return components


def _pick_height(components: List[Dict], target_keV: float, tol_keV: float):
    cands = [c for c in components if abs(c["m"] - target_keV) <= tol_keV and c["a"] > 0]
    if not cands:
        return None
    return max(cands, key=lambda c: c["a"])["a"]


def extract_quantitative_values():
    h_dir = DATA_DIR / "exp 2" / "H"
    h0_dir = DATA_DIR / "exp 2" / "H0"

    h_trials = []
    for labx in sorted(h_dir.rglob("*.labx")):
        fits = parse_labx_egaussfits(labx)
        comps = _collect_components(fits, positive_only=True)
        cu = _pick_height(comps, 8.044, 0.08)
        zn = _pick_height(comps, 8.635, 0.08)
        pb = _pick_height(comps, 10.556, 0.20)
        if cu and zn and pb:
            h_trials.append({
                "file": str(labx.relative_to(ROOT)),
                "Cu": cu, "Zn": zn, "Pb": pb,
                "fits": fits,
            })

    ref_map = {
        "Cu": h0_dir / "Copper",
        "Zn": h0_dir / "Zinc",
        "Pb": h0_dir / "lead",
    }
    ref_keV = {"Cu": 8.044, "Zn": 8.635, "Pb": 10.556}
    h0_trials = {k: [] for k in ref_map}
    for elem, folder in ref_map.items():
        for labx in sorted(folder.rglob("*.labx")):
            fits = parse_labx_egaussfits(labx)
            comps = _collect_components(fits, positive_only=True)
            val = _pick_height(comps, ref_keV[elem], 0.20 if elem == "Pb" else 0.08)
            if val:
                h0_trials[elem].append({"file": str(labx.relative_to(ROOT)), "H0": val, "fits": fits})

    if not h_trials:
        raise RuntimeError("No usable brass H trials extracted from labx files")
    if not all(h0_trials[k] for k in ("Cu", "Zn", "Pb")):
        raise RuntimeError("Missing H0 trials for one or more reference elements")

    h_vals = {el: np.array([t[el] for t in h_trials], dtype=float) for el in ("Cu", "Zn", "Pb")}
    h0_vals = {el: np.array([t["H0"] for t in h0_trials[el]], dtype=float) for el in ("Cu", "Zn", "Pb")}

    h_stats = {
        el: {"mean": float(np.mean(vals)),
             "std": float(np.std(vals, ddof=1 if len(vals) > 1 else 0)),
             "n": int(len(vals))}
        for el, vals in h_vals.items()
    }
    h0_stats = {
        el: {"mean": float(np.mean(vals)),
             "std": float(np.std(vals, ddof=1 if len(vals) > 1 else 0)),
             "n": int(len(vals))}
        for el, vals in h0_vals.items()
    }

    rho_h_over_h0 = {}
    for el in ("Cu", "Zn", "Pb"):
        rho_h_over_h0[el] = DENSITIES[el] * h_stats[el]["mean"] / h0_stats[el]["mean"]
    denom = sum(rho_h_over_h0.values())
    c_nominal = {el: rho_h_over_h0[el] / denom for el in rho_h_over_h0}

    rng = np.random.default_rng(42)
    n_mc = 50000
    mc = {}
    for el in ("Cu", "Zn", "Pb"):
        mu_h, sd_h = h_stats[el]["mean"], max(1e-9, h_stats[el]["std"])
        mu_h0, sd_h0 = h0_stats[el]["mean"], max(1e-9, h0_stats[el]["std"])
        h_draw = np.clip(rng.normal(mu_h, sd_h, n_mc), 1e-9, None)
        h0_draw = np.clip(rng.normal(mu_h0, sd_h0, n_mc), 1e-9, None)
        mc[el] = DENSITIES[el] * h_draw / h0_draw
    denom_mc = mc["Cu"] + mc["Zn"] + mc["Pb"]
    c_mc = {el: mc[el] / denom_mc for el in ("Cu", "Zn", "Pb")}
    c_stats = {
        el: {
            "mean": float(np.mean(c_mc[el])),
            "std": float(np.std(c_mc[el], ddof=1)),
            "percent": float(100.0 * np.mean(c_mc[el])),
            "percent_std": float(100.0 * np.std(c_mc[el], ddof=1)),
        }
        for el in ("Cu", "Zn", "Pb")
    }

    return {
        "H_trials": h_trials,
        "H0_trials": h0_trials,
        "H_stats": h_stats,
        "H0_stats": h0_stats,
        "rho_H_over_H0": rho_h_over_h0,
        "mass_fraction_nominal": c_nominal,
        "mass_fraction_mc": c_stats,
    }


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


GAUSS_LABELS = {
    8.044: "Cu Kα",
    8.635: "Zn Kα",
    8.905: "Cu Kβ",
    10.556: "Pb Lα",
}


def plot_brass_gaussian_deconvolution(brass_spec: Spectrum,
                                      h_trials: List[Dict], out_file: Path):
    """Plot deconvolution with Gaussians scaled to match the Part 1 qualitative spectrum."""
    best = max(h_trials, key=lambda t: t["Cu"])
    fits = best["fits"]
    comps = _collect_components(fits, positive_only=True)

    x = brass_spec.energy
    y = brass_spec.events

    cu_peak_data = float(np.max(y[(x >= 7.9) & (x <= 8.2)]))
    cu_amp_fit = best["Cu"]
    scale = cu_peak_data / cu_amp_fit if cu_amp_fit > 0 else 1.0

    window = (x >= 7.5) & (x <= 11.3)
    xw, yw = x[window], y[window]

    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.plot(xw, yw, color=INK, linewidth=1.0)

    colors = {"Cu Kα": BLUE, "Zn Kα": GOLD, "Cu Kβ": TEAL, "Pb Lα": PURPLE}
    total = np.zeros_like(xw)
    for c in comps:
        g = gaussian(xw, c["a"] * scale, c["m"], c["w"])
        total += g
        label_name = GAUSS_LABELS.get(round(c["m"], 3))
        if label_name is None:
            for ref_e, ref_name in GAUSS_LABELS.items():
                if abs(c["m"] - ref_e) < 0.05:
                    label_name = ref_name
                    break
        color = colors.get(label_name, "tab:blue")
        ax.plot(xw, g, color=color, linewidth=0.9)
        if label_name:
            i = int(np.argmax(g))
            ax.text(xw[i] + 0.05, g[i], label_name, color=color, fontsize=7.5, va="bottom")

    ax.plot(xw, total, color=RED, linewidth=1.0, linestyle="--")
    ax.text(0.97, 0.95, "measured", color=INK, transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    ax.text(0.97, 0.85, "sum of Gaussians", color=RED, transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel("Counts per channel")
    ax.set_ylim(bottom=0)
    save(fig, FIG_DIR, Path(out_file).stem)
    return {"file": best["file"], "scale_factor": scale}


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    qualitative = {}
    spectra_cache = {}
    for key, cfg in SAMPLE_CONFIG.items():
        spec = read_spectrum_csv(cfg["csv"])
        out_path = PLOTS_DIR / cfg["out"]
        peaks = plot_spectrum(key, spec, cfg["labels"], out_path)
        qualitative[key] = {
            "title": cfg["title"],
            "source_file": spec.source_file,
            "segment_count": spec.segment_count,
            "selected_segment": spec.selected_segment,
            "plot": str(out_path.relative_to(ROOT)),
            "identified_peaks": peaks,
        }
        spectra_cache[key] = spec

    quantitative = extract_quantitative_values()
    deconv_out = PLOTS_DIR / "brass_gaussian_deconvolution.png"
    deconv_meta = plot_brass_gaussian_deconvolution(
        spectra_cache["brass"], quantitative["H_trials"], deconv_out
    )
    quantitative["deconvolution_plot"] = str(deconv_out.relative_to(ROOT))
    quantitative["deconvolution_fit_source"] = deconv_meta

    payload = {
        "qualitative": qualitative,
        "quantitative": quantitative,
        "constants": {
            "densities_g_per_cm3": DENSITIES,
            "line_energies_keV": LINE_ENERGIES,
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote analysis summary: {OUT_JSON}")
    print(f"Plots directory: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
