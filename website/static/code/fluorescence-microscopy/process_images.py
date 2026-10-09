import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # plotstyle.py lives next to this script
from plotstyle import COL, apply_style, save  # noqa: E402

apply_style()
PIXEL_UM = 0.599  # from the OME metadata (20x objective, DP74 camera)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
OUT_DIR = Path(__file__).resolve().parents[1]

OME_NS = {"OME": "http://www.openmicroscopy.org/Schemas/OME/2015-01"}

IMAGE_MAP = {
    "Image_609": {"cell": "cheek", "mode": "fluorescence", "note": "overexposed"},
    "Image_610": {"cell": "cheek", "mode": "brightfield"},
    "Image_611": {"cell": "cheek", "mode": "fluorescence"},
    "Image_612": {"cell": "cheek", "mode": "brightfield"},
    "Image_613": {"cell": "onion", "mode": "brightfield"},
    "Image_614": {"cell": "onion", "mode": "fluorescence"},
    "Image_615": {"cell": "onion", "mode": "brightfield"},
    "Image_616": {"cell": "onion", "mode": "fluorescence"},
}

PANEL_IMAGES = {
    "Onion – Brightfield": "Image_613",
    "Onion – Fluorescence": "Image_614",
    "Cheek – Brightfield": "Image_610",
    "Cheek – Fluorescence": "Image_611",
}


def parse_ome(meta_str: str) -> dict:
    root = ET.fromstring(meta_str)
    info = {}

    acq = root.find(".//OME:AcquisitionDate", OME_NS)
    if acq is not None:
        info["acquisition_date"] = acq.text

    det = root.find(".//OME:Detector", OME_NS)
    if det is not None:
        info["detector"] = det.get("Model", "")
        info["gain"] = int(det.get("Gain", "0"))

    obj = root.find(".//OME:Objective", OME_NS)
    if obj is not None:
        info["magnification"] = int(float(obj.get("NominalMagnification", "0")))
        info["NA"] = round(float(obj.get("LensNA", "0")), 2)

    pixels = root.find(".//OME:Pixels", OME_NS)
    if pixels is not None:
        info["pixel_size_um"] = round(float(pixels.get("PhysicalSizeX", "0")), 4)

    plane = root.find(".//OME:Plane", OME_NS)
    if plane is not None:
        info["exposure_ms"] = round(float(plane.get("ExposureTime", "0")), 2)

    return info


def convert_and_extract():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_meta = []

    for stem in sorted(IMAGE_MAP):
        src = DATA_DIR / f"{stem}.tif"
        dst = PLOTS_DIR / f"{stem}.png"
        img = Image.open(src)
        img.save(dst, "PNG")

        ome_xml = img.tag_v2.get(270, "")
        meta = parse_ome(ome_xml) if ome_xml else {}
        meta["file"] = src.name
        meta.update(IMAGE_MAP[stem])
        all_meta.append(meta)

        print(f"  {src.name} -> {dst.name}")

    out = OUT_DIR / "image_metadata.json"
    out.write_text(json.dumps(all_meta, indent=2))
    print(f"Wrote {out}")
    return all_meta


def make_panel():
    """2x2 panel at column width: rows are cell types, columns imaging modes,
    each with a 100 um scale bar drawn from the metadata pixel size."""
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.75))
    for ax, (title, stem), letter in zip(axes.flat, PANEL_IMAGES.items(), "abcd"):
        img = np.array(Image.open(PLOTS_DIR / f"{stem}.png"))
        h, w = img.shape[:2]
        ax.imshow(img)
        ax.set_title(f"({letter}) " + title.replace(" – ", ", ").lower().capitalize(), fontsize=8, loc="left", pad=2)
        bar = 100 / PIXEL_UM
        ax.plot([w * 0.95 - bar, w * 0.95], [h * 0.92, h * 0.92], color="white", lw=2, solid_capstyle="butt")
        ax.text(w * 0.95 - bar / 2, h * 0.88, "100 µm", color="white", ha="center", va="bottom", fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=0.2, h_pad=0.6, w_pad=0.4)
    out = save(fig, Path(__file__).resolve().parents[1] / "report" / "figures", "comparison_panel")
    print(f"Wrote {out}")


def main():
    print("Converting TIFFs to PNG and extracting metadata...")
    convert_and_extract()
    print("Generating comparison panel...")
    make_panel()
    print("Done.")


if __name__ == "__main__":
    main()
