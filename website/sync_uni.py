# /// script
# requires-python = ">=3.12"
# dependencies = ["pikepdf>=9", "pillow>=11"]
# ///
"""Copy the coursework PDFs listed in data/coursework.toml from ~/-uni into the site.

    uv run website/sync_uni.py            # copy what changed
    uv run website/sync_uni.py --check    # only report what would change (exit 1 if anything)

~/-uni is the source of truth for reports; the website keeps published copies,
taken from what is COMMITTED in ~/-uni (so a half-edited report on disk is never
published: commit it in ~/-uni first). PDFs are published with oversized photos
scaled down (text, vector figures and links untouched), because phone photos
embedded at full resolution made some reports 20-80 MB.

data/coursework-sources.json records, for each published PDF, the hash of the
source it was copied from (how the sync knows a copy is current) and the title and
page count the /coursework/ page shows. PDFs in static/pdfs/experiments/ that no
entry lists any more are removed.

Set UNI=/some/path to read from somewhere else.
"""

import hashlib
import io
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

SITE = Path(__file__).resolve().parent
UNI = Path(os.environ.get("UNI", Path.home() / "-uni"))
PDFS = SITE / "static" / "pdfs" / "experiments"
RECORD = SITE / "data" / "coursework-sources.json"
MAX_PX = 2000   # longest side kept for embedded photos


def committed(src):
    """The file's contents as of the last commit in ~/-uni, or None if not committed."""
    r = subprocess.run(["git", "-C", str(UNI), "show", f"HEAD:{src}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def dirty_in_uni(src):
    r = subprocess.run(["git", "-C", str(UNI), "status", "--porcelain", "--", src],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def plan():
    """(source in ~/-uni, destination in static/) for every PDF the page links to."""
    data = tomllib.loads((SITE / "data" / "coursework.toml").read_text())
    folders = [c["folder"] for c in data["courses"]]
    pairs, slugs = [], set()
    for r in data["reports"]:
        slug = r["slug"]
        if slug in slugs:
            sys.exit(f"coursework.toml: slug {slug!r} is used twice")
        slugs.add(slug)
        if not any(r["report"].startswith(f + "/") for f in folders):
            sys.exit(f"coursework.toml: {r['report']} is in no listed course folder; "
                     "add a [[courses]] entry for it")
        pairs.append((r["report"], PDFS / f"{slug}.pdf"))
        if "slides" in r:
            pairs.append((r["slides"], PDFS / f"{slug}-slides.pdf"))
    return pairs


def describe(data):
    """Title (from the PDF's metadata, may be empty) and page count."""
    import pikepdf

    pdf = pikepdf.open(io.BytesIO(data))
    return {"title": str(pdf.docinfo.get("/Title", "")).strip(), "pages": len(pdf.pages)}


def publish_pdf(data, dest):
    """Write the PDF bytes to dest, scaling down embedded photos larger than MAX_PX."""
    import pikepdf
    from PIL import Image

    pdf = pikepdf.open(io.BytesIO(data))
    shrunk = False
    for obj in pdf.objects:
        if not (isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image"):
            continue
        if max(int(obj.Width), int(obj.Height)) <= MAX_PX or "/SMask" in obj or "/Mask" in obj:
            continue
        if int(obj.get("/BitsPerComponent", 8)) != 8:
            continue
        cs = obj.get("/ColorSpace")
        n = (int(cs[1].N) if isinstance(cs, pikepdf.Array) and cs[0] == "/ICCBased"
             else {"/DeviceRGB": 3, "/DeviceGray": 1}.get(str(cs)))
        if n not in (1, 3):
            continue
        img = pikepdf.PdfImage(obj).as_pil_image()
        scale = MAX_PX / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB" if n == 3 else "L").save(buf, "JPEG", quality=88)
        obj.write(buf.getvalue(), filter=pikepdf.Name.DCTDecode)   # keeps the colour profile
        obj.Width, obj.Height = img.size
        if "/DecodeParms" in obj:
            del obj.DecodeParms
        shrunk = True
    if shrunk:
        pdf.save(dest, compress_streams=True,
                 object_stream_mode=pikepdf.ObjectStreamMode.generate)
    if not shrunk or dest.stat().st_size >= len(data):
        dest.write_bytes(data)    # no gain: publish the original untouched


def main():
    check = "--check" in sys.argv
    pairs = plan()
    sources = {src: committed(src) for src, _ in pairs}
    missing = [src for src, data in sources.items() if data is None]
    if missing:
        sys.exit(f"not committed in {UNI} (git add + commit them there first):\n  "
                 + "\n  ".join(missing))
    for src in sources:
        if dirty_in_uni(src):
            print(f"note: {src} has uncommitted changes in ~/-uni; using the committed version")

    record = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    rel = lambda p: str(p.relative_to(SITE))
    changed = []
    for src, dest in pairs:
        data = sources[src]
        h = hashlib.sha256(data).hexdigest()
        if dest.exists() and record.get(rel(dest), {}).get("sha256") == h:
            continue
        changed.append(dest)
        if check:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        publish_pdf(data, dest)
        record[rel(dest)] = {"sha256": h, **describe(data)}

    # PDFs that no entry lists any more
    wanted = {dest for _, dest in pairs}
    stale = sorted(p for p in PDFS.glob("*.pdf") if p not in wanted)
    if not check:
        for p in stale:
            p.unlink()
        record = {k: record[k] for k in sorted(record) if SITE / k in wanted}
        RECORD.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")

    for p in changed:
        mb = f"  ({p.stat().st_size / 1e6:.1f} MB)" if p.exists() else ""
        print(("would update " if check else "updated ") + rel(p) + mb)
    for p in stale:
        print(("would remove " if check else "removed ") + rel(p))
    if not changed and not stale:
        print("coursework is in sync with ~/-uni")
    if check and (changed or stale):
        sys.exit(1)


if __name__ == "__main__":
    main()
