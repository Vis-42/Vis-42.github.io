# /// script
# requires-python = ">=3.12"
# dependencies = ["pikepdf>=9", "pillow>=11"]
# ///
"""Copy the coursework files listed in data/coursework.toml from ~/-uni into the site.

    uv run website/sync_uni.py            # copy what changed
    uv run website/sync_uni.py --check    # only report what would change (exit 1 if anything)

~/-uni is the source of truth for reports and analysis code; the website keeps
published copies. Report PDFs are published with oversized photos scaled down
(text, vector figures and links untouched), because phone photos embedded at
full resolution made some reports 20-80 MB. data/coursework-sources.json records
the hash of each source copied, which is how the sync knows a copy is current.

Set UNI=/some/path to read from somewhere else.
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

SITE = Path(__file__).resolve().parent
UNI = Path(os.environ.get("UNI", Path.home() / "-uni"))
PDFS = SITE / "static" / "pdfs" / "experiments"
CODE = SITE / "static" / "code"
RECORD = SITE / "data" / "coursework-sources.json"
MAX_PX = 2000   # longest side kept for embedded photos


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_in_uni(path):
    r = subprocess.run(["git", "-C", str(UNI), "ls-files", "--error-unmatch", str(path)],
                       capture_output=True)
    return r.returncode == 0


def plan():
    """(source in ~/-uni, destination in static/) for every file the cards publish."""
    cards = tomllib.loads((SITE / "data" / "coursework.toml").read_text())["experiments"]
    pairs = []
    for c in cards:
        slug = c["slug"]
        pairs.append((c["report"], PDFS / f"{slug}.pdf"))
        if "slides" in c:
            pairs.append((c["slides"], PDFS / f"{slug}-slides.pdf"))
        for link in c.get("code", []):
            pairs.append((link["src"], CODE / slug / Path(link["src"]).name))
    return pairs


def publish_pdf(src, dest):
    """Copy src to dest, scaling down embedded photos larger than MAX_PX."""
    import pikepdf
    from PIL import Image

    pdf = pikepdf.open(src)
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
    if not shrunk or dest.stat().st_size >= src.stat().st_size:
        shutil.copy2(src, dest)   # no gain: publish the original untouched


def main():
    check = "--check" in sys.argv
    pairs = plan()
    missing = [src for src, _ in pairs if not (UNI / src).is_file()]
    if missing:
        sys.exit(f"missing in {UNI}:\n  " + "\n  ".join(missing))

    record = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    rel = lambda p: str(p.relative_to(SITE))
    changed = []
    for src, dest in pairs:
        h = sha256(UNI / src)
        if dest.exists() and record.get(rel(dest)) == h:
            continue
        changed.append(dest)
        if check:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix == ".pdf":
            publish_pdf(UNI / src, dest)
        else:
            shutil.copy2(UNI / src, dest)
        record[rel(dest)] = h
        if not tracked_in_uni(src):
            print(f"note: {src} isn't committed in ~/-uni yet")

    # code files that no card lists any more
    wanted = {dest for _, dest in pairs}
    stale = [p for p in CODE.rglob("*") if p.is_file() and p not in wanted] if CODE.exists() else []
    if not check:
        for p in stale:
            p.unlink()
            record.pop(rel(p), None)
        record = {k: v for k, v in sorted(record.items()) if (SITE / k).exists()}
        RECORD.write_text(json.dumps(record, indent=1) + "\n")

    for p in changed:
        mb = f"  ({p.stat().st_size / 1e6:.1f} MB)" if p.suffix == ".pdf" and p.exists() else ""
        print(("would update " if check else "updated ") + rel(p) + mb)
    for p in stale:
        print(("would remove " if check else "removed ") + rel(p))
    if not changed and not stale:
        print("coursework is in sync with ~/-uni")
    if check and (changed or stale):
        sys.exit(1)


if __name__ == "__main__":
    main()
