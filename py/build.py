"""Export the site's marimo notebooks to WebAssembly apps for the website.

Run it from py/ with uv, so it uses the project's pinned marimo:

    uv run build.py                     # every notebook on the site
    uv run build.py reaction_diffusion qm_wavepacket
    uv run build.py --out /tmp/site     # build somewhere else (for testing)

Which notebooks are "on the site" comes from website/content/explorations/*/index.md
(the ones with an `app:` line), so there is no list to keep in sync.

Apps are exported in marimo's edit mode (visitors can change the code and re-run it)
with auto-run on load switched on.

Output, under website/static/apps/:
    <slug>/index.html   one per notebook
    assets/             marimo's frontend, shared by every app (~27 MB)
    _shared/            py/_shared/, which the notebooks load at runtime

All apps must be exported by the same marimo version, because they share assets/.
py/pyproject.toml pins it; bump it there and rebuild everything together.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "py"
SHARED = PY / "_shared"
SITE_APPS = ROOT / "website" / "static" / "apps"
EXPLORATIONS = ROOT / "website" / "content" / "explorations"


def site_slugs():
    slugs = []
    for page in sorted(EXPLORATIONS.glob("*/index.md")):
        if re.search(r"^app:\s*\S", page.read_text(), re.M):
            slugs.append(page.parent.name)
    return slugs


def check_shared_js():
    # mo.iframe flattens its HTML onto one line, so a // comment would swallow the
    # rest of the script. Only /* */ comments are safe in the shared JS.
    bad = []
    for js in [*SHARED.glob("*.js"), *SHARED.glob("*.html")]:
        for n, line in enumerate(js.read_text().splitlines(), 1):
            code = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`[^`]*`', '""', line)
            if re.search(r"(?<![:\\])//", code):
                bad.append(f"{js.relative_to(ROOT)}:{n}: {line.strip()[:80]}")
    if bad:
        sys.exit("// comments break mo.iframe; use /* */ instead:\n  " + "\n  ".join(bad))


def export(slug, out, tmp):
    nb = PY / slug / f"{slug}.py"
    dest = tmp / slug
    subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html-wasm",
         str(nb), "-o", str(dest), "--mode", "edit", "--force"],
        check=True, stdout=subprocess.DEVNULL,
    )
    html = (dest / "index.html").read_text()
    # Edit mode keeps the code editable in the browser, but by default it opens with
    # every cell un-run (no outputs, no animations). Make it run everything on load.
    off = '"auto_instantiate": false'
    if html.count(off) != 1:
        sys.exit(f"{slug}: expected one {off!r} in the export; did marimo change its format?")
    html = html.replace(off, '"auto_instantiate": true')
    # Point the export's ./assets, ./favicon.ico, ... at the shared ../assets/ directory.
    html = re.sub(r'((?:href|src)=")\./', r"\1../assets/", html)
    (out / slug).mkdir(parents=True, exist_ok=True)
    (out / slug / "index.html").write_text(html)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slugs", nargs="*", help="notebooks to build (default: all on the site)")
    ap.add_argument("--out", type=Path, default=SITE_APPS, help="apps directory to write into")
    args = ap.parse_args()

    try:
        import marimo
    except ImportError:
        sys.exit("marimo isn't installed here; run this from py/ as `uv run build.py`")

    known = site_slugs()
    slugs = args.slugs or known
    unknown = [s for s in slugs if s not in known]
    if unknown:
        sys.exit(f"not on the site (no app: in content/explorations): {', '.join(unknown)}")

    check_shared_js()
    out = args.out
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        last = None
        for slug in slugs:
            print(f"exporting {slug}")
            last = export(slug, out, tmp)

        # Refresh the shared frontend from this marimo version's export.
        assets = out / "assets"
        if assets.exists():
            shutil.rmtree(assets)
        assets.mkdir(parents=True)
        for item in last.iterdir():
            if item.name in ("index.html", "CLAUDE.md"):
                continue
            target = assets / item.name
            shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)

    if out.joinpath("_shared").exists():
        shutil.rmtree(out / "_shared")
    if SHARED.exists():
        shutil.copytree(SHARED, out / "_shared")

    if args.slugs and set(slugs) != set(known):
        print(f"note: assets/ is now from marimo {marimo.__version__}; rebuild the other notebooks too "
              "if they were exported with a different version")
    print(f"done: {len(slugs)} app(s) in {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
