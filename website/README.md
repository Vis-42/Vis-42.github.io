# Website — parthbhargava.net

Hugo static site, no theme, no framework. The deploy is fully automated: `../deploy-public.sh` pushes only the public files (website, notebooks, workflow) to `Vis-42/Vis-42.github.io`, and the GitHub Action there runs Hugo and publishes to Pages. Never push the `public/` directory.

```bash
# preview locally
cd website && hugo server --port 1313

# verify it builds before pushing
cd website && hugo --minify

# deploy
git push origin main && ./deploy-public.sh    # run from the repo root
```

---

## Where things live

All the CSS is in one `<style>` block in `layouts/_default/baseof.html`. That file also has the header, nav, sticker field, and footer. Do not add a separate stylesheet.

Content you would actually read lives in `content/`:
- `content/explorations/<slug>/index.md` — each exploration page (implementations of existing ideas)
- `content/projects/<slug>/index.md` — each projects page (original independent work)
- `content/blog/<slug>/index.md` — each blog post
- `content/photographs/<slug>.md` — each photograph entry
- `content/_index.md` — the homepage "about" text; the hero card and sticker field around it are in `layouts/_default/home.html`

The `/explorations/` listing is generated from the front matter of each exploration page, and `/coursework/` from `data/coursework.toml`, so there is nothing to keep in sync by hand.

---

## Adding or editing an exploration

**Edit the write-up:** Open `content/explorations/<slug>/index.md`. The body after the `---` front matter is standard Markdown and renders on the exploration page; its first 30 words are the summary on the `/explorations/` card.

**Update the live notebook:** Edit `../py/<slug>/<slug>.py` in marimo, then rebuild. Both run from `py/` through uv, which keeps the pinned marimo version:
```bash
cd ../py
uv run marimo edit <slug>/<slug>.py
uv run build.py <slug>        # or with no arguments: every notebook on the site
```
This exports the notebook to `static/apps/<slug>/index.html`, refreshes the shared marimo frontend in `static/apps/assets/`, and copies `py/_shared/` to `static/apps/_shared/`. All apps share `assets/`, so they must come from the same marimo version: the version is pinned in `py/pyproject.toml`, and if you bump it, rebuild everything.

The animations in every notebook are drawn by one shared engine, `py/_shared/canvas.html`. A notebook builds a spec (frames, panels, layout; the format is documented at the top of that file) and calls `canvas_anim(spec, height)`. To add a new kind of plot, add a branch to the engine. Use only `/* */` comments in it: `mo.iframe` puts the page on one line, so a `//` comment would break the script, and `build.py` refuses to build if it finds one.

**Change the thumbnail:** Replace `static/media/thumbs/<slug>.jpg`. JPEG only; keep it under 200 KB. The front matter `thumb:` field is the path from `static/`.

**Add a new exploration** (implementation of an existing idea):
1. Create `content/explorations/<new-slug>/index.md`:
```yaml
---
title: "Exploration Title"
slug: "<new-slug>"
weight: 14          # controls ordering; higher number = further down
thumb: "/media/thumbs/<new-slug>.jpg"
filed: "Topic · subtopic"
app: "/apps/<new-slug>/"
links:
  - label: "Notebook source (Python · marimo)"
    url: "https://github.com/Vis-42/Vis-42.github.io/blob/main/py/<new-slug>/<new-slug>.py"
  - label: "Original Julia notebook (upstream)"
    url: "https://github.com/Vis-42/Vis-42.github.io/tree/main/jl/<new-slug>"
---
Description paragraph here.
```
2. Put the thumbnail at `static/media/thumbs/<new-slug>.jpg`
3. Run `uv run build.py <new-slug>` from `py/`
4. Add `<new-slug>` to the `SLUGS` list in `../deploy-public.sh`, otherwise the notebook source is not published and the source link 404s
5. Optionally add a sticker to `layouts/_default/home.html` (see Stickers below)

**Add a new project** (original independent work):
1. Create `content/projects/<new-slug>/index.md` with the same front matter structure
2. No card in `home.html` — projects have their own `/projects/` listing page

---

## Adding a blog post

Create `content/blog/<slug>/index.md`:
```yaml
---
title: "Title of the post"
date: 2026-06-19
excerpt: "One or two sentences shown on the /blog/ listing page."
tags: ["optional", "tags"]
---
Post body in Markdown.
```
The post appears on `/blog/` sorted newest-first. No other changes needed.

---

## Adding a photograph

1. Copy the image to `static/media/photos/<filename>.jpg`
2. Create `content/photographs/<slug>.md`:
```yaml
---
title: "Short label"
date: 2026-06-19
src: "/media/photos/<filename>.jpg"
caption: "what it shows · place, year"
---
```
The photo appears on `/photographs/` in the grid. No other changes needed.

---

## Coursework (lab reports and analysis code)

Reports and analysis scripts live in `~/-uni` (the private repo `Vis-42/-uni`), which is the source of truth. The site keeps published copies, and `data/coursework.toml` lists every card on `/coursework/` with the `~/-uni` paths of its report, optional slides, and code.

To update a report or add one:
1. Compile it in `~/-uni` as usual (e.g. `typst compile --root . report/report.typ report/report.pdf`).
2. New report: add a card to `data/coursework.toml` (copy an existing one) and put its thumbnail in `static/media/thumbs/`.
3. From the repo root, run `uv run website/sync_uni.py`. It copies the report to `static/pdfs/experiments/<slug>.pdf`, slides to `<slug>-slides.pdf`, and scripts to `static/code/<slug>/`. Embedded photos larger than 2000 px are scaled down on the way (text, vector figures and links are untouched), because full-resolution phone photos made some reports 20–80 MB.
4. Commit and deploy.

`uv run website/sync_uni.py --check` lists anything out of date without changing files. The "Analysis Code" links point at the copies in `static/code/` on the public mirror on GitHub, so scripts must be synced before they show up there.

---

## Stickers

SVG doodles in `static/stickers/svg/` are used as CSS mask-image glyphs on the homepage. They are themed: the same SVG appears in the page's accent colour in both light and dark mode. To add a new sticker, put an SVG there and add a `<figure class="sticker line">` block in `layouts/_default/home.html`, following the existing pattern. The `data-rot` attribute sets the tilt angle (degrees); `style` controls position and size.

---

## If something doesn't update

- A PDF or image returning 404: check that the file is committed and that the path in the HTML matches the filename exactly (case-sensitive on the server).
- The live notebook doesn't load: check that `static/apps/<slug>/index.html` exists and that `static/apps/assets/` and `static/apps/_shared/canvas.html` exist. Rebuilding with `uv run build.py` (from `py/`) restores all three.
- The site doesn't update after push: open `https://github.com/Vis-42/Vis-42.github.io/actions` and check that the Action passed. Make sure the Pages source is set to **GitHub Actions** in the Pages settings.
