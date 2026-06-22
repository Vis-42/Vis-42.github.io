# Website — parthbhargava.net

Hugo static site, no theme, no framework. The deploy is fully automated: push to `parthb` and the GitHub Action on `Vis-42/Vis-42.github.io` runs Hugo and publishes to Pages. Never push the `public/` directory.

```bash
# preview locally
cd website && hugo server --port 1313

# verify it builds before pushing
cd website && hugo --minify

# deploy
git push origin main    # then: ./deploy-public.sh
```

---

## Where things live

All the CSS is in one `<style>` block in `layouts/_default/baseof.html`. That file also has the header, nav, sticker field, and footer. Do not add a separate stylesheet.

Content you would actually read lives in `content/`:
- `content/explorations/<slug>/index.md` — each exploration page (implementations of existing ideas)
- `content/projects/<slug>/index.md` — each projects page (original independent work)
- `content/blog/<slug>/index.md` — each blog post
- `content/photographs/<slug>.md` — each photograph entry
- `content/_index.md` — the homepage (just the title; body is in `layouts/_default/home.html`)

The homepage (`layouts/_default/home.html`) has its own hardcoded copy of each exploration description. If you update a description in its `content/explorations/<slug>/index.md`, update it in `home.html` too — they are intentionally duplicated so the homepage renders without Hugo data pipelines.

---

## Adding or editing an exploration

**Edit the write-up:** Open `content/explorations/<slug>/index.md`. The body after the `---` front matter is standard Markdown and renders on the exploration page. If you change the description, also update the matching paragraph in `layouts/_default/home.html`.

**Update the live notebook:** Edit `../py/<slug>/<slug>.py` in marimo, then rebuild:
```bash
marimo export html-wasm ../py/<slug>/<slug>.py -o static/apps/<slug> --mode edit
# keep only index.html, delete the rest of the export
find static/apps/<slug> -not -name "index.html" -not -path "static/apps/<slug>" -delete
perl -i '' -e 's{"\./}{"../_assets/}g' static/apps/<slug>/index.html
```
The `_assets/` directory is shared across all apps; do not rebuild it.

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
3. Export the notebook following the steps above
4. Add the exploration card and description to `layouts/_default/home.html` (copy the structure of an existing card)

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

## Publishing a lab report PDF

1. Compile with Typst: `cd ../manuscripts/typst && typst compile --root . <exp>/<report>.typ <exp>/<report>.pdf`
2. Copy the PDF to `static/pdfs/experiments/<slug>.pdf`
3. Commit and push. The coursework link in `home.html` must match the path exactly or it 404s.

---

## Stickers

SVG doodles in `static/stickers/svg/` are used as CSS mask-image glyphs on the homepage. They are themed: the same SVG appears in the page's accent colour in both light and dark mode. To add a new sticker, put an SVG there and add a `<figure class="sticker line">` block in `layouts/_default/home.html`, following the existing pattern. The `data-rot` attribute sets the tilt angle (degrees); `style` controls position and size.

---

## If something doesn't update

- A PDF or image returning 404: check that the file is committed and that the path in the HTML matches the filename exactly (case-sensitive on the server).
- The live notebook doesn't load: check that `static/apps/<slug>/index.html` exists and that all `../_assets/` paths resolve. If the `_assets/` directory is missing from `website/static/apps/`, run `git checkout HEAD -- website/static/apps/_assets/` to restore it.
- The site doesn't update after push: open `https://github.com/Vis-42/Vis-42.github.io/actions` and check that the Action passed. Make sure the Pages source is set to **GitHub Actions** in the Pages settings.
