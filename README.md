# parthbhargava.net

Source for my site at [parthbhargava.net](https://parthbhargava.net): interactive
physics notebooks (marimo, run in the browser via WebAssembly), the Hugo site that
publishes them, and the Julia/Pluto notebooks they are ported from.

- `website/` — the Hugo site
- `py/<project>/<project>.py` — the marimo notebook for each project
- `py/_shared/canvas.html` — the animation engine the notebooks share
- `py/build.py` — exports the notebooks into `website/static/apps/`
- `py/pyproject.toml`, `py/uv.lock` — the Python environment (`cd py && uv sync`)
- `jl/<project>/app.pluto.jl` — the original Julia/Pluto notebook

Built and deployed to GitHub Pages by the Action in `.github/workflows/deploy.yml`.
