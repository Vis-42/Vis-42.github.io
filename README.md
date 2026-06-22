# parthbhargava.net

Source for my site at [parthbhargava.net](https://parthbhargava.net): interactive
physics notebooks (marimo, run in the browser via WebAssembly), the Hugo site that
publishes them, and the Julia/Pluto notebooks they are ported from.

- `website/` — the Hugo site
- `py/<project>/<project>.py` — the marimo notebook for each project
- `jl/<project>/app.pluto.jl` — the original Julia/Pluto notebook

Built and deployed to GitHub Pages by the Action in `.github/workflows/deploy.yml`.
