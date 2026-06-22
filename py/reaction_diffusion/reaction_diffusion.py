import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Reaction-Diffusion")


@app.cell(hide_code=True)
def _():
    import json

    import marimo as mo
    import numpy as np

    return json, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Reaction-diffusion: how chemistry makes pattern

        Turing's idea, made concrete by the Gray-Scott model: two chemicals that react and diffuse
        at different rates can spontaneously break a uniform state into spots, stripes, and mazes. Two
        fields $u,v$ obey

        $$\dot u=D_u\nabla^2u-uv^2+F(1-u),\qquad \dot v=D_v\nabla^2v+uv^2-(F+k)v,$$

        with $v$ the slow, short-range activator and $u$ the fast-diffusing substrate. The animation is
        the $v$ field computed live on the grid and drawn frame by frame as it self-organises.
        """
    )
    return


@app.cell(hide_code=True)
def _(json, mo):
    def canvas_anim(spec, height="460px"):
        html = r"""<!doctype html><html><head><meta charset="utf-8"><style>
        html,body{margin:0;background:#0f1117;overflow:hidden}canvas{display:block;width:100vw;height:100vh}
        </style></head><body><canvas id="c"></canvas><script>
        const S = __SPEC__, cv = document.getElementById("c"), ctx = cv.getContext("2d");
        const DPR = Math.min(window.devicePixelRatio||1, 2);
        function fit(){const w=cv.offsetWidth||window.innerWidth||800;const h=cv.offsetHeight||window.innerHeight||450;if(w>0&&h>0){cv.width=w*DPR;cv.height=h*DPR;}}
        addEventListener("resize", fit); fit();
        let f = 0, last = 0, off = null, octx = null, img = null;
        function setup(n){ off = document.createElement("canvas"); off.width = n; off.height = n;
          octx = off.getContext("2d"); img = octx.createImageData(n, n); }
        function draw(){
          const W = cv.width, H = cv.height; ctx.fillStyle = "#0f1117"; ctx.fillRect(0,0,W,H);
          if (S.kind === "heatmap"){
            const n = S.n; if (!off) setup(n);
            const g = S.grid[f], d = img.data;
            for (let i=0;i<n*n;i++){
              const val = g[i]/255;
              d[4*i]   = 255*Math.min(1, val*1.6);
              d[4*i+1] = 255*Math.min(1, 0.25+val*0.9);
              d[4*i+2] = 255*Math.min(1, 0.45+val*0.7);
              d[4*i+3] = 255;
            }
            octx.putImageData(img, 0, 0);
            const s = Math.min(W,H), ox = (W-s)/2, oy = (H-s)/2;
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(off, ox, oy, s, s);
          }
        }
        function loop(t){ if (t-last > S.dt){ f = (f+1) % S.frames; last = t; } try{draw();}catch(e){}requestAnimationFrame(loop);}
        function _kick(){fit();if(cv.width>0&&cv.height>0){requestAnimationFrame(loop);}else{setTimeout(_kick,80);}}_kick();
        </script></body></html>""".replace("__SPEC__", json.dumps(spec))
        return mo.iframe(html, height=height)

    return (canvas_anim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The model and its patterns

        The pattern is selected by just two numbers, the feed rate $F$ and the kill rate $k$. Small
        changes move the system across a chemical phase diagram from isolated spots, to a replicating
        "mitosis" regime, to labyrinthine stripes. Diffusion is computed with the five-point
        Laplacian on a periodic grid; the activator diffuses more slowly than the substrate, which is
        the asymmetry Turing identified as the engine of pattern.
        """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def laplacian(a):
        return (np.roll(a, 1, 0) + np.roll(a, -1, 0) +
                np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a)

    def gray_scott(n, F, k, steps, store_every, Du=0.16, Dv=0.08, seed=0):
        rng = np.random.default_rng(seed)
        u = np.ones((n, n)); v = np.zeros((n, n))
        r = max(3, n // 12); c = n // 2
        u[c - r:c + r, c - r:c + r] = 0.50
        v[c - r:c + r, c - r:c + r] = 0.25
        v += 0.02 * rng.random((n, n))
        frames = []
        for s in range(steps):
            uvv = u * v * v
            u += Du * laplacian(u) - uvv + F * (1 - u)
            v += Dv * laplacian(v) + uvv - (F + k) * v
            if s % store_every == 0:
                frames.append(v.copy())
        return np.array(frames)
    return (gray_scott,)


@app.cell(hide_code=True)
def _(mo):
    preset = mo.ui.dropdown(
        {"spots (F=.035 k=.065)": (0.035, 0.065),
         "mitosis (F=.026 k=.061)": (0.026, 0.061),
         "stripes (F=.030 k=.057)": (0.030, 0.057),
         "maze (F=.029 k=.057)": (0.029, 0.057),
         "holes (F=.039 k=.058)": (0.039, 0.058),
         "custom": (0.035, 0.065)},
        value="spots (F=.035 k=.065)", label="preset",
    )
    F_sl  = mo.ui.slider(0.010, 0.060, value=0.035, step=0.002, label="feed rate F", show_value=True)
    k_sl  = mo.ui.slider(0.045, 0.075, value=0.065, step=0.002, label="kill rate k", show_value=True)
    N_sl  = mo.ui.slider(40, 80, value=56, step=8, label="grid N", show_value=True)
    mo.md(
        f"""
        **Controls.** Each preset selects a point in the Pearson (F, k) plane; override with the
        sliders to explore freely. The pattern grows on load and replays as a loop; larger grids
        take a little longer.
        {mo.hstack([preset, F_sl, k_sl, N_sl], justify="start", gap=2)}
        """
    )
    return F_sl, N_sl, k_sl, preset


@app.cell(hide_code=True)
def _(F_sl, N_sl, gray_scott, k_sl, np, preset):
    # Auto-grows on load with the chosen preset/sliders (no button gate).
    _use_custom = preset.selected_key == "custom" if hasattr(preset, "selected_key") else False
    _F = F_sl.value if _use_custom else preset.value[0]
    _k = k_sl.value if _use_custom else preset.value[1]
    _N = int(N_sl.value)
    _frames = gray_scott(n=_N, F=_F, k=_k, steps=3500, store_every=50, seed=0)
    _vmax = max(1e-6, float(_frames.max()))
    _v_mean = float(_frames[-1].mean())
    _v_std  = float(_frames[-1].std())
    _patterned = _v_std > 0.04
    grid_u8 = np.clip(_frames / _vmax * 255, 0, 255).astype("uint8").reshape(_frames.shape[0], -1)
    gs_stats = {"F": _F, "k": _k, "N": _N, "frames": int(_frames.shape[0]),
                "v_mean": _v_mean, "v_std": _v_std, "patterned": _patterned}
    return grid_u8, gs_stats


@app.cell(hide_code=True)
def _(canvas_anim, grid_u8, gs_stats, mo):
    _s = gs_stats
    _spec = {"kind": "heatmap", "n": _s["N"], "frames": _s["frames"], "dt": 60,
             "grid": grid_u8.tolist()}
    _state = "PATTERNED" if _s["patterned"] else "UNIFORM"
    mo.vstack([
        mo.md(
            f"**{_state}** &nbsp;|&nbsp; "
            f"F = {_s['F']:.3f}, k = {_s['k']:.3f} &nbsp;|&nbsp; "
            f"grid {_s['N']}×{_s['N']} &nbsp;|&nbsp; "
            f"⟨v⟩ = {_s['v_mean']:.3f} &nbsp;|&nbsp; "
            f"σ(v) = {_s['v_std']:.3f} &nbsp;|&nbsp; "
            f"frames = {_s['frames']}"
        ),
        canvas_anim(_spec, height="480px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **What this shows.** No template, no mould, no external map tells the chemistry where to put a
        spot. The pattern is an instability of the uniform state: diffusion, usually a smoother of
        differences, instead amplifies them because the activator and substrate diffuse at different
        rates. The same mechanism is a leading model for animal coat markings and many developmental
        patterns; change two rate constants and the morphology changes with them.
        """
    )
    return


if __name__ == "__main__":
    app.run()
