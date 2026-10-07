import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Active Matter")


@app.cell(hide_code=True)
def _():
    import json
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return json, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Active matter: collective motion and single-cell swimming

        Two models from scratch. The Vicsek model is about a crowd: N self-propelled particles
        aligning with their neighbours, swept over noise $\eta$ to trace the order parameter
        $\phi = |\langle e^{i\theta}\rangle|$ through its disorder-to-order transition.
        Run-and-tumble is about a single cell: straight runs interrupted by Poisson reorientations,
        setting a ballistic-to-diffusive crossover in the mean-squared displacement.

        Three panels share one animation clock: the flock coloured by heading, $\phi(t)$ with
        an orange dot tracking the current frame, and the $\phi$ vs $\eta$ phase diagram with
        the current noise level marked.
        """
    )
    return


@app.cell(hide_code=True)
def _(json, mo):
    import sys as _sys

    # The animation engine shared by every notebook is py/_shared/canvas.html;
    # the website serves the same file at /apps/_shared/canvas.html.
    if _sys.platform == "emscripten":
        from js import location
        from urllib.request import urlopen

        _engine = urlopen(f"{location.origin}/apps/_shared/canvas.html").read().decode()
    else:
        _engine = (mo.notebook_dir().parent / "_shared" / "canvas.html").read_text()

    def canvas_anim(spec, height="480px"):
        """Animate a spec (format: see the top of py/_shared/canvas.html)."""
        return mo.iframe(_engine.replace("__SPEC__", json.dumps(spec)), height=height)

    return (canvas_anim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 1: Vicsek model")
    return


@app.cell(hide_code=True)
def _(mo):
    eta_sl = mo.ui.slider(0.0, 6.0,  value=1.5,  step=0.1,  label="noise η",         show_value=True)
    rho_sl = mo.ui.slider(0.5, 4.0,  value=2.0,  step=0.25, label="density ρ",       show_value=True)
    N_sl   = mo.ui.slider(40,  240,   value=120,  step=20,   label="particles N",     show_value=True)
    v0_sl  = mo.ui.slider(0.05, 0.4,  value=0.15, step=0.05, label="speed v₀",       show_value=True)
    obsr_sl = mo.ui.slider(0.02, 0.25, value=0.08, step=0.01, label="obstacle r / L", show_value=True)
    mo.hstack([eta_sl, rho_sl, N_sl, v0_sl, obsr_sl], justify="start", gap=2)
    return N_sl, eta_sl, obsr_sl, rho_sl, v0_sl


@app.cell(hide_code=True)
def _(N_sl, eta_sl, np, rho_sl, v0_sl):
    _N   = int(N_sl.value)
    _eta = float(eta_sl.value)
    _rho = float(rho_sl.value)
    _v0  = float(v0_sl.value)
    _L   = float(np.sqrt(_N / _rho))
    _WARM   = 60
    _FRAMES = 140
    _rng = np.random.default_rng(42)

    def _step(x, y, th, N, L, v0, eta, rng):
        dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
        nb = (dx**2 + dy**2) < 1.0
        s = (nb * np.sin(th)).sum(1) / nb.sum(1)
        c = (nb * np.cos(th)).sum(1) / nb.sum(1)
        th2 = np.arctan2(s, c) + eta * rng.uniform(-0.5, 0.5, N)
        return (x + v0 * np.cos(th2)) % L, (y + v0 * np.sin(th2)) % L, th2

    _x = _rng.uniform(0, _L, _N); _y = _rng.uniform(0, _L, _N)
    _th = _rng.uniform(-np.pi, np.pi, _N)
    for _ in range(_WARM):
        _x, _y, _th = _step(_x, _y, _th, _N, _L, _v0, _eta, _rng)

    _xs_f, _ys_f, _ths_f, _phi_s = [], [], [], []
    for _ in range(_FRAMES):
        _x, _y, _th = _step(_x, _y, _th, _N, _L, _v0, _eta, _rng)
        _xs_f.append(np.round(_x, 2).tolist())
        _ys_f.append(np.round(_y, 2).tolist())
        _ths_f.append(np.round(_th, 2).tolist())
        _phi_s.append(float(np.abs(np.mean(np.exp(1j * _th)))))

    _phi_mean = float(np.mean(_phi_s[_FRAMES // 2:]))
    _regime = "ORDERED" if _phi_mean > 0.65 else "TRANSITION" if _phi_mean > 0.3 else "DISORDERED"

    vicsek_core = {
        "xs": _xs_f, "ys": _ys_f, "ths": _ths_f, "phi": _phi_s,
        "L": float(_L), "frames": _FRAMES,
        "eta": float(_eta),
        "eta_val": [float(_eta)] * _FRAMES,
        "phi_mean": _phi_mean, "regime": _regime,
        "step_fn": _step, "rng": _rng,
        "N": _N, "v0": float(_v0),
    }
    return vicsek_core,


@app.cell(hide_code=True)
def _(np, vicsek_core):
    # Phase-diagram sweep: small N and short runs so it always computes on load.
    _d = vicsek_core
    _N_sw = 45
    _L_sw = float(np.sqrt(_N_sw / 2.0))
    _rng_sw = np.random.default_rng(7)
    _eta_arr = np.linspace(0.2, 5.8, 11)
    _phi_arr = []
    for _e in _eta_arr:
        _xs2 = _rng_sw.uniform(0, _L_sw, _N_sw)
        _ys2 = _rng_sw.uniform(0, _L_sw, _N_sw)
        _ths2 = _rng_sw.uniform(-np.pi, np.pi, _N_sw)
        for _ in range(45):
            _xs2, _ys2, _ths2 = _d["step_fn"](_xs2, _ys2, _ths2, _N_sw, _L_sw, _d["v0"], _e, _rng_sw)
        _ps = [float(np.abs(np.mean(np.exp(1j * _ths2))))]
        for _ in range(30):
            _xs2, _ys2, _ths2 = _d["step_fn"](_xs2, _ys2, _ths2, _N_sw, _L_sw, _d["v0"], _e, _rng_sw)
            _ps.append(float(np.abs(np.mean(np.exp(1j * _ths2)))))
        _phi_arr.append(float(np.mean(_ps)))
    vicsek_data = {
        **_d,
        "eta_sweep": _eta_arr.tolist(),
        "phi_sweep": _phi_arr,
        "has_sweep": True,
    }
    return vicsek_data,


@app.cell(hide_code=True)
def _(canvas_anim, mo, obsr_sl, vicsek_data):
    _d = vicsek_data
    _panels = [
        {"kind": "live_vicsek",
         "N": _d["N"], "L": _d["L"], "v0": _d["v0"], "eta": _d["eta"],
         "R": 1.0, "obs_r": max(0.3, float(_d["L"]) * float(obsr_sl.value))},
        {"kind": "phi_t", "phi": _d["phi"]},
    ]
    _layout = [[0.0, 0.0, 0.50, 1.00], [0.50, 0.0, 0.50, 1.00]]
    if _d["has_sweep"] and len(_d["eta_sweep"]) > 0:
        _panels[1] = {"kind": "phi_t", "phi": _d["phi"]}
        _panels.append({"kind": "scatter_vline", "title": "Phase diagram  φ vs η", "xlabel": "noise η", "ylabel": "φ",
                         "ymin": 0.0, "ymax": 1.05,
                         "x": _d["eta_sweep"], "y": _d["phi_sweep"],
                         "dot_x": _d["eta_val"], "dot_y": _d["phi"]})
        _layout = [[0.0, 0.0, 0.50, 1.00], [0.50, 0.0, 0.50, 0.50], [0.50, 0.50, 0.50, 0.50]]
    _spec = {"frames": _d["frames"], "dt": 60, "panels": _panels, "layout": _layout}
    mo.vstack([
        mo.md(
            f"**{_d['regime']}** &nbsp;|&nbsp; "
            f"φ (final) = {_d['phi'][-1]:.2f} &nbsp;|&nbsp; "
            f"φ (mean, last half) = {_d['phi_mean']:.2f} &nbsp;|&nbsp; "
            f"box L = {_d['L']:.2f}"
            + (" &nbsp;|&nbsp; press 'compute phase diagram' above to add the φ vs η panel" if not _d["has_sweep"] else "")
        ),
        canvas_anim(_spec, height="500px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **The Vicsek transition.** Below a critical noise $\eta_c$ the system snaps into a
        flock: particles align headings and $\phi = |\langle e^{i\theta}\rangle|$ rises sharply.
        The top-right panel traces $\phi(t)$ for the current run; the orange dot marks the
        current frame. The bottom-right shows the full $\phi$ vs $\eta$ bifurcation curve,
        with the current noise level marked by the orange dot and vline.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 2: Run-and-tumble")
    return


@app.cell(hide_code=True)
def _(mo):
    v_sl   = mo.ui.slider(0.1, 1.0,  value=0.4,  step=0.05, label="speed v",       show_value=True)
    lam_sl = mo.ui.slider(0.02, 0.5, value=0.1,  step=0.02, label="tumble rate λ", show_value=True)
    kap_sl = mo.ui.slider(0.0, 0.4,  value=0.0,  step=0.05, label="confinement κ", show_value=True)
    mo.hstack([v_sl, lam_sl, kap_sl], justify="start", gap=2)
    return kap_sl, lam_sl, v_sl


@app.cell(hide_code=True)
def _(kap_sl, lam_sl, np, v_sl):
    _v   = float(v_sl.value)
    _lam = float(lam_sl.value)
    _kap = float(kap_sl.value)
    _NW  = 25
    _T   = 220
    _dt2 = 0.2
    _D_eff = _v**2 / (2 * _lam)
    _tau_rt = 1.0 / _lam
    _rng2 = np.random.default_rng(7)
    _px = _rng2.standard_normal((_NW,)) * 0.05
    _py = _rng2.standard_normal((_NW,)) * 0.05
    _ph = _rng2.uniform(-np.pi, np.pi, (_NW,))
    _x0, _y0 = _px.copy(), _py.copy()
    _paths_w = [[None] * _T for _ in range(_NW)]
    _msd_sim, _t_msd = [], []
    for _s in range(_T):
        _tumble = _rng2.random(_NW) < _lam * _dt2
        _ph = np.where(_tumble, _rng2.uniform(-np.pi, np.pi, _NW), _ph)
        _px = _px + _v * np.cos(_ph) * _dt2
        _py = _py + _v * np.sin(_ph) * _dt2
        if _kap > 0:
            _px -= _kap * _px * _dt2
            _py -= _kap * _py * _dt2
        for _w in range(_NW):
            _paths_w[_w][_s] = [round(float(_px[_w]), 3), round(float(_py[_w]), 3)]
        _t_now = (_s + 1) * _dt2
        _t_msd.append(float(_t_now))
        _msd_sim.append(float(np.mean((_px - _x0)**2 + (_py - _y0)**2)))

    _t_arr = np.array(_t_msd)
    _msd_th = (4 * _D_eff * (_t_arr + (2.0 / _lam) * (np.exp(-_lam * _t_arr) - 1))).tolist()
    _span = max(float(np.abs(_px).max()), float(np.abs(_py).max()), 1.0) * 2.5
    _conf_r = float(np.sqrt(_D_eff / _kap)) if _kap > 0 else 0.0

    rnt_data = {
        "paths": _paths_w, "t_msd": _t_msd,
        "msd_sim": _msd_sim, "msd_th": _msd_th,
        "frames": _T, "L": float(_span), "conf": _conf_r,
        "D_eff": float(_D_eff), "tau": float(_tau_rt),
    }
    return (rnt_data,)


@app.cell(hide_code=True)
def _(canvas_anim, mo, rnt_data):
    _d = rnt_data
    _spec2 = {
        "frames": _d["frames"], "dt": 55,
        "panels": [
            {"kind": "rnt", "title": "Run-and-tumble walkers", "xlabel": "position", "ylabel": "y",
             "paths": _d["paths"],
             "msd_sim": _d["msd_sim"], "msd_th": _d["msd_th"],
             "t_msd": _d["t_msd"],
             "L": _d["L"], "conf": _d["conf"]},
        ],
        "layout": [[0.0, 0.0, 1.0, 1.0]],
    }
    mo.vstack([
        mo.md(
            f"**D_eff = {_d['D_eff']:.3f}** &nbsp;|&nbsp; "
            f"τ = 1/λ = {_d['tau']:.2f} &nbsp;|&nbsp; "
            "left: walker ensemble (coloured trails), "
            "right: MSD log-log, blue = simulation, red dashed = theory"
        ),
        canvas_anim(_spec2, height="420px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **The MSD crossover.** At short times $t \ll \tau = 1/\lambda$ walkers have not yet
        tumbled and travel straight, so MSD $\propto t^2$ (ballistic). At long times
        $t \gg \tau$ many tumbles randomise the heading and MSD $= 4 D_\text{eff} t$
        (diffusive) with $D_\text{eff} = v^2/(2\lambda)$. The theory curve
        $4 D_\text{eff}[t + (2/\lambda)(e^{-\lambda t}-1)]$ interpolates between
        the two regimes. With confinement $\kappa > 0$ the MSD saturates at
        $\sim D_\text{eff}/\kappa$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
