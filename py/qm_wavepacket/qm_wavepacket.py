import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Quantum Wavepacket")


@app.cell(hide_code=True)
def _():
    import json
    import marimo as mo
    import numpy as np
    return json, mo, np


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
    mo.md(
        r"""
        # Quantum wavepacket: Schrodinger evolution and tunnelling

        A Gaussian wavepacket evolves under the 1D time-dependent Schrodinger equation
        $i\,\partial_t\psi = -\tfrac12\partial_x^2\psi + V(x)\psi$ (units $\hbar = m = 1$),
        integrated by the **split-step Fourier** method. Choose a potential and watch the
        packet spread, reflect, and tunnel. The top panel renders the complex wavefunction
        as an Argand helix in 3D perspective; the bottom tracks $|\psi(x,t)|^2$ live.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    pot_sl = mo.ui.dropdown(
        ["free", "barrier", "well", "harmonic", "double-barrier", "step", "custom"],
        value="barrier",
        label="potential",
    )
    A_sl   = mo.ui.slider(0.1, 3.0, value=0.8, step=0.05, label="amplitude A (x KE)", show_value=True)
    w_sl   = mo.ui.slider(0.2, 4.0, value=1.0, step=0.1,  label="barrier width w",    show_value=True)
    k0_sl  = mo.ui.slider(1.0, 6.0, value=3.0, step=0.25, label="wavenumber k0",      show_value=True)
    spf_sl = mo.ui.slider(1, 20,    value=6,   step=1,    label="steps per frame",    show_value=True)
    Vx_sl  = mo.ui.text(value="3 * x**2", label="custom V(x)  (use x, np)")
    mo.vstack([
        mo.hstack([pot_sl, A_sl, w_sl, k0_sl, spf_sl], justify="start", gap=2),
        mo.md(f"Custom V(x) expression (active when potential = custom): {Vx_sl}"),
    ])
    return A_sl, k0_sl, Vx_sl, pot_sl, spf_sl, w_sl


@app.cell(hide_code=True)
def _(np, A_sl, k0_sl, Vx_sl, pot_sl, spf_sl, w_sl):
    _pot  = pot_sl.value
    _A    = float(A_sl.value)
    _w    = float(w_sl.value)
    _k0   = float(k0_sl.value)
    _spf  = int(spf_sl.value)

    # ── Grid ──────────────────────────────────────────────────────────────
    _N   = 256
    _L   = 20.0
    _dx  = _L / _N
    _x   = np.linspace(-_L / 2, _L / 2, _N, endpoint=False)
    _k   = 2 * np.pi * np.fft.fftfreq(_N, d=_dx)

    # ── Initial state ─────────────────────────────────────────────────────
    _x0  = -5.0
    _sig = 0.5
    _psi = np.exp(-(_x - _x0) ** 2 / (4 * _sig ** 2)) * np.exp(1j * _k0 * _x)
    _psi = _psi / np.sqrt(np.sum(np.abs(_psi) ** 2) * _dx)

    # ── Potential (in units where KE = k0^2/2) ───────────────────────────
    _E   = 0.5 * _k0 ** 2
    _V   = np.zeros(_N)
    if _pot == "custom":
        try:
            _V = np.clip(
                eval(Vx_sl.value, {"np": np, "x": _x, "__builtins__": {}}),
                -20.0, 20.0,
            ).astype(float)
        except Exception:
            _V = np.zeros_like(_x)
    elif _pot == "barrier":
        _V[np.abs(_x) < _w / 2] = _A * _E
    elif _pot == "well":
        _V[np.abs(_x) < _w / 2] = -_A * _E
    elif _pot == "harmonic":
        _V = 0.5 * (_x / _w) ** 2 * _A * _E
    elif _pot == "double-barrier":
        _V[np.abs(np.abs(_x) - _w) < 0.2] = _A * _E
    elif _pot == "step":
        _V[_x > 0] = _A * _E

    # ── Absorbing boundary mask ───────────────────────────────────────────
    _edge   = 0.08 * _L
    _abs_m  = np.ones(_N)
    _d      = (np.abs(_x) - (_L / 2 - _edge)) / _edge
    _mask   = _d > 0
    _abs_m[_mask] = np.exp(-4.0 * _d[_mask] ** 2)

    # ── Split-step operators ──────────────────────────────────────────────
    _dt     = 0.005
    _kin    = np.exp(-1j * 0.5 * _k ** 2 * _dt)
    _halfV  = np.exp(-1j * _V * _dt / 2)

    # ── Pre-compute frames ────────────────────────────────────────────────
    _FRAMES     = 100
    _psi_re_f   = []
    _psi_im_f   = []
    _prob_f     = []
    _psi_cur    = _psi.copy()

    for _fi in range(_FRAMES):
        for _ in range(_spf):
            _psi_cur = _halfV * _psi_cur
            _psi_cur = np.fft.ifft(_kin * np.fft.fft(_psi_cur))
            _psi_cur = _halfV * _psi_cur
            _psi_cur *= _abs_m
        _psi_re_f.append(np.round(np.real(_psi_cur), 4).tolist())
        _psi_im_f.append(np.round(np.imag(_psi_cur), 4).tolist())
        _prob_f.append(np.round(np.abs(_psi_cur) ** 2, 5).tolist())

    # ── Potential markers for probability panel ───────────────────────────
    _markers = []
    if _pot == "barrier" or _pot == "well":
        _markers = [-_w / 2, _w / 2]
    elif _pot == "double-barrier":
        _markers = [-_w - 0.2, -_w + 0.2, _w - 0.2, _w + 0.2]
    elif _pot == "step":
        _markers = [0.0]
    elif _pot == "harmonic":
        _markers = [-_w, _w]

    # ── Per-frame camera follow: center of mass and spread ────────────────
    # The 3D view window tracks the packet's center of mass and widens as it
    # spreads, exactly like the Julia Axis3 camera-follow logic.
    _com_f, _halfw_f = [], []
    for _pf in _prob_f:
        _p = np.array(_pf)
        _s = float(np.sum(_p))
        if _s > 1e-30:
            _cm = float(np.sum(_x * _p) / _s)
            _sp = float(np.sqrt(max(np.sum((_x - _cm) ** 2 * _p) / _s, 1e-12)))
        else:
            _cm, _sp = 0.0, _sig
        _com_f.append(round(_cm, 4))
        _halfw_f.append(round(max(3.0 * _sp, 3.0 * _sig), 4))

    # ── Amplitude normalisation (so Re, Im, helix, prob share scale) ──────
    _amp = max(1e-12, float(np.max([np.max(np.abs(_psi_re_f)),
                                    np.max(np.abs(_psi_im_f))])))
    _pmax = max(1e-12, float(np.max(_prob_f)))

    # ── Stats for last frame ──────────────────────────────────────────────
    _prob_last  = np.array(_prob_f[-1])
    _norm_last  = float(np.sum(_prob_last) * _dx)
    _com_last   = float(np.sum(_x * _prob_last) * _dx / max(_norm_last, 1e-30))
    _T_proxy    = float(np.sum(_prob_last[_x > 0]) * _dx)
    _R_proxy    = float(np.sum(_prob_last[_x <= 0]) * _dx)

    # ── V scaled for display in both panels ──────────────────────────────
    _Vmax  = float(np.max(np.abs(_V))) if np.any(_V != 0) else 1.0
    _prob0max = float(np.max(np.abs(_psi) ** 2))
    _V_disp2d = np.round(_V / _Vmax * _prob0max * 0.7, 6).tolist()
    _V_disp3d = np.round(_V / _Vmax * 0.6, 6).tolist()

    qm_data = {
        "frames": _FRAMES,
        "psi_re": _psi_re_f,
        "psi_im": _psi_im_f,
        "prob": _prob_f,
        "V2d": _V_disp2d,
        "V3d": _V_disp3d,
        "x": np.round(_x, 4).tolist(),
        "markers": _markers,
        "com_f": _com_f,
        "halfw_f": _halfw_f,
        "amp": _amp,
        "pmax": _pmax,
        "norm": _norm_last,
        "T": _T_proxy,
        "R": _R_proxy,
        "com": _com_last,
        "k0": _k0,
        "pot": _pot,
        "A": _A,
        "E": float(_E),
    }
    return (qm_data,)


@app.cell(hide_code=True)
def _(mo, qm_data):
    _d = qm_data
    mo.md(
        f"**norm** {_d['norm']:.4f} &nbsp;|&nbsp; "
        f"**T proxy** (right half) {_d['T']:.3f} &nbsp;|&nbsp; "
        f"**R proxy** (left half) {_d['R']:.3f} &nbsp;|&nbsp; "
        f"**&lt;x&gt;** {_d['com']:.3f} &nbsp;|&nbsp; "
        f"k0 = {_d['k0']:.2f}, KE = {_d['E']:.2f}, potential = {_d['pot']}, A = {_d['A']:.2f}"
    )
    return


@app.cell(hide_code=True)
def _(canvas_anim, mo, qm_data):

    _d = qm_data
    _spec = {
        "frames": _d["frames"], "dt": 50,
        "panels": [
            {
                "kind": "wavepacket_3d",
                "psi_re": _d["psi_re"], "psi_im": _d["psi_im"], "prob": _d["prob"],
                "V3d": _d["V3d"], "x": _d["x"],
                "com_f": _d["com_f"], "halfw_f": _d["halfw_f"],
                "amp": _d["amp"], "pmax": _d["pmax"],
            },
            {
                "kind": "prob_density",
                "prob": _d["prob"], "V2d": _d["V2d"], "x": _d["x"], "markers": _d["markers"],
            },
        ],
        "layout": [[0.0, 0.0, 1.0, 0.60], [0.0, 0.60, 1.0, 0.40]],
    }
    canvas_anim(_spec, height="880px")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **The split-step method.** One timestep applies the potential half-kick, propagates freely
        in Fourier space, then applies the potential half-kick again (Strang splitting, error
        $O(\Delta t^2)$):

        $$\psi \to e^{-iV\Delta t/2}\,\psi,\quad
        \psi \to \mathcal{F}^{-1}\bigl[e^{-ik^2\Delta t/2}\,\mathcal{F}\psi\bigr],\quad
        \psi \to e^{-iV\Delta t/2}\,\psi.$$

        Units: $\hbar = m = 1$, so kinetic energy $E = k_0^2/2$ and all potentials are set in
        multiples of $E$. Amplitude $A < 1$ is classically forbidden (pure tunnelling). The
        **Argand helix** (top): cyan curve = $\mathrm{Re}(\psi)$ in the $x$-$z$ plane, amber =
        $\mathrm{Im}(\psi)$ in the $x$-$y$ plane, dark red = $|\psi|^2$ on the floor, violet =
        the full helix. Pitch equals the de Broglie wavelength $2\pi/k_0$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
