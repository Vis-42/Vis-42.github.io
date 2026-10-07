import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Coupled Oscillators")


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
        # Coupled oscillators: synchronisation and chimera states

        Two questions, same system. The Kuramoto model asks when N phase oscillators with
        spread natural frequencies fall into collective step: there is a sharp transition at
        a critical coupling $K_c = 2\sigma\sqrt{2/\pi}$ where the order parameter
        $r = |\langle e^{i\theta}\rangle|$ jumps from 0 to 1. Three live panels share one
        clock: the phase circle with the mean-field vector, $r(t)$ with a tracking dot,
        and the $r$ vs $K$ bifurcation curve with the current $K$ marked.

        The second question is stranger: can part of an *identical* ring synchronise while
        the rest stays incoherent? Yes, and it is called a chimera state.
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
    mo.md("## Part 1: Kuramoto synchronisation")
    return


@app.cell(hide_code=True)
def _(mo):
    K_sl   = mo.ui.slider(0.0, 6.0,  value=2.5,  step=0.25, label="coupling K",      show_value=True)
    sig_sl = mo.ui.slider(0.2, 2.0,  value=0.8,  step=0.1,  label="freq spread σ",   show_value=True)
    N_sl   = mo.ui.slider(20,  120,   value=50,   step=10,   label="oscillators N",   show_value=True)
    topo_sl = mo.ui.dropdown(["all-to-all", "Erdős-Rényi", "scale-free"],
                             value="all-to-all", label="network")
    mo.hstack([topo_sl, K_sl, sig_sl, N_sl], justify="start", gap=2)
    return K_sl, N_sl, sig_sl, topo_sl


@app.cell(hide_code=True)
def _(K_sl, N_sl, np, sig_sl, topo_sl):
    _K   = float(K_sl.value)
    _sig = float(sig_sl.value)
    _N   = int(N_sl.value)
    _dt  = 0.05
    _WARM   = 200
    _FRAMES = 200
    _rng = np.random.default_rng(0)
    _omega = _rng.normal(0, _sig, _N)
    _theta = _rng.uniform(-np.pi, np.pi, _N)
    _Kc_theory = 2 * _sig * np.sqrt(2 / np.pi)

    # ── build the coupling adjacency for the chosen topology ──
    def _adjacency(N, topo, rng):
        if topo == "Erdős-Rényi":
            p = min(1.0, 6.0 / N)   # mean degree ~6
            A = (rng.random((N, N)) < p).astype(float)
            A = np.triu(A, 1); A = A + A.T
        elif topo == "scale-free":
            # Barabasi-Albert preferential attachment, m=2
            m = 2; A = np.zeros((N, N))
            deg = np.ones(N) * 1e-9
            for _i in range(m):
                A[_i, (_i + 1) % m] = A[(_i + 1) % m, _i] = 1; deg[_i] += 1
            for _i in range(m, N):
                _pr = deg[:_i] / deg[:_i].sum()
                _tg = rng.choice(_i, size=min(m, _i), replace=False, p=_pr)
                for _j in _tg:
                    A[_i, _j] = A[_j, _i] = 1; deg[_i] += 1; deg[_j] += 1
            np.fill_diagonal(A, 0)
        else:
            A = np.ones((N, N)); np.fill_diagonal(A, 0)
        return A

    _A = _adjacency(_N, topo_sl.value, _rng)
    _kbar = max(1.0, float(_A.sum(1).mean()))   # mean degree for normalisation

    def _kura_step(th, omega, K, N, dt, A=_A, kbar=_kbar):
        diff = th[None, :] - th[:, None]
        dth = omega + (K / kbar) * (A * np.sin(diff)).sum(1)
        return th + dth * dt

    for _ in range(_WARM):
        _theta = _kura_step(_theta, _omega, _K, _N, _dt)

    _ths_f, _r_s = [], []
    for _ in range(_FRAMES):
        _theta = _kura_step(_theta, _omega, _K, _N, _dt)
        _ths_f.append((_theta % (2 * np.pi)).tolist())
        _r_s.append(float(np.abs(np.mean(np.exp(1j * _theta)))))

    _r_mean = float(np.mean(_r_s[_FRAMES // 2:]))
    _regime = "SYNCHRONISED" if _r_mean > 0.75 else "PARTIAL" if _r_mean > 0.3 else "INCOHERENT"

    # K sweep , no button gate
    _K_arr = np.linspace(0.1, 6.0, 16)
    _r_arr = []
    for _Kk in _K_arr:
        _th2 = _rng.uniform(-np.pi, np.pi, _N)
        for _ in range(150):
            _th2 = _kura_step(_th2, _omega, _Kk, _N, _dt)
        _rs2 = []
        for _ in range(80):
            _th2 = _kura_step(_th2, _omega, _Kk, _N, _dt)
            _rs2.append(float(np.abs(np.mean(np.exp(1j * _th2)))))
        _r_arr.append(float(np.mean(_rs2)))

    kura_data = {
        "ths": _ths_f, "r": _r_s, "frames": _FRAMES,
        "K_sweep": _K_arr.tolist(), "r_sweep": _r_arr,
        "Kc": float(_Kc_theory),
        "K_now": [float(_K)] * _FRAMES, "r_now": _r_s,
        "r_mean": _r_mean, "regime": _regime,
        "topo": topo_sl.value, "kbar": float(_kbar),
    }
    return (kura_data,)


@app.cell(hide_code=True)
def _(canvas_anim, kura_data, mo):
    _d = kura_data
    _spec = {
        "frames": _d["frames"], "dt": 55,
        "panels": [
            {"kind": "phases", "title": "Phases on unit circle", "theta": _d["ths"], "r_series": _d["r"]},
            {"kind": "timeseries", "title": "Order parameter r(t)", "xlabel": "frame", "ylabel": "r = |⟨eⁱᶿ⟩|",
             "ymin": 0.0, "ymax": 1.05, "grow": True, "dot": 0,
             "ys": [_d["r"]], "colors": ["#7dd3fc"], "t": list(range(_d["frames"]))},
            {"kind": "scatter_vline", "title": "Bifurcation: r vs K  (amber = Kc theory)", "xlabel": "coupling K", "ylabel": "r (steady state)",
             "ymin": 0.0, "ymax": 1.05,
             "x": _d["K_sweep"], "y": _d["r_sweep"], "Kc": _d["Kc"],
             "dot_x": _d["K_now"], "dot_y": _d["r_now"]},
        ],
        "layout": [
            [0.0,  0.0,  0.40, 1.0],
            [0.40, 0.0,  0.60, 0.50],
            [0.40, 0.50, 0.60, 0.50],
        ],
    }
    mo.vstack([
        mo.md(
            f"**{_d['regime']}** &nbsp;|&nbsp; "
            f"network: {_d['topo']} (mean degree {_d['kbar']:.1f}) &nbsp;|&nbsp; "
            f"r (mean, last half) = {_d['r_mean']:.2f} &nbsp;|&nbsp; "
            f"K_c theory = {_d['Kc']:.2f}"
        ),
        canvas_anim(_spec, height="500px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **The synchronisation transition.** The order parameter $r = |\langle e^{i\theta_i}\rangle|$
        measures how aligned the oscillator phases are: $r = 0$ is fully incoherent,
        $r = 1$ is fully synchronised. The analytic result $K_c = 2\sigma\sqrt{2/\pi}$
        marks the onset, visible as the inflection in the bifurcation curve (bottom right).
        The orange vector on the phase circle is the mean-field phasor $r e^{i\psi}$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 2: Chimera states")
    return


@app.cell(hide_code=True)
def _(mo):
    Kch_sl = mo.ui.slider(0.5, 4.0,  value=1.5,  step=0.25, label="coupling K",     show_value=True)
    R_sl   = mo.ui.slider(2,   20,    value=8,    step=1,    label="range R",        show_value=True)
    alp_sl = mo.ui.slider(0.0, 1.5,   value=1.46, step=0.05, label="phase lag α",   show_value=True)
    Nch_sl = mo.ui.slider(30,  80,    value=50,   step=10,   label="ring size N",   show_value=True)
    mo.hstack([Kch_sl, R_sl, alp_sl, Nch_sl], justify="start", gap=2)
    return Kch_sl, Nch_sl, R_sl, alp_sl


@app.cell(hide_code=True)
def _(np):
    def ks_step(th, N, K, R, alp, dt):
        dth = np.zeros(N)
        for k in range(-R, R + 1):
            if k == 0:
                continue
            dth += np.sin(np.roll(th, -k) - th - alp)
        return th + dt * (K / (2 * R)) * dth
    return (ks_step,)


@app.cell(hide_code=True)
def _(Kch_sl, Nch_sl, R_sl, alp_sl, ks_step, np):
    import matplotlib.cm as _cm
    _N   = int(Nch_sl.value)
    _K   = float(Kch_sl.value)
    _R   = int(R_sl.value)
    _alp = float(alp_sl.value)
    _dt  = 0.05
    _T_WARM = 300
    _T_REC  = 200
    _theta3 = np.random.default_rng(1).uniform(-np.pi, np.pi, _N)

    for _ in range(_T_WARM):
        _theta3 = ks_step(_theta3, _N, _K, _R, _alp, _dt)

    _history = []
    for _ in range(_T_REC):
        _theta3 = ks_step(_theta3, _N, _K, _R, _alp, _dt)
        _history.append(_theta3.copy())
    _history = np.array(_history)

    # space-time RGB (HSV colormap: hue = theta mod 2pi)
    _ht_norm = (_history % (2 * np.pi)) / (2 * np.pi)
    _rgba    = _cm.hsv(_ht_norm)
    _st_rgb  = (_rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8).flatten().tolist()

    # local order parameter at every time step (T x N), vectorised
    _nbr_idx   = np.array([(np.arange(i - _R, i + _R + 1) % _N) for i in range(_N)])
    _th_nbr    = _history[:, _nbr_idx]                              # (T, N, 2R+1)
    _local_r_f = np.abs(np.mean(np.exp(1j * _th_nbr), axis=2))     # (T, N)

    _theta_final = _history[-1]
    _global_r    = float(np.abs(np.mean(np.exp(1j * _theta_final))))
    _sigma_r     = float(np.std(_local_r_f[-1]))
    _is_chimera  = _sigma_r > 0.08 and _global_r < 0.85

    chimera_data  = {
        "T": _T_REC, "N": _N, "R": _R,
        "st_rgb": _st_rgb,
        "local_r_f": _local_r_f.tolist(),
    }
    chimera_label = (
        f"CHIMERA (σ_r = {_sigma_r:.2f}, |Z| = {_global_r:.2f})"
        if _is_chimera else
        f"not chimera (σ_r = {_sigma_r:.2f}, |Z| = {_global_r:.2f})"
    )
    return chimera_data, chimera_label


@app.cell(hide_code=True)
def _(canvas_anim, chimera_data, chimera_label, mo):
    _d = chimera_data
    _spec_ch = {
        "frames": _d["T"], "dt": 60,
        "panels": [
            {"kind": "space_time", "T": _d["T"], "N": _d["N"],
             "rgb": _d["st_rgb"], "title": "theta mod 2pi (space-time)"},
            {"kind": "local_r_frame", "T": _d["T"], "N": _d["N"],
             "lr": _d["local_r_f"],
             "title": "|r_i| local order", "xlabel": "oscillator i"},
        ],
        "layout": [[0.0, 0.0, 0.60, 1.0], [0.60, 0.0, 0.40, 1.0]],
    }
    mo.vstack([
        mo.md(f"**{chimera_label}**"),
        canvas_anim(_spec_ch, height="420px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Chimera states.** The Kuramoto-Sakaguchi model adds a phase lag $\alpha$ and
        restricts coupling to neighbours within range $R$. Near $\alpha \approx \pi/2$ this
        can produce a chimera: part of the ring phase-locks (high local $|r_i|$, flat in
        the space-time heatmap) while the rest stays incoherent (low $|r_i|$, turbulent).
        Both parts contain identical oscillators with $\omega_i = 0$. The frustration is
        geometric, not from disorder. Try $\alpha \approx 1.46$, $K \approx 1.5$, $R = 8$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 3: (K, R) phase diagram")
    return


@app.cell(hide_code=True)
def _(mo):
    kr_btn = mo.ui.run_button(label="Scan 6x6 (K, R) grid")
    mo.md(
        f"""
        Compute the chimera measure across a 6x6 grid of coupling K and neighbour range R
        at fixed alpha. Bright cells support chimera states.

        {kr_btn}
        """
    )
    return (kr_btn,)


@app.cell(hide_code=True)
def _(alp_sl, ks_step, kr_btn, mo, np):
    mo.stop(not kr_btn.value, mo.md("Press **Scan** to compute the (K, R) diagram."))
    _alp_kr  = float(alp_sl.value)
    _K_vals  = np.linspace(0.5, 3.5, 6)
    _R_vals  = [2, 4, 6, 8, 12, 16]
    _N_kr    = 40
    _T_KR    = 200
    _dt_kr   = 0.05
    _cmap_kr = np.zeros((6, 6))
    for _ki, _K in enumerate(_K_vals):
        for _ri, _R in enumerate(_R_vals):
            _th = np.random.default_rng(1).uniform(-np.pi, np.pi, _N_kr)
            for _ in range(_T_KR):
                _th = ks_step(_th, _N_kr, float(_K), int(_R), _alp_kr, _dt_kr)
            _nbr = np.array([(np.arange(i - int(_R), i + int(_R) + 1) % _N_kr) for i in range(_N_kr)])
            _lr  = np.abs(np.mean(np.exp(1j * _th[_nbr]), axis=1))
            _gr  = np.abs(np.mean(np.exp(1j * _th)))
            _cmap_kr[_ki, _ri] = float(np.std(_lr)) * (1.0 - float(_gr))
    kr_data = {"K_vals": _K_vals.tolist(), "R_vals": _R_vals, "cmap": _cmap_kr.tolist()}
    return (kr_data,)


@app.cell(hide_code=True)
def _(kr_data, mo, plt):
    _K  = kr_data["K_vals"]
    _R  = kr_data["R_vals"]
    _cm = kr_data["cmap"]
    fig_kr, ax_kr = plt.subplots(figsize=(5.5, 3.8))
    _im = ax_kr.pcolormesh(_R, _K, _cm, cmap="magma", shading="nearest", vmin=0)
    fig_kr.colorbar(_im, ax=ax_kr, label="chimera measure sigma(|r_i|) x (1-|Z|)")
    ax_kr.set_xlabel("neighbour range R")
    ax_kr.set_ylabel("coupling K")
    ax_kr.set_title("(K, R) phase diagram (alpha fixed, N = 40)")
    ax_kr.set_xticks(_R)
    ax_kr.set_yticks([round(v, 2) for v in _K])
    fig_kr.tight_layout()
    mo.vstack([
        mo.md("Bright = chimera-supporting. Chimera measure = sigma(|r_i|) x (1-|Z|); zero = fully sync or fully incoherent."),
        fig_kr,
    ])
    return


if __name__ == "__main__":
    app.run()
