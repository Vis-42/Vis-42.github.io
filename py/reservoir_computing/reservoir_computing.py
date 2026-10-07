import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Reservoir Computing")


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
        # Reservoir computing: predicting chaos with an echo-state network

        Chaos puts a hard wall in front of prediction: nearby trajectories separate exponentially,
        so past a horizon the future is unknowable from any finite measurement. This notebook builds
        an **echo-state network** (ESN) that learns Lorenz-63 or Rossler dynamics and then free-runs
        on its own output, chasing that wall in miniature. Training is a single closed-form ridge
        solve, no backpropagation.

        A live two-panel canvas shows the attractor in 3D (drag to orbit it) and the growing x(t) time series,
        with the true trajectory in blue and the ESN prediction trailing 20 frames behind in amber.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    N_sl = mo.ui.slider(40, 200, value=100, step=10,
                        label="reservoir size N", show_value=True)
    rho_sl = mo.ui.slider(0.5, 1.3, value=0.9, step=0.05,
                          label="spectral radius rho", show_value=True)
    sp_sl = mo.ui.slider(0.05, 0.5, value=0.1, step=0.05,
                         label="sparsity", show_value=True)
    sys_sl = mo.ui.dropdown(["Lorenz-63", "Rossler"],
                            value="Lorenz-63", label="system")
    ph_sl = mo.ui.slider(100, 600, value=400, step=50,
                         label="prediction steps", show_value=True)
    _controls = mo.vstack([
        mo.hstack([N_sl, rho_sl, sp_sl], justify="start", gap=2),
        mo.hstack([sys_sl, ph_sl], justify="start", gap=2),
    ])
    _controls
    return N_sl, ph_sl, rho_sl, sp_sl, sys_sl


@app.cell(hide_code=True)
def _(N_sl, ph_sl, rho_sl, sp_sl, sys_sl, np):
    # ── attractor integration (custom RK4, no scipy) ─────────────────────────

    def _lorenz_deriv(_u):
        _x, _y, _z = _u
        return np.array([10.0 * (_y - _x),
                         _x * (28.0 - _z) - _y,
                         _x * _y - (8.0 / 3.0) * _z])

    def _rossler_deriv(_u):
        _x, _y, _z = _u
        return np.array([- (_y + _z),
                         _x + 0.2 * _y,
                         0.2 + _z * (_x - 5.7)])

    def _rk4_step(_f, _u, _dt):
        _k1 = _f(_u)
        _k2 = _f(_u + 0.5 * _dt * _k1)
        _k3 = _f(_u + 0.5 * _dt * _k2)
        _k4 = _f(_u + _dt * _k3)
        return _u + (_dt / 6.0) * (_k1 + 2.0 * _k2 + 2.0 * _k3 + _k4)

    def _integrate(_f, _u0, _dt, _n):
        _traj = np.empty((_n, 3))
        _u = np.array(_u0, dtype=float)
        for _i in range(_n):
            _u = _rk4_step(_f, _u, _dt)
            _traj[_i] = _u
        return _traj

    _system = sys_sl.value
    if _system == "Lorenz-63":
        _f_sys = _lorenz_deriv
        _dt_sys = 0.01
        _u0_sys = [1.0, 0.0, 0.0]
        _lya = 0.906        # max Lyapunov exponent
        _tol_esn = 0.5
    else:
        _f_sys = _rossler_deriv
        _dt_sys = 0.05
        _u0_sys = [1.0, 0.0, 0.0]
        _lya = 0.071        # Rossler max Lyapunov exponent
        _tol_esn = 0.5

    _WASHOUT = 500
    _TRAIN   = 1000
    _PRED    = int(ph_sl.value)
    _TOTAL   = _WASHOUT + _TRAIN + _PRED + 200   # extra buffer

    _raw = _integrate(_f_sys, _u0_sys, _dt_sys, _TOTAL)

    # ── normalise ─────────────────────────────────────────────────────────────
    _mu  = _raw.mean(0)
    _std = _raw.std(0) + 1e-12
    _data = (_raw - _mu) / _std

    # ── build reservoir ───────────────────────────────────────────────────────
    _N    = int(N_sl.value)
    _sp   = float(sp_sl.value)
    _rho  = float(rho_sl.value)
    _rng  = np.random.default_rng(42)

    _W_in = (2.0 * _rng.random((_N, 3)) - 1.0) * 0.5
    _W_res = np.zeros((_N, _N))
    _n_edges = max(1, round(_N * _N * _sp))
    _idx = _rng.choice(_N * _N, size=_n_edges, replace=False)
    _W_res.flat[_idx] = _rng.standard_normal(_n_edges)
    _ev = np.abs(np.linalg.eigvals(_W_res))
    _rho_cur = _ev.max()
    if _rho_cur > 1e-8:
        _W_res *= _rho / _rho_cur

    # ── drive + collect ───────────────────────────────────────────────────────
    _state = np.zeros(_N)
    _coll  = np.zeros((_TRAIN - 1, _N + 3))
    _train_data = _data[:_WASHOUT + _TRAIN]
    for _t in range(len(_train_data) - 1):
        _state = np.tanh(_W_res @ _state + _W_in @ _train_data[_t])
        if _t >= _WASHOUT:
            _ti = _t - _WASHOUT
            _coll[_ti, :_N] = _state
            _coll[_ti, _N:] = _train_data[_t]

    _tgt = _train_data[_WASHOUT + 1:]
    _A   = _coll.T @ _coll + 1e-6 * np.eye(_N + 3)
    _W_out = np.linalg.solve(_A, _coll.T @ _tgt).T  # (3, N+3)

    # ── autonomous prediction ─────────────────────────────────────────────────
    _pred = np.zeros((_PRED, 3))
    _inp  = _train_data[_WASHOUT + _TRAIN - 1].copy()
    for _t in range(_PRED):
        _state = np.tanh(_W_res @ _state + _W_in @ _inp)
        _inp = _W_out @ np.concatenate([_state, _inp])
        _pred[_t] = _inp

    _true_test = _data[_WASHOUT + _TRAIN: _WASHOUT + _TRAIN + _PRED]
    _n_cmp = min(len(_pred), len(_true_test))
    _pred = _pred[:_n_cmp]
    _true_test = _true_test[:_n_cmp]

    # ── metrics ───────────────────────────────────────────────────────────────
    _rmse_val = float(np.sqrt(np.mean((_pred - _true_test) ** 2)))
    _ns = float(np.sqrt(np.mean(_true_test ** 2))) + 1e-12
    _errs = np.sqrt(np.sum((_pred - _true_test) ** 2, axis=1)) / _ns
    _cross = np.argmax(_errs > _tol_esn) if np.any(_errs > _tol_esn) else _n_cmp
    _T_lya = 1.0 / _lya
    _horizon_lt = float(_cross * _dt_sys / _T_lya)

    # ── Lyapunov exponent (divergence of nearby trajectory, Benettin proxy) ──
    _eps0  = 1e-6
    _u_ref = np.array(_u0_sys, dtype=float)
    # warm up reference
    for _ in range(_WASHOUT + 50):
        _u_ref = _rk4_step(_f_sys, _u_ref, _dt_sys)
    _u_pert = _u_ref + _eps0 * _rng.standard_normal(3)
    _u_pert /= np.linalg.norm(_u_pert - _u_ref) / _eps0
    _lya_sum = 0.0
    _lya_steps = 500
    for _ in range(_lya_steps):
        _u_ref  = _rk4_step(_f_sys, _u_ref,  _dt_sys)
        _u_pert = _rk4_step(_f_sys, _u_pert, _dt_sys)
        _d = np.linalg.norm(_u_pert - _u_ref)
        _lya_sum += np.log(_d / _eps0)
        _u_pert = _u_ref + (_u_pert - _u_ref) * (_eps0 / _d)
    _lya_est = float(_lya_sum / (_lya_steps * _dt_sys))

    # ── build canvas spec ──────────────────────────────────────────────────────
    # Attractor panel: project to (x, z) in physical (un-normalised) coords.
    # ESN predictions can blow up; clamp to physical attractor bounds.
    _phys_true = _true_test * _std + _mu
    _phys_pred = _pred       * _std + _mu

    _ax_lo = float(_phys_true[:, 0].min()) - 3.0
    _ax_hi = float(_phys_true[:, 0].max()) + 3.0
    _ay_lo = float(_phys_true[:, 1].min()) - 3.0
    _ay_hi = float(_phys_true[:, 1].max()) + 3.0
    _az_lo = float(_phys_true[:, 2].min()) - 3.0
    _az_hi = float(_phys_true[:, 2].max()) + 3.0

    _px_clamped = np.clip(_phys_pred[:, 0], _ax_lo, _ax_hi)
    _py_clamped = np.clip(_phys_pred[:, 1], _ay_lo, _ay_hi)
    _pz_clamped = np.clip(_phys_pred[:, 2], _az_lo, _az_hi)

    # Full ghost trajectory in physical coords (for attractor background).
    _ghost_raw  = _raw[_WASHOUT: _WASHOUT + _TRAIN + _PRED]
    _ghost_x = np.round(_ghost_raw[:, 0], 2).tolist()
    _ghost_y = np.round(_ghost_raw[:, 1], 2).tolist()
    _ghost_z = np.round(_ghost_raw[:, 2], 2).tolist()

    # bounds centre + half-span for the 3D box normalisation
    _cx = 0.5 * (_ax_lo + _ax_hi); _cy = 0.5 * (_ay_lo + _ay_hi); _cz = 0.5 * (_az_lo + _az_hi)
    _hs = 0.5 * max(_ax_hi - _ax_lo, _ay_hi - _ay_lo, _az_hi - _az_lo) + 1e-9

    _FRAMES = _n_cmp
    _canvas_spec = {
        "frames":  _FRAMES,
        "dt":      30,
        "kind":    "attractor_esn",
        "panels": [{"kind": "attractor_esn"}],
        "layout": [[0.0, 0.0, 1.0, 1.0]],
        "system":  _system,
        "cx": _cx, "cy": _cy, "cz": _cz, "hs": _hs,
        "ghost_x": _ghost_x, "ghost_y": _ghost_y, "ghost_z": _ghost_z,
        "true_x":  np.round(_phys_true[:, 0], 2).tolist(),
        "true_y":  np.round(_phys_true[:, 1], 2).tolist(),
        "true_z":  np.round(_phys_true[:, 2], 2).tolist(),
        "esn_x":   np.round(_px_clamped, 2).tolist(),
        "esn_y":   np.round(_py_clamped, 2).tolist(),
        "esn_z":   np.round(_pz_clamped, 2).tolist(),
        "ts_x_true": np.round(_phys_true[:, 0], 2).tolist(),
        "ts_x_esn":  np.round(_px_clamped, 2).tolist(),
    }

    esn_results = {
        "canvas_spec":  _canvas_spec,
        "N":            _N,
        "rho":          _rho,
        "sp":           _sp,
        "rmse":         _rmse_val,
        "horizon_lt":   _horizon_lt,
        "lya_est":      _lya_est,
        "system":       _system,
        "dt":           _dt_sys,
        "n_cmp":        _n_cmp,
    }
    return (esn_results,)


@app.cell(hide_code=True)
def _(canvas_anim, esn_results, mo):

    _spec = esn_results["canvas_spec"]
    _r = esn_results
    _ph_color = "#57c98a" if _r["horizon_lt"] > 3.0 else ("#e6a356" if _r["horizon_lt"] > 1.5 else "#e0635a")
    _stats_md = mo.md(
        f"**{_r['system']}** "
        f"| N = {_r['N']} "
        f"| rho = {_r['rho']:.2f} "
        f"| sparsity = {_r['sp']:.2f} "
        f"| RMSE = {_r['rmse']:.3f} "
        f"| Lyapunov est = {_r['lya_est']:.3f} s^-1 "
        f"| prediction horizon = **{_r['horizon_lt']:.2f} Lyapunov times**"
    )
    mo.vstack([_stats_md, canvas_anim(_spec, "520px")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## How it works

        An Echo State Network begins with a fixed, randomly wired recurrent reservoir: the input
        weights $W_\text{in}$ and the reservoir weight matrix $W_\text{res}$ are drawn once and
        never adjusted again. At each time step the reservoir updates by

        $$\mathbf r(t) = \tanh\!\bigl(W_\text{res}\,\mathbf r(t-1) + W_\text{in}\,\mathbf u(t)\bigr).$$

        After a washout period (500 steps, discarded so the reservoir forgets its zero initial
        condition), the extended state $[\mathbf r(t),\, \mathbf u(t)]$ is collected into a feature
        matrix. The output weights $W_\text{out}$ are found in a single batch ridge solve:

        $$W_\text{out} = (X^\top X + \alpha I)^{-1} X^\top Y.$$

        After training, the network runs autonomously: its own output is fed back as the next input.
        Errors compound exponentially, so prediction is finite by necessity, not by choice.

        **Spectral radius near 1** is the sweet spot. Below 1 the echo-state property holds
        (memory decays), but very low values forget too fast. Near 1 the reservoir retains the
        longest memory. Above 1 the dynamics become unstable and the echo-state property is lost.

        **Prediction horizon** is measured in Lyapunov times: the number of $1/\lambda_\text{max}$
        intervals before the normalised RMSE exceeds 50% of attractor amplitude. Three or more
        Lyapunov times is considered good for a small reservoir.
        """
    )
    return


if __name__ == "__main__":
    app.run()
