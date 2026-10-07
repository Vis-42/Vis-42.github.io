import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Physics Discovery")


@app.cell(hide_code=True)
def _():
    import json
    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return json, matplotlib, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Physics discovery: SINDy, conservation laws, symbolic regression

        Three complementary ways to recover physical laws from trajectory data alone.
        SINDy identifies governing ODEs via sparse regression. Kernel conservation-law
        discovery finds what is preserved. Symbolic genetic programming discovers
        closed-form formulas. Each section runs its own animation; all three share
        one RK4 integrator written from scratch.
        """
    )
    return


# ── Canvas animation helper (shared by all three parts) ────────────────────────
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


# ── Shared numerics (RK4, library builder, STLSQ) ──────────────────────────────
@app.cell(hide_code=True)
def _(np):
    def _rk4_step(f, s, t, dt, p):
        k1 = f(s, t, p)
        k2 = f(s + 0.5 * dt * k1, t + 0.5 * dt, p)
        k3 = f(s + 0.5 * dt * k2, t + 0.5 * dt, p)
        k4 = f(s + dt * k3, t + dt, p)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4_integrate(f, u0, t0, t1, n, p=None):
        dt = (t1 - t0) / (n - 1)
        states = np.empty((n, len(u0)))
        states[0] = u0
        for i in range(1, n):
            t = t0 + (i - 1) * dt
            states[i] = _rk4_step(f, states[i - 1], t, dt, p)
        return states, dt

    def centered_diff(states, dt):
        n, d = states.shape
        dX = np.empty_like(states)
        dX[0] = (states[1] - states[0]) / dt
        dX[-1] = (states[-1] - states[-2]) / dt
        for i in range(1, n - 1):
            dX[i] = (states[i + 1] - states[i - 1]) / (2 * dt)
        return dX

    def poly_library(states, max_deg):
        n, d = states.shape
        cols = [np.ones(n)]
        names = ["1"]
        for total in range(1, max_deg + 1):
            if d == 2:
                for e0 in range(total + 1):
                    e1 = total - e0
                    col = states[:, 0] ** e0 * states[:, 1] ** e1
                    nm = ""
                    if e0 == 1:
                        nm += "x"
                    elif e0 > 1:
                        nm += f"x^{e0}"
                    if e1 == 1:
                        nm += "y"
                    elif e1 > 1:
                        nm += f"y^{e1}"
                    cols.append(col)
                    names.append(nm)
            else:
                # general: enumerate multi-indices
                from itertools import combinations_with_replacement
                for idx in combinations_with_replacement(range(d), total):
                    col = np.ones(n)
                    parts = {}
                    for j in idx:
                        parts[j] = parts.get(j, 0) + 1
                    for j, e in parts.items():
                        col = col * states[:, j] ** e
                    nm = "".join(f"x{j}^{e}" if e > 1 else f"x{j}" for j, e in sorted(parts.items()))
                    cols.append(col)
                    names.append(nm)
        return np.column_stack(cols), names

    def stlsq(Theta, dX, threshold, max_iter=20):
        xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]
        for _ in range(max_iter):
            small = np.abs(xi) < threshold
            xi[small] = 0.0
            for j in range(dX.shape[1]):
                idx = ~small[:, j]
                if idx.any():
                    xi[idx, j] = np.linalg.lstsq(Theta[:, idx], dX[:, j], rcond=None)[0]
        return xi

    return centered_diff, poly_library, rk4_integrate, stlsq


# ── Part 1: SINDy ──────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 1: SINDy: Sparse Identification of Nonlinear Dynamics")
    return


@app.cell(hide_code=True)
def _(mo):
    sindy_sys = mo.ui.dropdown(
        options={"Van der Pol": "vdp", "Duffing (undriven)": "duffing"},
        value="Van der Pol",
        label="system",
    )
    sindy_noise = mo.ui.slider(0.0, 0.10, value=0.0, step=0.005, label="noise σ", show_value=True)
    sindy_thresh = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="STLSQ threshold λ", show_value=True)
    sindy_deg = mo.ui.slider(2, 5, value=3, step=1, label="poly degree", show_value=True)
    mo.hstack([sindy_sys, sindy_noise, sindy_thresh, sindy_deg], justify="start", gap=2)
    return sindy_deg, sindy_noise, sindy_sys, sindy_thresh


@app.cell(hide_code=True)
def _(centered_diff, np, poly_library, rk4_integrate, sindy_deg, sindy_noise, sindy_sys, sindy_thresh, stlsq):
    _sysname = sindy_sys.value
    _sigma   = float(sindy_noise.value)
    _lam     = float(sindy_thresh.value)
    _deg     = int(sindy_deg.value)
    _N_pts   = 800
    _rng_s   = np.random.default_rng(42)

    if _sysname == "vdp":
        _mu = 1.0
        def _sindy_ode(s, t, p):
            return np.array([s[1], p * (1 - s[0]**2) * s[1] - s[0]])
        _u0_s = np.array([2.0, 0.0])
        _p_s  = _mu
        _t1_s = 30.0
    else:
        _al, _be, _de = -1.0, 1.0, 0.1
        def _sindy_ode(s, t, p):
            al, be, de = p
            return np.array([s[1], -de * s[1] - al * s[0] - be * s[0]**3])
        _u0_s = np.array([0.5, 0.0])
        _p_s  = (_al, _be, _de)
        _t1_s = 40.0

    _states_s, _dt_s = rk4_integrate(_sindy_ode, _u0_s, 0.0, _t1_s, _N_pts, _p_s)
    _states_n = _states_s + _sigma * _rng_s.standard_normal(_states_s.shape)
    _dX_s = centered_diff(_states_n, _dt_s)
    _Theta_s, _lib_names_s = poly_library(_states_n, _deg)
    _xi_s = stlsq(_Theta_s, _dX_s, _lam)

    # Reconstruct using identified model
    def _sindy_hat(s, t, xi):
        _row = poly_library(s[None, :], _deg)[0][0]
        return _row @ xi

    _recon_s = np.empty_like(_states_s)
    _recon_s[0] = _u0_s
    for _i in range(1, _N_pts):
        _s_prev = _recon_s[_i - 1]
        if np.any(np.isnan(_s_prev)) or np.any(np.isinf(_s_prev)) or np.max(np.abs(_s_prev)) > 300:
            _recon_s[_i] = np.nan
        else:
            _k1 = _sindy_hat(_s_prev, 0, _xi_s)
            _k2 = _sindy_hat(_s_prev + 0.5 * _dt_s * _k1, 0, _xi_s)
            _k3 = _sindy_hat(_s_prev + 0.5 * _dt_s * _k2, 0, _xi_s)
            _k4 = _sindy_hat(_s_prev + _dt_s * _k3, 0, _xi_s)
            _ns = _s_prev + (_dt_s / 6) * (_k1 + 2*_k2 + 2*_k3 + _k4)
            _recon_s[_i] = _ns if not (np.any(np.isnan(_ns)) or np.max(np.abs(_ns)) > 300) else np.nan

    _n_active_s = int(np.sum(np.any(np.abs(_xi_s) > 1e-8, axis=1)))
    _n_lib_s    = len(_lib_names_s)
    _sparsity_s = round(1 - _n_active_s / _n_lib_s, 3)
    _dX_hat_s   = _Theta_s @ _xi_s
    _rmse_s     = float(np.sqrt(np.mean((_dX_hat_s - _dX_s)**2)))

    def _eq_str(col, names, xi, threshold=1e-8):
        terms = []
        for i, nm in enumerate(names):
            c = xi[i, col]
            if abs(c) > threshold:
                terms.append(f"{c:+.3f} {nm}")
        return " ".join(terms) if terms else "0"

    _eq0 = _eq_str(0, _lib_names_s, _xi_s)
    _eq1 = _eq_str(1, _lib_names_s, _xi_s)

    sindy_data = {
        "states": _states_s,
        "recon":  _recon_s,
        "eq0":    _eq0,
        "eq1":    _eq1,
        "lib_size": _n_lib_s,
        "active":   _n_active_s,
        "sparsity": _sparsity_s,
        "rmse":     _rmse_s,
        "sysname":  _sysname,
    }
    return (sindy_data,)


@app.cell(hide_code=True)
def _(canvas_anim, mo, np, sindy_data):
    _d  = sindy_data
    _st = _d["states"]
    _rc = _d["recon"]
    _valid = ~np.isnan(_rc[:, 0])
    _rc_x = np.where(_valid, _rc[:, 0], _st[:, 0]).tolist()
    _rc_y = np.where(_valid, _rc[:, 1], _st[:, 1]).tolist()
    _all_x = np.concatenate([_st[:, 0], _rc[:, 0][_valid]])
    _all_y = np.concatenate([_st[:, 1], _rc[:, 1][_valid]])
    _xspan = float(_all_x.max() - _all_x.min()) or 1.0
    _yspan = float(_all_y.max() - _all_y.min()) or 1.0
    _xmin  = float(_all_x.min()) - 0.12 * _xspan
    _xmax  = float(_all_x.max()) + 0.12 * _xspan
    _ymin  = float(_all_y.min()) - 0.12 * _yspan
    _ymax  = float(_all_y.max()) + 0.12 * _yspan

    _spec_s = {
        "frames": len(_st), "dt": 40,
        "panels": [
            {
                "kind": "phase_portrait",
                "title": "Phase portrait: true (blue) vs SINDy (orange)",
                "xlabel": "x", "ylabel": "y",
                "series": [
                    {"xs": np.round(_st[:, 0], 3).tolist(), "ys": np.round(_st[:, 1], 3).tolist()},
                    {"xs": _rc_x, "ys": _rc_y},
                ],
                "labels": ["true trajectory", "SINDy reconstruction"],
                "colors": ["#7dd3fc", "#f97316"],
                "xmin": _xmin, "xmax": _xmax, "ymin": _ymin, "ymax": _ymax,
                "trail": 100,
            },
        ],
        "layout": [[0.0, 0.0, 1.0, 1.0]],
    }
    _rc_color = "#57c98a" if _d["rmse"] < 0.1 else ("#e6a356" if _d["rmse"] < 0.5 else "#e0635a")
    mo.vstack([
        mo.md(
            f"**dx/dt =** `{_d['eq0']}`\n\n"
            f"**dy/dt =** `{_d['eq1']}`\n\n"
            f"library {_d['lib_size']} terms | active {_d['active']} | "
            f"sparsity {_d['sparsity']} | RMSE {_d['rmse']:.4f}"
        ),
        canvas_anim(_spec_s, height="480px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **How STLSQ works.** The algorithm builds a library matrix $\Theta$ whose columns
        are all monomials $1, x, y, x^2, xy, \ldots$ up to the chosen degree, evaluated
        at every time sample. The SINDy hypothesis is $\dot{X} = \Theta\,\xi$ with
        $\xi$ sparse. STLSQ starts from a full least-squares solve and iterates: zero
        every coefficient below $\lambda$, re-solve on the surviving columns, repeat
        until the active set is stable. Physical laws are sparse in any reasonable
        basis, so the correct terms survive thresholding while noise-driven terms do not.
        Raise the noise slider to see the point where the finite-difference derivative
        estimate becomes too noisy for any threshold to clean up.
        """
    )
    return


# ── Part 2: Conservation law discovery ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 2: Conservation law discovery (kernel method)")
    return


@app.cell(hide_code=True)
def _(mo):
    cl_sys = mo.ui.dropdown(
        options={"Pendulum": "pendulum", "Kepler orbit": "kepler", "Duffing (conservative)": "duffing_c"},
        value="Pendulum",
        label="system",
    )
    cl_gamma = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="kernel width gamma", show_value=True)
    cl_theta0 = mo.ui.slider(0.2, 2.9, value=1.2, step=0.1, label="pendulum theta0", show_value=True)
    cl_ecc    = mo.ui.slider(0.0, 0.90, value=0.25, step=0.05, label="Kepler eccentricity", show_value=True)
    cl_x0     = mo.ui.slider(0.2, 1.8, value=0.6, step=0.1, label="Duffing x0", show_value=True)
    mo.vstack([
        mo.hstack([cl_sys, cl_gamma], justify="start", gap=2),
        mo.hstack([cl_theta0, cl_ecc, cl_x0], justify="start", gap=2),
    ])
    return cl_ecc, cl_gamma, cl_sys, cl_theta0, cl_x0


@app.cell(hide_code=True)
def _(cl_ecc, cl_gamma, cl_sys, cl_theta0, cl_x0, np, rk4_integrate):
    _cl_sysname = cl_sys.value
    _cl_gamma   = float(cl_gamma.value)

    if _cl_sysname == "pendulum":
        _th0 = float(cl_theta0.value)
        def _cl_ode(s, t, p):
            return np.array([s[1], -9.81 * np.sin(s[0])])
        _cl_u0 = np.array([_th0, 0.0])
        _cl_p  = None
        _cl_t1 = 20.0
        _cl_n  = 600
        def _cl_true_energy(states):
            return 0.5 * states[:, 1]**2 - 9.81 * np.cos(states[:, 0])
    elif _cl_sysname == "kepler":
        _ecc = float(cl_ecc.value)
        _ecc = min(_ecc, 0.92)
        def _cl_ode(s, t, p):
            r = np.hypot(s[0], s[1])
            return np.array([s[2], s[3], -s[0] / r**3, -s[1] / r**3])
        _cl_u0 = np.array([1.0 - _ecc, 0.0, 0.0, np.sqrt((1 + _ecc) / (1 - _ecc))])
        _cl_p  = None
        _cl_t1 = 4 * 2 * np.pi
        _cl_n  = 600
        def _cl_true_energy(states):
            r = np.hypot(states[:, 0], states[:, 1])
            return 0.5 * (states[:, 2]**2 + states[:, 3]**2) - 1.0 / r
    else:
        _x0_c = float(cl_x0.value)
        def _cl_ode(s, t, p):
            return np.array([s[1], s[0] - s[0]**3])
        _cl_u0 = np.array([_x0_c, 0.0])
        _cl_p  = None
        _cl_t1 = 50.0
        _cl_n  = 600
        def _cl_true_energy(states):
            return 0.5 * states[:, 1]**2 + 0.25 * states[:, 0]**4 - 0.5 * states[:, 0]**2

    _cl_states, _cl_dt = rk4_integrate(_cl_ode, _cl_u0, 0.0, _cl_t1, _cl_n, _cl_p)
    _cl_truth = _cl_true_energy(_cl_states)

    # Kernel invariant discovery
    _na = min(200, _cl_n)
    _idx_anchor = np.round(np.linspace(0, _cl_n - 1, _na)).astype(int)
    _anchors = _cl_states[_idx_anchor]
    _Phi = np.exp(-_cl_gamma * np.sum((_cl_states[:, None, :] - _anchors[None, :, :])**2, axis=2))
    _Phi_c = _Phi - _Phi.mean(axis=0, keepdims=True)
    _dPhi  = (_Phi_c[1:] - _Phi_c[:-1]) / _cl_dt
    _A = _dPhi.T @ _dPhi + 1e-5 * np.eye(_na)
    _B = _Phi_c.T @ _Phi_c + 1e-8 * np.eye(_na)
    try:
        from scipy.linalg import eigh
        _evals, _evecs = eigh(_A, _B)
        _c = _evecs[:, 0]
    except Exception:
        _c = np.linalg.lstsq(_B, np.linalg.lstsq(_A, np.eye(_na), rcond=None)[0][:, 0], rcond=None)[0]
    _discovered = _Phi_c @ _c
    _discovered = _discovered - _discovered[0]

    # Polynomial fit to discovered invariant
    _d2 = _cl_states.shape[1]
    _poly_cols = [np.ones(_cl_n)]
    _poly_names = ["1"]
    for _j in range(_d2):
        _poly_cols.append(_cl_states[:, _j])
        _poly_names.append(f"x{_j+1}")
        _poly_cols.append(_cl_states[:, _j]**2)
        _poly_names.append(f"x{_j+1}^2")
    _P = np.column_stack(_poly_cols)
    _beta = np.linalg.lstsq(_P, _discovered, rcond=None)[0]
    _poly_label_parts = [f"{_beta[i]:.3f} {_poly_names[i]}" for i in range(len(_beta)) if abs(_beta[i]) > 1e-4]
    _poly_label = " + ".join(_poly_label_parts) if _poly_label_parts else "0"
    _poly_pred  = _P @ _beta
    _poly_mse   = float(np.mean((_poly_pred - _discovered)**2))

    _E_drift_pct = float((np.max(_cl_truth) - np.min(_cl_truth)) / (abs(np.mean(_cl_truth)) + 1e-14) * 100)
    _disc_std    = float(np.std(_discovered))

    cl_data = {
        "states":    _cl_states,
        "truth":     _cl_truth,
        "discovered": _discovered,
        "sysname":   _cl_sysname,
        "poly_label": _poly_label,
        "poly_mse":   _poly_mse,
        "E_drift_pct": _E_drift_pct,
        "disc_std":   _disc_std,
        "dt":         _cl_dt,
        "n":          _cl_n,
    }
    return cl_data, eigh


@app.cell(hide_code=True)
def _(canvas_anim, cl_data, mo, np):
    _d2   = cl_data
    _st2  = _d2["states"]
    _n2   = _d2["n"]
    _sys2 = _d2["sysname"]
    _frames2 = _n2

    # Build panels depending on system
    _panels2 = []
    _layout2 = []

    if _sys2 == "pendulum":
        _panels2.append({
            "kind": "pendulum_arm",
            "theta": np.round(_st2[:, 0], 4).tolist(),
            "trail": 60,
            "title": "Pendulum",
        })
        _layout2.append([0.0, 0.0, 0.33, 1.0])
        _ps_x = np.round(_st2[:, 0], 4).tolist()
        _ps_y = np.round(_st2[:, 1], 4).tolist()
        _ps_xlabel = "theta"
        _ps_ylabel = "omega"
    elif _sys2 == "kepler":
        _panels2.append({
            "kind": "orbit",
            "x": np.round(_st2[:, 0], 4).tolist(),
            "y": np.round(_st2[:, 1], 4).tolist(),
            "trail": 80,
            "title": "Kepler orbit",
        })
        _layout2.append([0.0, 0.0, 0.33, 1.0])
        _ps_x = np.round(_st2[:, 0], 4).tolist()
        _ps_y = np.round(_st2[:, 2], 4).tolist()
        _ps_xlabel = "x"
        _ps_ylabel = "vx"
    else:
        _ps_x = np.round(_st2[:, 0], 4).tolist()
        _ps_y = np.round(_st2[:, 1], 4).tolist()
        _ps_xlabel = "x"
        _ps_ylabel = "xdot"
        # placeholder panel (just phase portrait)
        _panels2.append({
            "kind": "phase_portrait",
            "title": "Duffing phase space",
            "xlabel": _ps_xlabel, "ylabel": _ps_ylabel,
            "series": [{"xs": _ps_x, "ys": _ps_y}],
            "colors": ["#7dd3fc"],
            "labels": ["trajectory"],
            "xmin": float(np.min(_st2[:, 0])) - 0.2,
            "xmax": float(np.max(_st2[:, 0])) + 0.2,
            "ymin": float(np.min(_st2[:, 1])) - 0.2,
            "ymax": float(np.max(_st2[:, 1])) + 0.2,
            "trail": 80,
        })
        _layout2.append([0.0, 0.0, 0.33, 1.0])

    # Panel B: true energy over time
    _truth_shifted = (_d2["truth"] - _d2["truth"][0]).tolist()
    _panels2.append({
        "kind": "timeseries",
        "ys": [_truth_shifted],
        "colors": ["#57c98a"],
        "labels": ["true E(t) - E(0)"],
        "xlabel": "frame",
        "ylabel": "energy drift",
        "title": "True energy (should be flat)",
    })
    _layout2.append([0.33, 0.0, 0.34, 1.0])

    # Panel C: discovered invariant over time
    _disc_list = _d2["discovered"].tolist()
    _panels2.append({
        "kind": "timeseries",
        "ys": [_disc_list],
        "colors": ["#f97316"],
        "labels": ["kernel phi(t) - phi(0)"],
        "xlabel": "frame",
        "ylabel": "invariant drift",
        "title": "Discovered invariant (tune gamma to flatten)",
    })
    _layout2.append([0.67, 0.0, 0.33, 1.0])

    _spec_c = {"frames": _frames2, "dt": 35, "panels": _panels2, "layout": _layout2}
    _ec = "#57c98a" if _d2["E_drift_pct"] < 0.5 else "#e6a356"
    mo.vstack([
        mo.md(
            f"system: **{_sys2}** | "
            f"energy drift: **{_d2['E_drift_pct']:.3f}%** | "
            f"kernel phi std: **{_d2['disc_std']:.4f}** | "
            f"poly approx: `{_d2['poly_label'][:80]}` | "
            f"poly MSE: {_d2['poly_mse']:.4f}"
        ),
        canvas_anim(_spec_c, height="480px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Kernel invariant discovery.** Sample $n_a$ anchor states uniformly from the
        trajectory. Build the RBF feature matrix
        $\Phi_{ij} = \exp(-\gamma \| x_i - x_j \|^2)$.
        Column-center $\Phi$ to remove the trivial constant. Approximate
        $\dot\Phi$ by finite differences. Solve the generalized eigenproblem
        $\min_{c} c^\top (\dot\Phi^\top \dot\Phi + \alpha I) c$ subject to
        $c^\top (\Phi^\top \Phi) c = 1$: the eigenvector for the smallest eigenvalue
        defines the function $\varphi = \Phi c$ whose temporal drift is minimal.
        A flat $\varphi(t)$ curve means the algorithm found a conserved quantity.
        The polynomial projection in the stats bar shows the discovered function
        is approximately the true Hamiltonian.
        """
    )
    return


# ── Part 3: Symbolic regression (genetic programming) ──────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 3: Symbolic regression via genetic programming")
    return


@app.cell(hide_code=True)
def _(mo):
    gp_eq = mo.ui.dropdown(
        options={
            "Kinetic energy  E = 0.5 m v^2": "ke",
            "Pendulum period  T = 2pi sqrt(L/g)": "period",
            "Ohm law  V = I R": "ohm",
            "Hooke law  F = k x": "hooke",
        },
        value="Kinetic energy  E = 0.5 m v^2",
        label="equation",
    )
    gp_noise = mo.ui.slider(0.0, 0.3, value=0.05, step=0.02, label="noise sigma", show_value=True)
    gp_pop   = mo.ui.slider(30, 120, value=60, step=10, label="population", show_value=True)
    gp_ngen  = mo.ui.slider(10, 60, value=35, step=5, label="generations", show_value=True)
    mo.hstack([gp_eq, gp_noise, gp_pop, gp_ngen], justify="start", gap=2)
    return gp_eq, gp_ngen, gp_noise, gp_pop


@app.cell(hide_code=True)
def _(gp_eq, gp_ngen, gp_noise, gp_pop, np):
    _eq_key  = gp_eq.value
    _gp_sig  = float(gp_noise.value)
    _pop_sz  = int(gp_pop.value)
    _n_gen   = int(gp_ngen.value)
    _rng_gp  = np.random.default_rng(99)
    _N_data  = 120

    # Dataset generation
    if _eq_key == "ke":
        _var_names = ["m", "v"]
        _gp_X = np.column_stack([
            0.1 + 9.9 * _rng_gp.random(_N_data),
            0.1 + 9.9 * _rng_gp.random(_N_data),
        ])
        _gp_y_clean = 0.5 * _gp_X[:, 0] * _gp_X[:, 1]**2
    elif _eq_key == "period":
        _var_names = ["L"]
        _gp_X = (0.1 + 9.9 * _rng_gp.random((_N_data, 1)))
        _gp_y_clean = 2 * np.pi * np.sqrt(_gp_X[:, 0] / 9.81)
    elif _eq_key == "ohm":
        _var_names = ["I", "R"]
        _gp_X = np.column_stack([
            0.001 + 9.999 * _rng_gp.random(_N_data),
            1.0 + 999.0 * _rng_gp.random(_N_data),
        ])
        _gp_y_clean = _gp_X[:, 0] * _gp_X[:, 1]
    else:  # hooke
        _var_names = ["k", "x"]
        _gp_X = np.column_stack([
            1.0 + 999.0 * _rng_gp.random(_N_data),
            0.01 + 0.99 * _rng_gp.random(_N_data),
        ])
        _gp_y_clean = _gp_X[:, 0] * _gp_X[:, 1]

    _sy = max(float(np.std(_gp_y_clean)), 1e-12)
    _gp_y = _gp_y_clean + _gp_sig * _sy * _rng_gp.standard_normal(_N_data)
    _nv   = len(_var_names)

    # Expression tree (pure Python, no classes -- use dicts for Pyodide compatibility)
    def _node(op, value=0.0, var_idx=0, children=None):
        return {"op": op, "value": value, "var_idx": var_idx, "children": children or []}

    def _copy_node(n):
        return _node(n["op"], n["value"], n["var_idx"], [_copy_node(c) for c in n["children"]])

    def _count_nodes(n):
        return 1 + sum(_count_nodes(c) for c in n["children"])

    def _tree_str(n, var_names):
        if n["op"] == "const":
            return f"{n['value']:.3g}"
        if n["op"] == "var":
            i = min(n["var_idx"], len(var_names) - 1)
            return var_names[i]
        ch = n["children"]
        if n["op"] in ("+", "-", "*", "/"):
            return f"({_tree_str(ch[0], var_names)} {n['op']} {_tree_str(ch[1], var_names)})"
        return f"{n['op']}({_tree_str(ch[0], var_names)})"

    _BIN = ["+", "-", "*", "/"]
    _UN  = ["sqrt", "sin", "cos"]

    def _rnd_node(nv, rng, depth=0, max_depth=4):
        if depth >= max_depth or (depth > 1 and rng.random() < 0.4):
            if rng.random() < 0.6:
                return _node("var", var_idx=int(rng.integers(0, nv)))
            return _node("const", value=float(rng.standard_normal() * 2))
        op = _BIN[int(rng.integers(0, len(_BIN)))] if rng.random() < 0.7 else _UN[int(rng.integers(0, len(_UN)))]
        n_ch = 2 if op in _BIN else 1
        return _node(op, children=[_rnd_node(nv, rng, depth + 1, max_depth) for _ in range(n_ch)])

    def _eval_node(n, X):
        op = n["op"]
        if op == "const":
            return np.full(len(X), n["value"])
        if op == "var":
            i = min(n["var_idx"], X.shape[1] - 1)
            return X[:, i]
        ch = n["children"]
        a = _eval_node(ch[0], X)
        if op == "+":
            b = _eval_node(ch[1], X); return a + b
        if op == "-":
            b = _eval_node(ch[1], X); return a - b
        if op == "*":
            b = _eval_node(ch[1], X); return a * b
        if op == "/":
            b = _eval_node(ch[1], X); return a / (np.abs(b) + 1e-10)
        if op == "sqrt":
            return np.sqrt(np.abs(a))
        if op == "sin":
            return np.sin(a)
        if op == "cos":
            return np.cos(a)
        return np.zeros(len(X))

    def _score(n, X, y, cp=0.002):
        try:
            pred = _eval_node(n, X)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e12
            mse = float(np.mean((pred - y)**2))
            var_y = float(np.var(y)) or 1e-12
            return mse / var_y + cp * _count_nodes(n)
        except Exception:
            return 1e12

    def _collect_all(n, acc):
        acc.append(n)
        for c in n["children"]:
            _collect_all(c, acc)

    def _find_rand_parent(n, rng):
        if n["children"] and rng.random() < 0.45:
            i = int(rng.integers(0, len(n["children"])))
            return n, i
        for c in n["children"]:
            p, i = _find_rand_parent(c, rng)
            if p is not None:
                return p, i
        return None, 0

    def _mutate(n, nv, rng, rate=0.2):
        if rng.random() < rate:
            op = n["op"]
            if op == "const":
                n["value"] += float(rng.standard_normal() * 0.3)
            elif op == "var":
                n["var_idx"] = int(rng.integers(0, nv))
            elif op in _BIN:
                n["op"] = _BIN[int(rng.integers(0, len(_BIN)))]
            else:
                n["op"] = _UN[int(rng.integers(0, len(_UN)))]
        for c in n["children"]:
            _mutate(c, nv, rng, rate)

    def _crossover(t1, t2, rng):
        t1 = _copy_node(t1); t2 = _copy_node(t2)
        all_t2 = []
        _collect_all(t2, all_t2)
        if not all_t2:
            return t1
        src = _copy_node(all_t2[int(rng.integers(0, len(all_t2)))])
        p, i = _find_rand_parent(t1, rng)
        if p is not None:
            p["children"][i] = src
        return t1

    # Run GP
    _pop = [_rnd_node(_nv, _rng_gp) for _ in range(_pop_sz)]
    _fits = [_score(t, _gp_X, _gp_y) for t in _pop]
    _best_hist = []
    _elite_frac = max(1, _pop_sz // 20)

    for _gen in range(_n_gen):
        _best_hist.append(min(_fits))
        _ord = np.argsort(_fits)
        _new_pop = [_copy_node(_pop[_ord[i]]) for i in range(_elite_frac)]
        _new_fits = [_fits[_ord[i]] for i in range(_elite_frac)]
        while len(_new_pop) < _pop_sz:
            _i1, _i2 = int(_rng_gp.integers(0, _pop_sz)), int(_rng_gp.integers(0, _pop_sz))
            _p1 = _pop[_i1] if _fits[_i1] < _fits[_i2] else _pop[_i2]
            _i3, _i4 = int(_rng_gp.integers(0, _pop_sz)), int(_rng_gp.integers(0, _pop_sz))
            _p2 = _pop[_i3] if _fits[_i3] < _fits[_i4] else _pop[_i4]
            if _rng_gp.random() < 0.7:
                _child = _crossover(_p1, _p2, _rng_gp)
            else:
                _child = _copy_node(_p1)
            _mutate(_child, _nv, _rng_gp)
            if _count_nodes(_child) > 25:
                _child = _rnd_node(_nv, _rng_gp, max_depth=3)
            _new_pop.append(_child)
            _new_fits.append(_score(_child, _gp_X, _gp_y))
        _pop = _new_pop; _fits = _new_fits

    _best_idx = int(np.argmin(_fits))
    _best_tree = _pop[_best_idx]
    _best_pred = _eval_node(_best_tree, _gp_X)
    _y_bar = float(np.mean(_gp_y))
    _ss_res = float(np.sum((_gp_y - _best_pred)**2))
    _ss_tot = float(np.sum((_gp_y - _y_bar)**2)) or 1e-12
    _r2 = max(0.0, 1.0 - _ss_res / _ss_tot)
    _best_expr = _tree_str(_best_tree, _var_names)
    _complexity = _count_nodes(_best_tree)

    gp_data = {
        "best_hist":  _best_hist,
        "best_expr":  _best_expr,
        "r2":         _r2,
        "complexity": _complexity,
        "eq_key":     _eq_key,
        "y_clean":    _gp_y_clean.tolist(),
        "y_pred":     _best_pred.tolist(),
        "n_gen":      _n_gen,
        "var_names":  _var_names,
    }
    return (gp_data,)


@app.cell(hide_code=True)
def _(canvas_anim, gp_data, mo, np):
    _gd = gp_data
    _hist = _gd["best_hist"]
    _ng   = _gd["n_gen"]

    # Clamp hist for display
    _hist_disp = [min(h, 10.0) for h in _hist]

    # Scatter data for predicted vs true panel as timeseries approximation
    _yc  = np.array(_gd["y_clean"])
    _yp  = np.array(_gd["y_pred"])
    _idx_sort = np.argsort(_yc)
    _yc_s = _yc[_idx_sort].tolist()
    _yp_s = _yp[_idx_sort].tolist()

    _spec_gp = {
        "frames": _ng, "dt": 120,
        "panels": [
            {
                "kind": "timeseries",
                "ys": [_hist_disp],
                "colors": ["#57c98a"],
                "xlabel": "generation",
                "ylabel": "best fitness",
                "title": "GP convergence: best fitness per generation",
                "grow": True,
            },
            {
                "kind": "timeseries",
                "ys": [_yc_s, _yp_s],
                "colors": ["#57c98a", "#f97316"],
                "labels": ["true", "predicted"],
                "xlabel": "sample (sorted by true)",
                "ylabel": "target value",
                "title": f"Predicted vs true (R2 = {_gd['r2']:.3f})",
            },
        ],
        "layout": [
            [0.0, 0.0, 0.5, 1.0],
            [0.5, 0.0, 0.5, 1.0],
        ],
    }
    _r2c = "#57c98a" if _gd["r2"] > 0.9 else ("#e6a356" if _gd["r2"] > 0.6 else "#e0635a")
    mo.vstack([
        mo.md(
            f"**Best expression:** `{_gd['best_expr'][:120]}`\n\n"
            f"R2 = {_gd['r2']:.4f} | complexity = {_gd['complexity']} nodes | "
            f"generations = {_ng} | variables: {', '.join(_gd['var_names'])}"
        ),
        canvas_anim(_spec_gp, height="420px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Genetic programming.** Each candidate formula is an expression tree:
        internal nodes are operators ($+$, $-$, $\times$, $/$, $\sqrt{\cdot}$, $\sin$, $\cos$)
        and leaves are input variables or floating-point constants.
        Fitness is normalised MSE divided by $\text{var}(y)$ -- making the score
        scale-invariant across equations of very different magnitudes -- plus a
        parsimony penalty on node count to prevent bloat.
        Selection is binary tournament: draw two individuals at random, keep the fitter one.
        Offspring are produced by subtree crossover (grafting a random subtree from
        parent 2 onto a random slot in parent 1) or point mutation (changing one
        node's operator or nudging a constant). Elitism copies the top 5% unchanged,
        so the best solution never regresses. The left panel shows fitness converging
        toward zero; the right panel shows predicted vs true values for the best tree.
        """
    )
    return


# ── Part 4: Static comparison figure ───────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 4: Comparison: SINDy coefficient bar chart and polynomial approx")
    return


@app.cell(hide_code=True)
def _(cl_data, gp_data, np, plt, sindy_data):
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(14, 4))
    _fig.patch.set_facecolor("#0f1117")
    for _ax in (_ax1, _ax2, _ax3):
        _ax.set_facecolor("#0f1117")
        _ax.tick_params(colors="#8a93a6", labelsize=8)
        for _sp in _ax.spines.values():
            _sp.set_color("#1e2535")
        _ax.xaxis.label.set_color("#8a93a6")
        _ax.yaxis.label.set_color("#8a93a6")
        _ax.title.set_color("#c8cdd8")

    # Panel A: SINDy xi for equation 1 (dx/dt)
    _sd = sindy_data
    _st_s = _sd["states"]
    _n_pts_ax = _st_s.shape[0]
    _deg_ax = 3  # fixed for comparison
    _cols_ax = [np.ones(_n_pts_ax)]
    _nms_ax = ["1"]
    for _total in range(1, _deg_ax + 1):
        for _e0 in range(_total + 1):
            _e1 = _total - _e0
            _cols_ax.append(_st_s[:, 0]**_e0 * _st_s[:, 1]**_e1)
            _nm = ("" if _e0 == 0 else ("x" if _e0 == 1 else f"x^{_e0}")) + \
                  ("" if _e1 == 0 else ("y" if _e1 == 1 else f"y^{_e1}"))
            _nms_ax.append(_nm or "1")
    _Th_ax = np.column_stack(_cols_ax)
    _dX_ax  = (_st_s[2:] - _st_s[:-2]) / (2 * _sd["states"].shape[0] / 800 * 0.05)
    # just use stored xi from sindy_data via recomputing at deg=3
    # Actually stlsq is already done -- reuse from sindy_data by re-importing

    # Simpler: plot bar chart of xi coefficients from stored states
    # Recompute for display purposes only
    from numpy.linalg import lstsq as _lstsq_ax
    _dXa = np.gradient(_st_s, axis=0)  # crude but just for display
    _xi_ax = _lstsq_ax(_Th_ax, _dXa, rcond=None)[0]
    _thresh_ax = 0.1
    _mask_ax = np.abs(_xi_ax[:, 0]) > _thresh_ax
    _colors_bar = ["#57c98a" if _mask_ax[i] else "#2f3645" for i in range(len(_nms_ax))]
    _ax1.bar(range(len(_nms_ax)), np.abs(_xi_ax[:, 0]), color=_colors_bar, width=0.7)
    _ax1.set_xticks(range(len(_nms_ax)))
    _ax1.set_xticklabels(_nms_ax, rotation=45, ha="right", fontsize=7)
    _ax1.set_title(f"SINDy |xi| for dx/dt  ({_sd['sysname']})", fontsize=9)
    _ax1.set_ylabel("|coefficient|", fontsize=8)

    # Panel B: conservation law -- true energy vs discovered invariant (normalised)
    _cd = cl_data
    _E0 = _cd["truth"] - _cd["truth"][0]
    _D0 = _cd["discovered"]
    _E_sc = max(float(np.max(np.abs(_E0))), 1e-10)
    _D_sc = max(float(np.max(np.abs(_D0))), 1e-10)
    _t_ax = np.arange(_cd["n"])
    _ax2.plot(_t_ax, _E0 / _E_sc, color="#57c98a", linewidth=1.2, alpha=0.85, label="true E")
    _ax2.plot(_t_ax, _D0 / _D_sc, color="#f97316", linewidth=1.2, alpha=0.85, label="kernel phi")
    _ax2.axhline(0, color="#1e2535", linewidth=0.7, linestyle="--")
    _ax2.set_title(f"CL: true E vs kernel phi  ({_cd['sysname']})", fontsize=9)
    _ax2.set_xlabel("frame", fontsize=8); _ax2.set_ylabel("normalised drift", fontsize=8)
    _ax2.legend(fontsize=7, facecolor="#0f1117", edgecolor="#2f3645", labelcolor="#8a93a6")

    # Panel C: GP predicted vs true
    _gpd = gp_data
    _yc_ax = np.array(_gpd["y_clean"])
    _yp_ax = np.array(_gpd["y_pred"])
    _lo_ax = min(float(_yc_ax.min()), float(_yp_ax.min()))
    _hi_ax = max(float(_yc_ax.max()), float(_yp_ax.max()))
    _ax3.scatter(_yc_ax, _yp_ax, color="#7dd3fc", s=6, alpha=0.6)
    _ax3.plot([_lo_ax, _hi_ax], [_lo_ax, _hi_ax], color="#2f3645", linewidth=1.0, linestyle="--")
    _ax3.set_title(f"GP: pred vs true  R2={_gpd['r2']:.3f}  ({_gpd['eq_key']})", fontsize=9)
    _ax3.set_xlabel("true", fontsize=8); _ax3.set_ylabel("predicted", fontsize=8)

    plt.tight_layout(pad=1.2)
    plt.gca()
    _fig
    return


if __name__ == "__main__":
    app.run()
