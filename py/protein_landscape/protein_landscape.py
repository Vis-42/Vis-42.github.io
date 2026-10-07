import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Protein Folding")


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
        # Protein folding: the Go model and free-energy funnels

        A 24-bead chain in 2D explores configuration space by Metropolis Monte Carlo.
        The Hamiltonian is a Go model: pairs of beads that are close in the native structure
        feel an attractive 12-10 Lennard-Jones well of depth epsilon; all other non-bonded
        pairs see only a repulsive r^-12 excluded-volume wall. Because only native contacts
        attract, the landscape is a single funnel toward the native state rather than a rugged
        multi-minima landscape.

        Q = (native contacts formed) / (total native contacts) is the reaction coordinate.
        F(Q) = -kT ln P(Q) is recovered by Boltzmann inversion of the sampled histogram.
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

    def canvas_anim(spec, height="500px"):
        """Animate a spec (format: see the top of py/_shared/canvas.html)."""
        return mo.iframe(_engine.replace("__SPEC__", json.dumps(spec)), height=height)

    return (canvas_anim,)


@app.cell(hide_code=True)
def _(mo):
    T_sl   = mo.ui.slider(0.3, 3.0, value=0.7,  step=0.1,  label="temperature T (kB units)", show_value=True)
    eps_sl = mo.ui.slider(0.2, 3.0, value=1.0,  step=0.2,  label="well depth epsilon",        show_value=True)
    nst_sl = mo.ui.slider(100, 2000, value=800, step=100,  label="MC steps",                  show_value=True)
    mo.md(
        f"""
        **Controls.** Temperature T in units of epsilon/kB. Below the folding temperature
        (~0.8 for this model) the chain collapses to the native state. Epsilon sets the depth
        of native-contact wells; higher epsilon strengthens the funnel. MC steps controls how
        long the trajectory runs (more steps = better sampling of F(Q)).

        {mo.hstack([T_sl, eps_sl, nst_sl], justify="start", gap=2)}
        """
    )
    return T_sl, eps_sl, nst_sl


@app.cell(hide_code=True)
def _(np):
    def _make_native(n=24):
        # compact 2D spiral native structure -- same as before but kept as a helper
        t = np.linspace(0, 3.0 * np.pi, n)
        r = 0.28 + 0.042 * t
        pos = np.c_[r * np.cos(t), r * np.sin(t)]
        return pos - pos.mean(0)

    def _native_contacts(nat, sep=3, cutoff=0.55):
        n = len(nat)
        C = []
        for i in range(n):
            for j in range(i + sep, n):
                if np.linalg.norm(nat[i] - nat[j]) < cutoff:
                    C.append((i, j))
        return C

    def _go_energy(pos, nat, C, r0=0.28, k_bond=80.0, eps=1.0, sigma=0.15):
        # backbone harmonic bonds
        d = pos[1:] - pos[:-1]
        E = 0.5 * k_bond * np.sum((np.linalg.norm(d, axis=1) - r0) ** 2)
        # native contact attractions (12-10 LJ)
        nc_set = set(C)
        for i, j in C:
            r = max(np.linalg.norm(pos[i] - pos[j]), 1e-6)
            rn = max(np.linalg.norm(nat[i] - nat[j]), 0.05)
            x = rn / r
            E += eps * (5 * x ** 12 - 6 * x ** 10)
        # non-native excluded volume (r^-12)
        n = len(pos)
        for i in range(n):
            for j in range(i + 3, n):
                if (i, j) not in nc_set:
                    r = max(np.linalg.norm(pos[i] - pos[j]), 0.02)
                    E += eps * (sigma / r) ** 12
        return E

    def _frac_Q(pos, nat, C, tol=1.25):
        if not C:
            return 0.0
        formed = sum(
            np.linalg.norm(pos[i] - pos[j]) < tol * max(np.linalg.norm(nat[i] - nat[j]), 0.05)
            for i, j in C
        )
        return formed / len(C)

    def _free_energy(Q_traj, T, n_bins=20):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = np.zeros(n_bins)
        for q in Q_traj:
            idx = min(int(q * n_bins), n_bins - 1)
            counts[idx] += 1
        counts += 0.5  # Jeffreys prior to avoid log(0)
        P = counts / counts.sum()
        F = -T * np.log(P)
        F -= F.min()
        return centers.tolist(), F.tolist()

    def run_go_mc(T, n_steps, eps, seed=42):
        rng = np.random.default_rng(seed)
        nat = _make_native(24)
        C = _native_contacts(nat)
        n = len(nat)
        # start from extended chain to see folding
        pos = np.c_[np.linspace(-1.5, 1.5, n), 0.05 * rng.standard_normal(n)]
        E = _go_energy(pos, nat, C, eps=eps)
        store_every = max(1, n_steps // 300)
        frame_xs, frame_ys, Q_traj, E_traj = [], [], [], []
        step_size = 0.10
        for s in range(n_steps):
            i = int(rng.integers(n))
            trial = pos.copy()
            trial[i] += step_size * rng.standard_normal(2)
            Et = _go_energy(trial, nat, C, eps=eps)
            dE = Et - E
            if dE <= 0 or rng.random() < np.exp(-dE / max(T, 1e-6)):
                pos = trial
                E = Et
            if s % store_every == 0:
                frame_xs.append(np.round(pos[:, 0], 3).tolist())
                frame_ys.append(np.round(pos[:, 1], 3).tolist())
                Q_traj.append(_frac_Q(pos, nat, C))
                E_traj.append(float(E))
        Q_bins, F_vals = _free_energy(Q_traj, T)
        return {
            "nat": nat,
            "C": C,
            "frame_xs": frame_xs,
            "frame_ys": frame_ys,
            "Q_traj": np.array(Q_traj),
            "E_traj": np.array(E_traj),
            "Q_bins": Q_bins,
            "F_vals": F_vals,
        }

    return run_go_mc,


@app.cell(hide_code=True)
def _(T_sl, eps_sl, nst_sl, run_go_mc):
    _T   = float(T_sl.value)
    _eps = float(eps_sl.value)
    _nst = int(nst_sl.value)
    go_result = run_go_mc(_T, _nst, _eps, seed=42)
    return (go_result,)


@app.cell(hide_code=True)
def _(T_sl, eps_sl, go_result, mo, np):
    _Q  = go_result["Q_traj"]
    _E  = go_result["E_traj"]
    _F  = go_result["F_vals"]
    _T  = float(T_sl.value)
    _eps = float(eps_sl.value)

    _mQ  = float(np.mean(_Q))
    _sQ  = float(np.std(_Q))
    _mE  = float(np.mean(_E))
    _dF  = float(max(_F) - min(_F))
    _nc  = len(go_result["C"])

    _col = "#57c98a" if _mQ > 0.6 else ("#e6a356" if _mQ > 0.4 else "#8a92a3")

    mo.md(
        f"""
        <div style="background:#161a23;border:1px solid #2f3645;border-radius:8px;
                    overflow:hidden;margin:4px 0 8px;font-family:ui-sans-serif,system-ui,sans-serif;
                    display:grid;grid-template-columns:repeat(5,1fr);">
          <div style="padding:12px 16px;border-right:1px solid #2f3645;">
            <div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">mean Q</div>
            <div style="font-size:14px;color:{_col};font-weight:600;font-family:ui-monospace,monospace;">{_mQ:.3f}</div>
          </div>
          <div style="padding:12px 16px;border-right:1px solid #2f3645;">
            <div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">sigma(Q)</div>
            <div style="font-size:14px;color:#eef0f5;font-family:ui-monospace,monospace;">{_sQ:.3f}</div>
          </div>
          <div style="padding:12px 16px;border-right:1px solid #2f3645;">
            <div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">mean E</div>
            <div style="font-size:14px;color:#eef0f5;font-family:ui-monospace,monospace;">{_mE:.3g}</div>
          </div>
          <div style="padding:12px 16px;border-right:1px solid #2f3645;">
            <div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">native contacts</div>
            <div style="font-size:14px;color:#eef0f5;font-family:ui-monospace,monospace;">{_nc}</div>
          </div>
          <div style="padding:12px 16px;">
            <div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">dF barrier</div>
            <div style="font-size:14px;color:#eef0f5;font-family:ui-monospace,monospace;">{_dF:.3f}</div>
          </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(canvas_anim, go_result, mo, np):
    _d  = go_result
    _NF = len(_d["frame_xs"])
    _Q  = _d["Q_traj"].tolist()
    _E  = _d["E_traj"].tolist()

    _nat_x = np.round(_d["nat"][:, 0], 4).tolist()
    _nat_y = np.round(_d["nat"][:, 1], 4).tolist()

    _E_min = float(min(_E))
    _E_max = float(max(_E))
    _E_pad = (_E_max - _E_min) * 0.05 + 0.1

    _spec = {
        "frames": _NF, "dt": 80,
        "panels": [
            {
                "kind": "chain", "title": "Chain conformation",
                "x": _d["frame_xs"],
                "y": _d["frame_ys"],
                "nat_x": _nat_x,
                "nat_y": _nat_y,
                "native_contacts": _d["C"],
                "Q_per_frame": [round(float(q), 3) for q in _Q],
            },
            {
                "kind": "timeseries",
                "ys": [[round(float(q), 4) for q in _Q]],
                "colors": ["#7dd3fc"],
                "ylabel": "Q", "title": "Q(t) — fraction native contacts", "xlabel": "MC sample",
                "dot": 0,
                "grow": True,
                "ymin": 0.0,
                "ymax": 1.0,
            },
            {
                "kind": "freeenergy", "title": "Free energy F(Q)  (kT units)", "xlabel": "Q (native contacts fraction)", "ylabel": "F (kT)",
                "Q_bins": _d["Q_bins"],
                "F_vals": _d["F_vals"],
                "Q_now": [round(float(q), 4) for q in _Q],
            },
            {
                "kind": "timeseries",
                "ys": [[round(float(e), 4) for e in _E]],
                "colors": ["#f87171"],
                "ylabel": "E", "title": "Energy E(t)", "xlabel": "MC sample",
                "dot": 0,
                "grow": True,
                "ymin": _E_min - _E_pad,
                "ymax": _E_max + _E_pad,
            },
        ],
        "layout": [
            [0.00, 0.00, 0.42, 1.00],
            [0.42, 0.00, 0.58, 0.34],
            [0.42, 0.34, 0.58, 0.33],
            [0.42, 0.67, 0.58, 0.33],
        ],
    }
    mo.vstack([
        mo.md(
            "**Live animation.** Left: chain beads coloured by local Q (blue = non-native contacts, "
            "red = native contacts formed). Dashed cyan lines show currently formed native contacts. "
            "Right panels: Q(t) time series (top), F(Q) free energy with amber dot tracking current Q (middle), "
            "and E(t) energy (bottom). All panels share one clock."
        ),
        canvas_anim(_spec, height="520px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Model details.** The 24-bead chain lives in 2D. The native structure is a compact
        Archimedean spiral. The Go Hamiltonian has three terms:

        1. Backbone harmonic spring with equilibrium length r0 = 0.28 and stiffness k = 80,
           keeping adjacent beads connected.

        2. Native-contact attraction: a 12-10 Lennard-Jones well of depth epsilon for each
           pair (i, j) with |i-j| >= 3 that are within 0.55 units in the native structure.
           The potential is eps*(5*(rn/r)^12 - 6*(rn/r)^10), with minimum at r = rn.

        3. Non-native excluded volume: a purely repulsive (sigma/r)^12 term for all other
           non-bonded pairs. No attraction -- this asymmetry is the whole Go model.

        Metropolis MC picks a bead at random, proposes a Gaussian displacement, and accepts
        with probability min(1, exp(-dE/T)). Q is histogrammed over the trajectory and
        F(Q) = -T * ln P(Q) gives the free energy profile. The Jeffreys prior (+0.5 per bin)
        prevents log(0) in sparsely sampled regions.

        At low T the chain folds (Q -> 1, F(Q) slopes down to Q=1). At high T the chain is
        denatured (Q fluctuates near 0, flat F(Q)). Near the folding temperature a barrier
        appears in F(Q) at intermediate Q, giving cooperative two-state folding kinetics.
        """
    )
    return


if __name__ == "__main__":
    app.run()
