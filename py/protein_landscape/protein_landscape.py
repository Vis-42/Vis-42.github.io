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
        # Protein folding: the Gō model and its free-energy landscape

        A 24-bead chain in 2D, simulated by Metropolis Monte Carlo. The Hamiltonian is a Gō
        model: pairs of beads that touch in the native structure attract through a 12-10
        Lennard-Jones well of depth $\varepsilon$, and every other non-bonded pair only repels.
        Because only native contacts attract, the native fold is the single lowest-energy
        state and the landscape is a funnel rather than a rugged maze.

        The reaction coordinate is $Q$, the fraction of native contacts formed. The free
        energy $F(Q) = -k_BT \ln P(Q)$ comes from the histogram of $Q$ over many independent
        replicas once they have equilibrated. Near the folding temperature it has two basins,
        folded and unfolded, separated by a barrier: the signature of two-state folding.
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
    T_sl = mo.ui.slider(0.10, 0.60, value=0.27, step=0.01, label="temperature k_BT / ε", show_value=True)
    start_sl = mo.ui.dropdown(["native", "extended"], value="native", label="animated chain starts")
    nst_sl = mo.ui.slider(1000, 6000, value=3000, step=500, label="MC steps", show_value=True)
    mo.md(
        f"""
        **Controls.** Temperature is in units of the contact energy ε, so ε itself is not a
        separate knob. Below the folding temperature (marked on the melting curve, about 0.27)
        the native state is stable; above it the chain unfolds. The animated chain can start
        folded or fully extended; the free energy always comes from 12 replicas started in the
        native state.

        {mo.hstack([T_sl, start_sl, nst_sl], justify="start", gap=2)}
        """
    )
    return T_sl, nst_sl, start_sl


@app.cell(hide_code=True)
def _(np):
    # Native fold: a compact meander, four rows of six beads with bond length A, like a
    # four-stranded sheet. Its 15 native contacts are the beads facing each other across
    # neighbouring rows.
    A, ROWS, COLS = 0.3, 4, 6
    N_BEADS = ROWS * COLS
    NATIVE = np.array([[(c if r % 2 == 0 else COLS - 1 - c) * A, r * A]
                       for r in range(ROWS) for c in range(COLS)])
    NATIVE = NATIVE - NATIVE.mean(0)
    _I, _J = np.triu_indices(N_BEADS, 3)                 # non-bonded pairs, |i - j| >= 3
    _dn = np.linalg.norm(NATIVE[_I] - NATIVE[_J], axis=1)
    _is_nat = _dn < 1.05 * A
    CONTACTS = np.c_[_I[_is_nat], _J[_is_nat]]
    RN = _dn[_is_nat]
    _rn_all = np.where(_is_nat, _dn, 1.0)
    K_BOND, SIGMA = 100.0, 0.22

    def go_energy(pos):
        """Energy of conformations pos with shape (..., N_BEADS, 2), in units of ε."""
        bl = np.linalg.norm(np.diff(pos, axis=-2), axis=-1)
        e_bond = 0.5 * K_BOND * ((bl - A) ** 2).sum(-1)
        d = np.maximum(np.linalg.norm(pos[..., _I, :] - pos[..., _J, :], axis=-1), 0.05)
        x = _rn_all / d
        e_pair = np.where(_is_nat, 5 * x ** 12 - 6 * x ** 10, (SIGMA / d) ** 12)
        return e_bond + e_pair.sum(-1)

    def frac_Q(pos):
        d = np.linalg.norm(pos[..., CONTACTS[:, 0], :] - pos[..., CONTACTS[:, 1], :], axis=-1)
        return (d < 1.2 * RN).mean(-1)

    def mc_run(pos, T, n_steps, rng, store_every=10):
        """Metropolis on R replicas at once: pos (R, N, 2), T scalar or shape (R,).
        Each step, every replica proposes either a single-bead nudge (70%) or a pivot (30%):
        the shorter end of the chain is rotated rigidly about a random bead. Pivots move
        whole segments at once, which local moves alone would take ages to do."""
        R = pos.shape[0]; ar = np.arange(R); idx = np.arange(N_BEADS)[None, :]
        T = np.broadcast_to(np.asarray(T, float), (R,))
        E = go_energy(pos); out_pos, out_Q, out_E = [], [], []
        for s in range(n_steps):
            trial = pos.copy()
            pivot = rng.random(R) < 0.3
            i = rng.integers(N_BEADS, size=R)
            trial[ar, i] += np.where(pivot[:, None], 0.0, 0.07 * rng.standard_normal((R, 2)))
            k = rng.integers(1, N_BEADS - 1, size=R)
            th = rng.normal(0.0, 0.6, R); c, sn = np.cos(th)[:, None], np.sin(th)[:, None]
            side = np.where((k < N_BEADS // 2)[:, None], idx < k[:, None], idx > k[:, None]) & pivot[:, None]
            o = trial[ar, k][:, None, :]; rel = trial - o
            rot = o + np.stack([c * rel[..., 0] - sn * rel[..., 1], sn * rel[..., 0] + c * rel[..., 1]], -1)
            trial = np.where(side[..., None], rot, trial)
            Et = go_energy(trial)
            ok = (Et <= E) | (rng.random(R) < np.exp(-np.clip((Et - E) / T, 0, 700)))
            pos = np.where(ok[:, None, None], trial, pos); E = np.where(ok, Et, E)
            if s % store_every == 0:
                out_pos.append(pos[0].copy()); out_Q.append(frac_Q(pos)); out_E.append(E.copy())
        return np.array(out_pos), np.array(out_Q), np.array(out_E)

    def extended_chain():
        return np.c_[np.arange(N_BEADS) * A - (N_BEADS - 1) * A / 2, np.zeros(N_BEADS)]

    return A, CONTACTS, NATIVE, N_BEADS, RN, extended_chain, mc_run


@app.cell(hide_code=True)
def _(NATIVE, mc_run, np):
    # Melting curve: 3 replicas at each of 9 temperatures, all in one batch, started native.
    _temps = np.linspace(0.12, 0.52, 9)
    _T = np.repeat(_temps, 3)
    _, _Q, _ = mc_run(np.repeat(NATIVE[None], len(_T), 0), _T, 3000, np.random.default_rng(1))
    _mq = _Q[len(_Q) // 2:].mean(0).reshape(len(_temps), 3).mean(1)
    # folding temperature: where <Q> crosses 1/2
    _k = int(np.argmax(_mq < 0.5)) if np.any(_mq < 0.5) else len(_mq) - 1
    _Tf = float(np.interp(0.5, [_mq[_k], _mq[_k - 1]], [_temps[_k], _temps[_k - 1]])) if _k > 0 else float(_temps[0])
    melt = {"T": _temps.tolist(), "Q": _mq.tolist(), "Tf": _Tf}
    return (melt,)


@app.cell(hide_code=True)
def _(NATIVE, T_sl, extended_chain, mc_run, nst_sl, np, start_sl):
    _T = float(T_sl.value); _n = int(nst_sl.value); _R = 12
    _rng = np.random.default_rng(42)
    _pos0 = np.repeat(NATIVE[None], _R + 1, 0)
    if start_sl.value == "extended":
        _pos0[0] = extended_chain() + 0.01 * _rng.standard_normal(NATIVE.shape)
    _store = max(1, _n // 300)
    _frames, _Q, _E = mc_run(_pos0, _T, _n, _rng, store_every=_store)

    # F(Q) from the 12 native-started replicas (index 1..12), second half of the run only
    _q_eq = _Q[len(_Q) // 2:, 1:].ravel()
    _edges = np.linspace(0, 1, 17); _cnt, _ = np.histogram(_q_eq, _edges)
    _centres = 0.5 * (_edges[1:] + _edges[:-1]); _m = _cnt > 0
    _F = -_T * np.log(_cnt[_m] / _cnt.sum()); _F -= _F.min()
    _qb, _Fv = _centres[_m], _F

    # barrier: needs a basin below Q = 0.5 and one above 0.6
    _lo = _qb < 0.5; _hi = _qb > 0.6
    if _lo.any() and _hi.any():
        _ilo = np.where(_lo)[0][np.argmin(_Fv[_lo])]; _ihi = np.where(_hi)[0][np.argmin(_Fv[_hi])]
        _top = _Fv[_ilo:_ihi + 1].max()
        _barrier = float(_top - max(_Fv[_ilo], _Fv[_ihi]))
        _dF = float(_Fv[_ihi] - _Fv[_ilo])
        _two_state = _barrier > 0.2 * _T
    else:
        _barrier = _dF = 0.0; _two_state = False

    go_result = {
        "frames": _frames, "Q_anim": _Q[:, 0], "E_anim": _E[:, 0],
        "Q_bins": _qb.tolist(), "F_vals": _Fv.tolist(),
        "mean_Q": float(_q_eq.mean()), "barrier": _barrier, "dF": _dF,
        "two_state": bool(_two_state), "T": _T, "start": start_sl.value,
    }
    return (go_result,)


@app.cell(hide_code=True)
def _(go_result, melt, mo):
    _g = go_result
    _mQ = _g["mean_Q"]
    _col = "#57c98a" if _mQ > 0.6 else ("#e6a356" if _mQ > 0.35 else "#8a92a3")
    _state = "folded" if _mQ > 0.6 else ("coexistence" if _mQ > 0.35 else "unfolded")
    _bar = (f"{_g['barrier']:.2f} ε" if _g["two_state"] else "one basin")
    _cells = [
        ("state", _state, _col),
        ("mean Q (replicas)", f"{_mQ:.2f}", "#eef0f5"),
        ("T / T_f", f"{_g['T'] / melt['Tf']:.2f}  (T_f ≈ {melt['Tf']:.2f})", "#eef0f5"),
        ("barrier ΔF‡", _bar, "#eef0f5"),
        ("F(folded) − F(unfolded)", f"{_g['dF']:+.2f} ε" if _g["two_state"] else "n/a", "#eef0f5"),
    ]
    _html = "".join(
        f'<div style="padding:12px 16px;{"border-right:1px solid #2f3645;" if i < len(_cells) - 1 else ""}">'
        f'<div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">{l}</div>'
        f'<div style="font-size:14px;color:{c};font-weight:600;font-family:ui-monospace,monospace;">{v}</div></div>'
        for i, (l, v, c) in enumerate(_cells))
    mo.md(
        '<div style="background:#161a23;border:1px solid #2f3645;border-radius:8px;overflow:hidden;'
        'margin:4px 0 8px;font-family:ui-sans-serif,system-ui,sans-serif;display:grid;'
        f'grid-template-columns:repeat(5,1fr);">{_html}</div>'
    )
    return


@app.cell(hide_code=True)
def _(A, CONTACTS, N_BEADS, RN, canvas_anim, go_result, melt, mo, np):
    _d = go_result
    _fr = _d["frames"]; _NF = len(_fr)
    _Q = np.round(_d["Q_anim"], 3).tolist()
    _span = (A * (N_BEADS - 1) + 0.4) if _d["start"] == "extended" else 2.4
    _spec = {
        "frames": _NF, "dt": 70,
        "panels": [
            {
                "kind": "chain", "title": f"Chain at T = {_d['T']:.2f} (red: native contacts formed)",
                "x": np.round(_fr[:, :, 0], 3).tolist(), "y": np.round(_fr[:, :, 1], 3).tolist(),
                "native": CONTACTS.tolist(), "rn": np.round(RN, 4).tolist(), "span": _span,
            },
            {
                "kind": "timeseries", "ys": [_Q], "colors": ["#7dd3fc"],
                "ylabel": "Q", "title": "Q(t), fraction of native contacts", "xlabel": "MC sample",
                "dot": 0, "grow": True, "ymin": 0.0, "ymax": 1.0,
            },
            {
                "kind": "freeenergy", "title": "F(Q) from 12 replicas", "xlabel": "Q", "ylabel": "F (ε)",
                "Q_bins": _d["Q_bins"], "F_vals": _d["F_vals"], "Q_now": _Q,
            },
            {
                "kind": "scatter_vline", "title": "Melting curve <Q> vs T", "xlabel": "k_BT / ε", "ylabel": "<Q>",
                "ymin": 0.0, "ymax": 1.05, "x": melt["T"], "y": melt["Q"],
                "dot_x": [_d["T"]] * _NF, "dot_y": _Q,
            },
        ],
        "layout": [
            [0.00, 0.00, 0.40, 1.00],
            [0.40, 0.00, 0.60, 0.34],
            [0.40, 0.34, 0.30, 0.66],
            [0.70, 0.34, 0.30, 0.66],
        ],
    }
    mo.vstack([
        mo.md(
            "**Live animation.** Left: the chain, with native contacts drawn in red while they are "
            "formed. Right: Q(t) for the animated chain; F(Q) with an orange dot at its current Q; "
            "and the melting curve with the current temperature marked. All panels share one clock."
        ),
        canvas_anim(_spec, height="540px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Model details.** The native structure is a meander: four rows of six beads with bond
        length 0.3, like a small four-stranded sheet. The Hamiltonian has three terms:

        1. Harmonic bonds of rest length 0.3 and stiffness 100 between consecutive beads.
        2. For each of the 15 native contacts (beads facing each other across neighbouring rows),
           a 12-10 well $\varepsilon\,[5(r_n/r)^{12} - 6(r_n/r)^{10}]$ with its minimum, $-\varepsilon$,
           at the native distance $r_n$.
        3. For every other pair with $|i-j| \ge 3$, pure repulsion $\varepsilon(0.22/r)^{12}$.

        A contact counts as formed when its beads are within $1.2\,r_n$. Each Monte Carlo step
        proposes either a small move of one bead or a pivot that rotates one end of the chain
        about a bead, and accepts it with probability $\min(1, e^{-\Delta E/k_BT})$.

        **Reading the panels.** The melting curve comes from 27 replicas at nine temperatures,
        each started folded, with the first half of each run discarded. $\langle Q\rangle$ drops
        steeply around $T_f \approx 0.27$: the chain melts cooperatively instead of losing its
        contacts one at a time. At the chosen temperature, $F(Q)$ uses 12 more replicas the same
        way. Well below $T_f$ it has one basin near $Q = 1$; well above, one basin at low $Q$;
        near $T_f$ the two basins coexist with a barrier between them, and the animated chain
        can be seen hopping across it.

        **What this does not show.** Folding from a fully extended chain. Set the animated chain
        to start extended and lower the temperature: it collapses and forms some contacts, then
        stalls at $Q \approx 0.3$ to $0.5$ with a partly zipped structure that cannot rearrange.
        Escaping those traps takes far more steps than a browser can run in a few seconds, so
        the free energy here is measured from the folded side, as in a melting simulation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
