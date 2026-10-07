import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Causal Inference")


@app.cell(hide_code=True)
def _():
    import json
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return json, mo, np, plt


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
        # Causal inference in dynamical systems

        Three tools, each built for a regime the others miss.

        **Transfer entropy** measures directed information flow: how much does $X$'s past
        reduce uncertainty about $Y$'s future? It works well for stochastic, noisy systems
        where standard correlation is blind to direction.

        **Convergent cross-mapping** (CCM) recovers coupling direction in deterministic,
        weakly coupled systems, exactly where TE and Granger causality fail, by asking
        whether $X$'s shadow manifold can reconstruct $Y$'s history.

        **Causal emergence** asks an orthogonal question: at what scale does the causal
        structure live? Coarse-graining a system sometimes produces a macro description
        that carries *more* causal power than the microdynamics beneath it.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 1: Transfer entropy")
    return


@app.cell(hide_code=True)
def _(mo):
    eps_te = mo.ui.slider(0.0, 1.0,  value=0.3,  step=0.05, label="coupling ε",    show_value=True)
    sig_te = mo.ui.slider(0.0, 0.3,  value=0.05, step=0.025, label="noise σ",     show_value=True)
    N_te   = mo.ui.slider(200, 2000, value=800,  step=100,   label="series length N", show_value=True)
    mo.hstack([eps_te, sig_te, N_te], justify="start", gap=2)
    return N_te, eps_te, sig_te


@app.cell(hide_code=True)
def _(N_te, eps_te, np, sig_te):
    def _henon_coupled(N, eps, sigma, rng):
        x = np.zeros(N); y = np.zeros(N)
        x[0] = 0.1; y[0] = 0.1
        for i in range(N - 1):
            nx = x[i] + sigma * rng.standard_normal()
            ny = y[i] + sigma * rng.standard_normal()
            x[i+1] = np.clip(1 - 1.4 * nx**2 + 0.3 * ny, -3, 3)
            y[i+1] = np.clip(1 - (1 - eps) * 1.4 * ny**2 - eps * 1.4 * nx * ny + 0.3 * nx, -3, 3)
        return x, y

    def _te_hist(x, y, bins=6):
        # TE(X→Y): how much does x_n reduce uncertainty about y_{n+1} given y_n
        xn = x[:-1]; yn = y[:-1]; yn1 = y[1:]
        exy = np.column_stack([np.digitize(xn, np.linspace(xn.min(), xn.max()+1e-9, bins+1)[1:-1]),
                                np.digitize(yn, np.linspace(yn.min(), yn.max()+1e-9, bins+1)[1:-1]),
                                np.digitize(yn1, np.linspace(yn1.min(), yn1.max()+1e-9, bins+1)[1:-1])])
        def _mi(a, b):
            pab = {}
            for r in zip(a, b): pab[r] = pab.get(r, 0) + 1
            pa = {}; pb = {}
            for (ai, bi), c in pab.items():
                pa[ai] = pa.get(ai, 0) + c; pb[bi] = pb.get(bi, 0) + c
            n = len(a); s = 0
            for (ai, bi), c in pab.items():
                if c > 0: s += (c/n) * np.log2((c/n) / ((pa[ai]/n) * (pb[bi]/n)))
            return s
        # TE = I(Y_{n+1}; X_n | Y_n) = I(Y_{n+1}; X_n, Y_n) - I(Y_{n+1}; Y_n)
        te_xy = max(0.0, _mi(exy[:, 2], exy[:, 0] * bins + exy[:, 1]) -
                         _mi(exy[:, 2], exy[:, 1]))
        # TE(Y→X)
        xn2 = y[:-1]; yn2 = x[:-1]; yn12 = x[1:]
        exy2 = np.column_stack([np.digitize(xn2, np.linspace(xn2.min(), xn2.max()+1e-9, bins+1)[1:-1]),
                                  np.digitize(yn2, np.linspace(yn2.min(), yn2.max()+1e-9, bins+1)[1:-1]),
                                  np.digitize(yn12, np.linspace(yn12.min(), yn12.max()+1e-9, bins+1)[1:-1])])
        te_yx = max(0.0, _mi(exy2[:, 2], exy2[:, 0] * bins + exy2[:, 1]) -
                         _mi(exy2[:, 2], exy2[:, 1]))
        return te_xy, te_yx

    _rng_te = np.random.default_rng(0)
    _N = int(N_te.value); _eps = float(eps_te.value); _sig = float(sig_te.value)
    _x, _y = _henon_coupled(_N, _eps, _sig, _rng_te)

    # TE sweep over epsilon
    _eps_arr = np.linspace(0.0, 1.0, 12)
    _te_xy_arr, _te_yx_arr = [], []
    for _e in _eps_arr:
        _xx, _yy = _henon_coupled(_N, _e, _sig, np.random.default_rng(1))
        _txy, _tyx = _te_hist(_xx, _yy)
        _te_xy_arr.append(_txy); _te_yx_arr.append(_tyx)

    _te_now_xy, _te_now_yx = _te_hist(_x, _y)
    _direction = "X→Y" if _te_now_xy > _te_now_yx else "Y→X"

    te_data = {
        "x": _x[:200].tolist(), "y": _y[:200].tolist(),
        "eps_arr": _eps_arr.tolist(),
        "te_xy": _te_xy_arr, "te_yx": _te_yx_arr,
        "te_now_xy": float(_te_now_xy), "te_now_yx": float(_te_now_yx),
        "eps_now": float(_eps), "direction": _direction,
    }
    return (te_data,)


@app.cell(hide_code=True)
def _(mo, np, plt, te_data):
    _d = te_data
    _fig_te, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    _eps_arr = np.array(_d["eps_arr"])
    _a1.plot(_d["x"][:150], _d["y"][:150], ".", ms=2, color="#7dd3fc", alpha=0.6)
    _a1.set_xlabel("x"); _a1.set_ylabel("y")
    _a1.set_title("Hénon attractor (x drives y)")
    _a2.plot(_eps_arr, _d["te_xy"], "o-", color="#7dd3fc", label="TE(X→Y)")
    _a2.plot(_eps_arr, _d["te_yx"], "o-", color="#9c3b2c", label="TE(Y→X)")
    _a2.axvline(_d["eps_now"], color="#f97316", lw=1, ls="--")
    _a2.set_xlabel("coupling ε"); _a2.set_ylabel("TE (bits)")
    _a2.set_title("transfer entropy vs ε")
    _a2.legend(fontsize=8)
    fig_te = _fig_te
    fig_te.tight_layout()
    mo.vstack([
        mo.md(
            f"**Dominant direction: {_d['direction']}** &nbsp;|&nbsp; "
            f"TE(X→Y) = {_d['te_now_xy']:.3f} bits &nbsp;|&nbsp; "
            f"TE(Y→X) = {_d['te_now_yx']:.3f} bits &nbsp;|&nbsp; "
            f"ΔTE = {_d['te_now_xy'] - _d['te_now_yx']:.3f}"
        ),
        fig_te,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As coupling $\varepsilon$ rises, TE(X$\to$Y) (blue) climbs while TE(Y$\to$X)
        (red) stays low, correctly identifying X as the driver. At $\varepsilon = 0$
        the two are equal: no coupling, no direction. The Hénon map is stochastic enough
        that this information-theoretic approach works; for deterministic systems like
        Lorenz, CCM is more appropriate.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 2: Convergent cross-mapping")
    return


@app.cell(hide_code=True)
def _(mo):
    sys_ccm = mo.ui.dropdown(["coupled logistic maps", "coupled Lorenz"],
                             value="coupled logistic maps", label="system")
    eps_ccm = mo.ui.slider(0.0, 0.5, value=0.2, step=0.025, label="coupling ε (X→Y)", show_value=True)
    E_sl    = mo.ui.slider(2, 4, value=2, step=1, label="embed dim E", show_value=True)
    mo.hstack([sys_ccm, eps_ccm, E_sl], justify="start", gap=2)
    return E_sl, eps_ccm, sys_ccm


@app.cell(hide_code=True)
def _(E_sl, eps_ccm, np, sys_ccm):
    def _logistic_coupled(N, eps, rng, rx=3.72, ry=3.72):
        x = np.zeros(N); y = np.zeros(N)
        x[0] = 0.4 + rng.uniform(-0.1, 0.1); y[0] = 0.2 + rng.uniform(-0.1, 0.1)
        for i in range(N - 1):
            x[i+1] = np.clip(x[i] * (rx - rx * x[i]), 0, 1)
            y[i+1] = np.clip(y[i] * (ry - ry * y[i] - eps * x[i]), 0, 1)
        return x, y

    def _lorenz_coupled(N, eps, rng, dt=0.03, sig=10.0, rho=28.0, beta=8/3):
        def dv(s, drive):
            x, y, z = s
            dx = sig * (y - x) + (0.0 if drive is None else eps * (drive - x))
            return np.array([dx, x * (rho - z) - y, x * y - beta * z])
        def rk4(s, drive):
            k1 = dv(s, drive); k2 = dv(s + 0.5*dt*k1, drive)
            k3 = dv(s + 0.5*dt*k2, drive); k4 = dv(s + dt*k3, drive)
            return s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        s1 = rng.uniform(-8, 8, 3); s2 = rng.uniform(-8, 8, 3)
        for _ in range(400):
            s1 = rk4(s1, None); s2 = rk4(s2, s1[0])
        xs, ys = np.empty(N), np.empty(N)
        for i in range(N):
            s1 = rk4(s1, None); s2 = rk4(s2, s1[0])
            xs[i] = s1[0]; ys[i] = s2[0]
        return xs, ys

    def _embed(ts, E):
        n = len(ts)
        return np.column_stack([ts[E-1-k:n-k] for k in range(E)])

    def _ccm_rho(shadow, target, E, lib_sizes):
        n = len(shadow); rhos = []
        for L in lib_sizes:
            lib = shadow[:L]; pred = []
            for i in range(L, min(L + 60, n)):
                d = np.sqrt(((lib - shadow[i])**2).sum(1))
                nn = np.argsort(d)[:E + 1]
                w = np.exp(-d[nn] / (d[nn[0]] + 1e-12)); w /= w.sum()
                pred.append(float((w * target[nn]).sum()))
            if len(pred) > 5:
                rhos.append(float(np.nan_to_num(np.corrcoef(target[L:L+len(pred)], pred)[0, 1])))
            else:
                rhos.append(0.0)
        return rhos

    _N = 700; _E = int(E_sl.value); _eps = float(eps_ccm.value)
    _rng = np.random.default_rng(3)
    if sys_ccm.value == "coupled Lorenz":
        _x, _y = _lorenz_coupled(_N, _eps, _rng)
    else:
        _x, _y = _logistic_coupled(_N, _eps, _rng)
    # normalise both series to [0,1] for display
    def _nrm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo + 1e-12)
    _xn, _yn = _nrm(_x), _nrm(_y)

    _MX = _embed(_x, _E); _MY = _embed(_y, _E)
    _yt = _y[_E-1:]; _xt = _x[_E-1:]
    _L_arr = list(range(20, min(300, len(_MX) - 70), 20))
    _rho_yx = _ccm_rho(_MX, _yt, _E, _L_arr)   # X's manifold predicts Y  -> X drives Y
    _rho_xy = _ccm_rho(_MY, _xt, _E, _L_arr)
    _dir = "X drives Y" if _rho_yx[-1] > _rho_xy[-1] else "Y drives X"

    # manifold points (x_t, x_{t-1}) coloured by aligned Y, capped for JSON size
    _cap = 320
    _mx0 = _MX[:_cap, 0]; _mx1 = _MX[:_cap, 1]
    _ycol = _nrm(_yt[:_cap])
    _frames = min(_cap, len(_mx0))

    ccm_data = {
        "frames": _frames,
        "L_arr": _L_arr, "rho_yx": _rho_yx, "rho_xy": _rho_xy,
        "direction": _dir, "system": sys_ccm.value,
        "panels": [
            {
                "kind": "timeseries",
                "title": "Coupled system (X, Y building up)",
                "xlabel": "time step", "ylabel": "normalised value",
                "ys": [np.round(_xn[:_cap], 4).tolist(), np.round(_yn[:_cap], 4).tolist()],
                "t": list(range(_frames)),
                "colors": ["#7dd3fc", "#f97316"],
                "labels": ["X", "Y"],
                "grow": True,
            },
            {
                "kind": "ccm_scatter",
                "title": "M_X shadow manifold (coloured by Y)",
                "xlabel": "X(t-1)", "ylabel": "X(t)",
                "x": np.round(_mx0, 4).tolist(),
                "y": np.round(_mx1, 4).tolist(),
                "col": np.round(_ycol, 4).tolist(),
            },
        ],
        "layout": [[0.0, 0.0, 0.45, 1.0], [0.45, 0.0, 0.55, 1.0]],
    }
    return ccm_data,


@app.cell(hide_code=True)
def _(canvas_anim, ccm_data, mo):
    _d = ccm_data

    mo.vstack([
        mo.md(
            f"**{_d['direction']}** &nbsp;|&nbsp; system: {_d['system']} &nbsp;|&nbsp; "
            "left: the coupled series unfolding, right: X's shadow manifold building up, "
            "each point coloured by the Y value at that time. A smooth colour gradient over "
            "the manifold means Y is recoverable from X, the signature of X driving Y."
        ),
        canvas_anim(_d, height="430px"),
    ])
    return


@app.cell(hide_code=True)
def _(ccm_data, mo, np, plt):
    _d = ccm_data
    _fig_ccm, _b2 = plt.subplots(1, 1, figsize=(6.5, 3.0))
    _b2.plot(_d["L_arr"], _d["rho_yx"], "o-", color="#7dd3fc", label="ρ(Y | M_X): X→Y")
    _b2.plot(_d["L_arr"], _d["rho_xy"], "o-", color="#9c3b2c", label="ρ(X | M_Y): Y→X")
    _b2.set_ylim(0, 1); _b2.set_xlabel("library size L"); _b2.set_ylabel("cross-map skill ρ")
    _b2.set_title("CCM convergence: the higher, more convergent curve is the driver")
    _b2.legend(fontsize=8)
    _fig_ccm.tight_layout()
    fig_ccm = _fig_ccm
    mo.vstack([
        mo.md("As the library grows, the cross-map skill converges. The direction with the "
              "higher converged skill is the true driver."),
        fig_ccm,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The shadow manifold of X (left) is the delay-embedded reconstruction
        $\mathbf{x}(t) = [x_t, x_{t-1}]$. If X drives Y then Y's value is
        predictable from X's shadow manifold (the colour gradient is smooth), and
        the cross-map skill $\rho$ converges as library size grows (right, blue).
        The reverse direction (red) converges lower if the coupling is asymmetric.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 3: Causal emergence in elementary cellular automata")
    return


@app.cell(hide_code=True)
def _(mo):
    rule_sl  = mo.ui.slider(0, 255,  value=110, step=1, label="ECA rule", show_value=True)
    block_sl = mo.ui.slider(2, 4,    value=2,   step=1, label="block size", show_value=True)
    mo.hstack([rule_sl, block_sl], justify="start", gap=2)
    return block_sl, rule_sl


@app.cell(hide_code=True)
def _(block_sl, np, rule_sl):
    def _eca_step(row, rule_bits):
        n = len(row)
        out = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            idx = (row[(i-1) % n] << 2) | (row[i] << 1) | row[(i+1) % n]
            out[i] = rule_bits[7 - idx]
        return out

    def _rule_bits(r):
        return np.array([(r >> i) & 1 for i in range(8)], dtype=np.uint8)

    def _run_eca(rule, width=80, steps=60):
        rb = _rule_bits(rule); row = np.zeros(width, dtype=np.uint8); row[width // 2] = 1
        history = [row.copy()]
        for _ in range(steps - 1):
            row = _eca_step(row, rb); history.append(row.copy())
        return np.array(history)

    def _eca_ei(rule):
        rb = _rule_bits(rule)
        out_dist = np.zeros(2)
        for inp in range(8):
            out_dist[rb[7 - inp]] += 1
        out_dist /= 8
        # EI = H(output) - H(output|input)
        # H(output|input) = 0 (deterministic), H(output) = Shannon entropy
        p = out_dist[out_dist > 0]
        return float(-np.sum(p * np.log2(p))) if len(p) > 1 else 0.0

    def _macro_ei(rule, block):
        # majority-vote coarse-graining: block cells → 1 macro cell
        rb = _rule_bits(rule)
        def _mv(bits): return 1 if sum(bits) > len(bits) / 2 else 0
        # enumerate all (block)-cell input patterns, run 1 step, coarse-grain
        from itertools import product as _prod
        macro_table = {}
        for inp_left in _prod([0, 1], repeat=block):
            for inp_mid in _prod([0, 1], repeat=block):
                for inp_right in _prod([0, 1], repeat=block):
                    row = list(inp_left) + list(inp_mid) + list(inp_right)
                    out = [rb[7 - ((row[(i-1) % len(row)] << 2) | (row[i] << 1) | row[(i+1) % len(row)])] for i in range(len(row))]
                    macro_in = (_mv(inp_left), _mv(inp_mid), _mv(inp_right))
                    macro_out = _mv(out[block:2*block])
                    key = macro_in[:2]
                    if key not in macro_table: macro_table[key] = {}
                    macro_table[key][macro_out] = macro_table[key].get(macro_out, 0) + 1
        ei = 0.0
        n_inputs = len(macro_table)
        for inp, out_counts in macro_table.items():
            total = sum(out_counts.values())
            p_out = np.array(list(out_counts.values()), float) / total
            h = -np.sum(p_out * np.log2(p_out + 1e-12))
            ei += h
        return 0.0 if n_inputs == 0 else float(1.0 - ei / n_inputs)

    _rule_sel = int(rule_sl.value); _block = int(block_sl.value)
    _spacetime = _run_eca(_rule_sel, width=80, steps=60)

    # full EI sweep over all 256 rules (fast, deterministic)
    _micro_ei = np.array([_eca_ei(r) for r in range(256)])
    _macro_ei_arr = np.array([_macro_ei(r, _block) for r in range(256)])
    _delta_ei = _macro_ei_arr - _micro_ei
    _wolfram_class = np.zeros(256, int)  # rough classification by output entropy
    for _r in range(256):
        _e = _micro_ei[_r]
        if _e < 0.1: _wolfram_class[_r] = 1
        elif _e < 0.5: _wolfram_class[_r] = 2
        elif _e < 0.9: _wolfram_class[_r] = 3
        else: _wolfram_class[_r] = 4

    eca_data = {
        "spacetime": _spacetime.tolist(),
        "rule": _rule_sel,
        "delta_ei": _delta_ei.tolist(),
        "micro_ei": _micro_ei.tolist(),
        "macro_ei": _macro_ei_arr.tolist(),
        "wolfram_class": _wolfram_class.tolist(),
        "dei_rule": float(_delta_ei[_rule_sel]),
        "micro_rule": float(_micro_ei[_rule_sel]),
        "macro_rule": float(_macro_ei_arr[_rule_sel]),
    }
    return (eca_data,)


@app.cell(hide_code=True)
def _(eca_data, mo, np, plt):
    _d = eca_data
    _fig_eca, (_c1, _c2, _c3) = plt.subplots(1, 3, figsize=(11, 3.2))

    _c1.imshow(np.array(_d["spacetime"]), cmap="binary", interpolation="nearest", aspect="auto")
    _c1.set_title(f"rule {_d['rule']} spacetime")
    _c1.set_xlabel("cell"); _c1.set_ylabel("time")

    _dei = np.array(_d["delta_ei"])
    _colors = ["#9c3b2c" if v > 0 else "#3c5469" for v in _dei]
    _c2.bar(range(256), _dei, color=_colors, width=1.0)
    _c2.axvline(_d["rule"], color="#f97316", lw=1.5)
    _c2.axhline(0, color="#2f3645", lw=0.8)
    _c2.set_xlabel("rule"); _c2.set_ylabel("ΔEI = macro - micro")
    _c2.set_title("causal emergence (all 256 rules)")

    _wc = np.array(_d["wolfram_class"])
    _cmap4 = ["#3c5469", "#7dd3fc", "#f97316", "#9c3b2c"]
    for _cls in [1, 2, 3, 4]:
        _mask = _wc == _cls
        _c3.scatter(np.array(_d["micro_ei"])[_mask], np.array(_d["macro_ei"])[_mask],
                    s=12, color=_cmap4[_cls - 1], label=f"class {_cls}", alpha=0.8)
    _c3.plot([0, 1], [0, 1], "--", color="#2f3645", lw=1)
    _c3.set_xlabel("micro EI"); _c3.set_ylabel("macro EI")
    _c3.set_title("micro vs macro EI")
    _c3.legend(fontsize=7, ncol=2)
    _fig_eca.tight_layout()
    fig_eca = _fig_eca

    mo.vstack([
        mo.md(
            f"**Rule {_d['rule']}:** "
            f"micro EI = {_d['micro_rule']:.2f} &nbsp;|&nbsp; "
            f"macro EI = {_d['macro_rule']:.2f} &nbsp;|&nbsp; "
            f"ΔEI = {_d['dei_rule']:.2f} "
            + ("(causal emergence)" if _d['dei_rule'] > 0 else "(no emergence)")
        ),
        fig_eca,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Causal emergence** occurs when a coarse-grained macro description has higher
        effective information (EI) than the micro rule, $\Delta\text{EI} > 0$ (red bars).
        EI measures how much the transition function constrains the output: a deterministic
        rule with uniform output distribution has EI = 0; a rule that maps every input to
        a distinct output has EI = 1. Rules in Wolfram class 4 (complex, like rule 110)
        tend to show the largest emergence: the micro dynamics are hard to predict but
        a coarse-grained view can carry more causal power.
        """
    )
    return


if __name__ == "__main__":
    app.run()
