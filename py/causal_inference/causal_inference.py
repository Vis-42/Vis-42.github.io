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
        weakly coupled systems, where TE and Granger causality struggle. If $X$ drives $Y$,
        then $Y$'s history carries an imprint of $X$, so $X$ can be reconstructed from
        $Y$'s shadow manifold. The test runs backwards from the arrow it detects.

        **Causal emergence** asks a different question: at what scale does the causal
        structure live? Coarse-graining a system can, in principle, give a macro description
        with more effective information than the micro rule. Part 3 measures how often a
        simple majority-vote coarse-graining does that for the 256 elementary cellular
        automata. The answer is: almost never.
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
    sig_te = mo.ui.slider(0.0, 0.3,  value=0.05, step=0.025, label="measurement noise σ", show_value=True)
    N_te   = mo.ui.slider(200, 2000, value=800,  step=100,   label="series length N", show_value=True)
    mo.hstack([eps_te, sig_te, N_te], justify="start", gap=2)
    return N_te, eps_te, sig_te


@app.cell(hide_code=True)
def _(N_te, eps_te, np, sig_te):
    def _henon_coupled(N, eps, sigma, rng, burn=200):
        # Two Henon maps, X autonomous and Y driven by X with strength eps:
        #   x[n+1] = 1.4 - x[n]^2 + 0.3 x[n-1]
        #   y[n+1] = 1.4 - (eps x[n] y[n] + (1 - eps) y[n]^2) + 0.3 y[n-1]
        # sigma is measurement noise, added after the dynamics so it cannot kick
        # the map off its attractor.
        x = np.zeros(N + burn + 1); y = np.zeros(N + burn + 1)
        x[:2] = rng.uniform(0, 0.5, 2); y[:2] = rng.uniform(0, 0.5, 2)
        for i in range(1, N + burn):
            x[i+1] = 1.4 - x[i]**2 + 0.3 * x[i-1]
            y[i+1] = 1.4 - (eps * x[i] * y[i] + (1 - eps) * y[i]**2) + 0.3 * y[i-1]
        x = x[burn+1:] + sigma * rng.standard_normal(N)
        y = y[burn+1:] + sigma * rng.standard_normal(N)
        return x, y

    _BINS = 6

    def _binned(a):
        # equal-occupancy bins, so every symbol is used equally often
        return np.digitize(a, np.quantile(a, np.linspace(0, 1, _BINS + 1)[1:-1]))

    def _H(*cols):
        idx = np.zeros(len(cols[0]), int)
        for c in cols:
            idx = idx * _BINS + c
        p = np.bincount(idx) / len(idx); p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def _te_raw(src, dst):
        # TE(src -> dst) = I(dst[n+1]; src[n] | dst[n])
        s = _binned(src[:-1]); d = _binned(dst[:-1]); d1 = _binned(dst[1:])
        return _H(d1, d) + _H(s, d) - _H(d1, s, d) - _H(d)

    def _te(src, dst, rng, n_sur=8):
        # A binned estimate is biased upwards on a finite series. Shuffling the
        # source destroys any coupling but keeps the bias, so subtract it.
        bias = np.mean([_te_raw(rng.permutation(src), dst) for _ in range(n_sur)])
        return max(0.0, _te_raw(src, dst) - bias)

    _N = int(N_te.value); _eps = float(eps_te.value); _sig = float(sig_te.value)
    _rng_te = np.random.default_rng(0)
    _x, _y = _henon_coupled(_N, _eps, _sig, _rng_te)
    _te_now_xy = _te(_x, _y, _rng_te); _te_now_yx = _te(_y, _x, _rng_te)

    # TE sweep over epsilon
    _eps_arr = np.linspace(0.0, 1.0, 12)
    _te_xy_arr, _te_yx_arr = [], []
    for _e in _eps_arr:
        _rng_sw = np.random.default_rng(1)
        _xx, _yy = _henon_coupled(_N, _e, _sig, _rng_sw)
        _te_xy_arr.append(_te(_xx, _yy, _rng_sw)); _te_yx_arr.append(_te(_yy, _xx, _rng_sw))

    _gap = _te_now_xy - _te_now_yx
    _direction = "X→Y" if _gap > 0.05 else "Y→X" if _gap < -0.05 else "no clear direction"

    te_data = {
        "x": _x[:400].tolist(), "y": _y[:400].tolist(),
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
    _a1.plot(_d["x"], _d["y"], ".", ms=2.5, color="#7dd3fc", alpha=0.7)
    _a1.set_xlabel("x(n)"); _a1.set_ylabel("y(n)")
    _a1.set_title("driver x against response y")
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
            f"**Detected: {_d['direction']}** &nbsp;|&nbsp; "
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
        Two Hénon maps: $X$ runs on its own, and $Y$ is pulled toward it with strength
        $\varepsilon$. As $\varepsilon$ rises, TE(X$\to$Y) (blue) climbs while TE(Y$\to$X)
        (red) stays near zero, which picks out X as the driver. At $\varepsilon = 0$ both
        vanish: no coupling, no direction. Past $\varepsilon \approx 0.7$ the two maps
        synchronise, $y_n = x_n$ exactly, and TE collapses again. Once $Y$ is a copy of $X$,
        $X$'s past says nothing about $Y$'s future that $Y$'s own past does not already say.
        That blind spot at strong coupling is a known limit of transfer entropy, and it is
        where the scatter on the left turns into a diagonal line.

        Each value is a histogram estimate (6 equal-occupancy bins) with the finite-sample
        bias removed by subtracting the average over shuffled copies of the source.
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

    def _lorenz_coupled(N, eps, rng, dt=0.01, sub=5, sig=10.0, rho=28.0, beta=8/3):
        # Two Lorenz systems; the first runs free, the second is pulled toward it
        # through its x equation with strength c = 12 eps. Sampled every sub*dt.
        c = 12.0 * eps
        def dv(s):
            x1, y1, z1, x2, y2, z2 = s
            return np.array([sig * (y1 - x1), x1 * (rho - z1) - y1, x1 * y1 - beta * z1,
                             sig * (y2 - x2) + c * (x1 - x2), x2 * (rho - z2) - y2, x2 * y2 - beta * z2])
        def rk4(s):
            k1 = dv(s); k2 = dv(s + 0.5*dt*k1); k3 = dv(s + 0.5*dt*k2); k4 = dv(s + dt*k3)
            return s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        s = rng.uniform(-8, 8, 6)
        for _ in range(1000):
            s = rk4(s)
        xs, ys = np.empty(N), np.empty(N)
        for i in range(N):
            for _ in range(sub):
                s = rk4(s)
            xs[i] = s[0]; ys[i] = s[3]
        return xs, ys

    def _embed(ts, E, tau):
        # rows are [x(t), x(t - tau), ..., x(t - (E-1) tau)]
        n = len(ts) - (E - 1) * tau
        return np.column_stack([ts[(E-1-k)*tau:(E-1-k)*tau + n] for k in range(E)])

    def _ccm_rho(shadow, target, E, lib_sizes, n_test=150):
        # Estimate target from the nearest neighbours on the shadow manifold, using
        # the first L points as the library and the last n_test points as the test set.
        n = len(shadow); test = np.arange(n - n_test, n); rhos = []
        for L in lib_sizes:
            d = np.sqrt(((shadow[test][:, None, :] - shadow[None, :L, :])**2).sum(2))
            nn = np.argsort(d, axis=1)[:, :E + 1]
            dn = np.take_along_axis(d, nn, 1)
            w = np.exp(-dn / (dn[:, :1] + 1e-12)); w /= w.sum(1, keepdims=True)
            pred = (w * target[:L][nn]).sum(1)
            rhos.append(float(np.nan_to_num(np.corrcoef(pred, target[test])[0, 1])))
        return rhos

    _E = int(E_sl.value); _eps = float(eps_ccm.value)
    _rng = np.random.default_rng(3)
    if sys_ccm.value == "coupled Lorenz":
        # a flow needs a longer series, a delay of a few samples and at least 3 dimensions
        _N = 1200; _tau = 2; _E = max(_E, 3)
        _x, _y = _lorenz_coupled(_N, _eps, _rng)
    else:
        _N = 700; _tau = 1
        _x, _y = _logistic_coupled(_N, _eps, _rng)
    def _nrm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo + 1e-12)
    _xn, _yn = _nrm(_x), _nrm(_y)

    _MX = _embed(_x, _E, _tau); _MY = _embed(_y, _E, _tau)
    _off = (_E - 1) * _tau
    _yt = _y[_off:]; _xt = _x[_off:]
    _L_arr = list(range(20, len(_MX) - 160, 40))
    # X drives Y  <=>  X can be read off Y's manifold
    _rho_x_from_my = _ccm_rho(_MY, _xt, _E, _L_arr)
    _rho_y_from_mx = _ccm_rho(_MX, _yt, _E, _L_arr)
    _a, _b = _rho_x_from_my[-1], _rho_y_from_mx[-1]
    if max(_a, _b) < 0.3:
        _dir = "no coupling detected"
    elif abs(_a - _b) < 0.1:
        _dir = "both directions cross-map (synchronised or two-way)"
    else:
        _dir = "X drives Y" if _a > _b else "Y drives X"

    # Y's shadow manifold, each point coloured by X at the same time
    _cap = 320
    _frames = min(_cap, len(_MY))
    _xcol = _nrm(_xt[:_cap])

    ccm_data = {
        "frames": _frames, "dt": 45,
        "L_arr": _L_arr, "rho_x_from_my": _rho_x_from_my, "rho_y_from_mx": _rho_y_from_mx,
        "direction": _dir, "system": sys_ccm.value, "E": _E, "tau": _tau,
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
                "title": "M_Y shadow manifold (coloured by X)",
                "xlabel": "Y(t)", "ylabel": f"Y(t-{_tau})",
                "x": np.round(_MY[:_cap, 0], 4).tolist(),
                "y": np.round(_MY[:_cap, 1], 4).tolist(),
                "col": np.round(_xcol, 4).tolist(),
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
            f"**{_d['direction']}** &nbsp;|&nbsp; system: {_d['system']} "
            f"(E = {_d['E']}, delay {_d['tau']}) &nbsp;|&nbsp; "
            f"ρ(X from M_Y) = {_d['rho_x_from_my'][-1]:.2f}, "
            f"ρ(Y from M_X) = {_d['rho_y_from_mx'][-1]:.2f} &nbsp;|&nbsp; "
            "left: the coupled series unfolding, right: Y's shadow manifold building up, "
            "each point coloured by the X value at that time. A smooth colour gradient over "
            "the manifold means X can be read off Y's history, the signature of X driving Y."
        ),
        canvas_anim(_d, height="430px"),
    ])
    return


@app.cell(hide_code=True)
def _(ccm_data, mo, np, plt):
    _d = ccm_data
    _fig_ccm, _b2 = plt.subplots(1, 1, figsize=(6.5, 3.0))
    _b2.plot(_d["L_arr"], _d["rho_x_from_my"], "o-", color="#7dd3fc", label="X from M_Y: tests X→Y")
    _b2.plot(_d["L_arr"], _d["rho_y_from_mx"], "o-", color="#e0635a", label="Y from M_X: tests Y→X")
    _b2.set_ylim(-0.2, 1); _b2.set_xlabel("library size L"); _b2.set_ylabel("cross-map skill ρ")
    _b2.set_title("CCM convergence")
    _b2.legend(fontsize=8)
    _fig_ccm.tight_layout()
    fig_ccm = _fig_ccm
    mo.vstack([
        mo.md("Skill that climbs and levels off as the library grows is the convergence CCM "
              "is named for. A curve that stays near zero means that variable leaves no trace "
              "in the other one."),
        fig_ccm,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The shadow manifold $M_Y$ is the delay embedding $[y_t, y_{t-\tau}, \ldots]$ of
        the response. If X drives Y, every state of Y has been shaped by X, so nearby
        points on $M_Y$ share nearby values of X: the colours vary smoothly, and X can be
        estimated from the nearest neighbours on $M_Y$ (blue curve). The reverse does not
        hold. X evolves without any knowledge of Y, so $M_X$ cannot predict Y and the red
        curve stays low. This is why CCM reads the arrow backwards: to test "X drives Y",
        reconstruct X from Y.

        Two things to try. At $\varepsilon = 0$ both curves sit near zero. At strong
        coupling Y locks onto X, both directions cross-map, and CCM alone can no longer
        tell one-way driving from synchrony. The Lorenz pair is a harder case: a flow
        sampled in time needs more data and a higher embedding dimension (the notebook
        uses at least 3), and its curves are noisier than the map's.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 3: Does coarse-graining a cellular automaton gain causal power?")
    return


@app.cell(hide_code=True)
def _(mo):
    rule_sl  = mo.ui.slider(0, 255,  value=110, step=1, label="ECA rule", show_value=True)
    block_sl = mo.ui.slider(2, 4,    value=2,   step=1, label="block size", show_value=True)
    mo.hstack([rule_sl, block_sl], justify="start", gap=2)
    return block_sl, rule_sl


@app.cell(hide_code=True)
def _(block_sl, np, rule_sl):
    # RB[r, i] is rule r's output for the neighbourhood whose bits (left, centre, right) spell i
    RB = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.uint8)

    def _run_eca(rule, width=80, steps=60):
        row = np.zeros(width, dtype=np.uint8); row[width // 2] = 1
        history = [row.copy()]
        for _ in range(steps - 1):
            idx = (np.roll(row, 1) << 2) | (row << 1) | np.roll(row, -1)
            row = RB[rule][idx]; history.append(row.copy())
        return np.array(history)

    def _h2(p):
        # binary entropy in bits, 0 at p = 0 and p = 1
        q = np.clip(p, 1e-12, 1 - 1e-12)
        return np.where((p <= 0) | (p >= 1), 0.0, -(q * np.log2(q) + (1 - q) * np.log2(1 - q)))

    # Effective information of a map from inputs to one output bit, with every input
    # made equally likely:  EI = H(average output) - average over inputs of H(output | input).
    # Micro: 8 neighbourhoods in, one cell out. The rule is deterministic, so EI = H(output).
    _micro_ei = _h2(RB.mean(1))

    def _macro_ei(b):
        # Macro: three blocks of b cells in, the middle block out, each block reduced
        # to one bit by majority vote (ties count as 0). All 2^(3b) micro patterns are
        # enumerated for all 256 rules at once.
        n = 3 * b
        pats = ((np.arange(2**n)[:, None] >> np.arange(n)[None, ::-1]) & 1).astype(np.uint8)
        mid = np.arange(b, 2 * b)
        idx = (pats[:, mid - 1] << 2) | (pats[:, mid] << 1) | pats[:, mid + 1]
        maj = lambda a: (a.sum(-1) * 2 > a.shape[-1]).astype(np.uint8)
        m_out = maj(RB[:, idx])                                             # (256, patterns)
        m_in = (maj(pats[:, :b]) << 2) | (maj(pats[:, b:2*b]) << 1) | maj(pats[:, 2*b:])
        p1 = np.stack([m_out[:, m_in == s].mean(1) for s in range(8)], 1)   # P(out = 1 | macro input)
        return _h2(p1.mean(1)) - _h2(p1).mean(1)

    _rule_sel = int(rule_sl.value); _block = int(block_sl.value)
    _spacetime = _run_eca(_rule_sel, width=80, steps=60)
    _macro_ei_arr = _macro_ei(_block)
    _delta_ei = _macro_ei_arr - _micro_ei
    _emergent = np.where(_delta_ei > 1e-9)[0]

    eca_data = {
        "spacetime": _spacetime.tolist(),
        "rule": _rule_sel, "block": _block,
        "delta_ei": _delta_ei.tolist(),
        "micro_ei": _micro_ei.tolist(),
        "macro_ei": _macro_ei_arr.tolist(),
        "emergent": _emergent.tolist(),
        "dei_rule": float(_delta_ei[_rule_sel]),
        "micro_rule": float(_micro_ei[_rule_sel]),
        "macro_rule": float(_macro_ei_arr[_rule_sel]),
    }
    return (eca_data,)


@app.cell(hide_code=True)
def _(eca_data, mo, np, plt):
    _d = eca_data
    _fig_eca, (_c1, _c2, _c3) = plt.subplots(1, 3, figsize=(11, 3.2))

    _c1.imshow(np.array(_d["spacetime"]), cmap="binary_r", interpolation="nearest", aspect="auto")
    _c1.set_title(f"rule {_d['rule']} spacetime")
    _c1.set_xlabel("cell"); _c1.set_ylabel("time")

    _dei = np.array(_d["delta_ei"])
    _colors = ["#e0635a" if v > 1e-9 else "#5b7fa3" for v in _dei]
    _c2.bar(range(256), _dei, color=_colors, width=1.0)
    _c2.axvline(_d["rule"], color="#f97316", lw=1.5)
    _c2.axhline(0, color="#8a93a6", lw=0.8)
    _c2.set_xlabel("rule"); _c2.set_ylabel("ΔEI = macro - micro (bits)")
    _c2.set_title(f"all 256 rules, blocks of {_d['block']}")

    _mi = np.array(_d["micro_ei"]); _ma = np.array(_d["macro_ei"])
    _c3.scatter(_mi, _ma, s=12, color="#7dd3fc", alpha=0.6)
    _c3.scatter([_mi[_d["rule"]]], [_ma[_d["rule"]]], s=60, color="#f97316", zorder=3,
                label=f"rule {_d['rule']}")
    _c3.plot([0, 1], [0, 1], "--", color="#8a93a6", lw=1)
    _c3.set_xlabel("micro EI (bits)"); _c3.set_ylabel("macro EI (bits)")
    _c3.set_title("above the diagonal = emergence")
    _c3.legend(fontsize=7)
    _fig_eca.tight_layout()
    fig_eca = _fig_eca

    _em = _d["emergent"]
    _em_txt = ("none" if not _em else ", ".join(str(r) for r in _em[:12]) + (" ..." if len(_em) > 12 else ""))
    mo.vstack([
        mo.md(
            f"**Rule {_d['rule']}:** "
            f"micro EI = {_d['micro_rule']:.2f} &nbsp;|&nbsp; "
            f"macro EI = {_d['macro_rule']:.2f} &nbsp;|&nbsp; "
            f"ΔEI = {_d['dei_rule']:+.2f} bits "
            + ("(macro scale gains)" if _d['dei_rule'] > 1e-9 else "(macro scale gains nothing)")
            + f" &nbsp;|&nbsp; rules with ΔEI > 0 at this block size: {len(_em)} of 256 ({_em_txt})"
        ),
        fig_eca,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Effective information** (EI) is the mutual information between a mechanism's
        input and its output when every input is made equally likely. Here the output is
        one bit, so EI runs from 0 (the output ignores the input) to 1 bit (the output is
        balanced and fully determined by the input). At the micro scale the mechanism is
        the rule itself: 8 neighbourhoods in, one cell out. It is deterministic, so its EI
        is just the entropy of its output column: 1 bit for a balanced rule like 30 or 90,
        0 for the constant rules 0 and 255.

        At the macro scale each block of cells is replaced by its majority bit and the
        mechanism becomes "three block bits in, the middle block bit out". The same macro
        input now covers many micro patterns that can disagree about the output, which
        adds noise and costs EI. **Causal emergence** is the case
        $\Delta\text{EI} = \text{EI}_\text{macro} - \text{EI}_\text{micro} > 0$ (red bars).

        For this coarse-graining it almost never happens. With blocks of 2 only two rules
        come out ahead (127 and 200); with blocks of 3 or 4, none do. Rules that survive
        best are the ones a majority vote commutes with: the identity (204) keeps its full
        bit, and the majority rule (232) keeps most of it. The chaotic and complex rules
        (30, 90, 110) lose nearly everything, because the majority bit of a block says
        little about what their cells will do next. A macro scale that beats the micro
        scale has to be chosen to fit the dynamics; a generic one does not.
        """
    )
    return


if __name__ == "__main__":
    app.run()
