import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Differentiable Pendulum")


@app.cell(hide_code=True)
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

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
        # The differentiable double pendulum

        The double pendulum is the cleanest mechanical chaos: deterministic, only four state
        variables, yet exquisitely sensitive to where it starts. This notebook does two things with
        it. First it integrates the equations of motion and shows the chaos directly, by launching two
        almost-identical pendulums and watching them peel apart. Then it treats the whole simulation
        as a **differentiable function** and fits its physical parameters (masses, lengths) by gradient
        descent on a trajectory-matching loss. The animation is the live integration, not a recording.
        """
    )
    return


@app.cell(hide_code=True)
def _(json, mo):
    def pendulum_anim(spec, height="430px"):
        html = r"""<!doctype html><html><head><meta charset="utf-8"><style>
        html,body{margin:0;background:#0f1117;overflow:hidden}canvas{display:block;width:100vw;height:100vh}
        </style></head><body><canvas id="c"></canvas><script>
        const S = __SPEC__, cv = document.getElementById("c"), ctx = cv.getContext("2d");
        const DPR = Math.min(window.devicePixelRatio||1, 2);
        function fit(){const w=cv.offsetWidth||window.innerWidth||800;const h=cv.offsetHeight||window.innerHeight||450;if(w>0&&h>0){cv.width=w*DPR;cv.height=h*DPR;}}
        addEventListener("resize", fit); fit();
        let f = 0, last = 0;
        function draw(){
          const W = cv.width, H = cv.height; ctx.fillStyle = "#0f1117"; ctx.fillRect(0,0,W,H);
          if (S.kind === "pendulum"){
            const cx = W/2, cy = H*0.36, sc = 0.34*Math.min(W,H)/(S.L1+S.L2);
            for (const pen of S.p){
              const t0 = Math.max(0, f-140);
              ctx.strokeStyle = pen.trail; ctx.lineWidth = 1.2*DPR; ctx.beginPath();
              for (let i=t0;i<=f;i++){
                const X = cx + sc*S.L1*Math.sin(pen.th1[i]) + sc*S.L2*Math.sin(pen.th2[i]);
                const Y = cy + sc*S.L1*Math.cos(pen.th1[i]) + sc*S.L2*Math.cos(pen.th2[i]);
                i===t0?ctx.moveTo(X,Y):ctx.lineTo(X,Y);
              }
              ctx.stroke();
              const t1 = pen.th1[f], t2 = pen.th2[f];
              const x1 = cx + sc*S.L1*Math.sin(t1), y1 = cy + sc*S.L1*Math.cos(t1);
              const x2 = x1 + sc*S.L2*Math.sin(t2), y2 = y1 + sc*S.L2*Math.cos(t2);
              ctx.strokeStyle = pen.color; ctx.lineWidth = 3*DPR;
              ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
              ctx.fillStyle = pen.color;
              ctx.beginPath(); ctx.arc(x1,y1,5*DPR,0,7); ctx.fill();
              ctx.beginPath(); ctx.arc(x2,y2,6*DPR,0,7); ctx.fill();
            }
            ctx.fillStyle = "#8a93a6"; ctx.beginPath(); ctx.arc(cx,cy,3*DPR,0,7); ctx.fill();
          }
        }
        function loop(t){ if (t-last > S.dt){ f = (f+1) % S.frames; last = t; } try{draw();}catch(e){}requestAnimationFrame(loop);}
        function _kick(){fit();if(cv.width>0&&cv.height>0){requestAnimationFrame(loop);}else{setTimeout(_kick,80);}}_kick();
        </script></body></html>""".replace("__SPEC__", json.dumps(spec))
        return mo.iframe(html, height=height)

    return (pendulum_anim,)


@app.cell(hide_code=True)
def _(np):
    def deriv(s, m1, m2, L1, L2, g=9.8):
        t1, w1, t2, w2 = s
        d = t1 - t2
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(d) ** 2
        a1 = (-m2 * L1 * w1 ** 2 * np.sin(d) * np.cos(d) + m2 * g * np.sin(t2) * np.cos(d)
              - m2 * L2 * w2 ** 2 * np.sin(d) - (m1 + m2) * g * np.sin(t1)) / den1
        den2 = (L2 / L1) * den1
        a2 = (m2 * L2 * w2 ** 2 * np.sin(d) * np.cos(d)
              + (m1 + m2) * (g * np.sin(t1) * np.cos(d) + L1 * w1 ** 2 * np.sin(d) - g * np.sin(t2))) / den2
        return np.array([w1, a1, w2, a2])

    def simulate(s0, m1, m2, L1, L2, dt, n):
        s = np.array(s0, float); out = np.empty((n, 4))
        for i in range(n):
            k1 = deriv(s, m1, m2, L1, L2)
            k2 = deriv(s + 0.5 * dt * k1, m1, m2, L1, L2)
            k3 = deriv(s + 0.5 * dt * k2, m1, m2, L1, L2)
            k4 = deriv(s + dt * k3, m1, m2, L1, L2)
            s = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            out[i] = s
        return out
    return deriv, simulate


@app.cell(hide_code=True)
def _(mo):
    theta0 = mo.ui.slider(60, 175, value=140, step=5, label="initial angle (°)", show_value=True)
    L2u = mo.ui.slider(0.6, 1.6, value=1.0, step=0.05, label="lower length L₂", show_value=True)
    mo.md(
        f"""
        **Two pendulums, launched a whisker apart.** Both start at the same angle except the second is
        nudged by 0.001 rad. Watch how long they stay together before chaos separates them.

        {mo.hstack([theta0, L2u], justify="start", gap=2)}
        """
    )
    return L2u, theta0


@app.cell(hide_code=True)
def _(L2u, np, simulate, theta0):
    th = np.radians(theta0.value)
    s0 = [th, 0.0, th, 0.0]
    s0b = [th, 0.0, th + 1e-3, 0.0]                # 0.001 rad apart
    A = simulate(s0, 1.0, 1.0, 1.0, L2u.value, dt=0.02, n=1400)
    B = simulate(s0b, 1.0, 1.0, 1.0, L2u.value, dt=0.02, n=1400)
    return A, B


@app.cell(hide_code=True)
def _(A, B, L2u, mo, np, pendulum_anim):
    _spec = {
        "kind": "pendulum", "title": "Chaos: two pendulums 0.001 rad apart", "frames": A.shape[0], "dt": 18, "L1": 1.0, "L2": float(L2u.value),
        "p": [
            {"th1": np.round(A[:, 0], 4).tolist(), "th2": np.round(A[:, 2], 4).tolist(),
             "color": "#7dd3fc", "trail": "rgba(125,211,252,0.35)"},
            {"th1": np.round(B[:, 0], 4).tolist(), "th2": np.round(B[:, 2], 4).tolist(),
             "color": "#e6a463", "trail": "rgba(230,164,99,0.35)"},
        ],
    }
    mo.vstack([
        mo.md("**The double pendulum, live.** Two copies, 0.001 rad apart at launch, with the lower-bob "
              "trails. They track, then diverge completely: deterministic, but unpredictable."),
        pendulum_anim(_spec, height="470px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Differentiable: fitting the pendulum to data

        Because the integrator is just arithmetic, the map from physical parameters
        $(m_1,m_2,L_1,L_2)$ to a trajectory is differentiable, and the mismatch to an observed
        trajectory is a loss we can descend:

        $$\mathcal L(\boldsymbol\phi)=\frac1T\sum_t\big\|\,\text{sim}(\boldsymbol\phi)_t-\text{obs}_t\big\|^2.$$

        Below, an "observed" run is generated from true parameters, then gradient descent (gradients by
        central finite differences) recovers them from a wrong start. The loss falls and the parameters
        converge, the same idea that underlies differentiable physics and system identification.
        """
    )
    return


@app.cell(hide_code=True)
def _(np, simulate):
    def traj_loss(phi, obs, s0, dt, n):
        m1, m2, L1, L2 = np.clip(phi, 0.1, 5.0)
        sim = simulate(s0, m1, m2, L1, L2, dt, n)
        return float(np.mean((sim - obs) ** 2))

    def fd_grad(phi, obs, s0, dt, n, eps=1e-3):
        g = np.zeros_like(phi)
        for i in range(len(phi)):
            hi = phi.copy(); hi[i] += eps; lo = phi.copy(); lo[i] -= eps
            g[i] = (traj_loss(hi, obs, s0, dt, n) - traj_loss(lo, obs, s0, dt, n)) / (2 * eps)
        return g
    return fd_grad, traj_loss


@app.cell(hide_code=True)
def _(mo):
    m1_sl = mo.ui.slider(0.4, 2.5, value=1.0, step=0.1, label="true m₁", show_value=True)
    m2_sl = mo.ui.slider(0.4, 2.5, value=1.2, step=0.1, label="true m₂", show_value=True)
    L1_sl = mo.ui.slider(0.5, 1.6, value=1.0, step=0.05, label="true L₁", show_value=True)
    L2_sl = mo.ui.slider(0.5, 1.6, value=0.9, step=0.05, label="true L₂", show_value=True)
    sig_sl = mo.ui.slider(0.0, 0.15, value=0.02, step=0.01, label="obs noise σ", show_value=True)
    ng_sl = mo.ui.slider(20, 80, value=45, step=5, label="gradient steps", show_value=True)
    mo.md(f"""
    **Set the true parameters, hide them in a noisy trajectory, and recover them.** Gradient
    descent starts from a deliberately wrong guess and walks back to the truth.
    {mo.hstack([m1_sl, m2_sl, L1_sl, L2_sl], justify="start", gap=2)}
    {mo.hstack([sig_sl, ng_sl], justify="start", gap=2)}
    """)
    return L1_sl, L2_sl, m1_sl, m2_sl, ng_sl, sig_sl


@app.cell(hide_code=True)
def _(L1_sl, L2_sl, fd_grad, m1_sl, m2_sl, ng_sl, np, sig_sl, simulate, traj_loss):
    _s0 = [0.5, 0.0, 0.4, 0.0]; _dt = 0.02; _n = 220
    true_phi = np.array([m1_sl.value, m2_sl.value, L1_sl.value, L2_sl.value], float)
    _rng = np.random.default_rng(0)
    _obs_clean = simulate(_s0, *true_phi, _dt, _n)
    _obs = _obs_clean + sig_sl.value * _rng.standard_normal(_obs_clean.shape)
    # deliberately wrong start
    _phi = true_phi * np.array([1.4, 0.7, 1.3, 1.25])
    _NG = int(ng_sl.value)
    _lh, _ph = [], [_phi.copy()]
    for _ in range(_NG):
        _lh.append(traj_loss(_phi, _obs, _s0, _dt, _n))
        _phi = np.clip(_phi - 4e-3 * fd_grad(_phi, _obs, _s0, _dt, _n), 0.1, 5.0)
        _ph.append(_phi.copy())
    inf_phi = _phi.copy()

    # true + inferred trajectories for the animation (noise free)
    _true_tr = simulate(_s0, *true_phi, _dt, _n)
    _inf_tr  = simulate(_s0, *inf_phi, _dt, _n)

    def _bobs(tr, L1, L2):
        _t1, _t2 = tr[:, 0], tr[:, 2]
        _x1 = L1 * np.sin(_t1); _y1 = -L1 * np.cos(_t1)
        _x2 = _x1 + L2 * np.sin(_t2); _y2 = _y1 - L2 * np.cos(_t2)
        return _x1, _y1, _x2, _y2

    _tx1, _ty1, _tx2, _ty2 = _bobs(_true_tr, true_phi[2], true_phi[3])
    _ix1, _iy1, _ix2, _iy2 = _bobs(_inf_tr, inf_phi[2], inf_phi[3])

    _DELAY = 8
    _sep = []
    for _f in range(_n):
        _g = max(0, _f - _DELAY)
        _sep.append(float(np.hypot(_tx2[_f] - _ix2[_g], _ty2[_f] - _iy2[_g])))

    _err = np.abs(inf_phi - true_phi) / np.abs(true_phi) * 100.0
    _rmse = float(np.sqrt(np.mean((_inf_tr - _true_tr) ** 2)))

    infer_data = {
        "frames": _n, "delay": _DELAY,
        "err": _err.tolist(), "final_loss": float(_lh[-1]), "rmse": _rmse,
        "true": true_phi.tolist(), "inf": inf_phi.tolist(),
        "panels": [
            {
                "kind": "pendulum_infer",
                "delay": _DELAY,
                "scale": float(true_phi[2] + true_phi[3] + 0.2),
                "tx1": np.round(_tx1, 4).tolist(), "ty1": np.round(_ty1, 4).tolist(),
                "tx2": np.round(_tx2, 4).tolist(), "ty2": np.round(_ty2, 4).tolist(),
                "ix1": np.round(_ix1, 4).tolist(), "iy1": np.round(_iy1, 4).tolist(),
                "ix2": np.round(_ix2, 4).tolist(), "iy2": np.round(_iy2, 4).tolist(),
            },
            {
                "kind": "timeseries",
                "title": "Gradient descent loss",
                "xlabel": "gradient step", "ylabel": "MSE",
                "ys": [[float(v) for v in _lh]],
                "colors": ["#9c3b2c"],
                "grow": True,
            },
            {
                "kind": "timeseries",
                "title": "Bob-2 separation (true vs inferred)",
                "xlabel": "frame", "ylabel": "distance",
                "ys": [_sep],
                "colors": ["#a594f2"],
                "grow": True,
            },
        ],
        "layout": [
            [0.00, 0.00, 0.45, 1.00],
            [0.45, 0.00, 0.55, 0.50],
            [0.45, 0.50, 0.55, 0.50],
        ],
    }
    return (infer_data,)


@app.cell(hide_code=True)
def _(canvas_anim, infer_data, mo):
    _d = infer_data

    _e = _d["err"]
    mo.vstack([
        mo.md(
            f"**Recovered** m₁ {_d['inf'][0]:.2f}, m₂ {_d['inf'][1]:.2f}, "
            f"L₁ {_d['inf'][2]:.2f}, L₂ {_d['inf'][3]:.2f} &nbsp;|&nbsp; "
            f"errors {_e[0]:.0f}% / {_e[1]:.0f}% / {_e[2]:.0f}% / {_e[3]:.0f}% &nbsp;|&nbsp; "
            f"final loss {_d['final_loss']:.1e} &nbsp;|&nbsp; trajectory RMSE {_d['rmse']:.3f}"
        ),
        canvas_anim(_d, height="460px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The inferred pendulum (green) chases the true one (blue dashed) with a short lag, and you can
        watch them lock together as the parameters converge. The loss falls by orders of magnitude and
        the bob separation collapses toward zero. Because the integrator is plain arithmetic, the whole
        simulator is differentiable, so the same finite-difference gradient that fits four numbers here
        scales up to the system identification problems behind real differentiable physics.
        """
    )
    return


if __name__ == "__main__":
    app.run()
