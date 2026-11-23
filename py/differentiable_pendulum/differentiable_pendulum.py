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
        as a **differentiable function** of its physical parameters and recovers them from a noisy
        trajectory by fitting the simulation to it. The animation is the live integration, not a
        recording.
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
    def deriv(s, mu, L1, L2, g=9.8):
        # mu = m2 / m1. Every term of the equations of motion carries one power of a mass,
        # so only the ratio can matter: doubling both masses changes nothing.
        t1, w1, t2, w2 = s
        d = t1 - t2; c, sn = np.cos(d), np.sin(d)
        den1 = (1 + mu) * L1 - mu * L1 * c ** 2
        a1 = (-mu * L1 * w1 ** 2 * sn * c + mu * g * np.sin(t2) * c
              - mu * L2 * w2 ** 2 * sn - (1 + mu) * g * np.sin(t1)) / den1
        den2 = (L2 / L1) * den1
        a2 = (mu * L2 * w2 ** 2 * sn * c
              + (1 + mu) * (g * np.sin(t1) * c + L1 * w1 ** 2 * sn - g * np.sin(t2))) / den2
        return np.array([w1, a1, w2, a2])

    def simulate(s0, mu, L1, L2, dt, n):
        """RK4 trajectory, shape (n, 4). If mu, L1, L2 are arrays of shape K, all K
        parameter sets run at once and the result has shape (n, 4, *K)."""
        mu, L1, L2 = np.asarray(mu, float), np.asarray(L1, float), np.asarray(L2, float)
        K = np.broadcast_shapes(mu.shape, L1.shape, L2.shape)
        s = np.broadcast_to(np.asarray(s0, float).reshape((4,) + (1,) * len(K)), (4,) + K).copy()
        out = np.empty((n, 4) + K)
        for i in range(n):
            k1 = deriv(s, mu, L1, L2)
            k2 = deriv(s + 0.5 * dt * k1, mu, L1, L2)
            k3 = deriv(s + 0.5 * dt * k2, mu, L1, L2)
            k4 = deriv(s + dt * k3, mu, L1, L2)
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
    A = simulate(s0, 1.0, 1.0, L2u.value, dt=0.02, n=1400)
    B = simulate(s0b, 1.0, 1.0, L2u.value, dt=0.02, n=1400)
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

        Because the integrator is just arithmetic, the map from physical parameters to a trajectory
        is differentiable, and the mismatch to an observed trajectory is a loss we can minimise:
        $$\mathcal L(\boldsymbol\phi)=\frac1{4T}\sum_t\big\|\,\text{sim}(\boldsymbol\phi)_t-\text{obs}_t\big\|^2 .$$
        The parameters are $\boldsymbol\phi = (m_2/m_1,\,L_1,\,L_2)$. The two masses cannot be
        recovered separately, only their ratio, because scaling both by the same factor leaves the
        motion unchanged. Below, an "observed" run is generated from true parameters with
        measurement noise added, and the fit recovers them from a wrong start.

        Two details make the fit work. It uses **Levenberg-Marquardt**, which builds the Jacobian
        of the trajectory with respect to the parameters (by finite differences, all perturbed
        runs integrated together in one batch) and takes Gauss-Newton steps, damped when a step
        would raise the loss. And it fits a **growing horizon**: the first 0.6 s of motion, then
        1.2 s, 2.2 s and finally all 4.4 s. A long oscillating trajectory makes the loss surface
        ripple with local minima, one for every way the simulated swing can fall a whole period
        behind the real one. A short horizon has no such minima, and its answer is close enough
        to start the next stage in the right valley.
        """
    )
    return


@app.cell(hide_code=True)
def _(np, simulate):
    def lm_fit(obs, s0, dt, phi0, stages=(30, 60, 110, 220), iters=12, h=1e-4):
        """Levenberg-Marquardt in log-parameters (keeps them positive), over a growing horizon.
        Returns the final parameters and the parameter value after every accepted step."""
        p = np.log(np.asarray(phi0, float)); path = [np.exp(p)]
        for m in stages:
            m = min(m, len(obs)); lam = 1e-2
            for _ in range(iters):
                P = np.column_stack([p] + [p + h * e for e in np.eye(3)])   # centre + 3 nudges
                tr = simulate(s0, *np.exp(P), dt, m)                       # (m, 4, 4), one batch
                r = (tr[..., 0] - obs[:m]).ravel()
                J = np.stack([(tr[..., j + 1] - tr[..., 0]).ravel() / h for j in range(3)], 1)
                cost = float(np.mean(r ** 2)); A = J.T @ J
                while lam < 1e8:
                    step = np.linalg.solve(A + lam * np.diag(np.diag(A)), -J.T @ r)
                    pn = p + np.clip(step, -0.5, 0.5)
                    if float(np.mean((simulate(s0, *np.exp(pn), dt, m) - obs[:m]) ** 2)) < cost:
                        p = pn; lam = max(lam / 3, 1e-7); break
                    lam *= 4
                if lam >= 1e8:
                    break
                path.append(np.exp(p))
        return np.exp(p), np.array(path)
    return (lm_fit,)


@app.cell(hide_code=True)
def _(mo):
    mu_sl = mo.ui.slider(0.2, 3.0, value=1.2, step=0.1, label="true m₂/m₁", show_value=True)
    L1_sl = mo.ui.slider(0.5, 1.6, value=1.0, step=0.05, label="true L₁", show_value=True)
    L2_sl = mo.ui.slider(0.5, 1.6, value=0.9, step=0.05, label="true L₂", show_value=True)
    sig_sl = mo.ui.slider(0.0, 0.15, value=0.02, step=0.01, label="obs noise σ (rad, rad/s)", show_value=True)
    mo.md(f"""
    **Set the true parameters, hide them in a noisy trajectory, and recover them.** The fit
    starts from a guess that is off by 30 to 40 percent in every parameter.
    {mo.hstack([mu_sl, L1_sl, L2_sl, sig_sl], justify="start", gap=2)}
    """)
    return L1_sl, L2_sl, mu_sl, sig_sl


@app.cell(hide_code=True)
def _(L1_sl, L2_sl, lm_fit, mu_sl, np, sig_sl, simulate):
    _s0 = [0.5, 0.0, 0.4, 0.0]; _dt = 0.02; _n = 220
    true_phi = np.array([mu_sl.value, L1_sl.value, L2_sl.value], float)
    _rng = np.random.default_rng(0)
    _obs_clean = simulate(_s0, *true_phi, _dt, _n)
    _obs = _obs_clean + sig_sl.value * _rng.standard_normal(_obs_clean.shape)

    _phi0 = true_phi * np.array([0.6, 1.3, 1.25])          # deliberately wrong start
    inf_phi, _path = lm_fit(_obs, _s0, _dt, _phi0)

    # loss over the full 4.4 s for every step of the fit, all steps integrated in one batch
    _all = simulate(_s0, *_path.T, _dt, _n)               # (n, 4, steps)
    _lh = np.mean((_all - _obs[..., None]) ** 2, axis=(0, 1))

    _true_tr = _obs_clean
    _inf_tr = simulate(_s0, *inf_phi, _dt, _n)

    def _bobs(tr, L1, L2):
        _t1, _t2 = tr[:, 0], tr[:, 2]
        _x1 = L1 * np.sin(_t1); _y1 = -L1 * np.cos(_t1)
        _x2 = _x1 + L2 * np.sin(_t2); _y2 = _y1 - L2 * np.cos(_t2)
        return _x1, _y1, _x2, _y2

    _tx1, _ty1, _tx2, _ty2 = _bobs(_true_tr, true_phi[1], true_phi[2])
    _ix1, _iy1, _ix2, _iy2 = _bobs(_inf_tr, inf_phi[1], inf_phi[2])
    _sep = np.hypot(_tx2 - _ix2, _ty2 - _iy2).tolist()
    _err = np.abs(inf_phi - true_phi) / np.abs(true_phi) * 100.0
    _rmse = float(np.sqrt(np.mean((_inf_tr - _true_tr) ** 2)))

    infer_data = {
        "frames": _n, "dt": 35,
        "err": _err.tolist(), "final_loss": float(_lh[-1]), "start_loss": float(_lh[0]),
        "noise_floor": float(sig_sl.value) ** 2, "rmse": _rmse, "steps": len(_lh) - 1,
        "true": true_phi.tolist(), "inf": inf_phi.tolist(),
        "panels": [
            {
                "kind": "pendulum_infer",
                "delay": 0,
                "scale": float(true_phi[1] + true_phi[2] + 0.2),
                "tx1": np.round(_tx1, 4).tolist(), "ty1": np.round(_ty1, 4).tolist(),
                "tx2": np.round(_tx2, 4).tolist(), "ty2": np.round(_ty2, 4).tolist(),
                "ix1": np.round(_ix1, 4).tolist(), "iy1": np.round(_iy1, 4).tolist(),
                "ix2": np.round(_ix2, 4).tolist(), "iy2": np.round(_iy2, 4).tolist(),
            },
            {
                "kind": "timeseries",
                "title": "Full-trajectory loss at each fit step",
                "xlabel": "fit step", "ylabel": "MSE (log)",
                "ys": [np.log10(np.maximum(_lh, 1e-12)).tolist()],
                "colors": ["#e0635a"],
                "grow": True,
            },
            {
                "kind": "timeseries",
                "title": "Lower-bob separation, true vs fitted",
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
            f"**Recovered** m₂/m₁ {_d['inf'][0]:.3f}, L₁ {_d['inf'][1]:.3f}, L₂ {_d['inf'][2]:.3f} "
            f"(true {_d['true'][0]:.2f}, {_d['true'][1]:.2f}, {_d['true'][2]:.2f}) &nbsp;|&nbsp; "
            f"errors {_e[0]:.1f}% / {_e[1]:.1f}% / {_e[2]:.1f}% &nbsp;|&nbsp; "
            f"loss {_d['start_loss']:.2e} → {_d['final_loss']:.2e} in {_d['steps']} steps "
            f"(noise floor σ² = {_d['noise_floor']:.1e}) &nbsp;|&nbsp; trajectory RMSE {_d['rmse']:.4f}"
        ),
        canvas_anim(_d, height="460px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The fitted pendulum (orange) is drawn over the true one (blue dashed). When the fit has
        worked the two move as one and the separation panel stays near zero; the loss curve
        drops from the wrong start to the noise floor, where the only mismatch left is the
        measurement noise itself, which no choice of parameters can explain. Every step of the
        fit costs one batched integration of four parameter sets, so finite differences are
        affordable for three parameters. With hundreds of parameters you would differentiate
        through the integrator instead (automatic differentiation or the adjoint method), which
        is what differentiable-physics libraries do; the fitting idea is the same.

        Things to try: push the noise up and watch the errors grow but stay bounded, or set
        $m_2/m_1$ small, where the lower bob barely tugs on the upper one and its mass becomes
        hard to see in the data.
        """
    )
    return


if __name__ == "__main__":
    app.run()
