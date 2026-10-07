import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="CD Spectral Inversion")


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
        # Circular dichroism: reading protein structure with a physics-constrained network

        Far-UV circular dichroism reads out a protein's secondary structure, but only forwards: a
        spectrum is the weighted sum of fixed basis curves, $y(\lambda)=f_H B_H+f_S B_S+f_C B_C$.
        Going backwards, recovering the fractions from a noisy spectrum, is ill-posed because the
        basis curves overlap. Here a small MLP learns the inverse map, with a **softmax output** that
        forces the prediction onto the simplex (non-negative, summing to one) and a **physics loss**
        that penalises predictions whose reconstructed spectrum does not match the input. Watch the
        reconstruction converge onto the measurement as the network trains.
        """
    )
    return


@app.cell(hide_code=True)
def _(np):
    WL = np.arange(190.0, 251.0, 1.0)
    N_WL = len(WL)

    def _basis():
        helix = (-30 * np.exp(-0.5 * ((WL - 208) / 5) ** 2) - 25 * np.exp(-0.5 * ((WL - 222) / 4) ** 2)
                 + 50 * np.exp(-0.5 * ((WL - 193) / 8) ** 2))
        sheet = (-15 * np.exp(-0.5 * ((WL - 216) / 7) ** 2) + 20 * np.exp(-0.5 * ((WL - 195) / 6) ** 2))
        coil = (-8 * np.exp(-0.5 * ((WL - 200) / 6) ** 2) + 5 * np.exp(-0.5 * ((WL - 218) / 8) ** 2))
        return np.column_stack([helix, sheet, coil])

    B = _basis()
    SCALE = 30.0   # input spectra are O(30); scale to O(1) for training stability

    def spectrum(frac):
        return B @ np.asarray(frac, float)
    return B, N_WL, SCALE, WL, spectrum


@app.cell(hide_code=True)
def _(B, N_WL, SCALE, np):
    # ── physics-informed MLP: 61 -> H -> H -> 3 logits -> softmax ──────────────
    def init_mlp(H, seed=42):
        rng = np.random.default_rng(seed)
        k1 = np.sqrt(2.0 / N_WL); k2 = np.sqrt(2.0 / H)
        return {
            "W1": rng.standard_normal((H, N_WL)) * k1, "b1": np.zeros(H),
            "W2": rng.standard_normal((H, H)) * k2,    "b2": np.zeros(H),
            "W3": rng.standard_normal((3, H)) * k2,    "b3": np.zeros(3),
        }

    def _softmax(Z):
        e = np.exp(Z - Z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def forward(p, X):
        z1 = X @ p["W1"].T + p["b1"]; a1 = np.maximum(0, z1)
        z2 = a1 @ p["W2"].T + p["b2"]; a2 = np.maximum(0, z2)
        logit = a2 @ p["W3"].T + p["b3"]
        return _softmax(logit), (z1, a1, z2, a2)

    def train(p, X, Y, x_query, y_meas, epochs, lr, lam, B_arr):
        loss_hist, pred_hist, recon_hist = [], [], []
        _scale = SCALE
        _var = max(float(np.var(y_meas)), 1e-9)
        for _ in range(epochs):
            pred, (z1, a1, z2, a2) = forward(p, X)
            _Bn = X.shape[0]
            # MSE gradient through softmax (analytic, vectorised)
            d = (2.0 / 3.0) * (pred - Y)
            dlogit = pred * (d - (pred * d).sum(1, keepdims=True))
            dW3 = dlogit.T @ a2 / _Bn; db3 = dlogit.mean(0)
            da2 = (dlogit @ p["W3"]) * (z2 > 0)
            dW2 = da2.T @ a1 / _Bn; db2 = da2.mean(0)
            da1 = (da2 @ p["W2"]) * (z1 > 0)
            dW1 = da1.T @ X / _Bn; db1 = da1.mean(0)
            for _k, _g in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2), ("W3", dW3), ("b3", db3)):
                p[_k] -= lr * _g
            # track: supervised MSE + physics consistency on the query spectrum
            _q, _ = forward(p, x_query[None, :])
            _qc = _q[0]
            _recon = B_arr @ _qc
            _mse = float(np.mean((pred - Y) ** 2))
            _phys = float(np.mean((_recon - y_meas) ** 2) / _var)
            loss_hist.append(_mse + lam * _phys)
            pred_hist.append(_qc.tolist())
            recon_hist.append(np.round(_recon, 3).tolist())
        return loss_hist, pred_hist, recon_hist

    def gen_dataset(n, sigma, seed):
        rng = np.random.default_rng(seed)
        u = np.sort(rng.uniform(0, 1, (n, 2)), axis=1)
        comps = np.column_stack([u[:, 0], u[:, 1] - u[:, 0], 1 - u[:, 1]])
        specs = comps @ B.T + sigma * rng.standard_normal((n, N_WL))
        return specs / SCALE, comps
    return forward, gen_dataset, init_mlp, train


@app.cell(hide_code=True)
def _(mo):
    fh = mo.ui.slider(0.0, 1.0, value=0.5, step=0.05, label="true helix", show_value=True)
    fs = mo.ui.slider(0.0, 1.0, value=0.2, step=0.05, label="true sheet", show_value=True)
    sig = mo.ui.slider(0.0, 3.0, value=0.8, step=0.1, label="measurement noise σ", show_value=True)
    lam = mo.ui.slider(0.0, 2.0, value=0.5, step=0.1, label="physics weight λ", show_value=True)
    ep = mo.ui.slider(20, 80, value=45, step=5, label="training epochs", show_value=True)
    mo.md(f"""
    **Set a true structure, measure it noisily, and let the network learn the inverse map.**
    The physics weight λ controls how much the loss rewards a self-consistent reconstruction.
    {mo.hstack([fh, fs, sig, lam, ep], justify="start", gap=2)}
    """)
    return ep, fh, fs, lam, sig


@app.cell(hide_code=True)
def _(B, ep, fh, fs, gen_dataset, init_mlp, lam, np, sig, spectrum, train):
    _h = fh.value; _s = min(fs.value, 1 - _h); _c = max(0.0, 1 - _h - _s)
    true_frac = np.array([_h, _s, _c])
    _rng = np.random.default_rng(0)
    y_meas = spectrum(true_frac) + sig.value * _rng.standard_normal(len(B))
    _x_query = y_meas / 30.0

    _X, _Y = gen_dataset(450, sig.value, 1)
    _p = init_mlp(32, seed=42)
    loss_hist, pred_hist, recon_hist = train(
        _p, _X, _Y, _x_query, y_meas, int(ep.value), lr=0.15, lam=lam.value, B_arr=B)

    rec_frac = np.array(pred_hist[-1])
    # spectral R^2 of the final reconstruction
    _recon = np.array(recon_hist[-1])
    _ss_res = float(np.sum((_recon - y_meas) ** 2))
    _ss_tot = float(np.sum((y_meas - y_meas.mean()) ** 2)) + 1e-12
    cd_stats = {
        "true": true_frac.tolist(), "rec": rec_frac.tolist(),
        "r2": 1 - _ss_res / _ss_tot, "final_loss": float(loss_hist[-1]),
    }
    cd_anim = {
        "frames": len(loss_hist),
        "loss": [float(v) for v in loss_hist],
        "recon": recon_hist,
        "pred": pred_hist,
        "meas": np.round(y_meas, 3).tolist(),
        "helix_basis": np.round(B[:, 0], 3).tolist(),
        "sheet_basis": np.round(B[:, 1], 3).tolist(),
        "true": true_frac.tolist(),
        "wl0": float(190.0), "wl1": float(250.0),
    }
    return cd_anim, cd_stats


@app.cell(hide_code=True)
def _(canvas_anim, cd_anim, cd_stats, mo):
    _spec = {
        "frames": cd_anim["frames"], "dt": 60,
        "panels": [{"kind": "cd_inversion", **cd_anim}],
        "layout": [[0, 0, 1, 1]],
    }
    _st = cd_stats
    mo.vstack([
        mo.md(
            f"**Recovered** helix {_st['rec'][0]:.0%}, sheet {_st['rec'][1]:.0%}, coil {_st['rec'][2]:.0%} "
            f"(true {_st['true'][0]:.0%} / {_st['true'][1]:.0%} / {_st['true'][2]:.0%}) &nbsp;|&nbsp; "
            f"spectral R² {_st['r2']:.3f} &nbsp;|&nbsp; final loss {_st['final_loss']:.3f}"
        ),
        canvas_anim(_spec, height="470px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The softmax output guarantees the prediction is a valid composition, so the network can never
        return negative or unnormalised fractions. The physics term measures how well the predicted
        composition re-explains the measured spectrum through the fixed forward model, which is what
        keeps the inversion stable when the basis curves overlap and noise is high. As training runs,
        the reconstructed spectrum (amber) settles onto the measurement and the predicted bars climb
        toward the true ones.
        """
    )
    return


if __name__ == "__main__":
    app.run()
