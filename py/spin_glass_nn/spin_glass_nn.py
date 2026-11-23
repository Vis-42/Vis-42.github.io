import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Spin Glass & Neural Nets")


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
        # Spin glass physics of neural networks

        Training a network is a walk on a high-dimensional loss landscape, and the
        statistical physics of **spin glasses** is the standard language for landscapes
        like that: many saddles, few true minima, and a link between how high you are and
        how many directions still lead down. The object to look at is the **Hessian**, the
        matrix of second derivatives of the loss, and its eigenvalues.

        Two numbers summarise the spectrum. The **index** is the fraction of eigenvalues
        that are negative: directions along which the loss still curves downward. The
        **inverse participation ratio** (IPR) of the eigenvalue magnitudes says how the
        curvature is shared out: near $1/P$ it is spread over all $P$ directions, near 1 a
        single direction carries it all. Here a small two-layer MLP learns a 2-D
        classification task, and the decision boundary, the loss and the full Hessian
        spectrum are tracked on one shared animation clock.
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
def _(np):
    # Dataset generators

    def make_spiral(n_per_class=60, noise=0.10, seed=42):
        rng = np.random.default_rng(seed)
        n = n_per_class * 2
        X = np.zeros((n, 2)); y = np.zeros(n, dtype=int)
        for c in range(2):
            ix = slice(c * n_per_class, (c + 1) * n_per_class)
            r = np.linspace(0.0, 1.0, n_per_class)
            t = np.linspace(c * 4.0, (c + 2) * 4.0, n_per_class) + rng.standard_normal(n_per_class) * noise
            X[ix, 0] = r * np.sin(t)
            X[ix, 1] = r * np.cos(t)
            y[ix] = c
        return X, y

    def make_xor(N=200, noise=0.10, seed=42):
        rng = np.random.default_rng(seed)
        X = 2.0 * rng.random((N, 2)) - 1.0 + rng.standard_normal((N, 2)) * noise
        y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(int)
        return X, y

    def make_checkerboard(N=200, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((N, 2))
        y = ((np.floor(X[:, 0]).astype(int) + np.floor(X[:, 1]).astype(int)) % 2 == 0).astype(int)
        return X, y

    return make_checkerboard, make_spiral, make_xor


@app.cell(hide_code=True)
def _(np):
    # MLP: input=2, hidden=H, output=1 (sigmoid), tanh hidden activations

    def mlp_init(H, seed=42):
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / 2)
        s2 = np.sqrt(2.0 / H)
        W1 = rng.standard_normal((H, 2)) * s1
        b1 = np.zeros(H)
        W2 = rng.standard_normal(H) * s2
        b2 = np.zeros(1)
        return np.concatenate([W1.ravel(), b1, W2, b2])

    def mlp_unpack(th, H):
        i = 0
        W1 = th[i:i + H * 2].reshape(H, 2); i += H * 2
        b1 = th[i:i + H]; i += H
        W2 = th[i:i + H]; i += H
        b2 = th[i]
        return W1, b1, W2, b2

    def mlp_forward(th, X, H):
        W1, b1, W2, b2 = mlp_unpack(th, H)
        a1 = np.tanh(X @ W1.T + b1)
        logit = a1 @ W2 + b2
        return 1.0 / (1.0 + np.exp(-logit))

    def mlp_loss(th, X, y, H):
        p = np.clip(mlp_forward(th, X, H), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def mlp_grad(th, X, y, H):
        W1, b1, W2, b2 = mlp_unpack(th, H)
        n = len(y)
        z1 = X @ W1.T + b1
        a1 = np.tanh(z1)
        p = np.clip(1.0 / (1.0 + np.exp(-(a1 @ W2 + b2))), 1e-7, 1.0 - 1e-7)
        dz2 = (p - y) / n
        gW2 = a1.T @ dz2
        gb2 = dz2.sum()
        da1 = np.outer(dz2, W2) * (1.0 - a1 ** 2)
        gW1 = da1.T @ X
        gb1 = da1.sum(0)
        return np.concatenate([gW1.ravel(), gb1, gW2, [gb2]])

    def mlp_hess_eigs(th, X, y, H, eps=1e-4):
        """Eigenvalues of the full P x P Hessian. Row k is the change in the analytic
        gradient when parameter k is nudged, so the whole matrix costs 2P gradients."""
        n = len(th)
        Hm = np.empty((n, n))
        for k in range(n):
            e = np.zeros(n); e[k] = eps
            Hm[k] = (mlp_grad(th + e, X, y, H) - mlp_grad(th - e, X, y, H)) / (2 * eps)
        return np.linalg.eigvalsh(0.5 * (Hm + Hm.T))

    def mlp_ipr(curvatures):
        a = np.abs(curvatures)
        s = a.sum()
        if s < 1e-12:
            return 1.0
        return float(((a / s) ** 2).sum())

    return mlp_forward, mlp_grad, mlp_hess_eigs, mlp_init, mlp_ipr, mlp_loss, mlp_unpack


@app.cell(hide_code=True)
def _(mo):
    dataset_sl = mo.ui.dropdown(
        options={"Spiral": "spiral", "XOR": "xor", "Checkerboard": "checkerboard"},
        value="Spiral",
        label="Dataset",
    )
    H_sl = mo.ui.slider(6, 32, value=16, step=2, label="Hidden units H", show_value=True)
    lr_sl = mo.ui.slider(0.005, 0.10, value=0.03, step=0.005, label="Learning rate (Adam)", show_value=True)
    epochs_sl = mo.ui.slider(100, 600, value=400, step=100, label="Epochs", show_value=True)
    mo.md(
        f"""
        **Controls.**
        {mo.hstack([dataset_sl, H_sl, lr_sl, epochs_sl], justify="start", gap=2)}
        """
    )
    return H_sl, dataset_sl, epochs_sl, lr_sl


@app.cell(hide_code=True)
def _(
    H_sl,
    dataset_sl,
    epochs_sl,
    lr_sl,
    make_checkerboard,
    make_spiral,
    make_xor,
    mlp_forward,
    mlp_grad,
    mlp_hess_eigs,
    mlp_init,
    mlp_ipr,
    mlp_loss,
    np,
):
    _ds = dataset_sl.value
    _H = int(H_sl.value)
    _lr = float(lr_sl.value)
    _n_epochs = int(epochs_sl.value)
    _snap_every = max(5, _n_epochs // 60)

    # Build dataset
    if _ds == "spiral":
        _X, _y = make_spiral(n_per_class=60, noise=0.10, seed=42)
    elif _ds == "xor":
        _X, _y = make_xor(N=200, noise=0.10, seed=42)
    else:
        _X, _y = make_checkerboard(N=200, seed=42)

    # Normalise X to roughly [-1, 1] for display
    _xmn = _X.min(0); _xmx = _X.max(0)
    _xrng = np.maximum(_xmx - _xmn, 1e-6)
    _Xn = (_X - _xmn) / _xrng * 2.0 - 1.0

    # Grid for decision boundary heatmap, 50x50
    _GN = 50
    _gv = np.linspace(-1.0, 1.0, _GN)
    _GX, _GY = np.meshgrid(_gv, _gv)
    _grid_pts = np.c_[_GX.ravel(), _GY.ravel()]

    # Training loop with snapshots
    _th = mlp_init(_H, seed=42)
    _snap_epochs = []
    _snap_loss = []
    _snap_ipr = []
    _snap_grid = []    # list of flat uint8 lists (heatmap frames)
    _snap_hess = []    # Hessian eigenvalues at each snap
    _snap_neg = []     # fraction of negative eigenvalues (the index)

    # Full-batch Adam. Plain gradient descent at these learning rates barely moves
    # off the initial plateau in a few hundred steps.
    _m = np.zeros_like(_th); _v = np.zeros_like(_th)
    for _ep in range(1, _n_epochs + 1):
        _g = mlp_grad(_th, _Xn, _y, _H)
        _m = 0.9 * _m + 0.1 * _g; _v = 0.999 * _v + 0.001 * _g * _g
        _th = _th - _lr * (_m / (1 - 0.9 ** _ep)) / (np.sqrt(_v / (1 - 0.999 ** _ep)) + 1e-8)
        if _ep == 1 or _ep % _snap_every == 0 or _ep == _n_epochs:
            _pb = mlp_forward(_th, _grid_pts, _H).reshape(_GN, _GN)
            # Flip y so increasing row = increasing y coordinate
            _snap_grid.append((_pb[::-1, :] * 255).astype("uint8").ravel().tolist())
            _snap_loss.append(mlp_loss(_th, _Xn, _y, _H))
            _hd = mlp_hess_eigs(_th, _Xn, _y, _H)
            _snap_ipr.append(mlp_ipr(_hd))
            _snap_neg.append(float(np.mean(_hd < -1e-4 * np.abs(_hd).max())))
            _snap_hess.append(_hd)
            _snap_epochs.append(_ep)

    # Histogram bins from all snapshots combined (fixed edges); the few eigenvalues
    # beyond the 2nd/98th percentile are counted in the end bins rather than dropped.
    _all_curv = np.concatenate(_snap_hess)
    _edge_lo = float(np.percentile(_all_curv, 2))
    _edge_hi = float(np.percentile(_all_curv, 98))
    _n_bins = 30
    _edges = np.linspace(_edge_lo, _edge_hi, _n_bins + 1)
    _hess_counts = []
    for _hd in _snap_hess:
        _c, _ = np.histogram(np.clip(_hd, _edge_lo, _edge_hi), bins=_edges)
        _hess_counts.append(_c.tolist())

    # Data points for overlay: normalised to [0,1] in display space
    # grid spans [-1,1] so fraction = (v+1)/2
    _pts_overlay = [
        [float((_Xn[i, 0] + 1.0) / 2.0), float(1.0 - (_Xn[i, 1] + 1.0) / 2.0), int(_y[i])]
        for i in range(len(_y))
    ]

    _n_snaps = len(_snap_epochs)
    _n_params = len(_th)
    _final_loss = _snap_loss[-1]
    _final_ipr = _snap_ipr[-1]
    _final_pr = 1.0 / max(_final_ipr, 1e-9)
    _acc = float(np.mean((mlp_forward(_th, _Xn, _H) > 0.5) == _y))

    training_result = {
        "n_snaps": _n_snaps,
        "snap_epochs": _snap_epochs,
        "snap_loss": _snap_loss,
        "snap_ipr": _snap_ipr,
        "snap_neg": _snap_neg,
        "acc": _acc,
        "snap_grid": _snap_grid,
        "hess_counts": _hess_counts,
        "hess_edges": _edges.tolist(),
        "pts_overlay": _pts_overlay,
        "n_params": _n_params,
        "final_loss": _final_loss,
        "final_ipr": _final_ipr,
        "final_pr": _final_pr,
        "GN": _GN,
    }
    return (training_result,)


@app.cell(hide_code=True)
def _(mo, training_result):
    _r = training_result
    _loss_color = (
        "#57c98a" if _r["final_loss"] < 0.35
        else "#e6a356" if _r["final_loss"] < 0.60
        else "#e0635a"
    )
    mo.md(
        f"""
        <div style="font-family:ui-sans-serif,system-ui,sans-serif;display:grid;grid-template-columns:repeat(4,1fr);
        border:1px solid #232936;border-radius:6px;overflow:hidden;margin:4px 0;background:#161a23;">
          <div style="padding:13px 18px;border-right:1px solid #232936;">
            <div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;">Final loss / accuracy</div>
            <div style="font-size:18px;font-weight:600;font-family:ui-monospace,monospace;color:{_loss_color};">{_r["final_loss"]:.3f} / {_r["acc"]:.0%}</div>
          </div>
          <div style="padding:13px 18px;border-right:1px solid #232936;">
            <div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;">IPR, first to last</div>
            <div style="font-size:18px;color:#a594f2;font-weight:600;font-family:ui-monospace,monospace;">{_r["snap_ipr"][0]:.2f} → {_r["final_ipr"]:.2f}</div>
          </div>
          <div style="padding:13px 18px;border-right:1px solid #232936;">
            <div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;">Negative eigenvalues, first to last</div>
            <div style="font-size:16px;color:#57c98a;font-weight:600;font-family:ui-monospace,monospace;">{_r["snap_neg"][0]:.0%} → {_r["snap_neg"][-1]:.0%}</div>
          </div>
          <div style="padding:13px 18px;">
            <div style="font-size:10px;color:#626b7c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;">Snapshots / params</div>
            <div style="font-size:16px;color:#e6a356;font-weight:600;font-family:ui-monospace,monospace;">{_r["n_snaps"]} / {_r["n_params"]}</div>
          </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(canvas_anim, mo, training_result):
    _r = training_result
    _spec = {
        "frames": _r["n_snaps"],
        "dt": 300,
        "panels": [
            {
                "kind": "heatmap_pts", "title": "Decision boundary (probability field)",
                "n": _r["GN"],
                "grid": _r["snap_grid"],
                "pts": _r["pts_overlay"],
                "epochs": _r["snap_epochs"],
            },
            {
                "kind": "timeseries", "title": "Loss, IPR and index vs epoch", "xlabel": "epoch", "ylabel": "value",
                "ys": [_r["snap_loss"], _r["snap_ipr"], _r["snap_neg"]],
                "t": _r["snap_epochs"],
                "colors": ["#e0635a", "#7dd3fc", "#57c98a"],
                "labels": ["loss", "IPR", "index"],
                "ymin": 0.0,
                "grow": True,
            },
            {
                "kind": "histogram_live", "title": "Hessian eigenvalue spectrum", "xlabel": "eigenvalue", "ylabel": "count",
                "counts": _r["hess_counts"],
                "edges": _r["hess_edges"],
            },
        ],
        "layout": [
            [0.00, 0.00, 0.45, 1.00],
            [0.45, 0.00, 0.55, 0.50],
            [0.45, 0.50, 0.55, 0.50],
        ],
    }
    mo.vstack([
        mo.md(
            "**Three panels, one epoch clock.** "
            "Left: the network's output probability as a heatmap (blue = class 0, red = class 1) with the data points. "
            "Top-right: loss (red), IPR (cyan) and the fraction of negative Hessian eigenvalues (green). "
            "Bottom-right: histogram of all Hessian eigenvalues at the current snapshot."
        ),
        canvas_anim(_spec, height="520px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **What this shows.**

        At initialisation the network sits on a saddle, not in a basin: about half of the
        Hessian eigenvalues are negative, and the curvature is spread over many directions,
        so the IPR is low. As the loss falls the negative eigenvalues disappear one by one
        (the green index curve drops toward zero) and the spectrum splits in two. Most
        eigenvalues collapse into a spike at zero, flat directions the data does not
        constrain, while a handful of large positive ones carry almost all the curvature
        and pull the IPR up. On the spiral the IPR keeps climbing as long as training
        runs; on XOR it rises early and then levels off.

        The falling index is the spin-glass part of the story. In mean-field glass models
        the typical critical point at high energy is a saddle with many downhill
        directions, and the number of downhill directions shrinks as the energy drops, so
        that only near the bottom of the landscape are critical points true minima. A
        gradient-based optimiser on a small network traces the same relation between
        height and index. The spike-plus-outliers shape at the end is the generic picture
        for trained networks: a few stiff directions, many soft ones.

        **How it is computed.** The gradient is analytic. Each row of the Hessian is a
        central finite difference of that gradient in one parameter, so the full
        $P \times P$ matrix costs $2P$ gradient evaluations per snapshot, and its
        eigenvalues come from a symmetric eigensolver. With $P$ around 65 to 130 that is
        cheap enough to do at every snapshot. An eigenvalue counts as negative when it is
        below $-10^{-4}$ times the largest magnitude in the spectrum. Training is
        full-batch Adam.
        """
    )
    return


if __name__ == "__main__":
    app.run()
