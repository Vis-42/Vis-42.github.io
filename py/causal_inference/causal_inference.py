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
def _(ccm_data, json, mo):
    _d = ccm_data
    _JS = r"""<!doctype html><html><head><meta charset="utf-8"><style>
    html,body{margin:0;background:#0f1117;overflow:hidden}canvas{display:block;width:100vw;height:100vh}
    </style></head><body><canvas id="c"></canvas><script>
    /* ============================================================
   CANVAS ENGINE v3 — shared template for all marimo notebooks
   Injected by replacing __SPEC__ with json.dumps(spec).

   Spec root fields:
     frames, dt, panels[], layout[], bg (optional "#rrggbb")

   Per-panel fields (all optional where sensible):
     kind       – "timeseries"|"scatter"|"heatmap"|"particles"|...
     title      – panel title shown at top
     xlabel     – x-axis label (may include units e.g. "position x (nm)")
     ylabel     – y-axis label
     xmin,xmax  – explicit data range (auto-computed if absent)
     ymin,ymax  – explicit data range (auto-computed if absent)
     logx,logy  – true for log-scale axis
   ============================================================ */

const S=__SPEC__,cv=document.getElementById("c"),ctx=cv.getContext("2d");
const DPR=Math.min(window.devicePixelRatio||1,2);
const BG=S.bg||"#0f1117";

/* ── Canvas sizing ─────────────────────────────────────────── */
function fit(){const w=cv.offsetWidth||window.innerWidth||900,h=cv.offsetHeight||window.innerHeight||500;if(w>0&&h>0){cv.width=w*DPR;cv.height=h*DPR;}}
addEventListener("resize",fit);

/* ── Animation state ────────────────────────────────────────── */
let frame=0,last=0,playing=true,speed=1.0;

/* ── Per-panel zoom/pan state ───────────────────────────────── */
const VP=S.panels.map(()=>({zoom:1,px:0,py:0}));

/* ── Mouse/touch state for zoom+pan ────────────────────────── */
let drag=false,dragPi=-1,dragX=0,dragY=0;
function panelAt(cx,cy){
  const W=cv.width,H=cv.height;
  for(let i=0;i<S.layout.length;i++){
    const[lx,ly,lw,lh]=S.layout[i];
    if(cx>=lx*W&&cx<=(lx+lw)*W&&cy>=ly*H&&cy<=(ly+lh)*H)return i;
  }return -1;
}
cv.addEventListener("mousedown",e=>{drag=true;const r=cv.getBoundingClientRect();dragX=e.clientX-r.left;dragY=e.clientY-r.top;dragPi=panelAt(dragX*DPR,dragY*DPR);});
cv.addEventListener("mousemove",e=>{
  if(!drag||dragPi<0)return;
  const r=cv.getBoundingClientRect(),nx=(e.clientX-r.left),ny=(e.clientY-r.top);
  const dx=(nx-dragX)/cv.clientWidth,dy=(ny-dragY)/cv.clientHeight;
  VP[dragPi].px+=dx;VP[dragPi].py+=dy;dragX=nx;dragY=ny;
});
cv.addEventListener("mouseup",()=>drag=false);
cv.addEventListener("mouseleave",()=>drag=false);
cv.addEventListener("wheel",e=>{
  const r=cv.getBoundingClientRect(),cx=(e.clientX-r.left)*DPR,cy=(e.clientY-r.top)*DPR;
  const pi=panelAt(cx,cy);if(pi<0)return;
  const factor=Math.exp(-e.deltaY*0.001);
  VP[pi].zoom=Math.max(0.15,Math.min(20,VP[pi].zoom*factor));
  e.preventDefault();
},{passive:false});
cv.addEventListener("dblclick",e=>{
  const r=cv.getBoundingClientRect(),cx=(e.clientX-r.left)*DPR,cy=(e.clientY-r.top)*DPR;
  const pi=panelAt(cx,cy);if(pi>=0){VP[pi].zoom=1;VP[pi].px=0;VP[pi].py=0;}
});
cv.addEventListener("touchstart",e=>{const t=e.touches[0];const r=cv.getBoundingClientRect();dragX=t.clientX-r.left;dragY=t.clientY-r.top;dragPi=panelAt(dragX*DPR,dragY*DPR);drag=true;e.preventDefault();},{passive:false});
cv.addEventListener("touchmove",e=>{if(!drag||dragPi<0)return;const t=e.touches[0];const r=cv.getBoundingClientRect();const nx=t.clientX-r.left,ny=t.clientY-r.top;VP[dragPi].px+=(nx-dragX)/cv.clientWidth;VP[dragPi].py+=(ny-dragY)/cv.clientHeight;dragX=nx;dragY=ny;e.preventDefault();},{passive:false});
cv.addEventListener("touchend",()=>drag=false);

/* ── Tick generation ────────────────────────────────────────── */
function niceNum(x,round){if(x===0)return 0;const s=x<0?-1:1;x=Math.abs(x);const e=Math.floor(Math.log10(x)),f=x/Math.pow(10,e);let nf;if(round)nf=f<1.5?1:f<3?2:f<7?5:10;else nf=f<=1?1:f<=2?2:f<=5?5:10;return s*nf*Math.pow(10,e);}
function genTicks(vmin,vmax,n=5){
  if(vmax<=vmin)return[vmin,vmax];
  const d=niceNum((vmax-vmin)/(n-1),true);
  const lo=Math.ceil(vmin/d)*d,hi=Math.floor(vmax/d)*d;
  const t=[];for(let v=lo;v<=hi+d*0.5;v+=d)t.push(parseFloat(v.toPrecision(10)));
  return t.length?t:[vmin,vmax];
}
function logTicks(vmin,vmax){
  const lo=Math.floor(Math.log10(Math.max(vmin,1e-30))),hi=Math.ceil(Math.log10(Math.max(vmax,1e-30)));
  const t=[];for(let e=lo;e<=hi;e++){t.push(Math.pow(10,e));if(hi-lo<3){t.push(2*Math.pow(10,e));t.push(5*Math.pow(10,e));}}
  return t.filter(v=>v>=vmin*0.99&&v<=vmax*1.01);
}
function fmtN(v){
  if(v===0)return"0";const a=Math.abs(v);
  if(a>=1e4||a<1e-3)return v.toExponential(1).replace("e+","e").replace("e0","e").replace("e-0","e-");
  if(a>=100)return v.toFixed(0);if(a>=10)return v.toFixed(1);if(a>=1)return v.toFixed(2);
  return v.toFixed(3).replace(/0+$/,"").replace(/\.$/,"");
}

/* ── Core 2D axis renderer with grid + ticks ───────────────── */
function axes2D(p,lx,ly,lw,lh,xd,yd,vp){
  const W=cv.width,H=cv.height;
  const px=Math.round(lx*W),py=Math.round(ly*H),pw=Math.round(lw*W),ph=Math.round(lh*H);
  const hasTitle=p.title&&p.title.length>0;
  const ML=46*DPR,MR=8*DPR,MT=(hasTitle?22:8)*DPR,MB=28*DPR;
  const mx=px+ML,my=py+MT,mw=pw-ML-MR,mh=ph-MT-MB;

  /* data→screen transforms with zoom+pan */
  const z=vp.zoom;
  const toX=v=>mx+mw*((v-xd.min)/(xd.max-xd.min)*z+vp.px);
  const toY=v=>my+mh*(1-(v-yd.min)/(yd.max-yd.min)*z+vp.py);
  /* screen→data */
  const fromX=sx=>(sx-mx)/mw/z*(xd.max-xd.min)+xd.min-vp.px*(xd.max-xd.min)/z;

  /* clip to plot area */
  ctx.save();ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.clip();

  /* grid */
  const xticks=xd.log?logTicks(xd.min,xd.max):genTicks(xd.min,xd.max,6);
  const yticks=yd.log?logTicks(yd.min,yd.max):genTicks(yd.min,yd.max,5);
  ctx.strokeStyle="rgba(255,255,255,0.055)";ctx.lineWidth=1;
  for(const v of xticks){const x=xd.log?toX(Math.log10(v)):toX(v);if(x<mx||x>mx+mw)continue;ctx.beginPath();ctx.moveTo(x,my);ctx.lineTo(x,my+mh);ctx.stroke();}
  for(const v of yticks){const y=yd.log?toY(Math.log10(v)):toY(v);if(y<my||y>my+mh)continue;ctx.beginPath();ctx.moveTo(mx,y);ctx.lineTo(mx+mw,y);ctx.stroke();}
  ctx.restore();

  /* axis border */
  ctx.strokeStyle="#243040";ctx.lineWidth=1.2;
  ctx.strokeRect(mx,my,mw,mh);

  /* tick marks + labels */
  ctx.fillStyle="#8a93a6";ctx.font=(9.5*DPR)+"px monospace";
  ctx.textAlign="center";
  for(const v of xticks){
    const x=xd.log?toX(Math.log10(v)):toX(v);if(x<mx-1||x>mx+mw+1)continue;
    ctx.fillStyle="#1e2535";ctx.fillRect(x-0.5,my+mh,1,4*DPR);
    ctx.fillStyle="#8a93a6";ctx.fillText(xd.log?("10^"+Math.round(Math.log10(v))):fmtN(v),x,py+ph-2*DPR);
  }
  ctx.textAlign="right";
  for(const v of yticks){
    const y=yd.log?toY(Math.log10(v)):toY(v);if(y<my-1||y>my+mh+1)continue;
    ctx.fillStyle="#1e2535";ctx.fillRect(mx-4*DPR,y-0.5,4*DPR,1);
    ctx.fillStyle="#8a93a6";ctx.fillText(yd.log?("10^"+Math.round(Math.log10(v))):fmtN(v),mx-5*DPR,y+3.5*DPR);
  }

  /* axis labels + title */
  ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";
  if(p.xlabel){ctx.textAlign="center";ctx.fillText(p.xlabel,mx+mw/2,py+ph);}
  if(p.ylabel){ctx.save();ctx.translate(px+11*DPR,my+mh/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillText(p.ylabel,0,0);ctx.restore();}
  if(p.title){ctx.fillStyle="#b3bac8";ctx.font=(10.5*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(p.title,mx+mw/2,py+13*DPR);}

  return{mx,my,mw,mh,toX,toY,fromX};
}

/* ── Legend helper ──────────────────────────────────────────── */
function legend(px,py,pw,ph,entries){
  const lx=px+pw-8*DPR,ly=py+14*DPR;
  ctx.font=(9*DPR)+"px monospace";ctx.textAlign="right";
  entries.forEach(([label,col],i)=>{
    const y=ly+i*13*DPR;
    ctx.fillStyle=col;ctx.fillRect(lx-28*DPR,y-6*DPR,22*DPR,2.5*DPR);
    ctx.fillStyle="#8a93a6";ctx.fillText(label,lx,y);
  });
}

/* ── HSV helper ─────────────────────────────────────────────── */
function hsv(h,s,v){let r,g,b,i=Math.floor(h*6),f=h*6-i,p=v*(1-s),q=v*(1-f*s),t=v*(1-(1-f)*s);switch(i%6){case 0:r=v,g=t,b=p;break;case 1:r=q,g=v,b=p;break;case 2:r=p,g=v,b=t;break;case 3:r=p,g=q,b=v;break;case 4:r=t,g=p,b=v;break;case 5:r=v,g=p,b=q;break;}return`rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;}

/* ── 3D orbit camera ────────────────────────────────────────── */
let az3=0.52,el3=0.38,zoom3=1.0,drag3=false,lx3=0,ly3=0;
function P3(wx,wy,wz,cxp,cyp,sc){const ca=Math.cos(az3),sa=Math.sin(az3);const x1=wx*ca+wy*sa,y1=-wx*sa+wy*ca,z1=wz;const ce=Math.cos(el3),se=Math.sin(el3);return[cxp+x1*sc*zoom3,cyp-(-y1*se+z1*ce)*sc*zoom3];}

/* ── Panel rendering ────────────────────────────────────────── */
function drawPanel(p,pi,lx,ly,lw,lh){
  const W=cv.width,H=cv.height;
  const px=Math.round(lx*W),py=Math.round(ly*H),pw=Math.round(lw*W),ph=Math.round(lh*H);
  ctx.save();ctx.beginPath();ctx.rect(px,py,pw,ph);ctx.clip();
  const f=Math.min(frame,S.frames-1);
  const vp=VP[pi];

  /* ─────────────── PARTICLES ─────────────── */
  if(p.kind==="particles"){
    const xs=p.x[f],ys=p.y[f],ths=p.theta?p.theta[f]:null,L=p.L;
    const sc=Math.min(pw,ph)*0.9/L,ox=px+(pw-L*sc)/2,oy=py+(ph-L*sc)/2,r=Math.max(1.5,(p.r||0.3)*sc);
    ctx.strokeStyle="#243040";ctx.lineWidth=1;ctx.strokeRect(ox,oy,L*sc,L*sc);
    for(let i=0;i<xs.length;i++){
      const clr=ths?hsv(((ths[i]%(2*Math.PI))+2*Math.PI)%(2*Math.PI)/(2*Math.PI),0.85,0.95):"#7dd3fc";
      ctx.fillStyle=clr;ctx.beginPath();ctx.arc(ox+xs[i]*sc,oy+ys[i]*sc,r,0,7);ctx.fill();
    }
    /* scale bar */
    const bar=Math.round(L/4)*1,barPx=bar*sc;
    ctx.strokeStyle="#5a6880";ctx.lineWidth=1.5*DPR;
    ctx.beginPath();ctx.moveTo(ox+4*DPR,oy+L*sc-8*DPR);ctx.lineTo(ox+4*DPR+barPx,oy+L*sc-8*DPR);ctx.stroke();
    ctx.fillStyle="#5a6880";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText(fmtN(bar),ox+4*DPR,oy+L*sc-12*DPR);
    if(p.title){ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText(p.title,px+4*DPR,py+12*DPR);}
    if(p.eta_val){ctx.fillStyle="#626b7c";ctx.fillText("η="+p.eta_val[f].toFixed(2)+"  N="+xs.length,px+4*DPR,py+22*DPR);}
  }

  /* ─────────────── TIMESERIES (single or multi-series growing line) ─── */
  else if(p.kind==="timeseries"||p.kind==="phi_t"||p.kind==="r_t"||p.kind==="grow_curve"){
    const ys_arr=p.ys||p.ys_arr||(p.phi?[p.phi]:(p.r?[p.r]:(p.y?[p.y]:[])));
    const n=ys_arr[0]?ys_arr[0].length:0;
    const ts=p.t||(Array.from({length:n},(_,i)=>i));
    let ymin=p.ymin!==undefined?p.ymin:Infinity,ymax=p.ymax!==undefined?p.ymax:-Infinity;
    if(ymin===Infinity)for(const ya of ys_arr)for(const v of ya){if(v<ymin)ymin=v;if(v>ymax)ymax=v;}
    ymin=p.ymin!==undefined?p.ymin:ymin-Math.abs(ymax-ymin)*0.05;
    ymax=p.ymax!==undefined?p.ymax:ymax+Math.abs(ymax-ymin)*0.05;
    const xd={min:ts[0],max:ts[n-1]||1,log:false},yd={min:ymin,max:ymax||1,log:!!p.logy};
    const{mx,my,mw,mh,toX,toY}=axes2D(p,lx,ly,lw,lh,xd,yd,vp);
    ctx.save();ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.clip();
    const colors=p.colors||["#7dd3fc","#f97316","#a594f2","#5ed28c","#9c3b2c"];
    const upTo=p.grow===false?n:Math.min(f+1,n);
    for(let si=0;si<ys_arr.length;si++){
      const ya=ys_arr[si];ctx.strokeStyle=colors[si%colors.length];ctx.lineWidth=2*DPR;ctx.beginPath();
      for(let i=0;i<upTo;i++){const x=toX(ts[i]),y=yd.log?toY(Math.log10(Math.max(ya[i],1e-30))):toY(ya[i]);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
      ctx.stroke();
    }
    /* moving dot on first series */
    const di=p.dot!==undefined?p.dot:0;if(di<ys_arr.length&&upTo>0){
      const iv=Math.min(upTo-1,n-1);const x=toX(ts[iv]),y=yd.log?toY(Math.log10(Math.max(ys_arr[di][iv],1e-30))):toY(ys_arr[di][iv]);
      ctx.fillStyle="#f97316";ctx.beginPath();ctx.arc(x,y,4.5*DPR,0,7);ctx.fill();
    }
    ctx.restore();
    /* vlines */
    if(p.vlines)for(const vl of p.vlines){const x=toX(vl.x);if(x<mx||x>mx+mw)continue;ctx.strokeStyle=vl.color||"#3a4255";ctx.lineWidth=1;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(x,my);ctx.lineTo(x,my+mh);ctx.stroke();ctx.setLineDash([]);if(vl.label){ctx.fillStyle=vl.color||"#5a6880";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText(vl.label,x+3*DPR,my+10*DPR);}}
    if(p.labels||p.colors)legend(px,py,pw,ph,(p.labels||[]).map((l,i)=>[l,(p.colors||colors)[i%colors.length]]));
  }

  /* ─────────────── SCATTER / PHASE_DIAG / BIFURCATION ─── */
  else if(p.kind==="scatter"||p.kind==="scatter_vline"||p.kind==="phase_diag"||p.kind==="bifurcation"){
    const xs=p.x||p.eta_sweep||p.K_sweep||[];
    const ys=p.y||p.phi_sweep||p.r_sweep||[];
    let xmin=p.xmin!==undefined?p.xmin:Math.min(...xs);
    let xmax=p.xmax!==undefined?p.xmax:Math.max(...xs);
    let ymin=p.ymin!==undefined?p.ymin:Math.min(...ys)*0.95;
    let ymax=p.ymax!==undefined?p.ymax:Math.max(...ys)*1.05||1.05;
    const xd={min:xmin,max:xmax,log:false},yd={min:ymin,max:ymax,log:false};
    const{mx,my,mw,mh,toX,toY}=axes2D(p,lx,ly,lw,lh,xd,yd,vp);
    ctx.save();ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.clip();
    /* sweep curve */
    if(xs.length>1){ctx.strokeStyle=p.color||"#9c3b2c";ctx.lineWidth=2.5*DPR;ctx.beginPath();for(let i=0;i<xs.length;i++){const x=toX(xs[i]),y=toY(ys[i]);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.stroke();}
    /* static vlines (theory markers) */
    if(p.Kc!==undefined){const x=toX(p.Kc);ctx.strokeStyle="#4a5568";ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,my);ctx.lineTo(x,my+mh);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle="#4a5568";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText("Kc="+fmtN(p.Kc),x,my+10*DPR);}
    if(p.vlines)for(const vl of p.vlines){const x=toX(vl.x);if(x<mx||x>mx+mw)continue;ctx.strokeStyle=vl.color||"#4a5568";ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,my);ctx.lineTo(x,my+mh);ctx.stroke();ctx.setLineDash([]);if(vl.label){ctx.fillStyle=vl.color||"#5a6880";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(vl.label,x,my+10*DPR);}}
    /* current-frame dot + vline */
    const dotx=p.dot_x?p.dot_x[f]:(p.eta_now?p.eta_now[f]:(p.K_now?p.K_now[f]:null));
    const doty=p.dot_y?p.dot_y[f]:(p.phi_now?p.phi_now[f]:(p.r_now?p.r_now[f]:null));
    if(dotx!==null&&dotx!==undefined){const x=toX(dotx);ctx.strokeStyle="#f97316";ctx.lineWidth=1;ctx.setLineDash([2,2]);ctx.beginPath();ctx.moveTo(x,my);ctx.lineTo(x,my+mh);ctx.stroke();ctx.setLineDash([]);if(doty!==null){const y=toY(doty);ctx.fillStyle="#f97316";ctx.beginPath();ctx.arc(x,y,5*DPR,0,7);ctx.fill();}}
    ctx.restore();
  }

  /* ─────────────── PHASES (Kuramoto circle) ─── */
  else if(p.kind==="phases"){
    const ths=p.theta[f],rv=p.r_series?p.r_series[f]:0;
    const R=Math.min(pw,ph)*0.38,cxc=px+pw/2,cyc=py+ph/2;
    ctx.strokeStyle="#1e2535";ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(cxc,cyc,R,0,7);ctx.stroke();
    /* reference circle ticks at 0/90/180/270 */
    ctx.fillStyle="#3a4255";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="center";
    for(const [a,l] of [[0,"0"],[Math.PI/2,"π/2"],[Math.PI,"π"],[3*Math.PI/2,"3π/2"]]){
      ctx.fillText(l,cxc+Math.cos(a)*(R+10*DPR),cyc-Math.sin(a)*(R+10*DPR)+3*DPR);
    }
    for(const th of ths){ctx.fillStyle="#7dd3fc";ctx.beginPath();ctx.arc(cxc+R*Math.cos(th),cyc-R*Math.sin(th),3*DPR,0,7);ctx.fill();}
    /* mean-field vector */
    const psi=Math.atan2(ths.reduce((a,t)=>a+Math.sin(t),0),ths.reduce((a,t)=>a+Math.cos(t),0));
    ctx.strokeStyle="#f97316";ctx.lineWidth=3*DPR;ctx.beginPath();ctx.moveTo(cxc,cyc);ctx.lineTo(cxc+R*rv*Math.cos(psi),cyc-R*rv*Math.sin(psi));ctx.stroke();
    ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText("r="+rv.toFixed(3),px+5*DPR,py+14*DPR);
    if(p.title){ctx.fillStyle="#b3bac8";ctx.fillText(p.title,px+5*DPR,py+26*DPR);}
  }

  /* ─────────────── HEATMAP (with colorbar) ─── */
  else if(p.kind==="heatmap"||p.kind==="heatmap_pts"){
    const n=p.n,g=p.grid[f];
    if(!p._oc){p._oc=document.createElement("canvas");p._oc.width=n;p._oc.height=n;p._octx=p._oc.getContext("2d");p._img=p._octx.createImageData(n,n);}
    const d=p._img.data;
    if(p.kind==="heatmap"){
      for(let i=0;i<n*n;i++){const v=g[i]/255;d[4*i]=Math.round(255*(0.06+0.35*v));d[4*i+1]=Math.round(255*(0.18+0.55*v));d[4*i+2]=Math.round(255*(0.25+0.55*v));d[4*i+3]=255;}
    } else {
      for(let i=0;i<n*n;i++){const v=g[i]/255;d[4*i]=Math.round(255*(0.18+0.60*v));d[4*i+1]=Math.round(255*(0.42-0.08*v));d[4*i+2]=Math.round(255*(0.75-0.42*v));d[4*i+3]=255;}
    }
    p._octx.putImageData(p._img,0,0);
    const cbW=16*DPR,s=Math.min(pw-cbW-4*DPR,ph)*0.9,ox=px+(pw-cbW-4*DPR-s)/2,oy=py+(ph-s)/2;
    ctx.imageSmoothingEnabled=true;ctx.drawImage(p._oc,ox,oy,s,s);
    /* colorbar */
    const cbX=ox+s+4*DPR,cbY=oy,cbH=s;
    const grad=ctx.createLinearGradient(0,cbY+cbH,0,cbY);
    if(p.kind==="heatmap"){grad.addColorStop(0,"rgb(15,46,64)");grad.addColorStop(0.5,"rgb(26,88,100)");grad.addColorStop(1,"rgb(72,171,156)");}
    else{grad.addColorStop(0,"rgb(46,107,191)");grad.addColorStop(0.5,"rgb(170,142,191)");grad.addColorStop(1,"rgb(237,156,192)");}
    ctx.fillStyle=grad;ctx.fillRect(cbX,cbY,cbW,cbH);
    ctx.strokeStyle="#3a4255";ctx.lineWidth=1;ctx.strokeRect(cbX,cbY,cbW,cbH);
    ctx.fillStyle="#8a93a6";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="left";
    ctx.fillText("1",cbX+cbW+3*DPR,cbY+8*DPR);ctx.fillText("0",cbX+cbW+3*DPR,cbY+cbH+4*DPR);
    /* data points overlay */
    if(p.pts)for(const pt of p.pts){const cx2=ox+pt[0]*s,cy2=oy+pt[1]*s;ctx.fillStyle=pt[2]?"#0f1117":"#ffffff";ctx.beginPath();ctx.arc(cx2,cy2,3*DPR,0,7);ctx.fill();}
    if(p.title){ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(p.title,px+pw/2,py+12*DPR);}
  }

  /* ─────────────── RNT (walker paths + MSD log-log) ─── */
  else if(p.kind==="rnt"){
    const lW=pw*0.46,L2=p.L,cxw=px+lW/2,cyw=py+ph/2;
    const sc=Math.min(lW,ph)*0.85/L2;
    if(p.conf>0){ctx.strokeStyle="#2f3645";ctx.lineWidth=1;ctx.beginPath();ctx.arc(cxw,cyw,p.conf*sc,0,7);ctx.stroke();}
    const upTo=Math.min(f+1,p.paths[0].length);
    for(let w=0;w<p.paths.length;w++){
      const clr=`hsla(${w*47},55%,68%,0.6)`;ctx.strokeStyle=clr;ctx.lineWidth=1;ctx.beginPath();
      for(let i=Math.max(0,upTo-80);i<upTo;i++){const[wx,wy]=p.paths[w][i];const sx=cxw+wx*sc,sy=cyw-wy*sc;i===Math.max(0,upTo-80)?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}
      ctx.stroke();
      if(upTo>0){const[wx,wy]=p.paths[w][upTo-1];ctx.fillStyle=clr;ctx.beginPath();ctx.arc(cxw+wx*sc,cyw-wy*sc,3*DPR,0,7);ctx.fill();}
    }
    ctx.fillStyle="#5a6880";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText("walkers",px+3*DPR,py+12*DPR);
    /* MSD right panel with proper log axes */
    const rPnl={kind:"timeseries",title:"MSD vs time",xlabel:"time t",ylabel:"⟨r²⟩",logy:true,grow:true,dot:0,ys:[p.msd_sim,p.msd_th],colors:["#7dd3fc","#9c3b2c"],labels:["simulation","theory"],t:p.t_msd,ymin:Math.min(...p.msd_sim.filter(v=>v>0))*0.5,ymax:Math.max(...p.msd_sim)*2};
    drawPanel(rPnl,pi,lx+lw*0.48,ly,lw*0.52,lh);
  }

  /* ─────────────── 3D WAVEPACKET HELIX ─── */
  else if(p.kind==="wavepacket_3d"){
    const f_=Math.min(frame,S.frames-1);
    const re=p.psi_re[f_],im=p.psi_im[f_],prob=p.prob[f_],V3=p.V3d;
    const xArr=p.x,com=p.com_f[f_],halfw=p.halfw_f[f_];
    const amp=p.amp,pmax=p.pmax;
    const cxp=px+pw*0.5,cyp=py+ph*0.52,sc3=Math.min(pw,ph)*0.34;
    function wxOf(i){return Math.max(-1.05,Math.min(1.05,(xArr[i]-com)/halfw));}
    function Pq(wx,wy,wz){return P3(wx,wy,wz,cxp,cyp,sc3);}
    /* box + grid */
    const C=[];for(const sx of[-1,1])for(const sy of[-1,1])for(const sz of[-1,1])C.push(Pq(sx,sy,sz));
    ctx.strokeStyle="rgba(255,255,255,0.09)";ctx.lineWidth=1;
    [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]].forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(C[a][0],C[a][1]);ctx.lineTo(C[b][0],C[b][1]);ctx.stroke();});
    /* floor grid */
    ctx.strokeStyle="rgba(255,255,255,0.07)";
    for(let g=-1;g<=1;g+=0.5){const a=Pq(g,0,-1),b=Pq(g,0,1);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();const c=Pq(-1,0,g),d=Pq(1,0,g);ctx.beginPath();ctx.moveTo(c[0],c[1]);ctx.lineTo(d[0],d[1]);ctx.stroke();}
    /* axis labels */
    ctx.fillStyle="#5a6880";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="center";
    const lx2=Pq(1.2,0,0),lim=Pq(0,0,1.25),lre=Pq(-1.1,1.3,0);
    ctx.fillText("x",lx2[0],lx2[1]);ctx.fillText("Im(ψ)",lim[0],lim[1]);ctx.fillText("Re(ψ)",lre[0],lre[1]);
    /* V on floor */
    let vmax=0;if(V3)for(let i=0;i<V3.length;i++)if(Math.abs(V3[i])>vmax)vmax=Math.abs(V3[i]);
    if(vmax>1e-9){ctx.strokeStyle="rgba(0,217,217,0.5)";ctx.lineWidth=1.5*DPR;ctx.beginPath();for(let i=0;i<xArr.length;i++){const q=Pq(wxOf(i),0,V3[i]/vmax*0.65);i===0?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);}ctx.stroke();}
    /* |psi|^2 (amber) */
    ctx.strokeStyle="#f5992e";ctx.lineWidth=2.4*DPR;ctx.beginPath();for(let i=0;i<xArr.length;i++){const q=Pq(wxOf(i),0,prob[i]/pmax);i===0?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);}ctx.stroke();
    /* Im(psi) green */
    ctx.strokeStyle="rgba(51,191,128,0.9)";ctx.lineWidth=2.2*DPR;ctx.beginPath();for(let i=0;i<xArr.length;i++){const q=Pq(wxOf(i),im[i]/amp,0);i===0?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);}ctx.stroke();
    /* Re(psi) blue */
    ctx.strokeStyle="rgba(90,133,230,0.9)";ctx.lineWidth=2.2*DPR;ctx.beginPath();for(let i=0;i<xArr.length;i++){const q=Pq(wxOf(i),0,re[i]/amp);i===0?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);}ctx.stroke();
    /* helix violet */
    ctx.strokeStyle="rgba(179,77,242,0.92)";ctx.lineWidth=2.5*DPR;ctx.beginPath();for(let i=0;i<xArr.length;i++){const q=Pq(wxOf(i),im[i]/amp,re[i]/amp);i===0?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);}ctx.stroke();
    legend(px,py,pw,ph,[["ψ helix","#b34df2"],["Re(ψ)","#5a85e6"],["Im(ψ)","#33bf80"],["|ψ|²","#f5992e"]]);
    ctx.fillStyle="#4a5568";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText("drag to orbit · scroll zoom · dblclick reset",px+5*DPR,py+ph-8*DPR);
  }

  /* ─────────────── PROB_DENSITY (2D wavepacket) ─── */
  else if(p.kind==="prob_density"){
    const prob=p.prob[f],V2=p.V2d,xArr=p.x;
    const dx_nm=xArr[1]-xArr[0]; /* in m; display in nm */
    const xnm=xArr.map(v=>v*1e9); /* → nm */
    const pnl={kind:"timeseries",title:p.title||"|ψ(x,t)|²  (amber)  ·  V(x) scaled (cyan)",xlabel:"position x (nm)",ylabel:"amplitude",ymin:0,ymax:Math.max(...prob)*1.15,ys:[prob],colors:["#f5992e"],labels:["|ψ|²"],t:xnm,grow:false};
    /* inject V overlay before rendering */
    const savedKind=p.kind;
    drawPanel(pnl,pi,lx,ly,lw,lh);
    /* now overlay V manually */
    const xd2={min:xnm[0],max:xnm[xnm.length-1],log:false};
    const yd2={min:0,max:Math.max(...prob)*1.15,log:false};
    const ML=46*DPR,MR=8*DPR,MT=22*DPR,MB=28*DPR;
    const mx2=Math.round(lx*cv.width)+ML,my2=Math.round(ly*cv.height)+MT,mw2=Math.round(lw*cv.width)-ML-MR,mh2=Math.round(lh*cv.height)-MT-MB;
    const toX2=v=>mx2+mw2*(v-xd2.min)/(xd2.max-xd2.min);
    const toY2=v=>my2+mh2*(1-v/yd2.max);
    ctx.save();ctx.beginPath();ctx.rect(mx2,my2,mw2,mh2);ctx.clip();
    if(V2&&V2.some(v=>Math.abs(v)>1e-9)){
      let vmax=0;for(const v of V2)if(Math.abs(v)>vmax)vmax=Math.abs(v);
      const vscl=yd2.max*0.5/vmax;
      ctx.strokeStyle="rgba(0,217,217,0.5)";ctx.lineWidth=1.5*DPR;ctx.beginPath();
      for(let i=0;i<xnm.length;i++){const X=toX2(xnm[i]),Y=toY2(Math.max(0,V2[i]*vscl));i===0?ctx.moveTo(X,Y):ctx.lineTo(X,Y);}ctx.stroke();
    }
    if(p.markers)for(const m of p.markers){const X=toX2(m*1e9);ctx.strokeStyle="rgba(249,115,22,0.55)";ctx.lineWidth=1.1*DPR;ctx.setLineDash([3*DPR,3*DPR]);ctx.beginPath();ctx.moveTo(X,my2);ctx.lineTo(X,my2+mh2);ctx.stroke();ctx.setLineDash([]);}
    ctx.restore();
    legend(Math.round(lx*cv.width),Math.round(ly*cv.height),Math.round(lw*cv.width),Math.round(lh*cv.height),[["V(x)","rgba(0,217,217,0.7)"],["barrier","rgba(249,115,22,0.7)"]]);
  }

  /* ─────────────── CHAIN (protein folding) ─── */
  else if(p.kind==="chain"){
    const xs=p.x[f],ys=p.y[f],n=xs.length;
    let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
    for(let i=0;i<n;i++){if(xs[i]<xmin)xmin=xs[i];if(xs[i]>xmax)xmax=xs[i];if(ys[i]<ymin)ymin=ys[i];if(ys[i]>ymax)ymax=ys[i];}
    const span=Math.max(xmax-xmin,ymax-ymin,1)*1.4;
    const cxc=(xmin+xmax)/2,cyc=(ymin+ymax)/2;
    const sc=Math.min(pw,ph)*0.85/span,oxc=px+pw/2,oyc=py+ph/2;
    function ts(x,y){return[oxc+(x-cxc)*sc,oyc-(y-cyc)*sc];}
    ctx.strokeStyle="#4a5568";ctx.lineWidth=2*DPR;ctx.beginPath();
    for(let i=0;i<n-1;i++){const[ax,ay]=ts(xs[i],ys[i]);const[bx,by]=ts(xs[i+1],ys[i+1]);i===0?ctx.moveTo(ax,ay):ctx.lineTo(bx,by);}ctx.stroke();
    if(p.native){ctx.strokeStyle="#9c3b2c";ctx.lineWidth=0.8*DPR;ctx.setLineDash([2,2]);ctx.beginPath();for(const[a,b]of p.native){const[ax,ay]=ts(xs[a],ys[a]);const[bx,by]=ts(xs[b],ys[b]);ctx.moveTo(ax,ay);ctx.lineTo(bx,by);}ctx.stroke();ctx.setLineDash([]);}
    for(let i=0;i<n;i++){const[sx,sy]=ts(xs[i],ys[i]);ctx.fillStyle=i===0?"#f97316":i===n-1?"#a855f7":"#7dd3fc";ctx.beginPath();ctx.arc(sx,sy,4*DPR,0,7);ctx.fill();}
    if(p.title){ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(p.title,px+pw/2,py+12*DPR);}
  }

  /* ─────────────── FREE ENERGY F(Q) ─── */
  else if(p.kind==="freeenergy"){
    const qb=p.Q_bins,fv=p.F_vals,qn=p.Q_now?p.Q_now[f]:null;
    const ymin=Math.min(...fv),ymax=Math.max(...fv)+0.5;
    const pnl={kind:"scatter",title:p.title||"Free energy F(Q)",xlabel:"Q (native fraction)",ylabel:"F (kT)",x:qb,y:fv,ymin,ymax,color:"#9c3b2c"};
    drawPanel(pnl,pi,lx,ly,lw,lh);
    if(qn!==null){/* overlay moving dot */
      const ML=46*DPR,MT=22*DPR,mw2=Math.round(lw*cv.width)-ML-8*DPR,mh2=Math.round(lh*cv.height)-MT-28*DPR;
      const mx2=Math.round(lx*cv.width)+ML,my2=Math.round(ly*cv.height)+MT;
      const toX2=v=>mx2+mw2*(v-qb[0])/(qb[qb.length-1]-qb[0]);
      const toY2=v=>my2+mh2*(1-(v-ymin)/(ymax-ymin));
      /* find F at Q=qn by interpolation */
      let fi=0;for(let i=0;i<qb.length;i++)if(qb[i]<=qn)fi=i;
      const fAtQ=fv[fi];
      ctx.fillStyle="#f97316";ctx.beginPath();ctx.arc(toX2(qn),toY2(fAtQ),5*DPR,0,7);ctx.fill();
    }
  }

  /* ─────────────── SWEEP CURVE (percolation S(p) / chi(p)) ─── */
  else if(p.kind==="sweep_curve"){
    const data=p.data,pv=S.p_arr||p.p_arr||[],n=data.length;
    const ymax=Math.max(...data)*1.1+0.01;
    const pnl={kind:"timeseries",title:p.title,xlabel:p.xlabel||"bond prob. p",ylabel:p.ylabel,ymin:0,ymax,ys:[data],colors:[p.color||"#7dd3fc"],grow:true,t:pv};
    drawPanel(pnl,pi,lx,ly,lw,lh);
    /* p_c vline + current dot */
    if(p.pc!==null&&p.pc!==undefined){
      const ML=46*DPR,MT=22*DPR,mw2=Math.round(lw*cv.width)-ML-8*DPR,mh2=Math.round(lh*cv.height)-MT-28*DPR;
      const mx2=Math.round(lx*cv.width)+ML,my2=Math.round(ly*cv.height)+MT;
      const toX2=v=>mx2+mw2*(v-pv[0])/(pv[pv.length-1]-pv[0]);
      const toY2=v=>my2+mh2*(1-v/ymax);
      const vx=toX2(p.pc);ctx.strokeStyle="rgba(139,148,158,0.4)";ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(vx,my2);ctx.lineTo(vx,my2+mh2);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle="#8a93a6";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText("p_c="+fmtN(p.pc),vx,my2+10*DPR);
      const ci=Math.min(f,n-1);
      ctx.fillStyle="#f97316";ctx.beginPath();ctx.arc(toX2(pv[ci]),toY2(data[ci]),4.5*DPR,0,7);ctx.fill();
    }
  }

  /* ─────────────── GENERIC CURVE (CD spectrum, etc.) ─── */
  else if(p.kind==="curve"){
    const yy=p.y[Math.min(f,p.y.length-1)],ymax=p.ymax||1,ymin=p.ymin||-ymax;
    const wl0=p.wl0||190,wl1=p.wl1||250,N2=yy.length;
    const pnl={kind:"timeseries",title:p.title||p.label?.[f]||"",xlabel:p.xlabel||"wavelength (nm)",ylabel:p.ylabel||"CD",ymin,ymax,ys:[yy],colors:["#7dd3fc"],grow:false,t:Array.from({length:N2},(_,i)=>wl0+(wl1-wl0)*i/(N2-1))};
    drawPanel(pnl,pi,lx,ly,lw,lh);
  }

  /* ─────────────── ATTRACTOR 3D (reservoir computing) ─── */
  else if(p.kind==="attractor_esn"||p.kind==="attractor"){
    /* handled inline in reservoir_computing — kept here as fallback */
    const cxp=px+pw*0.5,cyp=py+ph*0.52,sc3=Math.min(pw,ph)*0.38;
    function NP(ax,ay,az_,i){return P3((ax[i]-S.cx)/S.hs,(ay[i]-S.cy)/S.hs,(az_[i]-S.cz)/S.hs,cxp,cyp,sc3);}
    /* box */
    const C=[];for(const sx of[-1,1])for(const sy of[-1,1])for(const sz of[-1,1])C.push(P3(sx,sy,sz,cxp,cyp,sc3));
    ctx.strokeStyle="rgba(255,255,255,0.07)";ctx.lineWidth=1;
    [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]].forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(C[a][0],C[a][1]);ctx.lineTo(C[b][0],C[b][1]);ctx.stroke();});
    const TRAIL=120,DELAY=20;
    /* ghost */
    ctx.strokeStyle="rgba(180,190,210,0.07)";ctx.lineWidth=1;ctx.beginPath();for(let i=0;i<S.ghost_x.length;i++){const q=NP(S.ghost_x,S.ghost_y,S.ghost_z,i);i===0?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);}ctx.stroke();
    /* true */
    const t0=Math.max(0,f-TRAIL);ctx.strokeStyle="#7dd3fc";ctx.lineWidth=2*DPR;ctx.beginPath();for(let i=t0;i<=f;i++){const q=NP(S.true_x,S.true_y,S.true_z,i);i===t0?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);}ctx.stroke();
    const tq=NP(S.true_x,S.true_y,S.true_z,f);ctx.fillStyle="#7dd3fc";ctx.beginPath();ctx.arc(tq[0],tq[1],4.5*DPR,0,7);ctx.fill();
    /* esn */
    const fd=Math.max(0,f-DELAY),e0=Math.max(0,fd-TRAIL);
    if(fd>0){ctx.strokeStyle="#f97316";ctx.lineWidth=2*DPR;ctx.beginPath();for(let i=e0;i<=fd;i++){const q=NP(S.esn_x,S.esn_y,S.esn_z,i);i===e0?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);}ctx.stroke();const eq=NP(S.esn_x,S.esn_y,S.esn_z,fd);ctx.fillStyle="#f97316";ctx.beginPath();ctx.arc(eq[0],eq[1],4.5*DPR,0,7);ctx.fill();}
    legend(px,py,pw,ph,[["true","#7dd3fc"],["ESN (20 frames behind)","#f97316"]]);
    ctx.fillStyle="#4a5568";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText("drag to orbit · scroll zoom",px+5*DPR,py+ph-7*DPR);
  }

  /* ─────────────── CCM SHADOW MANIFOLD (growing coloured scatter) ─── */
  else if(p.kind==="ccm_scatter"){
    const xs=p.x,ys=p.y,cols=p.col;
    const xmin=Math.min(...xs),xmax=Math.max(...xs);
    const ymin=Math.min(...ys),ymax=Math.max(...ys);
    const xd={min:xmin,max:xmax,log:false},yd={min:ymin,max:ymax,log:false};
    const{mx,my,mw,mh,toX,toY}=axes2D(p,lx,ly,lw,lh,xd,yd,vp);
    ctx.save();ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.clip();
    const upTo=Math.min(f+1,xs.length);
    for(let i=0;i<upTo;i++){
      const c=cols[i];
      const r=Math.round(125*(1-c)+15*c),g2=Math.round(211*(1-c)+163*c),b2=Math.round(252*(1-c)+99*c);
      ctx.fillStyle=`rgb(${r},${g2},${b2})`;
      ctx.beginPath();ctx.arc(toX(xs[i]),toY(ys[i]),2.5*DPR,0,7);ctx.fill();
    }
    ctx.restore();
    if(p.title){ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(p.title,px+pw/2,py+12*DPR);}
    if(p.xlabel){ctx.fillStyle="#626b7c";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(p.xlabel,px+pw/2,py+ph-3*DPR);}
  }

  /* ─────────────── PASS-THROUGH (let panel handle its own draw) ─── */
  else if(typeof p._draw==="function"){p._draw(ctx,px,py,pw,ph,f,DPR);}

  ctx.restore();
}

/* ── Controls overlay ───────────────────────────────────────── */
function drawControls(){
  const W=cv.width,H=cv.height,y=H-14*DPR;
  ctx.fillStyle="rgba(15,17,23,0.75)";ctx.fillRect(0,y-8*DPR,W,22*DPR);
  const btn=playing?"⏸":"▶",spd="×"+speed.toFixed(1);
  ctx.fillStyle="#8a93a6";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText(btn+"  "+spd+"  frame "+frame+"/"+S.frames+"  scroll=zoom  drag=pan  dblclick=reset",8*DPR,y+5*DPR);
}

/* ── Main draw ──────────────────────────────────────────────── */
function draw(){
  ctx.fillStyle=BG;ctx.fillRect(0,0,cv.width,cv.height);
  for(let i=0;i<S.panels.length;i++) try{drawPanel(S.panels[i],i,...S.layout[i]);}catch(e){}
  drawControls();
}

/* ── Controls (click to pause, double-click to reset, key S for speed) ── */
cv.addEventListener("click",e=>{
  const r=cv.getBoundingClientRect(),y=e.clientY-r.top;
  if(y>cv.clientHeight-20){playing=!playing;}
});
addEventListener("keydown",e=>{
  if(e.key==="s"||e.key==="S"){speed=speed<4?speed+0.5:0.5;}
  if(e.key===" "){playing=!playing;}
  if(e.key==="r"||e.key==="R"){frame=0;VP.forEach(v=>{v.zoom=1;v.px=0;v.py=0;});}
});

/* ── 3D orbit events (only apply when 3D panel hovered) ─────── */
cv.addEventListener("mousedown",e=>{const r=cv.getBoundingClientRect();lx3=e.clientX-r.left;ly3=e.clientY-r.top;drag3=true;});
cv.addEventListener("mousemove",e=>{if(!drag3)return;const r=cv.getBoundingClientRect();az3+=(e.clientX-r.left-lx3)*0.01;el3-=(e.clientY-r.top-ly3)*0.01;el3=Math.max(-1.45,Math.min(1.45,el3));lx3=e.clientX-r.left;ly3=e.clientY-r.top;});
addEventListener("mouseup",()=>{drag3=false;drag=false;});
cv.addEventListener("wheel",e=>{zoom3=Math.max(0.3,Math.min(3,zoom3*Math.exp(-e.deltaY*0.001)));},{passive:true});

/* ── Animation loop ─────────────────────────────────────────── */
function loop(t){if(playing&&t-last>S.dt/speed){frame=(frame+1)%S.frames;last=t;}try{draw();}catch(e){}requestAnimationFrame(loop);}
function _kick(){fit();if(cv.width>0&&cv.height>0){requestAnimationFrame(loop);}else{setTimeout(_kick,80);}}_kick();

    </script></body></html>""".replace("__SPEC__", json.dumps(_d))
    mo.vstack([
        mo.md(
            f"**{_d['direction']}** &nbsp;|&nbsp; system: {_d['system']} &nbsp;|&nbsp; "
            "left: the coupled series unfolding, right: X's shadow manifold building up, "
            "each point coloured by the Y value at that time. A smooth colour gradient over "
            "the manifold means Y is recoverable from X, the signature of X driving Y."
        ),
        mo.iframe(_JS, height="430px"),
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
