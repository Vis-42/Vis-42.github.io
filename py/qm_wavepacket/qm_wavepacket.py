import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Quantum Wavepacket")


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
        # Quantum wavepacket: Schrodinger evolution and tunnelling

        A Gaussian wavepacket evolves under the 1D time-dependent Schrodinger equation
        $i\,\partial_t\psi = -\tfrac12\partial_x^2\psi + V(x)\psi$ (units $\hbar = m = 1$),
        integrated by the **split-step Fourier** method. Choose a potential and watch the
        packet spread, reflect, and tunnel. The top panel renders the complex wavefunction
        as an Argand helix in 3D perspective; the bottom tracks $|\psi(x,t)|^2$ live.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    pot_sl = mo.ui.dropdown(
        ["free", "barrier", "well", "harmonic", "double-barrier", "step"],
        value="barrier",
        label="potential",
    )
    A_sl = mo.ui.slider(0.1, 3.0, value=0.8, step=0.05, label="amplitude A (x KE)", show_value=True)
    w_sl = mo.ui.slider(0.2, 4.0, value=1.0, step=0.1, label="barrier width w", show_value=True)
    k0_sl = mo.ui.slider(1.0, 6.0, value=3.0, step=0.25, label="wavenumber k0", show_value=True)
    spf_sl = mo.ui.slider(1, 20, value=6, step=1, label="steps per frame", show_value=True)
    mo.hstack([pot_sl, A_sl, w_sl, k0_sl, spf_sl], justify="start", gap=2)
    return A_sl, k0_sl, pot_sl, spf_sl, w_sl


@app.cell(hide_code=True)
def _(np, A_sl, k0_sl, pot_sl, spf_sl, w_sl):
    _pot  = pot_sl.value
    _A    = float(A_sl.value)
    _w    = float(w_sl.value)
    _k0   = float(k0_sl.value)
    _spf  = int(spf_sl.value)

    # ── Grid ──────────────────────────────────────────────────────────────
    _N   = 256
    _L   = 20.0
    _dx  = _L / _N
    _x   = np.linspace(-_L / 2, _L / 2, _N, endpoint=False)
    _k   = 2 * np.pi * np.fft.fftfreq(_N, d=_dx)

    # ── Initial state ─────────────────────────────────────────────────────
    _x0  = -5.0
    _sig = 0.5
    _psi = np.exp(-(_x - _x0) ** 2 / (4 * _sig ** 2)) * np.exp(1j * _k0 * _x)
    _psi = _psi / np.sqrt(np.sum(np.abs(_psi) ** 2) * _dx)

    # ── Potential (in units where KE = k0^2/2) ───────────────────────────
    _E   = 0.5 * _k0 ** 2
    _V   = np.zeros(_N)
    if _pot == "barrier":
        _V[np.abs(_x) < _w / 2] = _A * _E
    elif _pot == "well":
        _V[np.abs(_x) < _w / 2] = -_A * _E
    elif _pot == "harmonic":
        _V = 0.5 * (_x / _w) ** 2 * _A * _E
    elif _pot == "double-barrier":
        _V[np.abs(np.abs(_x) - _w) < 0.2] = _A * _E
    elif _pot == "step":
        _V[_x > 0] = _A * _E

    # ── Absorbing boundary mask ───────────────────────────────────────────
    _edge   = 0.08 * _L
    _abs_m  = np.ones(_N)
    _d      = (np.abs(_x) - (_L / 2 - _edge)) / _edge
    _mask   = _d > 0
    _abs_m[_mask] = np.exp(-4.0 * _d[_mask] ** 2)

    # ── Split-step operators ──────────────────────────────────────────────
    _dt     = 0.005
    _kin    = np.exp(-1j * 0.5 * _k ** 2 * _dt)
    _halfV  = np.exp(-1j * _V * _dt / 2)

    # ── Pre-compute frames ────────────────────────────────────────────────
    _FRAMES     = 100
    _psi_re_f   = []
    _psi_im_f   = []
    _prob_f     = []
    _psi_cur    = _psi.copy()

    for _fi in range(_FRAMES):
        for _ in range(_spf):
            _psi_cur = _halfV * _psi_cur
            _psi_cur = np.fft.ifft(_kin * np.fft.fft(_psi_cur))
            _psi_cur = _halfV * _psi_cur
            _psi_cur *= _abs_m
        _psi_re_f.append(np.round(np.real(_psi_cur), 4).tolist())
        _psi_im_f.append(np.round(np.imag(_psi_cur), 4).tolist())
        _prob_f.append(np.round(np.abs(_psi_cur) ** 2, 5).tolist())

    # ── Potential markers for probability panel ───────────────────────────
    _markers = []
    if _pot == "barrier" or _pot == "well":
        _markers = [-_w / 2, _w / 2]
    elif _pot == "double-barrier":
        _markers = [-_w - 0.2, -_w + 0.2, _w - 0.2, _w + 0.2]
    elif _pot == "step":
        _markers = [0.0]
    elif _pot == "harmonic":
        _markers = [-_w, _w]

    # ── Per-frame camera follow: center of mass and spread ────────────────
    # The 3D view window tracks the packet's center of mass and widens as it
    # spreads, exactly like the Julia Axis3 camera-follow logic.
    _com_f, _halfw_f = [], []
    for _pf in _prob_f:
        _p = np.array(_pf)
        _s = float(np.sum(_p))
        if _s > 1e-30:
            _cm = float(np.sum(_x * _p) / _s)
            _sp = float(np.sqrt(max(np.sum((_x - _cm) ** 2 * _p) / _s, 1e-12)))
        else:
            _cm, _sp = 0.0, _sig
        _com_f.append(round(_cm, 4))
        _halfw_f.append(round(max(3.0 * _sp, 3.0 * _sig), 4))

    # ── Amplitude normalisation (so Re, Im, helix, prob share scale) ──────
    _amp = max(1e-12, float(np.max([np.max(np.abs(_psi_re_f)),
                                    np.max(np.abs(_psi_im_f))])))
    _pmax = max(1e-12, float(np.max(_prob_f)))

    # ── Stats for last frame ──────────────────────────────────────────────
    _prob_last  = np.array(_prob_f[-1])
    _norm_last  = float(np.sum(_prob_last) * _dx)
    _com_last   = float(np.sum(_x * _prob_last) * _dx / max(_norm_last, 1e-30))
    _T_proxy    = float(np.sum(_prob_last[_x > 0]) * _dx)
    _R_proxy    = float(np.sum(_prob_last[_x <= 0]) * _dx)

    # ── V scaled for display in both panels ──────────────────────────────
    _Vmax  = float(np.max(np.abs(_V))) if np.any(_V != 0) else 1.0
    _prob0max = float(np.max(np.abs(_psi) ** 2))
    _V_disp2d = np.round(_V / _Vmax * _prob0max * 0.7, 6).tolist()
    _V_disp3d = np.round(_V / _Vmax * 0.6, 6).tolist()

    qm_data = {
        "frames": _FRAMES,
        "psi_re": _psi_re_f,
        "psi_im": _psi_im_f,
        "prob": _prob_f,
        "V2d": _V_disp2d,
        "V3d": _V_disp3d,
        "x": np.round(_x, 4).tolist(),
        "markers": _markers,
        "com_f": _com_f,
        "halfw_f": _halfw_f,
        "amp": _amp,
        "pmax": _pmax,
        "norm": _norm_last,
        "T": _T_proxy,
        "R": _R_proxy,
        "com": _com_last,
        "k0": _k0,
        "pot": _pot,
        "A": _A,
        "E": float(_E),
    }
    return (qm_data,)


@app.cell(hide_code=True)
def _(mo, qm_data):
    _d = qm_data
    mo.md(
        f"**norm** {_d['norm']:.4f} &nbsp;|&nbsp; "
        f"**T proxy** (right half) {_d['T']:.3f} &nbsp;|&nbsp; "
        f"**R proxy** (left half) {_d['R']:.3f} &nbsp;|&nbsp; "
        f"**&lt;x&gt;** {_d['com']:.3f} &nbsp;|&nbsp; "
        f"k0 = {_d['k0']:.2f}, KE = {_d['E']:.2f}, potential = {_d['pot']}, A = {_d['A']:.2f}"
    )
    return


@app.cell(hide_code=True)
def _(json, mo, qm_data):
    _JS = r"""<!doctype html><html><head><meta charset="utf-8"><style>
    html,body{margin:0;background:#0f1117;overflow:hidden;user-select:none}
    canvas{display:block;width:100vw;height:100vh;cursor:grab}
    canvas.drag{cursor:grabbing}
    #hint{position:absolute;top:8px;right:10px;font:10px monospace;color:#4a5568;pointer-events:none}
    </style></head><body><canvas id="c"></canvas><div id="hint">drag to orbit the 3D space &middot; scroll to zoom</div><script>
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

    </script></body></html>"""

    _d = qm_data
    _spec = {
        "frames": _d["frames"], "dt": 50,
        "panels": [
            {
                "kind": "wavepacket_3d",
                "psi_re": _d["psi_re"], "psi_im": _d["psi_im"], "prob": _d["prob"],
                "V3d": _d["V3d"], "x": _d["x"],
                "com_f": _d["com_f"], "halfw_f": _d["halfw_f"],
                "amp": _d["amp"], "pmax": _d["pmax"],
            },
            {
                "kind": "prob_density",
                "prob": _d["prob"], "V2d": _d["V2d"], "x": _d["x"], "markers": _d["markers"],
            },
        ],
        "layout": [[0.0, 0.0, 1.0, 0.60], [0.0, 0.60, 1.0, 0.40]],
    }
    mo.iframe(_JS.replace("__SPEC__", json.dumps(_spec)), height="880px")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **The split-step method.** One timestep applies the potential half-kick, propagates freely
        in Fourier space, then applies the potential half-kick again (Strang splitting, error
        $O(\Delta t^2)$):

        $$\psi \to e^{-iV\Delta t/2}\,\psi,\quad
        \psi \to \mathcal{F}^{-1}\bigl[e^{-ik^2\Delta t/2}\,\mathcal{F}\psi\bigr],\quad
        \psi \to e^{-iV\Delta t/2}\,\psi.$$

        Units: $\hbar = m = 1$, so kinetic energy $E = k_0^2/2$ and all potentials are set in
        multiples of $E$. Amplitude $A < 1$ is classically forbidden (pure tunnelling). The
        **Argand helix** (top): cyan curve = $\mathrm{Re}(\psi)$ in the $x$-$z$ plane, amber =
        $\mathrm{Im}(\psi)$ in the $x$-$y$ plane, dark red = $|\psi|^2$ on the floor, violet =
        the full helix. Pitch equals the de Broglie wavelength $2\pi/k_0$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
