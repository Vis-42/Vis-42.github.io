import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Driven Quadruple Pendulum")


@app.cell(hide_code=True)
def _():
    import json
    import marimo as mo
    import numpy as np
    return json, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Driven damped quadruple pendulum

    Four equal masses on rigid rods with a periodic torque applied at the first pivot and viscous
    damping on each joint. PC3236 at NUS.

    The mass matrix $M(\theta)$ couples all four angles through cosine terms; the RHS carries centrifugal
    couplings, gravity, and the drive torque $\tau_0\cos(\omega_d t)$ at pivot 1 only. Even mild drive
    amplitudes push the outer joints into irregular, non-repeating motion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    tau0_sl = mo.ui.slider(0.0, 12.0, value=8.0, step=0.25, label="drive τ₀ (N·m)", show_value=True)
    omega_sl = mo.ui.slider(1.0, 12.0, value=5.03, step=0.1, label="ω_d (rad/s)", show_value=True)
    b_sl = mo.ui.slider(0.0, 0.3, value=0.10, step=0.01, label="damping b", show_value=True)
    mo.md(f"""{mo.hstack([tau0_sl, omega_sl, b_sl], justify="start", gap=2)}""")
    return tau0_sl, omega_sl, b_sl


@app.cell(hide_code=True)
def _(np, tau0_sl, omega_sl, b_sl):
    _g = 9.81
    _L = [0.40, 0.35, 0.30, 0.25]
    _m = [1.0, 1.0, 1.0, 1.0]
    _tau0 = tau0_sl.value
    _wd   = omega_sl.value
    _b    = b_sl.value

    def _M(th):
        L, m = _L, _m
        Mm = np.zeros((4, 4))
        cmsum = [sum(m[k:]) for k in range(4)]
        for i in range(4):
            Mm[i, i] = cmsum[i] * L[i] ** 2
        for i in range(4):
            for j in range(i + 1, 4):
                v = sum(m[k] for k in range(j, 4)) * L[i] * L[j] * np.cos(th[i] - th[j])
                Mm[i, j] = Mm[j, i] = v
        return Mm

    def _F(th, om, t):
        L, m, g = _L, _m, _g
        F = np.zeros(4)
        cmsum = [sum(m[k:]) for k in range(4)]
        for i in range(4):
            F[i] -= cmsum[i] * g * L[i] * np.sin(th[i])
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                c = sum(m[k] for k in range(max(i, j), 4)) * L[i] * L[j]
                if j < i:
                    F[i] += c * om[j] ** 2 * np.sin(th[j] - th[i])
                else:
                    F[i] -= c * om[j] ** 2 * np.sin(th[i] - th[j])
        F[0] += _tau0 * np.cos(_wd * t)
        F -= _b * om
        return F

    def _derivs(t, y):
        th, om = y[:4], y[4:]
        return np.concatenate([om, np.linalg.solve(_M(th), _F(th, om, t))])

    def _rk4(t, y, dt):
        k1 = _derivs(t, y)
        k2 = _derivs(t + dt / 2, y + dt / 2 * k1)
        k3 = _derivs(t + dt / 2, y + dt / 2 * k2)
        k4 = _derivs(t + dt, y + dt * k3)
        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    DT = 0.01; T_END = 12.0; MAX_FRAMES = 800
    _n = int(T_END / DT)
    _step = max(1, _n // MAX_FRAMES)
    _y = np.zeros(8); _y[0] = 0.3; _y[1] = 0.1
    _t = 0.0; _hist = []
    for _i in range(_n):
        if _i % _step == 0:
            _hist.append(_y[:4].tolist())
        _y = _rk4(_t, _y, DT); _t += DT

    sim = {"th": _hist, "n": len(_hist), "L": _L}
    return (sim,)


@app.cell(hide_code=True)
def _(json, mo, sim):
    _JS = r"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:#0f1117;overflow:hidden}canvas{display:block;width:100vw;height:100vh}
</style></head><body><canvas id="c"></canvas><script>
const S=__SPEC__,cv=document.getElementById("c"),ctx=cv.getContext("2d");
const DPR=Math.min(window.devicePixelRatio||1,2);
function fit(){const w=cv.offsetWidth||window.innerWidth||800,h=cv.offsetHeight||window.innerHeight||520;if(w>0&&h>0){cv.width=w*DPR;cv.height=h*DPR;}}
addEventListener("resize",fit);fit();
const VP=S.panels.map(()=>({zoom:1,px:0,py:0}));
let frame=0,playing=true,last=0;
function legend(lx,ly,lw,lh,entries){
  const y0=ly+lh-14*DPR;
  entries.forEach(([l,c],i)=>{ctx.fillStyle=c;ctx.fillRect(lx+5*DPR+i*72*DPR,y0,8*DPR,8*DPR);ctx.fillStyle="#8a93a6";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText(l,lx+15*DPR+i*72*DPR,y0+7*DPR);});
}
function axes2D(title,xlabel,lx,ly,lw,lh,xd,yd){
  const pL=34*DPR,pR=8*DPR,pT=18*DPR,pB=22*DPR;
  const mx=lx+pL,my=ly+pT,mw=lw-pL-pR,mh=lh-pT-pB;
  function toX(v){return mx+(v-xd.min)/(xd.max-xd.min||1)*mw;}
  function toY(v){return my+mh-(v-yd.min)/(yd.max-yd.min||1)*mh;}
  ctx.strokeStyle="#1e2535";ctx.lineWidth=1;
  for(let i=0;i<=5;i++){const x=mx+i*mw/5;ctx.beginPath();ctx.moveTo(x,my);ctx.lineTo(x,my+mh);ctx.stroke();}
  for(let i=0;i<=4;i++){const y=my+i*mh/4;ctx.beginPath();ctx.moveTo(mx,y);ctx.lineTo(mx+mw,y);ctx.stroke();}
  ctx.strokeStyle="#2a3448";ctx.lineWidth=1;ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.stroke();
  ctx.fillStyle="#4a5568";ctx.font=(8*DPR)+"px monospace";ctx.textAlign="center";
  for(let i=0;i<=5;i+=2){const v=xd.min+i*(xd.max-xd.min)/5;ctx.fillText(v.toFixed(0),mx+i*mw/5,my+mh+10*DPR);}
  ctx.textAlign="right";
  for(let i=0;i<=4;i+=2){const v=yd.min+i*(yd.max-yd.min)/4;ctx.fillText(v.toFixed(1),mx-3*DPR,my+mh-i*mh/4+4*DPR);}
  if(title){ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(title,lx+lw/2,ly+12*DPR);}
  if(xlabel){ctx.fillStyle="#626b7c";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(xlabel,lx+lw/2,ly+lh-4*DPR);}
  return{mx,my,mw,mh,toX,toY};
}
function drawPanel(p,i,lx,ly,lw,lh){
  const f=frame;
  ctx.save();ctx.beginPath();ctx.rect(lx,ly,lw,lh);ctx.clip();
  if(p.kind==="pendulum_chain"){
    const Ls=p.L,thF=p.th[f];
    const cx=lx+lw*0.5,cy=ly+lh*0.10;
    const totalL=Ls.reduce((a,b)=>a+b,0);
    const sc=Math.min(lw,lh)*0.76/totalL;
    let bx=cx,by=cy;const bobs=[];
    for(let j=0;j<Ls.length;j++){
      const nx=bx+sc*Ls[j]*Math.sin(thF[j]),ny=by+sc*Ls[j]*Math.cos(thF[j]);
      bobs.push([nx,ny]);bx=nx;by=ny;
    }
    let rx=cx,ry=cy;
    for(let j=0;j<bobs.length;j++){
      ctx.strokeStyle=p.colors[j]||"#7dd3fc";ctx.lineWidth=2.5*DPR;
      ctx.beginPath();ctx.moveTo(rx,ry);ctx.lineTo(bobs[j][0],bobs[j][1]);ctx.stroke();
      rx=bobs[j][0];ry=bobs[j][1];
    }
    const rs=[9,7,6,5];
    for(let j=0;j<bobs.length;j++){
      ctx.fillStyle=p.colors[j]||"#7dd3fc";
      ctx.beginPath();ctx.arc(bobs[j][0],bobs[j][1],rs[j]*DPR,0,7);ctx.fill();
    }
    ctx.fillStyle="#8a93a6";ctx.beginPath();ctx.arc(cx,cy,3*DPR,0,7);ctx.fill();
    legend(lx,ly,lw,lh,(p.colors||[]).map((c,j)=>["θ"+(j+1),c]));
  }
  else if(p.kind==="timeseries"){
    const ys_arr=p.ys||[],n=ys_arr[0]?ys_arr[0].length:0;
    const ts=p.t||Array.from({length:n},(_,k)=>k);
    let ymin=Infinity,ymax=-Infinity;
    for(const ya of ys_arr)for(const v of ya){if(v<ymin)ymin=v;if(v>ymax)ymax=v;}
    ymin-=Math.abs(ymax-ymin)*0.07;ymax+=Math.abs(ymax-ymin)*0.07;
    const xd={min:ts[0]||0,max:ts[n-1]||1},yd={min:ymin||0,max:ymax||1};
    const{mx,my,mw,mh,toX,toY}=axes2D(p.title,p.xlabel,lx,ly,lw,lh,xd,yd);
    ctx.save();ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.clip();
    const colors=p.colors||["#7dd3fc","#f97316","#a594f2","#e0635a"];
    const upTo=Math.min(f+1,n);
    for(let si=0;si<ys_arr.length;si++){
      const ya=ys_arr[si];ctx.strokeStyle=colors[si%colors.length];ctx.lineWidth=2*DPR;ctx.beginPath();
      for(let k=0;k<upTo;k++){const x=toX(ts[k]),y=toY(ya[k]);k===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
      ctx.stroke();
    }
    ctx.restore();
    if(p.labels)legend(lx,ly,lw,lh,p.labels.map((l,j)=>[l,(p.colors||colors)[j%colors.length]]));
  }
  ctx.restore();
}
function draw(){
  const W=cv.width,H=cv.height;
  ctx.fillStyle="#0f1117";ctx.fillRect(0,0,W,H);
  for(let i=0;i<S.panels.length;i++){
    const lay=S.layout[i];
    drawPanel(S.panels[i],i,lay[0]*W,lay[1]*H,lay[2]*W,lay[3]*H);
  }
  ctx.fillStyle="#2a3448";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="left";
  ctx.fillText("frame "+frame+"/"+S.frames+"  click to pause",6*DPR,H-5*DPR);
}
function loop(t){if(playing&&t-last>S.dt){frame=(frame+1)%S.frames;last=t;}try{draw();}catch(e){}requestAnimationFrame(loop);}
cv.addEventListener("click",()=>{playing=!playing;});
let _s=false;
const _ro=new ResizeObserver(()=>{fit();if(!_s&&cv.width>0){_s=true;requestAnimationFrame(loop);}});
_ro.observe(cv);setTimeout(()=>{fit();if(!_s&&cv.width>0){_s=true;requestAnimationFrame(loop);}},100);
</script></body></html>"""

    _th = sim["th"]
    _spec = {
        "frames": sim["n"], "dt": 20,
        "panels": [
            {
                "kind": "pendulum_chain",
                "L": sim["L"],
                "th": _th,
                "colors": ["#7dd3fc", "#a594f2", "#f97316", "#e0635a"],
            },
            {
                "kind": "timeseries",
                "title": "θ₁ and θ₂",
                "xlabel": "frame",
                "ys": [[row[0] for row in _th], [row[1] for row in _th]],
                "colors": ["#7dd3fc", "#a594f2"],
                "labels": ["θ₁", "θ₂"],
            },
            {
                "kind": "timeseries",
                "title": "θ₃ and θ₄",
                "xlabel": "frame",
                "ys": [[row[2] for row in _th], [row[3] for row in _th]],
                "colors": ["#f97316", "#e0635a"],
                "labels": ["θ₃", "θ₄"],
            },
        ],
        "layout": [
            [0.00, 0.00, 0.40, 1.00],
            [0.40, 0.00, 0.60, 0.50],
            [0.40, 0.50, 0.60, 0.50],
        ],
    }
    mo.vstack([
        mo.md("**Quadruple pendulum, live.** Click to pause."),
        mo.iframe(_JS.replace("__SPEC__", json.dumps(_spec)), height="520px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo, sim, tau0_sl, omega_sl, b_sl):
    _last = sim["th"][-1] if sim["th"] else [0, 0, 0, 0]
    mo.md(f"""
    **Drive:** τ₀ = {tau0_sl.value:.2f} N·m, ω_d = {omega_sl.value:.2f} rad/s, b = {b_sl.value:.2f}.
    Final angles: θ₁ = {_last[0]:.2f}, θ₂ = {_last[1]:.2f}, θ₃ = {_last[2]:.2f}, θ₄ = {_last[3]:.2f} rad.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Equations of motion.** The Lagrangian gives a $4\times4$ mass-matrix system:

    $$M(\theta)\,\ddot\theta = F(\theta,\dot\theta,t)$$

    where

    $$M_{ij} = \Bigl(\sum_{k\geq\max(i,j)} m_k\Bigr) L_i L_j \cos(\theta_i-\theta_j)$$

    and $F_i$ collects gravity ($-M_{ii}g/L_i\cdot\sin\theta_i$), centrifugal coupling
    (off-diagonal angular-velocity terms), the drive torque $\tau_0\cos(\omega_d t)$ at $i=1$
    only, and viscous damping $-b\dot\theta_i$.
    RK4 at $\Delta t = 10^{-2}$ s, 12 s trajectory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bifurcation diagram

    Sweep the drive amplitude $\tau_0$ from 0 to 12 N·m. At each amplitude record
    $\theta_1$ stroboscopically (once per drive period $T_d = 2\pi/\omega_d$) after
    discarding a transient. A single point means periodic motion; a vertical cloud means
    chaos via period-doubling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bif_btn = mo.ui.run_button(label="Compute bifurcation diagram")
    mo.md(f"""The sweep takes a few seconds. {bif_btn}""")
    return (bif_btn,)


@app.cell(hide_code=True)
def _(bif_btn, mo, np, omega_sl, b_sl):
    mo.stop(not bif_btn.value, mo.md("Press **Compute** above."))
    _g   = 9.81
    _L   = [0.40, 0.35, 0.30, 0.25]
    _m   = [1.0,  1.0,  1.0,  1.0]
    _b   = float(b_sl.value)
    _wd  = float(omega_sl.value)
    _Td  = 2 * np.pi / _wd
    _DT  = 0.01
    _N_TRANS = 20   # periods discarded
    _N_REC   = 30   # periods recorded

    def _M4(th):
        Mm = np.zeros((4, 4))
        cmsum = [sum(_m[k:]) for k in range(4)]
        for i in range(4):
            Mm[i, i] = cmsum[i] * _L[i] ** 2
        for i in range(4):
            for j in range(i + 1, 4):
                v = sum(_m[k] for k in range(j, 4)) * _L[i] * _L[j] * np.cos(th[i] - th[j])
                Mm[i, j] = Mm[j, i] = v
        return Mm

    def _F4(th, om, t, tau0):
        F = np.zeros(4)
        cmsum = [sum(_m[k:]) for k in range(4)]
        for i in range(4):
            F[i] -= cmsum[i] * _g * _L[i] * np.sin(th[i])
        for i in range(4):
            for j in range(4):
                if i == j: continue
                c = sum(_m[k] for k in range(max(i, j), 4)) * _L[i] * _L[j]
                if j < i:
                    F[i] += c * om[j] ** 2 * np.sin(th[j] - th[i])
                else:
                    F[i] -= c * om[j] ** 2 * np.sin(th[i] - th[j])
        F[0] += tau0 * np.cos(_wd * t)
        F -= _b * om
        return F

    def _rk4b(t, y, dt, tau0):
        def d(t_, y_): return np.concatenate([y_[4:], np.linalg.solve(_M4(y_[:4]), _F4(y_[:4], y_[4:], t_, tau0))])
        k1 = d(t, y); k2 = d(t+dt/2, y+dt/2*k1); k3 = d(t+dt/2, y+dt/2*k2); k4 = d(t+dt, y+dt*k3)
        return y + (dt/6)*(k1+2*k2+2*k3+k4)

    _tau_vals = np.linspace(0, 12, 80)
    _bif_tau  = []
    _bif_th1  = []
    for _tau0 in _tau_vals:
        _y = np.zeros(8); _y[0] = 0.3; _y[1] = 0.1
        _t = 0.0
        # transient
        _n_trans = int(_N_TRANS * _Td / _DT)
        for _ in range(_n_trans):
            _y = _rk4b(_t, _y, _DT, _tau0); _t += _DT
        # record stroboscopic samples
        _n_rec = int(_N_REC * _Td / _DT)
        _step  = max(1, int(_Td / _DT))
        for _i in range(_n_rec):
            _y = _rk4b(_t, _y, _DT, _tau0); _t += _DT
            if _i % _step == 0:
                _bif_tau.append(float(_tau0))
                _bif_th1.append(float((_y[0] + np.pi) % (2 * np.pi) - np.pi))

    bif_data = {"tau": _bif_tau, "th1": _bif_th1}
    return (bif_data,)


@app.cell(hide_code=True)
def _(bif_data, mo, plt):
    fig_bif, ax_bif = plt.subplots(figsize=(7, 3.5))
    ax_bif.scatter(bif_data["tau"], bif_data["th1"],
                   s=0.6, color="#7dd3fc", alpha=0.5, linewidths=0)
    ax_bif.set_xlabel("drive amplitude tau_0 (N m)")
    ax_bif.set_ylabel("theta_1 stroboscopic (rad)")
    ax_bif.set_title("bifurcation diagram: period-doubling route to chaos in theta_1")
    ax_bif.set_xlim(0, 12)
    ax_bif.set_ylim(-np.pi, np.pi)
    fig_bif.tight_layout()
    mo.vstack([
        mo.md("Single point per tau_0 = periodic; vertical band = period-doubled; dense cloud = chaos."),
        fig_bif,
    ])
    return


if __name__ == "__main__":
    app.run()
