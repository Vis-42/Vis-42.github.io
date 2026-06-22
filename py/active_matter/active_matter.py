import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Active Matter")


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
        # Active matter: collective motion and single-cell swimming

        Two models from scratch. The Vicsek model is about a crowd: N self-propelled particles
        aligning with their neighbours, swept over noise $\eta$ to trace the order parameter
        $\phi = |\langle e^{i\theta}\rangle|$ through its disorder-to-order transition.
        Run-and-tumble is about a single cell: straight runs interrupted by Poisson reorientations,
        setting a ballistic-to-diffusive crossover in the mean-squared displacement.

        Three panels share one animation clock: the flock coloured by heading, $\phi(t)$ with
        an orange dot tracking the current frame, and the $\phi$ vs $\eta$ phase diagram with
        the current noise level marked.
        """
    )
    return


@app.cell(hide_code=True)
def _(json, mo):
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
let _obsPanel=null,dragObs=false;
function panelAt(cx,cy){
  const W=cv.width,H=cv.height;
  for(let i=0;i<S.layout.length;i++){
    const[lx,ly,lw,lh]=S.layout[i];
    if(cx>=lx*W&&cx<=(lx+lw)*W&&cy>=ly*H&&cy<=(ly+lh)*H)return i;
  }return -1;
}
cv.addEventListener("mousedown",e=>{
  const r=cv.getBoundingClientRect();
  const cx=(e.clientX-r.left)*DPR,cy=(e.clientY-r.top)*DPR;
  if(_obsPanel&&_obsPanel._lv&&_obsPanel._ctx){
    const{ox,oy,sc}=_obsPanel._ctx,obs=_obsPanel._lv.obs;
    const osx=ox+obs.x*sc,osy=oy+obs.y*sc,osr=(obs.r+0.3)*sc;
    if((cx-osx)**2+(cy-osy)**2<(osr+12*DPR)**2){dragObs=true;return;}
  }
  drag=true;dragX=e.clientX-r.left;dragY=e.clientY-r.top;dragPi=panelAt(cx,cy);
});
cv.addEventListener("mousemove",e=>{
  if(dragObs&&_obsPanel&&_obsPanel._ctx){
    const r=cv.getBoundingClientRect();
    const cx=(e.clientX-r.left)*DPR,cy=(e.clientY-r.top)*DPR;
    const{ox,oy,sc,L}=_obsPanel._ctx,obs=_obsPanel._lv.obs;
    obs.x=Math.max(obs.r,(Math.min(L-obs.r,(cx-ox)/sc)));
    obs.y=Math.max(obs.r,(Math.min(L-obs.r,(cy-oy)/sc)));
    return;
  }
  if(!drag||dragPi<0)return;
  const r=cv.getBoundingClientRect(),nx=(e.clientX-r.left),ny=(e.clientY-r.top);
  const dx=(nx-dragX)/cv.clientWidth,dy=(ny-dragY)/cv.clientHeight;
  VP[dragPi].px+=dx;VP[dragPi].py+=dy;dragX=nx;dragY=ny;
});
cv.addEventListener("mouseup",()=>{drag=false;dragObs=false;});
cv.addEventListener("mouseleave",()=>{drag=false;dragObs=false;});
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
      if(ths){
        const cx_=ox+xs[i]*sc,cy_=oy+ys[i]*sc,AL=r*2.4,AW=r*0.75;
        ctx.save();ctx.translate(cx_,cy_);ctx.rotate(ths[i]);
        ctx.fillStyle=clr;ctx.beginPath();ctx.moveTo(AL,0);ctx.lineTo(-AW,-AW);ctx.lineTo(-AW,AW);ctx.closePath();ctx.fill();
        ctx.restore();
      } else {
        ctx.fillStyle=clr;ctx.beginPath();ctx.arc(ox+xs[i]*sc,oy+ys[i]*sc,r,0,7);ctx.fill();
      }
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

  /* ─────────────── LIVE VICSEK (JS-side simulation + draggable obstacle) ─── */
  else if(p.kind==="live_vicsek"){
    if(!p._lv){
      const N=p.N,L=p.L,v0=p.v0,eta=p.eta,R=p.R||1.0,obs_r=p.obs_r||Math.max(0.8,L*0.08);
      const x_=new Float32Array(N),y_=new Float32Array(N),th_=new Float32Array(N);
      for(let i=0;i<N;i++){x_[i]=Math.random()*L;y_[i]=Math.random()*L;th_[i]=(Math.random()-0.5)*2*Math.PI;}
      const na_=new Float32Array(N);
      /* warm-up */
      for(let w=0;w<40;w++){
        for(let i=0;i<N;i++){let c=0,s=0,n=0;for(let j=0;j<N;j++){let dx=x_[j]-x_[i],dy=y_[j]-y_[i];dx-=L*Math.round(dx/L);dy-=L*Math.round(dy/L);if(dx*dx+dy*dy<R*R){c+=Math.cos(th_[j]);s+=Math.sin(th_[j]);n++;}}na_[i]=(n>0?Math.atan2(s/n,c/n):th_[i])+(Math.random()-0.5)*eta;}
        for(let i=0;i<N;i++){x_[i]=(x_[i]+v0*Math.cos(na_[i])+L)%L;y_[i]=(y_[i]+v0*Math.sin(na_[i])+L)%L;th_[i]=na_[i];}
      }
      p._lv={N,L,v0,eta,R,x:x_,y:y_,th:th_,na:na_,obs:{x:L*0.35,y:L*0.5,r:obs_r},_lf:-1};
      _obsPanel=p;
    }
    const st=p._lv;
    if(frame!==st._lf){
      st._lf=frame;
      const{N,L,v0,eta,R,x,y,th,na,obs}=st,R2=R*R,obsr2=(obs.r+0.35)*(obs.r+0.35);
      for(let i=0;i<N;i++){let c=0,s=0,n=0;for(let j=0;j<N;j++){let dx=x[j]-x[i],dy=y[j]-y[i];dx-=L*Math.round(dx/L);dy-=L*Math.round(dy/L);if(dx*dx+dy*dy<R2){c+=Math.cos(th[j]);s+=Math.sin(th[j]);n++;}}na[i]=(n>0?Math.atan2(s/n,c/n):th[i])+(Math.random()-0.5)*eta;}
      for(let i=0;i<N;i++){
        const vx=v0*Math.cos(na[i]),vy=v0*Math.sin(na[i]);
        let nx_=(x[i]+vx+L)%L,ny_=(y[i]+vy+L)%L;
        const ddx=nx_-obs.x,ddy=ny_-obs.y,dd2=ddx*ddx+ddy*ddy;
        if(dd2<obsr2){const d=Math.sqrt(dd2)||1;const onx=ddx/d,ony=ddy/d;nx_=obs.x+(obs.r+0.35)*onx;ny_=obs.y+(obs.r+0.35)*ony;nx_=(nx_+L)%L;ny_=(ny_+L)%L;const dot=vx*onx+vy*ony;na[i]=Math.atan2(vy-2*dot*ony,vx-2*dot*onx);}
        x[i]=nx_;y[i]=ny_;th[i]=na[i];
      }
    }
    const{N,L,x,y,th,obs}=st;
    const sc_=Math.min(pw,ph)*0.9/L,ox_=px+(pw-L*sc_)/2,oy_=py+(ph-L*sc_)/2;
    p._ctx={ox:ox_,oy:oy_,sc:sc_,L};
    ctx.strokeStyle="#243040";ctx.lineWidth=1;ctx.strokeRect(ox_,oy_,L*sc_,L*sc_);
    const r_=Math.max(2,(p.r||0.3)*sc_),AL=r_*2.4,AW=r_*0.75;
    for(let i=0;i<N;i++){
      const hue=((th[i]%(2*Math.PI))+2*Math.PI)%(2*Math.PI)/(2*Math.PI);
      const cx_=ox_+x[i]*sc_,cy_=oy_+y[i]*sc_;
      ctx.save();ctx.translate(cx_,cy_);ctx.rotate(th[i]);
      ctx.fillStyle=hsv(hue,0.85,0.95);
      ctx.beginPath();ctx.moveTo(AL,0);ctx.lineTo(-AW,-AW);ctx.lineTo(-AW,AW);ctx.closePath();ctx.fill();
      ctx.restore();
    }
    const osx=ox_+obs.x*sc_,osy=oy_+obs.y*sc_,osr=obs.r*sc_;
    ctx.beginPath();ctx.arc(osx,osy,osr,0,7);
    ctx.fillStyle="rgba(80,30,10,0.28)";ctx.fill();
    ctx.strokeStyle="rgba(220,110,40,0.9)";ctx.lineWidth=2.5*DPR;ctx.stroke();
    let cs_=0,sn_=0;for(let i=0;i<N;i++){cs_+=Math.cos(th[i]);sn_+=Math.sin(th[i]);}
    const phi_=Math.sqrt(cs_*cs_+sn_*sn_)/N;
    ctx.fillStyle="#626b7c";ctx.font=(9*DPR)+"px monospace";ctx.textAlign="left";
    ctx.fillText("η="+st.eta.toFixed(2)+"  φ="+phi_.toFixed(2)+"  drag the obstacle",px+4*DPR,py+ph-10*DPR);
    if(p.title){ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="left";ctx.fillText(p.title,px+4*DPR,py+12*DPR);}
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
addEventListener("mouseup",()=>{drag3=false;drag=false;dragObs=false;});
cv.addEventListener("wheel",e=>{zoom3=Math.max(0.3,Math.min(3,zoom3*Math.exp(-e.deltaY*0.001)));},{passive:true});

/* ── Animation loop ─────────────────────────────────────────── */
function loop(t){if(playing&&t-last>S.dt/speed){frame=(frame+1)%S.frames;last=t;}try{draw();}catch(e){}requestAnimationFrame(loop);}
function _kick(){fit();if(cv.width>0&&cv.height>0){requestAnimationFrame(loop);}else{setTimeout(_kick,80);}}_kick();

    </script></body></html>"""

    def canvas_anim(spec, height="480px"):
        import json as _j
        return mo.iframe(_JS.replace("__SPEC__", _j.dumps(spec)), height=height)

    return canvas_anim,


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 1: Vicsek model")
    return


@app.cell(hide_code=True)
def _(mo):
    eta_sl = mo.ui.slider(0.0, 6.0,  value=1.5,  step=0.1,  label="noise η",    show_value=True)
    rho_sl = mo.ui.slider(0.5, 4.0,  value=2.0,  step=0.25, label="density ρ",  show_value=True)
    N_sl   = mo.ui.slider(40,  240,   value=120,  step=20,   label="particles N", show_value=True)
    v0_sl  = mo.ui.slider(0.05, 0.4,  value=0.15, step=0.05, label="speed v₀",  show_value=True)
    mo.hstack([eta_sl, rho_sl, N_sl, v0_sl], justify="start", gap=2)
    return N_sl, eta_sl, rho_sl, v0_sl


@app.cell(hide_code=True)
def _(N_sl, eta_sl, np, rho_sl, v0_sl):
    _N   = int(N_sl.value)
    _eta = float(eta_sl.value)
    _rho = float(rho_sl.value)
    _v0  = float(v0_sl.value)
    _L   = float(np.sqrt(_N / _rho))
    _WARM   = 60
    _FRAMES = 140
    _rng = np.random.default_rng(42)

    def _step(x, y, th, N, L, v0, eta, rng):
        dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
        nb = (dx**2 + dy**2) < 1.0
        s = (nb * np.sin(th)).sum(1) / nb.sum(1)
        c = (nb * np.cos(th)).sum(1) / nb.sum(1)
        th2 = np.arctan2(s, c) + eta * rng.uniform(-0.5, 0.5, N)
        return (x + v0 * np.cos(th2)) % L, (y + v0 * np.sin(th2)) % L, th2

    _x = _rng.uniform(0, _L, _N); _y = _rng.uniform(0, _L, _N)
    _th = _rng.uniform(-np.pi, np.pi, _N)
    for _ in range(_WARM):
        _x, _y, _th = _step(_x, _y, _th, _N, _L, _v0, _eta, _rng)

    _xs_f, _ys_f, _ths_f, _phi_s = [], [], [], []
    for _ in range(_FRAMES):
        _x, _y, _th = _step(_x, _y, _th, _N, _L, _v0, _eta, _rng)
        _xs_f.append(np.round(_x, 2).tolist())
        _ys_f.append(np.round(_y, 2).tolist())
        _ths_f.append(np.round(_th, 2).tolist())
        _phi_s.append(float(np.abs(np.mean(np.exp(1j * _th)))))

    _phi_mean = float(np.mean(_phi_s[_FRAMES // 2:]))
    _regime = "ORDERED" if _phi_mean > 0.65 else "TRANSITION" if _phi_mean > 0.3 else "DISORDERED"

    vicsek_core = {
        "xs": _xs_f, "ys": _ys_f, "ths": _ths_f, "phi": _phi_s,
        "L": float(_L), "frames": _FRAMES,
        "eta": float(_eta),
        "eta_val": [float(_eta)] * _FRAMES,
        "phi_mean": _phi_mean, "regime": _regime,
        "step_fn": _step, "rng": _rng,
        "N": _N, "v0": float(_v0),
    }
    return vicsek_core,


@app.cell(hide_code=True)
def _(np, vicsek_core):
    # Phase-diagram sweep: small N and short runs so it always computes on load.
    _d = vicsek_core
    _N_sw = 45
    _L_sw = float(np.sqrt(_N_sw / 2.0))
    _rng_sw = np.random.default_rng(7)
    _eta_arr = np.linspace(0.2, 5.8, 11)
    _phi_arr = []
    for _e in _eta_arr:
        _xs2 = _rng_sw.uniform(0, _L_sw, _N_sw)
        _ys2 = _rng_sw.uniform(0, _L_sw, _N_sw)
        _ths2 = _rng_sw.uniform(-np.pi, np.pi, _N_sw)
        for _ in range(45):
            _xs2, _ys2, _ths2 = _d["step_fn"](_xs2, _ys2, _ths2, _N_sw, _L_sw, _d["v0"], _e, _rng_sw)
        _ps = [float(np.abs(np.mean(np.exp(1j * _ths2))))]
        for _ in range(30):
            _xs2, _ys2, _ths2 = _d["step_fn"](_xs2, _ys2, _ths2, _N_sw, _L_sw, _d["v0"], _e, _rng_sw)
            _ps.append(float(np.abs(np.mean(np.exp(1j * _ths2)))))
        _phi_arr.append(float(np.mean(_ps)))
    vicsek_data = {
        **_d,
        "eta_sweep": _eta_arr.tolist(),
        "phi_sweep": _phi_arr,
        "has_sweep": True,
    }
    return vicsek_data,


@app.cell(hide_code=True)
def _(canvas_anim, mo, vicsek_data):
    _d = vicsek_data
    _panels = [
        {"kind": "live_vicsek",
         "N": _d["N"], "L": _d["L"], "v0": _d["v0"], "eta": _d["eta"],
         "R": 1.0, "obs_r": max(0.8, float(_d["L"]) * 0.08)},
        {"kind": "phi_t", "phi": _d["phi"]},
    ]
    _layout = [[0.0, 0.0, 0.50, 1.00], [0.50, 0.0, 0.50, 1.00]]
    if _d["has_sweep"] and len(_d["eta_sweep"]) > 0:
        _panels[1] = {"kind": "phi_t", "phi": _d["phi"]}
        _panels.append({"kind": "scatter_vline", "title": "Phase diagram  φ vs η", "xlabel": "noise η", "ylabel": "φ",
                         "ymin": 0.0, "ymax": 1.05,
                         "x": _d["eta_sweep"], "y": _d["phi_sweep"],
                         "dot_x": _d["eta_val"], "dot_y": _d["phi"]})
        _layout = [[0.0, 0.0, 0.50, 1.00], [0.50, 0.0, 0.50, 0.50], [0.50, 0.50, 0.50, 0.50]]
    _spec = {"frames": _d["frames"], "dt": 60, "panels": _panels, "layout": _layout}
    mo.vstack([
        mo.md(
            f"**{_d['regime']}** &nbsp;|&nbsp; "
            f"φ (final) = {_d['phi'][-1]:.2f} &nbsp;|&nbsp; "
            f"φ (mean, last half) = {_d['phi_mean']:.2f} &nbsp;|&nbsp; "
            f"box L = {_d['L']:.2f}"
            + (" &nbsp;|&nbsp; press 'compute phase diagram' above to add the φ vs η panel" if not _d["has_sweep"] else "")
        ),
        canvas_anim(_spec, height="500px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **The Vicsek transition.** Below a critical noise $\eta_c$ the system snaps into a
        flock: particles align headings and $\phi = |\langle e^{i\theta}\rangle|$ rises sharply.
        The top-right panel traces $\phi(t)$ for the current run; the orange dot marks the
        current frame. The bottom-right shows the full $\phi$ vs $\eta$ bifurcation curve,
        with the current noise level marked by the orange dot and vline.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 2: Run-and-tumble")
    return


@app.cell(hide_code=True)
def _(mo):
    v_sl   = mo.ui.slider(0.1, 1.0,  value=0.4,  step=0.05, label="speed v",       show_value=True)
    lam_sl = mo.ui.slider(0.02, 0.5, value=0.1,  step=0.02, label="tumble rate λ", show_value=True)
    kap_sl = mo.ui.slider(0.0, 0.4,  value=0.0,  step=0.05, label="confinement κ", show_value=True)
    mo.hstack([v_sl, lam_sl, kap_sl], justify="start", gap=2)
    return kap_sl, lam_sl, v_sl


@app.cell(hide_code=True)
def _(kap_sl, lam_sl, np, v_sl):
    _v   = float(v_sl.value)
    _lam = float(lam_sl.value)
    _kap = float(kap_sl.value)
    _NW  = 25
    _T   = 220
    _dt2 = 0.2
    _D_eff = _v**2 / (2 * _lam)
    _tau_rt = 1.0 / _lam
    _rng2 = np.random.default_rng(7)
    _px = _rng2.standard_normal((_NW,)) * 0.05
    _py = _rng2.standard_normal((_NW,)) * 0.05
    _ph = _rng2.uniform(-np.pi, np.pi, (_NW,))
    _x0, _y0 = _px.copy(), _py.copy()
    _paths_w = [[None] * _T for _ in range(_NW)]
    _msd_sim, _t_msd = [], []
    for _s in range(_T):
        _tumble = _rng2.random(_NW) < _lam * _dt2
        _ph = np.where(_tumble, _rng2.uniform(-np.pi, np.pi, _NW), _ph)
        _px = _px + _v * np.cos(_ph) * _dt2
        _py = _py + _v * np.sin(_ph) * _dt2
        if _kap > 0:
            _px -= _kap * _px * _dt2
            _py -= _kap * _py * _dt2
        for _w in range(_NW):
            _paths_w[_w][_s] = [round(float(_px[_w]), 3), round(float(_py[_w]), 3)]
        _t_now = (_s + 1) * _dt2
        _t_msd.append(float(_t_now))
        _msd_sim.append(float(np.mean((_px - _x0)**2 + (_py - _y0)**2)))

    _t_arr = np.array(_t_msd)
    _msd_th = (4 * _D_eff * (_t_arr + (2.0 / _lam) * (np.exp(-_lam * _t_arr) - 1))).tolist()
    _span = max(float(np.abs(_px).max()), float(np.abs(_py).max()), 1.0) * 2.5
    _conf_r = float(np.sqrt(_D_eff / _kap)) if _kap > 0 else 0.0

    rnt_data = {
        "paths": _paths_w, "t_msd": _t_msd,
        "msd_sim": _msd_sim, "msd_th": _msd_th,
        "frames": _T, "L": float(_span), "conf": _conf_r,
        "D_eff": float(_D_eff), "tau": float(_tau_rt),
    }
    return (rnt_data,)


@app.cell(hide_code=True)
def _(canvas_anim, mo, rnt_data):
    _d = rnt_data
    _spec2 = {
        "frames": _d["frames"], "dt": 55,
        "panels": [
            {"kind": "rnt", "title": "Run-and-tumble walkers", "xlabel": "position", "ylabel": "y",
             "paths": _d["paths"],
             "msd_sim": _d["msd_sim"], "msd_th": _d["msd_th"],
             "t_msd": _d["t_msd"],
             "L": _d["L"], "conf": _d["conf"]},
        ],
        "layout": [[0.0, 0.0, 1.0, 1.0]],
    }
    mo.vstack([
        mo.md(
            f"**D_eff = {_d['D_eff']:.3f}** &nbsp;|&nbsp; "
            f"τ = 1/λ = {_d['tau']:.2f} &nbsp;|&nbsp; "
            "left: walker ensemble (coloured trails), "
            "right: MSD log-log, blue = simulation, red dashed = theory"
        ),
        canvas_anim(_spec2, height="420px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **The MSD crossover.** At short times $t \ll \tau = 1/\lambda$ walkers have not yet
        tumbled and travel straight, so MSD $\propto t^2$ (ballistic). At long times
        $t \gg \tau$ many tumbles randomise the heading and MSD $= 4 D_\text{eff} t$
        (diffusive) with $D_\text{eff} = v^2/(2\lambda)$. The theory curve
        $4 D_\text{eff}[t + (2/\lambda)(e^{-\lambda t}-1)]$ interpolates between
        the two regimes. With confinement $\kappa > 0$ the MSD saturates at
        $\sim D_\text{eff}/\kappa$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
