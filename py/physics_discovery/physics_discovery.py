import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Physics Discovery")


@app.cell(hide_code=True)
def _():
    import json
    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return json, matplotlib, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Physics discovery: SINDy, conservation laws, symbolic regression

        Three complementary ways to recover physical laws from trajectory data alone.
        SINDy identifies governing ODEs via sparse regression. Kernel conservation-law
        discovery finds what is preserved. Symbolic genetic programming discovers
        closed-form formulas. Each section runs its own animation; all three share
        one RK4 integrator written from scratch.
        """
    )
    return


# ── Canvas animation helper (shared by all three parts) ────────────────────────
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

  /* ─────────────── MULTI-SERIES PHASE PORTRAIT (scatter with series[]) ─── */
  else if(p.kind==="phase_portrait"){
    const seriesList=p.series||[{xs:p.x||[],ys:p.y||[]}];
    const allX=seriesList.flatMap(s=>s.xs),allY=seriesList.flatMap(s=>s.ys);
    const xd={min:p.xmin!==undefined?p.xmin:Math.min(...allX),max:p.xmax!==undefined?p.xmax:Math.max(...allX),log:false};
    const yd={min:p.ymin!==undefined?p.ymin:Math.min(...allY),max:p.ymax!==undefined?p.ymax:Math.max(...allY),log:false};
    const{mx,my,mw,mh,toX,toY}=axes2D(p,lx,ly,lw,lh,xd,yd,vp);
    ctx.save();ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.clip();
    const colors=p.colors||["#7dd3fc","#f97316","#a594f2"];
    const trail=p.trail||80;
    const upTo=Math.min(f+1,seriesList[0]?.xs?.length||0);
    for(let si=0;si<seriesList.length;si++){
      const sx=seriesList[si].xs,sy=seriesList[si].ys,c=colors[si%colors.length];
      const t0=Math.max(0,upTo-trail);
      ctx.strokeStyle=c+(si===0?"":"80");ctx.lineWidth=(si===0?2:1.5)*DPR;ctx.beginPath();
      for(let i=t0;i<upTo;i++){const x=toX(sx[i]),y=toY(sy[i]);i===t0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.stroke();
      if(upTo>0){ctx.fillStyle=c;ctx.beginPath();ctx.arc(toX(sx[upTo-1]),toY(sy[upTo-1]),4*DPR,0,7);ctx.fill();}
    }
    ctx.restore();
    if(p.labels)legend(px,py,pw,ph,p.labels.map((l,i)=>[l,(p.colors||colors)[i%colors.length]]));
  }

  /* ─────────────── PENDULUM ARM ─── */
  else if(p.kind==="pendulum_arm"){
    const th=p.theta[f],cx=px+pw*0.5,cy=py+ph*0.22,sc=Math.min(pw,ph)*0.55;
    const bx=cx+sc*Math.sin(th),by=cy+sc*Math.cos(th);
    const t0=Math.max(0,f-(p.trail||60));
    ctx.strokeStyle="rgba(125,211,252,0.12)";ctx.lineWidth=1.5;ctx.beginPath();
    for(let i=t0;i<f;i++){const bxi=cx+sc*Math.sin(p.theta[i]),byi=cy+sc*Math.cos(p.theta[i]);i===t0?ctx.moveTo(bxi,byi):ctx.lineTo(bxi,byi);}ctx.stroke();
    ctx.strokeStyle="#7dd3fc";ctx.lineWidth=2.5*DPR;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(bx,by);ctx.stroke();
    ctx.fillStyle="#9c3b2c";ctx.beginPath();ctx.arc(bx,by,7*DPR,0,7);ctx.fill();
    ctx.fillStyle="#8a93a6";ctx.beginPath();ctx.arc(cx,cy,3*DPR,0,7);ctx.fill();
    if(p.title){ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(p.title,px+pw/2,py+12*DPR);}
  }

  /* ─────────────── 2D ORBIT ─── */
  else if(p.kind==="orbit"){
    const xs=p.x,ys_=p.y;
    const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin_=Math.min(...ys_),ymax_=Math.max(...ys_);
    const pad=(xmax-xmin)*0.15||0.5;
    const xd={min:xmin-pad,max:xmax+pad,log:false},yd={min:ymin_-pad,max:ymax_+pad,log:false};
    const{mx,my,mw,mh,toX,toY}=axes2D(p,lx,ly,lw,lh,xd,yd,vp);
    ctx.save();ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.clip();
    ctx.strokeStyle="rgba(125,211,252,0.08)";ctx.lineWidth=1;ctx.beginPath();
    for(let i=0;i<xs.length;i++){const x=toX(xs[i]),y=toY(ys_[i]);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.stroke();
    const t0=Math.max(0,f-(p.trail||80));
    ctx.strokeStyle="#7dd3fc";ctx.lineWidth=2*DPR;ctx.beginPath();
    for(let i=t0;i<=f;i++){const x=toX(xs[i]),y=toY(ys_[i]);i===t0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.stroke();
    ctx.fillStyle="#9c3b2c";ctx.beginPath();ctx.arc(toX(xs[f]),toY(ys_[f]),5*DPR,0,7);ctx.fill();
    ctx.restore();
    if(p.title){ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="center";ctx.fillText(p.title,px+pw/2,py+12*DPR);}
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

    def canvas_anim(spec, height="480px"):
        import json as _j
        return mo.iframe(_JS.replace("__SPEC__", _j.dumps(spec)), height=height)

    return (canvas_anim,)


# ── Shared numerics (RK4, library builder, STLSQ) ──────────────────────────────
@app.cell(hide_code=True)
def _(np):
    def _rk4_step(f, s, t, dt, p):
        k1 = f(s, t, p)
        k2 = f(s + 0.5 * dt * k1, t + 0.5 * dt, p)
        k3 = f(s + 0.5 * dt * k2, t + 0.5 * dt, p)
        k4 = f(s + dt * k3, t + dt, p)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4_integrate(f, u0, t0, t1, n, p=None):
        dt = (t1 - t0) / (n - 1)
        states = np.empty((n, len(u0)))
        states[0] = u0
        for i in range(1, n):
            t = t0 + (i - 1) * dt
            states[i] = _rk4_step(f, states[i - 1], t, dt, p)
        return states, dt

    def centered_diff(states, dt):
        n, d = states.shape
        dX = np.empty_like(states)
        dX[0] = (states[1] - states[0]) / dt
        dX[-1] = (states[-1] - states[-2]) / dt
        for i in range(1, n - 1):
            dX[i] = (states[i + 1] - states[i - 1]) / (2 * dt)
        return dX

    def poly_library(states, max_deg):
        n, d = states.shape
        cols = [np.ones(n)]
        names = ["1"]
        for total in range(1, max_deg + 1):
            if d == 2:
                for e0 in range(total + 1):
                    e1 = total - e0
                    col = states[:, 0] ** e0 * states[:, 1] ** e1
                    nm = ""
                    if e0 == 1:
                        nm += "x"
                    elif e0 > 1:
                        nm += f"x^{e0}"
                    if e1 == 1:
                        nm += "y"
                    elif e1 > 1:
                        nm += f"y^{e1}"
                    cols.append(col)
                    names.append(nm)
            else:
                # general: enumerate multi-indices
                from itertools import combinations_with_replacement
                for idx in combinations_with_replacement(range(d), total):
                    col = np.ones(n)
                    parts = {}
                    for j in idx:
                        parts[j] = parts.get(j, 0) + 1
                    for j, e in parts.items():
                        col = col * states[:, j] ** e
                    nm = "".join(f"x{j}^{e}" if e > 1 else f"x{j}" for j, e in sorted(parts.items()))
                    cols.append(col)
                    names.append(nm)
        return np.column_stack(cols), names

    def stlsq(Theta, dX, threshold, max_iter=20):
        xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]
        for _ in range(max_iter):
            small = np.abs(xi) < threshold
            xi[small] = 0.0
            for j in range(dX.shape[1]):
                idx = ~small[:, j]
                if idx.any():
                    xi[idx, j] = np.linalg.lstsq(Theta[:, idx], dX[:, j], rcond=None)[0]
        return xi

    return centered_diff, poly_library, rk4_integrate, stlsq


# ── Part 1: SINDy ──────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 1: SINDy: Sparse Identification of Nonlinear Dynamics")
    return


@app.cell(hide_code=True)
def _(mo):
    sindy_sys = mo.ui.dropdown(
        options={"Van der Pol": "vdp", "Duffing (undriven)": "duffing"},
        value="Van der Pol",
        label="system",
    )
    sindy_noise = mo.ui.slider(0.0, 0.10, value=0.0, step=0.005, label="noise σ", show_value=True)
    sindy_thresh = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="STLSQ threshold λ", show_value=True)
    sindy_deg = mo.ui.slider(2, 5, value=3, step=1, label="poly degree", show_value=True)
    mo.hstack([sindy_sys, sindy_noise, sindy_thresh, sindy_deg], justify="start", gap=2)
    return sindy_deg, sindy_noise, sindy_sys, sindy_thresh


@app.cell(hide_code=True)
def _(centered_diff, np, poly_library, rk4_integrate, sindy_deg, sindy_noise, sindy_sys, sindy_thresh, stlsq):
    _sysname = sindy_sys.value
    _sigma   = float(sindy_noise.value)
    _lam     = float(sindy_thresh.value)
    _deg     = int(sindy_deg.value)
    _N_pts   = 800
    _rng_s   = np.random.default_rng(42)

    if _sysname == "vdp":
        _mu = 1.0
        def _sindy_ode(s, t, p):
            return np.array([s[1], p * (1 - s[0]**2) * s[1] - s[0]])
        _u0_s = np.array([2.0, 0.0])
        _p_s  = _mu
        _t1_s = 30.0
    else:
        _al, _be, _de = -1.0, 1.0, 0.1
        def _sindy_ode(s, t, p):
            al, be, de = p
            return np.array([s[1], -de * s[1] - al * s[0] - be * s[0]**3])
        _u0_s = np.array([0.5, 0.0])
        _p_s  = (_al, _be, _de)
        _t1_s = 40.0

    _states_s, _dt_s = rk4_integrate(_sindy_ode, _u0_s, 0.0, _t1_s, _N_pts, _p_s)
    _states_n = _states_s + _sigma * _rng_s.standard_normal(_states_s.shape)
    _dX_s = centered_diff(_states_n, _dt_s)
    _Theta_s, _lib_names_s = poly_library(_states_n, _deg)
    _xi_s = stlsq(_Theta_s, _dX_s, _lam)

    # Reconstruct using identified model
    def _sindy_hat(s, t, xi):
        _row = poly_library(s[None, :], _deg)[0][0]
        return _row @ xi

    _recon_s = np.empty_like(_states_s)
    _recon_s[0] = _u0_s
    for _i in range(1, _N_pts):
        _s_prev = _recon_s[_i - 1]
        if np.any(np.isnan(_s_prev)) or np.any(np.isinf(_s_prev)) or np.max(np.abs(_s_prev)) > 300:
            _recon_s[_i] = np.nan
        else:
            _k1 = _sindy_hat(_s_prev, 0, _xi_s)
            _k2 = _sindy_hat(_s_prev + 0.5 * _dt_s * _k1, 0, _xi_s)
            _k3 = _sindy_hat(_s_prev + 0.5 * _dt_s * _k2, 0, _xi_s)
            _k4 = _sindy_hat(_s_prev + _dt_s * _k3, 0, _xi_s)
            _ns = _s_prev + (_dt_s / 6) * (_k1 + 2*_k2 + 2*_k3 + _k4)
            _recon_s[_i] = _ns if not (np.any(np.isnan(_ns)) or np.max(np.abs(_ns)) > 300) else np.nan

    _n_active_s = int(np.sum(np.any(np.abs(_xi_s) > 1e-8, axis=1)))
    _n_lib_s    = len(_lib_names_s)
    _sparsity_s = round(1 - _n_active_s / _n_lib_s, 3)
    _dX_hat_s   = _Theta_s @ _xi_s
    _rmse_s     = float(np.sqrt(np.mean((_dX_hat_s - _dX_s)**2)))

    def _eq_str(col, names, xi, threshold=1e-8):
        terms = []
        for i, nm in enumerate(names):
            c = xi[i, col]
            if abs(c) > threshold:
                terms.append(f"{c:+.3f} {nm}")
        return " ".join(terms) if terms else "0"

    _eq0 = _eq_str(0, _lib_names_s, _xi_s)
    _eq1 = _eq_str(1, _lib_names_s, _xi_s)

    sindy_data = {
        "states": _states_s,
        "recon":  _recon_s,
        "eq0":    _eq0,
        "eq1":    _eq1,
        "lib_size": _n_lib_s,
        "active":   _n_active_s,
        "sparsity": _sparsity_s,
        "rmse":     _rmse_s,
        "sysname":  _sysname,
    }
    return (sindy_data,)


@app.cell(hide_code=True)
def _(canvas_anim, mo, np, sindy_data):
    _d  = sindy_data
    _st = _d["states"]
    _rc = _d["recon"]
    _valid = ~np.isnan(_rc[:, 0])
    _rc_x = np.where(_valid, _rc[:, 0], _st[:, 0]).tolist()
    _rc_y = np.where(_valid, _rc[:, 1], _st[:, 1]).tolist()
    _all_x = np.concatenate([_st[:, 0], _rc[:, 0][_valid]])
    _all_y = np.concatenate([_st[:, 1], _rc[:, 1][_valid]])
    _xspan = float(_all_x.max() - _all_x.min()) or 1.0
    _yspan = float(_all_y.max() - _all_y.min()) or 1.0
    _xmin  = float(_all_x.min()) - 0.12 * _xspan
    _xmax  = float(_all_x.max()) + 0.12 * _xspan
    _ymin  = float(_all_y.min()) - 0.12 * _yspan
    _ymax  = float(_all_y.max()) + 0.12 * _yspan

    _spec_s = {
        "frames": len(_st), "dt": 40,
        "panels": [
            {
                "kind": "phase_portrait",
                "title": "Phase portrait: true (blue) vs SINDy (orange)",
                "xlabel": "x", "ylabel": "y",
                "series": [
                    {"xs": np.round(_st[:, 0], 3).tolist(), "ys": np.round(_st[:, 1], 3).tolist()},
                    {"xs": _rc_x, "ys": _rc_y},
                ],
                "labels": ["true trajectory", "SINDy reconstruction"],
                "colors": ["#7dd3fc", "#f97316"],
                "xmin": _xmin, "xmax": _xmax, "ymin": _ymin, "ymax": _ymax,
                "trail": 100,
            },
        ],
        "layout": [[0.0, 0.0, 1.0, 1.0]],
    }
    _rc_color = "#57c98a" if _d["rmse"] < 0.1 else ("#e6a356" if _d["rmse"] < 0.5 else "#e0635a")
    mo.vstack([
        mo.md(
            f"**dx/dt =** `{_d['eq0']}`\n\n"
            f"**dy/dt =** `{_d['eq1']}`\n\n"
            f"library {_d['lib_size']} terms | active {_d['active']} | "
            f"sparsity {_d['sparsity']} | RMSE {_d['rmse']:.4f}"
        ),
        canvas_anim(_spec_s, height="480px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **How STLSQ works.** The algorithm builds a library matrix $\Theta$ whose columns
        are all monomials $1, x, y, x^2, xy, \ldots$ up to the chosen degree, evaluated
        at every time sample. The SINDy hypothesis is $\dot{X} = \Theta\,\xi$ with
        $\xi$ sparse. STLSQ starts from a full least-squares solve and iterates: zero
        every coefficient below $\lambda$, re-solve on the surviving columns, repeat
        until the active set is stable. Physical laws are sparse in any reasonable
        basis, so the correct terms survive thresholding while noise-driven terms do not.
        Raise the noise slider to see the point where the finite-difference derivative
        estimate becomes too noisy for any threshold to clean up.
        """
    )
    return


# ── Part 2: Conservation law discovery ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 2: Conservation law discovery (kernel method)")
    return


@app.cell(hide_code=True)
def _(mo):
    cl_sys = mo.ui.dropdown(
        options={"Pendulum": "pendulum", "Kepler orbit": "kepler", "Duffing (conservative)": "duffing_c"},
        value="Pendulum",
        label="system",
    )
    cl_gamma = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="kernel width gamma", show_value=True)
    cl_theta0 = mo.ui.slider(0.2, 2.9, value=1.2, step=0.1, label="pendulum theta0", show_value=True)
    cl_ecc    = mo.ui.slider(0.0, 0.90, value=0.25, step=0.05, label="Kepler eccentricity", show_value=True)
    cl_x0     = mo.ui.slider(0.2, 1.8, value=0.6, step=0.1, label="Duffing x0", show_value=True)
    mo.vstack([
        mo.hstack([cl_sys, cl_gamma], justify="start", gap=2),
        mo.hstack([cl_theta0, cl_ecc, cl_x0], justify="start", gap=2),
    ])
    return cl_ecc, cl_gamma, cl_sys, cl_theta0, cl_x0


@app.cell(hide_code=True)
def _(cl_ecc, cl_gamma, cl_sys, cl_theta0, cl_x0, np, rk4_integrate):
    _cl_sysname = cl_sys.value
    _cl_gamma   = float(cl_gamma.value)

    if _cl_sysname == "pendulum":
        _th0 = float(cl_theta0.value)
        def _cl_ode(s, t, p):
            return np.array([s[1], -9.81 * np.sin(s[0])])
        _cl_u0 = np.array([_th0, 0.0])
        _cl_p  = None
        _cl_t1 = 20.0
        _cl_n  = 600
        def _cl_true_energy(states):
            return 0.5 * states[:, 1]**2 - 9.81 * np.cos(states[:, 0])
    elif _cl_sysname == "kepler":
        _ecc = float(cl_ecc.value)
        _ecc = min(_ecc, 0.92)
        def _cl_ode(s, t, p):
            r = np.hypot(s[0], s[1])
            return np.array([s[2], s[3], -s[0] / r**3, -s[1] / r**3])
        _cl_u0 = np.array([1.0 - _ecc, 0.0, 0.0, np.sqrt((1 + _ecc) / (1 - _ecc))])
        _cl_p  = None
        _cl_t1 = 4 * 2 * np.pi
        _cl_n  = 600
        def _cl_true_energy(states):
            r = np.hypot(states[:, 0], states[:, 1])
            return 0.5 * (states[:, 2]**2 + states[:, 3]**2) - 1.0 / r
    else:
        _x0_c = float(cl_x0.value)
        def _cl_ode(s, t, p):
            return np.array([s[1], s[0] - s[0]**3])
        _cl_u0 = np.array([_x0_c, 0.0])
        _cl_p  = None
        _cl_t1 = 50.0
        _cl_n  = 600
        def _cl_true_energy(states):
            return 0.5 * states[:, 1]**2 + 0.25 * states[:, 0]**4 - 0.5 * states[:, 0]**2

    _cl_states, _cl_dt = rk4_integrate(_cl_ode, _cl_u0, 0.0, _cl_t1, _cl_n, _cl_p)
    _cl_truth = _cl_true_energy(_cl_states)

    # Kernel invariant discovery
    _na = min(200, _cl_n)
    _idx_anchor = np.round(np.linspace(0, _cl_n - 1, _na)).astype(int)
    _anchors = _cl_states[_idx_anchor]
    _Phi = np.exp(-_cl_gamma * np.sum((_cl_states[:, None, :] - _anchors[None, :, :])**2, axis=2))
    _Phi_c = _Phi - _Phi.mean(axis=0, keepdims=True)
    _dPhi  = (_Phi_c[1:] - _Phi_c[:-1]) / _cl_dt
    _A = _dPhi.T @ _dPhi + 1e-5 * np.eye(_na)
    _B = _Phi_c.T @ _Phi_c + 1e-8 * np.eye(_na)
    try:
        from scipy.linalg import eigh
        _evals, _evecs = eigh(_A, _B)
        _c = _evecs[:, 0]
    except Exception:
        _c = np.linalg.lstsq(_B, np.linalg.lstsq(_A, np.eye(_na), rcond=None)[0][:, 0], rcond=None)[0]
    _discovered = _Phi_c @ _c
    _discovered = _discovered - _discovered[0]

    # Polynomial fit to discovered invariant
    _d2 = _cl_states.shape[1]
    _poly_cols = [np.ones(_cl_n)]
    _poly_names = ["1"]
    for _j in range(_d2):
        _poly_cols.append(_cl_states[:, _j])
        _poly_names.append(f"x{_j+1}")
        _poly_cols.append(_cl_states[:, _j]**2)
        _poly_names.append(f"x{_j+1}^2")
    _P = np.column_stack(_poly_cols)
    _beta = np.linalg.lstsq(_P, _discovered, rcond=None)[0]
    _poly_label_parts = [f"{_beta[i]:.3f} {_poly_names[i]}" for i in range(len(_beta)) if abs(_beta[i]) > 1e-4]
    _poly_label = " + ".join(_poly_label_parts) if _poly_label_parts else "0"
    _poly_pred  = _P @ _beta
    _poly_mse   = float(np.mean((_poly_pred - _discovered)**2))

    _E_drift_pct = float((np.max(_cl_truth) - np.min(_cl_truth)) / (abs(np.mean(_cl_truth)) + 1e-14) * 100)
    _disc_std    = float(np.std(_discovered))

    cl_data = {
        "states":    _cl_states,
        "truth":     _cl_truth,
        "discovered": _discovered,
        "sysname":   _cl_sysname,
        "poly_label": _poly_label,
        "poly_mse":   _poly_mse,
        "E_drift_pct": _E_drift_pct,
        "disc_std":   _disc_std,
        "dt":         _cl_dt,
        "n":          _cl_n,
    }
    return cl_data, eigh


@app.cell(hide_code=True)
def _(canvas_anim, cl_data, mo, np):
    _d2   = cl_data
    _st2  = _d2["states"]
    _n2   = _d2["n"]
    _sys2 = _d2["sysname"]
    _frames2 = _n2

    # Build panels depending on system
    _panels2 = []
    _layout2 = []

    if _sys2 == "pendulum":
        _panels2.append({
            "kind": "pendulum_arm",
            "theta": np.round(_st2[:, 0], 4).tolist(),
            "trail": 60,
            "title": "Pendulum",
        })
        _layout2.append([0.0, 0.0, 0.33, 1.0])
        _ps_x = np.round(_st2[:, 0], 4).tolist()
        _ps_y = np.round(_st2[:, 1], 4).tolist()
        _ps_xlabel = "theta"
        _ps_ylabel = "omega"
    elif _sys2 == "kepler":
        _panels2.append({
            "kind": "orbit",
            "x": np.round(_st2[:, 0], 4).tolist(),
            "y": np.round(_st2[:, 1], 4).tolist(),
            "trail": 80,
            "title": "Kepler orbit",
        })
        _layout2.append([0.0, 0.0, 0.33, 1.0])
        _ps_x = np.round(_st2[:, 0], 4).tolist()
        _ps_y = np.round(_st2[:, 2], 4).tolist()
        _ps_xlabel = "x"
        _ps_ylabel = "vx"
    else:
        _ps_x = np.round(_st2[:, 0], 4).tolist()
        _ps_y = np.round(_st2[:, 1], 4).tolist()
        _ps_xlabel = "x"
        _ps_ylabel = "xdot"
        # placeholder panel (just phase portrait)
        _panels2.append({
            "kind": "phase_portrait",
            "title": "Duffing phase space",
            "xlabel": _ps_xlabel, "ylabel": _ps_ylabel,
            "series": [{"xs": _ps_x, "ys": _ps_y}],
            "colors": ["#7dd3fc"],
            "labels": ["trajectory"],
            "xmin": float(np.min(_st2[:, 0])) - 0.2,
            "xmax": float(np.max(_st2[:, 0])) + 0.2,
            "ymin": float(np.min(_st2[:, 1])) - 0.2,
            "ymax": float(np.max(_st2[:, 1])) + 0.2,
            "trail": 80,
        })
        _layout2.append([0.0, 0.0, 0.33, 1.0])

    # Panel B: true energy over time
    _truth_shifted = (_d2["truth"] - _d2["truth"][0]).tolist()
    _panels2.append({
        "kind": "timeseries",
        "ys": [_truth_shifted],
        "colors": ["#57c98a"],
        "labels": ["true E(t) - E(0)"],
        "xlabel": "frame",
        "ylabel": "energy drift",
        "title": "True energy (should be flat)",
    })
    _layout2.append([0.33, 0.0, 0.34, 1.0])

    # Panel C: discovered invariant over time
    _disc_list = _d2["discovered"].tolist()
    _panels2.append({
        "kind": "timeseries",
        "ys": [_disc_list],
        "colors": ["#f97316"],
        "labels": ["kernel phi(t) - phi(0)"],
        "xlabel": "frame",
        "ylabel": "invariant drift",
        "title": "Discovered invariant (tune gamma to flatten)",
    })
    _layout2.append([0.67, 0.0, 0.33, 1.0])

    _spec_c = {"frames": _frames2, "dt": 35, "panels": _panels2, "layout": _layout2}
    _ec = "#57c98a" if _d2["E_drift_pct"] < 0.5 else "#e6a356"
    mo.vstack([
        mo.md(
            f"system: **{_sys2}** | "
            f"energy drift: **{_d2['E_drift_pct']:.3f}%** | "
            f"kernel phi std: **{_d2['disc_std']:.4f}** | "
            f"poly approx: `{_d2['poly_label'][:80]}` | "
            f"poly MSE: {_d2['poly_mse']:.4f}"
        ),
        canvas_anim(_spec_c, height="480px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Kernel invariant discovery.** Sample $n_a$ anchor states uniformly from the
        trajectory. Build the RBF feature matrix
        $\Phi_{ij} = \exp(-\gamma \| x_i - x_j \|^2)$.
        Column-center $\Phi$ to remove the trivial constant. Approximate
        $\dot\Phi$ by finite differences. Solve the generalized eigenproblem
        $\min_{c} c^\top (\dot\Phi^\top \dot\Phi + \alpha I) c$ subject to
        $c^\top (\Phi^\top \Phi) c = 1$: the eigenvector for the smallest eigenvalue
        defines the function $\varphi = \Phi c$ whose temporal drift is minimal.
        A flat $\varphi(t)$ curve means the algorithm found a conserved quantity.
        The polynomial projection in the stats bar shows the discovered function
        is approximately the true Hamiltonian.
        """
    )
    return


# ── Part 3: Symbolic regression (genetic programming) ──────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 3: Symbolic regression via genetic programming")
    return


@app.cell(hide_code=True)
def _(mo):
    gp_eq = mo.ui.dropdown(
        options={
            "Kinetic energy  E = 0.5 m v^2": "ke",
            "Pendulum period  T = 2pi sqrt(L/g)": "period",
            "Ohm law  V = I R": "ohm",
            "Hooke law  F = k x": "hooke",
        },
        value="Kinetic energy  E = 0.5 m v^2",
        label="equation",
    )
    gp_noise = mo.ui.slider(0.0, 0.3, value=0.05, step=0.02, label="noise sigma", show_value=True)
    gp_pop   = mo.ui.slider(30, 120, value=60, step=10, label="population", show_value=True)
    gp_ngen  = mo.ui.slider(10, 60, value=35, step=5, label="generations", show_value=True)
    mo.hstack([gp_eq, gp_noise, gp_pop, gp_ngen], justify="start", gap=2)
    return gp_eq, gp_ngen, gp_noise, gp_pop


@app.cell(hide_code=True)
def _(gp_eq, gp_ngen, gp_noise, gp_pop, np):
    _eq_key  = gp_eq.value
    _gp_sig  = float(gp_noise.value)
    _pop_sz  = int(gp_pop.value)
    _n_gen   = int(gp_ngen.value)
    _rng_gp  = np.random.default_rng(99)
    _N_data  = 120

    # Dataset generation
    if _eq_key == "ke":
        _var_names = ["m", "v"]
        _gp_X = np.column_stack([
            0.1 + 9.9 * _rng_gp.random(_N_data),
            0.1 + 9.9 * _rng_gp.random(_N_data),
        ])
        _gp_y_clean = 0.5 * _gp_X[:, 0] * _gp_X[:, 1]**2
    elif _eq_key == "period":
        _var_names = ["L"]
        _gp_X = (0.1 + 9.9 * _rng_gp.random((_N_data, 1)))
        _gp_y_clean = 2 * np.pi * np.sqrt(_gp_X[:, 0] / 9.81)
    elif _eq_key == "ohm":
        _var_names = ["I", "R"]
        _gp_X = np.column_stack([
            0.001 + 9.999 * _rng_gp.random(_N_data),
            1.0 + 999.0 * _rng_gp.random(_N_data),
        ])
        _gp_y_clean = _gp_X[:, 0] * _gp_X[:, 1]
    else:  # hooke
        _var_names = ["k", "x"]
        _gp_X = np.column_stack([
            1.0 + 999.0 * _rng_gp.random(_N_data),
            0.01 + 0.99 * _rng_gp.random(_N_data),
        ])
        _gp_y_clean = _gp_X[:, 0] * _gp_X[:, 1]

    _sy = max(float(np.std(_gp_y_clean)), 1e-12)
    _gp_y = _gp_y_clean + _gp_sig * _sy * _rng_gp.standard_normal(_N_data)
    _nv   = len(_var_names)

    # Expression tree (pure Python, no classes -- use dicts for Pyodide compatibility)
    def _node(op, value=0.0, var_idx=0, children=None):
        return {"op": op, "value": value, "var_idx": var_idx, "children": children or []}

    def _copy_node(n):
        return _node(n["op"], n["value"], n["var_idx"], [_copy_node(c) for c in n["children"]])

    def _count_nodes(n):
        return 1 + sum(_count_nodes(c) for c in n["children"])

    def _tree_str(n, var_names):
        if n["op"] == "const":
            return f"{n['value']:.3g}"
        if n["op"] == "var":
            i = min(n["var_idx"], len(var_names) - 1)
            return var_names[i]
        ch = n["children"]
        if n["op"] in ("+", "-", "*", "/"):
            return f"({_tree_str(ch[0], var_names)} {n['op']} {_tree_str(ch[1], var_names)})"
        return f"{n['op']}({_tree_str(ch[0], var_names)})"

    _BIN = ["+", "-", "*", "/"]
    _UN  = ["sqrt", "sin", "cos"]

    def _rnd_node(nv, rng, depth=0, max_depth=4):
        if depth >= max_depth or (depth > 1 and rng.random() < 0.4):
            if rng.random() < 0.6:
                return _node("var", var_idx=int(rng.integers(0, nv)))
            return _node("const", value=float(rng.standard_normal() * 2))
        op = _BIN[int(rng.integers(0, len(_BIN)))] if rng.random() < 0.7 else _UN[int(rng.integers(0, len(_UN)))]
        n_ch = 2 if op in _BIN else 1
        return _node(op, children=[_rnd_node(nv, rng, depth + 1, max_depth) for _ in range(n_ch)])

    def _eval_node(n, X):
        op = n["op"]
        if op == "const":
            return np.full(len(X), n["value"])
        if op == "var":
            i = min(n["var_idx"], X.shape[1] - 1)
            return X[:, i]
        ch = n["children"]
        a = _eval_node(ch[0], X)
        if op == "+":
            b = _eval_node(ch[1], X); return a + b
        if op == "-":
            b = _eval_node(ch[1], X); return a - b
        if op == "*":
            b = _eval_node(ch[1], X); return a * b
        if op == "/":
            b = _eval_node(ch[1], X); return a / (np.abs(b) + 1e-10)
        if op == "sqrt":
            return np.sqrt(np.abs(a))
        if op == "sin":
            return np.sin(a)
        if op == "cos":
            return np.cos(a)
        return np.zeros(len(X))

    def _score(n, X, y, cp=0.002):
        try:
            pred = _eval_node(n, X)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e12
            mse = float(np.mean((pred - y)**2))
            var_y = float(np.var(y)) or 1e-12
            return mse / var_y + cp * _count_nodes(n)
        except Exception:
            return 1e12

    def _collect_all(n, acc):
        acc.append(n)
        for c in n["children"]:
            _collect_all(c, acc)

    def _find_rand_parent(n, rng):
        if n["children"] and rng.random() < 0.45:
            i = int(rng.integers(0, len(n["children"])))
            return n, i
        for c in n["children"]:
            p, i = _find_rand_parent(c, rng)
            if p is not None:
                return p, i
        return None, 0

    def _mutate(n, nv, rng, rate=0.2):
        if rng.random() < rate:
            op = n["op"]
            if op == "const":
                n["value"] += float(rng.standard_normal() * 0.3)
            elif op == "var":
                n["var_idx"] = int(rng.integers(0, nv))
            elif op in _BIN:
                n["op"] = _BIN[int(rng.integers(0, len(_BIN)))]
            else:
                n["op"] = _UN[int(rng.integers(0, len(_UN)))]
        for c in n["children"]:
            _mutate(c, nv, rng, rate)

    def _crossover(t1, t2, rng):
        t1 = _copy_node(t1); t2 = _copy_node(t2)
        all_t2 = []
        _collect_all(t2, all_t2)
        if not all_t2:
            return t1
        src = _copy_node(all_t2[int(rng.integers(0, len(all_t2)))])
        p, i = _find_rand_parent(t1, rng)
        if p is not None:
            p["children"][i] = src
        return t1

    # Run GP
    _pop = [_rnd_node(_nv, _rng_gp) for _ in range(_pop_sz)]
    _fits = [_score(t, _gp_X, _gp_y) for t in _pop]
    _best_hist = []
    _elite_frac = max(1, _pop_sz // 20)

    for _gen in range(_n_gen):
        _best_hist.append(min(_fits))
        _ord = np.argsort(_fits)
        _new_pop = [_copy_node(_pop[_ord[i]]) for i in range(_elite_frac)]
        _new_fits = [_fits[_ord[i]] for i in range(_elite_frac)]
        while len(_new_pop) < _pop_sz:
            _i1, _i2 = int(_rng_gp.integers(0, _pop_sz)), int(_rng_gp.integers(0, _pop_sz))
            _p1 = _pop[_i1] if _fits[_i1] < _fits[_i2] else _pop[_i2]
            _i3, _i4 = int(_rng_gp.integers(0, _pop_sz)), int(_rng_gp.integers(0, _pop_sz))
            _p2 = _pop[_i3] if _fits[_i3] < _fits[_i4] else _pop[_i4]
            if _rng_gp.random() < 0.7:
                _child = _crossover(_p1, _p2, _rng_gp)
            else:
                _child = _copy_node(_p1)
            _mutate(_child, _nv, _rng_gp)
            if _count_nodes(_child) > 25:
                _child = _rnd_node(_nv, _rng_gp, max_depth=3)
            _new_pop.append(_child)
            _new_fits.append(_score(_child, _gp_X, _gp_y))
        _pop = _new_pop; _fits = _new_fits

    _best_idx = int(np.argmin(_fits))
    _best_tree = _pop[_best_idx]
    _best_pred = _eval_node(_best_tree, _gp_X)
    _y_bar = float(np.mean(_gp_y))
    _ss_res = float(np.sum((_gp_y - _best_pred)**2))
    _ss_tot = float(np.sum((_gp_y - _y_bar)**2)) or 1e-12
    _r2 = max(0.0, 1.0 - _ss_res / _ss_tot)
    _best_expr = _tree_str(_best_tree, _var_names)
    _complexity = _count_nodes(_best_tree)

    gp_data = {
        "best_hist":  _best_hist,
        "best_expr":  _best_expr,
        "r2":         _r2,
        "complexity": _complexity,
        "eq_key":     _eq_key,
        "y_clean":    _gp_y_clean.tolist(),
        "y_pred":     _best_pred.tolist(),
        "n_gen":      _n_gen,
        "var_names":  _var_names,
    }
    return (gp_data,)


@app.cell(hide_code=True)
def _(canvas_anim, gp_data, mo, np):
    _gd = gp_data
    _hist = _gd["best_hist"]
    _ng   = _gd["n_gen"]

    # Clamp hist for display
    _hist_disp = [min(h, 10.0) for h in _hist]

    # Scatter data for predicted vs true panel as timeseries approximation
    _yc  = np.array(_gd["y_clean"])
    _yp  = np.array(_gd["y_pred"])
    _idx_sort = np.argsort(_yc)
    _yc_s = _yc[_idx_sort].tolist()
    _yp_s = _yp[_idx_sort].tolist()

    _spec_gp = {
        "frames": _ng, "dt": 120,
        "panels": [
            {
                "kind": "timeseries",
                "ys": [_hist_disp],
                "colors": ["#57c98a"],
                "xlabel": "generation",
                "ylabel": "best fitness",
                "title": "GP convergence: best fitness per generation",
                "grow": True,
            },
            {
                "kind": "timeseries",
                "ys": [_yc_s, _yp_s],
                "colors": ["#57c98a", "#f97316"],
                "labels": ["true", "predicted"],
                "xlabel": "sample (sorted by true)",
                "ylabel": "target value",
                "title": f"Predicted vs true (R2 = {_gd['r2']:.3f})",
            },
        ],
        "layout": [
            [0.0, 0.0, 0.5, 1.0],
            [0.5, 0.0, 0.5, 1.0],
        ],
    }
    _r2c = "#57c98a" if _gd["r2"] > 0.9 else ("#e6a356" if _gd["r2"] > 0.6 else "#e0635a")
    mo.vstack([
        mo.md(
            f"**Best expression:** `{_gd['best_expr'][:120]}`\n\n"
            f"R2 = {_gd['r2']:.4f} | complexity = {_gd['complexity']} nodes | "
            f"generations = {_ng} | variables: {', '.join(_gd['var_names'])}"
        ),
        canvas_anim(_spec_gp, height="420px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Genetic programming.** Each candidate formula is an expression tree:
        internal nodes are operators ($+$, $-$, $\times$, $/$, $\sqrt{\cdot}$, $\sin$, $\cos$)
        and leaves are input variables or floating-point constants.
        Fitness is normalised MSE divided by $\text{var}(y)$ -- making the score
        scale-invariant across equations of very different magnitudes -- plus a
        parsimony penalty on node count to prevent bloat.
        Selection is binary tournament: draw two individuals at random, keep the fitter one.
        Offspring are produced by subtree crossover (grafting a random subtree from
        parent 2 onto a random slot in parent 1) or point mutation (changing one
        node's operator or nudging a constant). Elitism copies the top 5% unchanged,
        so the best solution never regresses. The left panel shows fitness converging
        toward zero; the right panel shows predicted vs true values for the best tree.
        """
    )
    return


# ── Part 4: Static comparison figure ───────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 4: Comparison: SINDy coefficient bar chart and polynomial approx")
    return


@app.cell(hide_code=True)
def _(cl_data, gp_data, np, plt, sindy_data):
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(14, 4))
    _fig.patch.set_facecolor("#0f1117")
    for _ax in (_ax1, _ax2, _ax3):
        _ax.set_facecolor("#0f1117")
        _ax.tick_params(colors="#8a93a6", labelsize=8)
        for _sp in _ax.spines.values():
            _sp.set_color("#1e2535")
        _ax.xaxis.label.set_color("#8a93a6")
        _ax.yaxis.label.set_color("#8a93a6")
        _ax.title.set_color("#c8cdd8")

    # Panel A: SINDy xi for equation 1 (dx/dt)
    _sd = sindy_data
    from itertools import combinations_with_replacement as _cwr
    _st_s = _sd["states"]
    _n_pts_ax = _st_s.shape[0]
    _deg_ax = 3  # fixed for comparison
    _cols_ax = [np.ones(_n_pts_ax)]
    _nms_ax = ["1"]
    for _total in range(1, _deg_ax + 1):
        for _e0 in range(_total + 1):
            _e1 = _total - _e0
            _cols_ax.append(_st_s[:, 0]**_e0 * _st_s[:, 1]**_e1)
            _nm = ("" if _e0 == 0 else ("x" if _e0 == 1 else f"x^{_e0}")) + \
                  ("" if _e1 == 0 else ("y" if _e1 == 1 else f"y^{_e1}"))
            _nms_ax.append(_nm or "1")
    _Th_ax = np.column_stack(_cols_ax)
    from scipy.linalg import eigh as _eigh_ax
    _dX_ax  = (_st_s[2:] - _st_s[:-2]) / (2 * _sd["states"].shape[0] / 800 * 0.05)
    # just use stored xi from sindy_data via recomputing at deg=3
    # Actually stlsq is already done -- reuse from sindy_data by re-importing

    # Simpler: plot bar chart of xi coefficients from stored states
    # Recompute for display purposes only
    from numpy.linalg import lstsq as _lstsq_ax
    _dXa = np.gradient(_st_s, axis=0)  # crude but just for display
    _xi_ax = _lstsq_ax(_Th_ax, _dXa, rcond=None)[0]
    _thresh_ax = 0.1
    _mask_ax = np.abs(_xi_ax[:, 0]) > _thresh_ax
    _colors_bar = ["#57c98a" if _mask_ax[i] else "#2f3645" for i in range(len(_nms_ax))]
    _ax1.bar(range(len(_nms_ax)), np.abs(_xi_ax[:, 0]), color=_colors_bar, width=0.7)
    _ax1.set_xticks(range(len(_nms_ax)))
    _ax1.set_xticklabels(_nms_ax, rotation=45, ha="right", fontsize=7)
    _ax1.set_title(f"SINDy |xi| for dx/dt  ({_sd['sysname']})", fontsize=9)
    _ax1.set_ylabel("|coefficient|", fontsize=8)

    # Panel B: conservation law -- true energy vs discovered invariant (normalised)
    _cd = cl_data
    _E0 = _cd["truth"] - _cd["truth"][0]
    _D0 = _cd["discovered"]
    _E_sc = max(float(np.max(np.abs(_E0))), 1e-10)
    _D_sc = max(float(np.max(np.abs(_D0))), 1e-10)
    _t_ax = np.arange(_cd["n"])
    _ax2.plot(_t_ax, _E0 / _E_sc, color="#57c98a", linewidth=1.2, alpha=0.85, label="true E")
    _ax2.plot(_t_ax, _D0 / _D_sc, color="#f97316", linewidth=1.2, alpha=0.85, label="kernel phi")
    _ax2.axhline(0, color="#1e2535", linewidth=0.7, linestyle="--")
    _ax2.set_title(f"CL: true E vs kernel phi  ({_cd['sysname']})", fontsize=9)
    _ax2.set_xlabel("frame", fontsize=8); _ax2.set_ylabel("normalised drift", fontsize=8)
    _ax2.legend(fontsize=7, facecolor="#0f1117", edgecolor="#2f3645", labelcolor="#8a93a6")

    # Panel C: GP predicted vs true
    _gpd = gp_data
    _yc_ax = np.array(_gpd["y_clean"])
    _yp_ax = np.array(_gpd["y_pred"])
    _lo_ax = min(float(_yc_ax.min()), float(_yp_ax.min()))
    _hi_ax = max(float(_yc_ax.max()), float(_yp_ax.max()))
    _ax3.scatter(_yc_ax, _yp_ax, color="#7dd3fc", s=6, alpha=0.6)
    _ax3.plot([_lo_ax, _hi_ax], [_lo_ax, _hi_ax], color="#2f3645", linewidth=1.0, linestyle="--")
    _ax3.set_title(f"GP: pred vs true  R2={_gpd['r2']:.3f}  ({_gpd['eq_key']})", fontsize=9)
    _ax3.set_xlabel("true", fontsize=8); _ax3.set_ylabel("predicted", fontsize=8)

    plt.tight_layout(pad=1.2)
    plt.gca()
    _fig
    return


if __name__ == "__main__":
    app.run()
