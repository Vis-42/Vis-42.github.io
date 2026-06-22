import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Coupled Oscillators")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Coupled oscillators: synchronisation and chimera states

        Two questions, same system. The Kuramoto model asks when N phase oscillators with
        spread natural frequencies fall into collective step: there is a sharp transition at
        a critical coupling $K_c = 2\sigma\sqrt{2/\pi}$ where the order parameter
        $r = |\langle e^{i\theta}\rangle|$ jumps from 0 to 1. Three live panels share one
        clock: the phase circle with the mean-field vector, $r(t)$ with a tracking dot,
        and the $r$ vs $K$ bifurcation curve with the current $K$ marked.

        The second question is stranger: can part of an *identical* ring synchronise while
        the rest stays incoherent? Yes, and it is called a chimera state.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
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

  /* ─────────────── SPACE-TIME (chimera heatmap building frame-by-frame) ─── */
  else if(p.kind==="space_time"){
    const T=p.T,N2=p.N;
    if(!p._off){p._off=document.createElement("canvas");p._off.width=N2;p._off.height=T;p._oct=p._off.getContext("2d");const img=p._oct.createImageData(N2,T);const d=img.data,rgb=p.rgb;for(let i=0;i<T*N2;i++){d[4*i]=rgb[3*i];d[4*i+1]=rgb[3*i+1];d[4*i+2]=rgb[3*i+2];d[4*i+3]=255;}p._oct.putImageData(img,0,0);}
    const fShow=Math.min(f+1,T);
    const pL=36*DPR,pR=6*DPR,pT=20*DPR,pB=24*DPR;
    const mx=px+pL,my=py+pT,mw=pw-pL-pR,mh=ph-pT-pB;
    ctx.save();ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.clip();
    ctx.imageSmoothingEnabled=false;
    ctx.drawImage(p._off,0,0,N2,fShow,mx,my,mw,mh*fShow/T);
    ctx.strokeStyle="#f97316";ctx.lineWidth=1.5*DPR;ctx.beginPath();ctx.moveTo(mx,my+mh*fShow/T);ctx.lineTo(mx+mw,my+mh*fShow/T);ctx.stroke();
    ctx.restore();
    ctx.fillStyle="#9aa3b0";ctx.font=(10*DPR)+"px monospace";ctx.textAlign="center";
    if(p.title)ctx.fillText(p.title,px+pw/2,py+13*DPR);
    ctx.fillText("oscillator i",mx+mw/2,py+ph-2*DPR);
    ctx.save();ctx.translate(px+11*DPR,my+mh/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillText("time",0,0);ctx.restore();
    ctx.fillStyle="#8a93a6";ctx.font=(8*DPR)+"px monospace";ctx.textAlign="center";
    [0,Math.round(N2/2),N2].forEach(v=>{ctx.fillText(v,mx+mw*v/N2,py+ph-4*DPR);});
    ctx.textAlign="right";[0,Math.round(T/2),T].forEach(v=>{ctx.fillText(v,mx-2*DPR,my+mh*(1-v/T)+3*DPR);});
  }

  /* ─────────────── LOCAL_R_FRAME (local order param bar chart at current frame) ─── */
  else if(p.kind==="local_r_frame"){
    const T=p.T,N2=p.N,t_=Math.min(f,T-1);
    const lr=p.lr[t_],ymax=1.05;
    const xd={min:0,max:N2,log:false},yd={min:0,max:ymax,log:false};
    const{mx,my,mw,mh,toX,toY}=axes2D(p,lx,ly,lw,lh,xd,yd,{zoom:1,px:0,py:0});
    ctx.save();ctx.beginPath();ctx.rect(mx,my,mw,mh);ctx.clip();
    const mean_r=lr.reduce((a,b)=>a+b,0)/N2;
    ctx.strokeStyle="#9c3b2c";ctx.lineWidth=1*DPR;ctx.setLineDash([4,4]);
    const ym=toY(mean_r);ctx.beginPath();ctx.moveTo(mx,ym);ctx.lineTo(mx+mw,ym);ctx.stroke();ctx.setLineDash([]);
    const bw=Math.max(1.5,mw/N2*0.8);
    for(let i=0;i<N2;i++){const x=toX(i+0.5),y=toY(lr[i]),yb=toY(0);ctx.fillStyle="#7dd3fc";ctx.fillRect(x-bw/2,y,bw,yb-y);}
    ctx.restore();
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

    import json as _json
    def canvas_anim(spec, height="480px"):
        return mo.iframe(_JS.replace("__SPEC__", _json.dumps(spec)), height=height)

    return canvas_anim,


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 1: Kuramoto synchronisation")
    return


@app.cell(hide_code=True)
def _(mo):
    K_sl   = mo.ui.slider(0.0, 6.0,  value=2.5,  step=0.25, label="coupling K",      show_value=True)
    sig_sl = mo.ui.slider(0.2, 2.0,  value=0.8,  step=0.1,  label="freq spread σ",   show_value=True)
    N_sl   = mo.ui.slider(20,  120,   value=50,   step=10,   label="oscillators N",   show_value=True)
    topo_sl = mo.ui.dropdown(["all-to-all", "Erdős-Rényi", "scale-free"],
                             value="all-to-all", label="network")
    mo.hstack([topo_sl, K_sl, sig_sl, N_sl], justify="start", gap=2)
    return K_sl, N_sl, sig_sl, topo_sl


@app.cell(hide_code=True)
def _(K_sl, N_sl, np, sig_sl, topo_sl):
    _K   = float(K_sl.value)
    _sig = float(sig_sl.value)
    _N   = int(N_sl.value)
    _dt  = 0.05
    _WARM   = 200
    _FRAMES = 200
    _rng = np.random.default_rng(0)
    _omega = _rng.normal(0, _sig, _N)
    _theta = _rng.uniform(-np.pi, np.pi, _N)
    _Kc_theory = 2 * _sig * np.sqrt(2 / np.pi)

    # ── build the coupling adjacency for the chosen topology ──
    def _adjacency(N, topo, rng):
        if topo == "Erdős-Rényi":
            p = min(1.0, 6.0 / N)   # mean degree ~6
            A = (rng.random((N, N)) < p).astype(float)
            A = np.triu(A, 1); A = A + A.T
        elif topo == "scale-free":
            # Barabasi-Albert preferential attachment, m=2
            m = 2; A = np.zeros((N, N))
            deg = np.ones(N) * 1e-9
            for _i in range(m):
                A[_i, (_i + 1) % m] = A[(_i + 1) % m, _i] = 1; deg[_i] += 1
            for _i in range(m, N):
                _pr = deg[:_i] / deg[:_i].sum()
                _tg = rng.choice(_i, size=min(m, _i), replace=False, p=_pr)
                for _j in _tg:
                    A[_i, _j] = A[_j, _i] = 1; deg[_i] += 1; deg[_j] += 1
            np.fill_diagonal(A, 0)
        else:
            A = np.ones((N, N)); np.fill_diagonal(A, 0)
        return A

    _A = _adjacency(_N, topo_sl.value, _rng)
    _kbar = max(1.0, float(_A.sum(1).mean()))   # mean degree for normalisation

    def _kura_step(th, omega, K, N, dt, A=_A, kbar=_kbar):
        diff = th[None, :] - th[:, None]
        dth = omega + (K / kbar) * (A * np.sin(diff)).sum(1)
        return th + dth * dt

    for _ in range(_WARM):
        _theta = _kura_step(_theta, _omega, _K, _N, _dt)

    _ths_f, _r_s = [], []
    for _ in range(_FRAMES):
        _theta = _kura_step(_theta, _omega, _K, _N, _dt)
        _ths_f.append((_theta % (2 * np.pi)).tolist())
        _r_s.append(float(np.abs(np.mean(np.exp(1j * _theta)))))

    _r_mean = float(np.mean(_r_s[_FRAMES // 2:]))
    _regime = "SYNCHRONISED" if _r_mean > 0.75 else "PARTIAL" if _r_mean > 0.3 else "INCOHERENT"

    # K sweep , no button gate
    _K_arr = np.linspace(0.1, 6.0, 16)
    _r_arr = []
    for _Kk in _K_arr:
        _th2 = _rng.uniform(-np.pi, np.pi, _N)
        for _ in range(150):
            _th2 = _kura_step(_th2, _omega, _Kk, _N, _dt)
        _rs2 = []
        for _ in range(80):
            _th2 = _kura_step(_th2, _omega, _Kk, _N, _dt)
            _rs2.append(float(np.abs(np.mean(np.exp(1j * _th2)))))
        _r_arr.append(float(np.mean(_rs2)))

    kura_data = {
        "ths": _ths_f, "r": _r_s, "frames": _FRAMES,
        "K_sweep": _K_arr.tolist(), "r_sweep": _r_arr,
        "Kc": float(_Kc_theory),
        "K_now": [float(_K)] * _FRAMES, "r_now": _r_s,
        "r_mean": _r_mean, "regime": _regime,
        "topo": topo_sl.value, "kbar": float(_kbar),
    }
    return (kura_data,)


@app.cell(hide_code=True)
def _(canvas_anim, kura_data, mo):
    _d = kura_data
    _spec = {
        "frames": _d["frames"], "dt": 55,
        "panels": [
            {"kind": "phases", "title": "Phases on unit circle", "theta": _d["ths"], "r_series": _d["r"]},
            {"kind": "timeseries", "title": "Order parameter r(t)", "xlabel": "frame", "ylabel": "r = |⟨eⁱᶿ⟩|",
             "ymin": 0.0, "ymax": 1.05, "grow": True, "dot": 0,
             "ys": [_d["r"]], "colors": ["#7dd3fc"], "t": list(range(_d["frames"]))},
            {"kind": "scatter_vline", "title": "Bifurcation: r vs K  (amber = Kc theory)", "xlabel": "coupling K", "ylabel": "r (steady state)",
             "ymin": 0.0, "ymax": 1.05,
             "x": _d["K_sweep"], "y": _d["r_sweep"], "Kc": _d["Kc"],
             "dot_x": _d["K_now"], "dot_y": _d["r_now"]},
        ],
        "layout": [
            [0.0,  0.0,  0.40, 1.0],
            [0.40, 0.0,  0.60, 0.50],
            [0.40, 0.50, 0.60, 0.50],
        ],
    }
    mo.vstack([
        mo.md(
            f"**{_d['regime']}** &nbsp;|&nbsp; "
            f"network: {_d['topo']} (mean degree {_d['kbar']:.1f}) &nbsp;|&nbsp; "
            f"r (mean, last half) = {_d['r_mean']:.2f} &nbsp;|&nbsp; "
            f"K_c theory = {_d['Kc']:.2f}"
        ),
        canvas_anim(_spec, height="500px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **The synchronisation transition.** The order parameter $r = |\langle e^{i\theta_i}\rangle|$
        measures how aligned the oscillator phases are: $r = 0$ is fully incoherent,
        $r = 1$ is fully synchronised. The analytic result $K_c = 2\sigma\sqrt{2/\pi}$
        marks the onset, visible as the inflection in the bifurcation curve (bottom right).
        The orange vector on the phase circle is the mean-field phasor $r e^{i\psi}$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 2: Chimera states")
    return


@app.cell(hide_code=True)
def _(mo):
    Kch_sl = mo.ui.slider(0.5, 4.0,  value=1.5,  step=0.25, label="coupling K",     show_value=True)
    R_sl   = mo.ui.slider(2,   20,    value=8,    step=1,    label="range R",        show_value=True)
    alp_sl = mo.ui.slider(0.0, 1.5,   value=1.46, step=0.05, label="phase lag α",   show_value=True)
    Nch_sl = mo.ui.slider(30,  80,    value=50,   step=10,   label="ring size N",   show_value=True)
    mo.hstack([Kch_sl, R_sl, alp_sl, Nch_sl], justify="start", gap=2)
    return Kch_sl, Nch_sl, R_sl, alp_sl


@app.cell(hide_code=True)
def _(np):
    def ks_step(th, N, K, R, alp, dt):
        dth = np.zeros(N)
        for k in range(-R, R + 1):
            if k == 0:
                continue
            dth += np.sin(np.roll(th, -k) - th - alp)
        return th + dt * (K / (2 * R)) * dth
    return (ks_step,)


@app.cell(hide_code=True)
def _(Kch_sl, Nch_sl, R_sl, alp_sl, ks_step, np):
    import matplotlib.cm as _cm
    _N   = int(Nch_sl.value)
    _K   = float(Kch_sl.value)
    _R   = int(R_sl.value)
    _alp = float(alp_sl.value)
    _dt  = 0.05
    _T_WARM = 300
    _T_REC  = 200
    _theta3 = np.random.default_rng(1).uniform(-np.pi, np.pi, _N)

    for _ in range(_T_WARM):
        _theta3 = ks_step(_theta3, _N, _K, _R, _alp, _dt)

    _history = []
    for _ in range(_T_REC):
        _theta3 = ks_step(_theta3, _N, _K, _R, _alp, _dt)
        _history.append(_theta3.copy())
    _history = np.array(_history)

    # space-time RGB (HSV colormap: hue = theta mod 2pi)
    _ht_norm = (_history % (2 * np.pi)) / (2 * np.pi)
    _rgba    = _cm.hsv(_ht_norm)
    _st_rgb  = (_rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8).flatten().tolist()

    # local order parameter at every time step (T x N), vectorised
    _nbr_idx   = np.array([(np.arange(i - _R, i + _R + 1) % _N) for i in range(_N)])
    _th_nbr    = _history[:, _nbr_idx]                              # (T, N, 2R+1)
    _local_r_f = np.abs(np.mean(np.exp(1j * _th_nbr), axis=2))     # (T, N)

    _theta_final = _history[-1]
    _global_r    = float(np.abs(np.mean(np.exp(1j * _theta_final))))
    _sigma_r     = float(np.std(_local_r_f[-1]))
    _is_chimera  = _sigma_r > 0.08 and _global_r < 0.85

    chimera_data  = {
        "T": _T_REC, "N": _N, "R": _R,
        "st_rgb": _st_rgb,
        "local_r_f": _local_r_f.tolist(),
    }
    chimera_label = (
        f"CHIMERA (σ_r = {_sigma_r:.2f}, |Z| = {_global_r:.2f})"
        if _is_chimera else
        f"not chimera (σ_r = {_sigma_r:.2f}, |Z| = {_global_r:.2f})"
    )
    return chimera_data, chimera_label


@app.cell(hide_code=True)
def _(canvas_anim, chimera_data, chimera_label, mo):
    _d = chimera_data
    _spec_ch = {
        "frames": _d["T"], "dt": 60,
        "panels": [
            {"kind": "space_time", "T": _d["T"], "N": _d["N"],
             "rgb": _d["st_rgb"], "title": "theta mod 2pi (space-time)"},
            {"kind": "local_r_frame", "T": _d["T"], "N": _d["N"],
             "lr": _d["local_r_f"],
             "title": "|r_i| local order", "xlabel": "oscillator i"},
        ],
        "layout": [[0.0, 0.0, 0.60, 1.0], [0.60, 0.0, 0.40, 1.0]],
    }
    mo.vstack([
        mo.md(f"**{chimera_label}**"),
        canvas_anim(_spec_ch, height="420px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Chimera states.** The Kuramoto-Sakaguchi model adds a phase lag $\alpha$ and
        restricts coupling to neighbours within range $R$. Near $\alpha \approx \pi/2$ this
        can produce a chimera: part of the ring phase-locks (high local $|r_i|$, flat in
        the space-time heatmap) while the rest stays incoherent (low $|r_i|$, turbulent).
        Both parts contain identical oscillators with $\omega_i = 0$. The frustration is
        geometric, not from disorder. Try $\alpha \approx 1.46$, $K \approx 1.5$, $R = 8$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Part 3: (K, R) phase diagram")
    return


@app.cell(hide_code=True)
def _(mo):
    kr_btn = mo.ui.run_button(label="Scan 6x6 (K, R) grid")
    mo.md(
        f"""
        Compute the chimera measure across a 6x6 grid of coupling K and neighbour range R
        at fixed alpha. Bright cells support chimera states.

        {kr_btn}
        """
    )
    return (kr_btn,)


@app.cell(hide_code=True)
def _(alp_sl, ks_step, kr_btn, mo, np):
    mo.stop(not kr_btn.value, mo.md("Press **Scan** to compute the (K, R) diagram."))
    _alp_kr  = float(alp_sl.value)
    _K_vals  = np.linspace(0.5, 3.5, 6)
    _R_vals  = [2, 4, 6, 8, 12, 16]
    _N_kr    = 40
    _T_KR    = 200
    _dt_kr   = 0.05
    _cmap_kr = np.zeros((6, 6))
    for _ki, _K in enumerate(_K_vals):
        for _ri, _R in enumerate(_R_vals):
            _th = np.random.default_rng(1).uniform(-np.pi, np.pi, _N_kr)
            for _ in range(_T_KR):
                _th = ks_step(_th, _N_kr, float(_K), int(_R), _alp_kr, _dt_kr)
            _nbr = np.array([(np.arange(i - int(_R), i + int(_R) + 1) % _N_kr) for i in range(_N_kr)])
            _lr  = np.abs(np.mean(np.exp(1j * _th[_nbr]), axis=1))
            _gr  = np.abs(np.mean(np.exp(1j * _th)))
            _cmap_kr[_ki, _ri] = float(np.std(_lr)) * (1.0 - float(_gr))
    kr_data = {"K_vals": _K_vals.tolist(), "R_vals": _R_vals, "cmap": _cmap_kr.tolist()}
    return (kr_data,)


@app.cell(hide_code=True)
def _(kr_data, mo, plt):
    _K  = kr_data["K_vals"]
    _R  = kr_data["R_vals"]
    _cm = kr_data["cmap"]
    fig_kr, ax_kr = plt.subplots(figsize=(5.5, 3.8))
    _im = ax_kr.pcolormesh(_R, _K, _cm, cmap="magma", shading="nearest", vmin=0)
    fig_kr.colorbar(_im, ax=ax_kr, label="chimera measure sigma(|r_i|) x (1-|Z|)")
    ax_kr.set_xlabel("neighbour range R")
    ax_kr.set_ylabel("coupling K")
    ax_kr.set_title("(K, R) phase diagram (alpha fixed, N = 40)")
    ax_kr.set_xticks(_R)
    ax_kr.set_yticks([round(v, 2) for v in _K])
    fig_kr.tight_layout()
    mo.vstack([
        mo.md("Bright = chimera-supporting. Chimera measure = sigma(|r_i|) x (1-|Z|); zero = fully sync or fully incoherent."),
        fig_kr,
    ])
    return


if __name__ == "__main__":
    app.run()
