/* ===== MobileNetV2 на чистом JS (без библиотек): картинка -> вектор из 1280 чисел ===== */
const MNV2 = (function(){
  let G=null, W=null;
  function f16(u){            // float16 -> float32
    const s=(u&0x8000)?-1:1, e=(u>>10)&0x1f, m=u&0x3ff;
    if(e===0) return s*Math.pow(2,-14)*(m/1024);
    if(e===31) return m?NaN:s*Infinity;
    return s*Math.pow(2,e-15)*(1+m/1024);
  }
  async function load(jsonUrl,binUrl){
    G=await (await fetch(jsonUrl)).json();
    const buf=await (await fetch(binUrl)).arrayBuffer();
    const u=new Uint16Array(buf), n=u.length;
    W=new Float32Array(n);
    const lut=new Float32Array(65536); for(let i=0;i<65536;i++) lut[i]=f16(i);
    for(let i=0;i<n;i++) W[i]=lut[u[i]];
  }
  function conv(x,H,Wd,Cin,o){
    const k=o.k,s=o.s,p=o.p,g=o.g,Co=o.co,cg=o.cg;
    const Ho=Math.floor((H+2*p-k)/s)+1, Wo=Math.floor((Wd+2*p-k)/s)+1;
    const out=new Float32Array(Co*Ho*Wo);
    const w=W, wo=o.w, bo=o.b;
    if(g===1 && k===1){                       // поточечная 1×1
      const HW=Ho*Wo;
      if(s===1){
        for(let co=0;co<Co;co++){
          const ob=co*HW, b=w[bo+co];
          for(let i=0;i<HW;i++) out[ob+i]=b;
          for(let ci=0;ci<Cin;ci++){
            const wt=w[wo+co*Cin+ci]; if(wt===0) continue;
            const ib=ci*HW;
            for(let i=0;i<HW;i++) out[ob+i]+=wt*x[ib+i];
          }
        }
      } else { throw new Error('stride1x1'); }
    } else if(g===1){                          // обычная k×k (первый слой)
      for(let co=0;co<Co;co++){
        for(let oy=0;oy<Ho;oy++) for(let ox=0;ox<Wo;ox++){
          let a=w[bo+co];
          for(let ci=0;ci<Cin;ci++) for(let ky=0;ky<k;ky++){
            const iy=oy*s-p+ky; if(iy<0||iy>=H) continue;
            for(let kx=0;kx<k;kx++){
              const ix=ox*s-p+kx; if(ix<0||ix>=Wd) continue;
              a+=w[wo+((co*Cin+ci)*k+ky)*k+kx]*x[(ci*H+iy)*Wd+ix];
            }
          }
          out[(co*Ho+oy)*Wo+ox]=a;
        }
      }
    } else {                                    // depthwise k×k
      for(let c=0;c<Co;c++){
        const wb=wo+c*k*k, b=w[bo+c], ib=c*H*Wd, ob=c*Ho*Wo;
        for(let oy=0;oy<Ho;oy++) for(let ox=0;ox<Wo;ox++){
          let a=b;
          for(let ky=0;ky<k;ky++){
            const iy=oy*s-p+ky; if(iy<0||iy>=H) continue;
            for(let kx=0;kx<k;kx++){
              const ix=ox*s-p+kx; if(ix<0||ix>=Wd) continue;
              a+=w[wb+ky*k+kx]*x[ib+iy*Wd+ix];
            }
          }
          out[ob+oy*Wo+ox]=a;
        }
      }
    }
    return {d:out,H:Ho,W:Wo,C:Co};
  }
  /* rgba — Uint8ClampedArray size×size*4 → Float32Array(1280) */
  function embed(rgba,size){
    const T=new Array(G.nt), n=size*size, x=new Float32Array(3*n);
    const mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225];
    for(let i=0;i<n;i++) for(let c=0;c<3;c++) x[c*n+i]=(rgba[i*4+c]/255-mean[c])/std[c];
    T[0]={d:x,H:size,W:size,C:3};
    for(const o of G.ops){
      if(o.t==='conv'){ const a=T[o.i]; T[o.o]=conv(a.d,a.H,a.W,a.C,o); }
      else if(o.t==='clip'){ const a=T[o.i], d=new Float32Array(a.d.length); for(let i=0;i<d.length;i++){const v=a.d[i]; d[i]=v<0?0:(v>6?6:v);} T[o.o]={d,H:a.H,W:a.W,C:a.C}; }
      else if(o.t==='add'){ const a=T[o.i], b=T[o.j], d=new Float32Array(a.d.length); for(let i=0;i<d.length;i++) d[i]=a.d[i]+b.d[i]; T[o.o]={d,H:a.H,W:a.W,C:a.C}; }
      else if(o.t==='gap'){ const a=T[o.i], hw=a.H*a.W, d=new Float32Array(a.C); for(let c=0;c<a.C;c++){let s=0; for(let i=0;i<hw;i++) s+=a.d[c*hw+i]; d[c]=s/hw;} T[o.o]={d}; }
    }
    return T[G.out].d;
  }
  return {load,embed};
})();
