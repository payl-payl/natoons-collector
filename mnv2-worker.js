/* Воркер: считает нейросеть в фоне, чтобы страница не тормозила */
importScripts('mnv2.js');
onmessage=async e=>{
  const m=e.data;
  if(m.t==='load'){
    try{ await MNV2.load(m.json,m.bin); postMessage({t:'ready'}); }
    catch(err){ postMessage({t:'error',msg:String(err)}); }
  } else if(m.t==='embed'){
    const v=MNV2.embed(new Uint8ClampedArray(m.rgba),m.size);
    postMessage({t:'emb',id:m.id,v},[v.buffer]);
  }
};
