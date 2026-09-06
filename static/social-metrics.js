/* Shared eight-cell presentation for current and older Three Chamber summaries. */
function threeChamberMetrics(s) {
 const side=s.stranger_side||s.target_side, valid=n=>typeof n==='number'&&Number.isFinite(n)&&n>=0;
 const chambers=['left','center','right'].map(k=>s[k+'_chamber_seconds']);
 const total=chambers.every(valid)?chambers.reduce((a,b)=>a+b,0):null;
 const nose=k=>s.nose_scoreable_fraction>0&&valid(s[k+'_nose_seconds'])?s[k+'_nose_seconds']:null;
 const pct=n=>valid(n)&&total>0?100*n/total:null;
 const role=k=>['left','right'].includes(side)&&k!=='center'?` (${k===side?'Stranger':'Object'})`:'';
 const cells=[];
 for(const kind of ['Chamber','Zone']) {
  for(const k of ['left','center','right']) {
   const value=kind==='Chamber'?s[k+'_chamber_seconds']:k==='center'?null:nose(k);
   cells.push({label:k[0].toUpperCase()+k.slice(1)+' '+kind+role(k)+' Time (s)',value:valid(value)?value.toFixed(1)+' s':kind==='Zone'&&k==='center'?'N/A':'—'});
  }
  const value=pct(kind==='Chamber'?s[side+'_chamber_seconds']:nose(side));
  cells.push({label:kind+' Stranger Interaction %',value:value===null?'—':value.toFixed(2)+'%'});
 }
 return cells;
}
