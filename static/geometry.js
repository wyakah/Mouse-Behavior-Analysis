// Geometry used by both editors; source pixels remain the authoritative coordinates.
window.ChamberGeometry={
 circlePoints(c,side){return Array.from({length:128},(_,i)=>{const a=i*Math.PI/64;return [c[side][0]+c.diameter_px/2*Math.cos(a),c[side][1]+c.diameter_px/2*Math.sin(a)];});},
 project(arena,x,y){return this.transform([[0,0],[1,0],[1,1],[0,1]],arena,x,y);},
 unproject(arena,x,y){return this.transform(arena,[[0,0],[1,0],[1,1],[0,1]],x,y);},
 transform(from,to,x,y){
  if(from.length!==4||to.length!==4)return [NaN,NaN];
  const m=[];
  from.forEach(([u,v],i)=>{const [a,b]=to[i];m.push([u,v,1,0,0,0,-a*u,-a*v,a],[0,0,0,u,v,1,-b*u,-b*v,b]);});
  for(let i=0;i<8;i++){let pivot=i;for(let j=i+1;j<8;j++)if(Math.abs(m[j][i])>Math.abs(m[pivot][i]))pivot=j;[m[i],m[pivot]]=[m[pivot],m[i]];if(Math.abs(m[i][i])<1e-10)return [NaN,NaN];const d=m[i][i];for(let k=i;k<9;k++)m[i][k]/=d;for(let j=0;j<8;j++)if(j!==i){const q=m[j][i];for(let k=i;k<9;k++)m[j][k]-=q*m[i][k];}}
  const h=m.map(r=>r[8]),d=h[6]*x+h[7]*y+1;return [(h[0]*x+h[1]*y+h[2])/d,(h[3]*x+h[4]*y+h[5])/d];
 }
};
