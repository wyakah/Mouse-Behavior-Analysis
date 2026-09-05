'use strict';
// Cage proposals are an independent review layer, never an annotation source.
let cageImage=new Image(),cageDraft=null,cageDrag=null,cageRunning=false,cageChanged=false,cageClipEnd=null;
const cageCanvas=$('cage-map'),cageContext=cageCanvas.getContext('2d');
function cageInputs(){if(!cageDraft)return;['x1','y1','x2','y2'].forEach((key,i)=>$('cage-'+key).value=Math.round(cageDraft.crop_xyxy[i]));$('cage-floor').value=Math.round(cageDraft.floor_y);}
function drawCage(){
 if(!cageImage.complete||!cageImage.naturalWidth)return;
 cageCanvas.width=cageImage.naturalWidth;cageCanvas.height=cageImage.naturalHeight;cageContext.drawImage(cageImage,0,0);
 if(!cageDraft)return;
 const [x1,y1,x2,y2]=cageDraft.crop_xyxy;
 if(!(x2>x1&&y2>y1))return;
 cageContext.fillStyle='#10211c88';cageContext.fillRect(0,0,cageCanvas.width,cageCanvas.height);
 cageContext.drawImage(cageImage,x1,y1,x2-x1,y2-y1,x1,y1,x2-x1,y2-y1);
 cageContext.lineWidth=Math.max(3,cageCanvas.width/450);cageContext.strokeStyle='#66e2ab';cageContext.strokeRect(x1,y1,x2-x1,y2-y1);
 cageContext.strokeStyle='#edbd61';cageContext.beginPath();cageContext.moveTo(x1,cageDraft.floor_y);cageContext.lineTo(x2,cageDraft.floor_y);cageContext.stroke();
}
function currentCageRun(){return [...(session?.cage_runs||[])].reverse().find(r=>{const a=r.result.mapping,b=session?.cage_mapping;return b&&a.floor_y===b.floor_y&&a.crop_xyxy.every((v,i)=>v===b.crop_xyxy[i]);});}
function cageUrl(file){return `/api/stereotypy/sessions/${session.id}/cage/${currentCageRun().id}/${file}`;}
function cageRefresh(){
 cageDraft=session.cage_mapping?structuredClone(session.cage_mapping):null;cageChanged=false;
 cageImage=new Image();cageImage.onload=drawCage;cageImage.src=`/api/stereotypy/sessions/${session.id}/frame/0`;
 for(const key of ['x1','y1','x2','y2','floor'])$('cage-'+key).value='';
 cageInputs();$('cage-view').value='source';$('cage-status').textContent=cageDraft?'Cage saved':'Draw a rectangle or enter pixel bounds';
 $('cage-panel').open=!cageDraft;cageResults();if(currentCageRun()){$('cage-view').value='review';switchCageVideo();}
}
function cageResults(){
 const run=currentCageRun();$('cage-results').hidden=!run;if(!run)return;
 $('cage-coverage').textContent=`${run.result.frame_count.toLocaleString()} frames processed · ${run.result.proposal_coverage_percent.toFixed(1)}% of recording time has a body proposal. This is proposal coverage, not accuracy. Behavior measurements remain manual.`;
 $('cage-features').href=cageUrl('frame-features.csv')+'?download=1';$('cage-manifest').href=cageUrl('manifest.json');
 $('cage-windows').replaceChildren();
 for(const clip of run.result.review_windows){const b=document.createElement('button');b.textContent=`${clip.start_s.toFixed(1)}–${clip.end_s.toFixed(1)} s · ${clip.reason}`;b.onclick=()=>{player.pause();exact=null;$('exact-frame').hidden=true;player.currentTime=clip.start_s;cageClipEnd=clip.end_s;player.play().catch(e=>notice(e.message,true));notice(`Review ${clip.start_s.toFixed(1)}–${clip.end_s.toFixed(1)} s: ${clip.reason}. No behavior label has been added.`);player.scrollIntoView({block:'center',behavior:'smooth'});};$('cage-windows').append(b);}
}
function cagePoint(event){const rect=cageCanvas.getBoundingClientRect();return [Math.max(0,Math.min(cageCanvas.width,(event.clientX-rect.left)*cageCanvas.width/rect.width)),Math.max(0,Math.min(cageCanvas.height,(event.clientY-rect.top)*cageCanvas.height/rect.height))];}
cageCanvas.onpointerdown=e=>{if(cageRunning||busy)return;cageDrag=cagePoint(e);cageCanvas.setPointerCapture(e.pointerId);};
cageCanvas.onpointermove=e=>{if(!cageDrag)return;const [x,y]=cagePoint(e),[a,b]=cageDrag;cageDraft={crop_xyxy:[Math.min(x,a),Math.min(y,b),Math.max(x,a),Math.max(y,b)],floor_y:Math.min(y,b)+Math.abs(y-b)*.8};cageChanged=true;cageInputs();drawCage();};
cageCanvas.onpointerup=()=>{cageDrag=null;};cageCanvas.onpointercancel=()=>{cageDrag=null;};
for(const key of ['x1','y1','x2','y2','floor'])$('cage-'+key).oninput=()=>{cageDraft={crop_xyxy:['x1','y1','x2','y2'].map(k=>Number($('cage-'+k).value)),floor_y:Number($('cage-floor').value)};cageChanged=true;drawCage();};
async function saveCage(){if(!cageDraft)throw Error('Draw the cage rectangle first.');const result=await api(`/api/stereotypy/sessions/${session.id}/cage`,{revision:session.cage_revision||0,mapping:cageDraft});session.cage_mapping=result.mapping;session.cage_revision=result.revision;cageChanged=false;$('cage-status').textContent='Cage saved';if($('cage-view').value==='review'){$('cage-view').value='source';switchCageVideo();}cageResults();}
action('cage-save',saveCage);
action('cage-run',async()=>{
 if(cageRunning)return;
 if(dirty)throw Error('Save your behavior annotations before starting the cage review.');
 cageRunning=true;busy=true;
 const controls=[...document.querySelectorAll('#review input,#review select,#review button,#sessions,[data-view]')],disabled=controls.map(e=>e.disabled);controls.forEach(e=>e.disabled=true);
 try{
  if(cageChanged||!session.cage_mapping)await saveCage();
  let job=await api(`/api/stereotypy/sessions/${session.id}/cage/analyze`,{});
  while(['queued','running'].includes(job.status)){$('cage-status').textContent=job.message;await new Promise(r=>setTimeout(r,1000));job=await api(`/api/jobs/${job.id}`);}
  if(job.status!=='complete')throw Error(job.message);
  session.cage_runs=[...(session.cage_runs||[]).filter(r=>r.id!==job.run.id),job.run];cageResults();$('cage-status').textContent='Focused review ready';$('cage-panel').open=false;$('cage-view').value='review';switchCageVideo();
 }finally{controls.forEach((e,i)=>e.disabled=disabled[i]);cageRunning=false;busy=false;}
});
player.addEventListener('timeupdate',()=>{if(cageClipEnd!==null&&player.currentTime>=cageClipEnd){player.pause();cageClipEnd=null;}});
function switchCageVideo(){cageClipEnd=null;const t=player.currentTime;player.pause();exact=null;$('exact-frame').hidden=true;player.src=$('cage-view').value==='review'?cageUrl('review.mp4'):`/api/video?video=${encodeURIComponent(session.video)}`;player.addEventListener('loadedmetadata',()=>{player.currentTime=t;},{once:true});}
$('cage-view').onchange=switchCageVideo;
window.addEventListener('beforeunload',e=>{if(cageChanged){e.preventDefault();e.returnValue='';}});
