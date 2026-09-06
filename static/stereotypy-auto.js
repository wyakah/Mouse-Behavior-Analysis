const $=s=>document.querySelector(s);
let queue={entries:[]},busy=false,current=null,timer=null,viewingSetup=false;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,body){const r=await fetch(url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');return data;}
const liveView=new LiveAnalysisView($('#live-analysis'),{apiBase:'/api/stereotypy/batches',stereotypy:true});
const setup=new RecordingSetup($('#recording-setup'),{entries:()=>queue.entries,disabled:()=>busy,change:()=>api('/api/stereotypy/draft',{entries:queue.entries}),seed:async video=>({video,id:video.split('/').pop().replace(/^[a-f0-9]{8}_/,'').split('_')[0]}),continue:start});
$('#recording-setup [data-continue]').textContent='Analyze videos';
function notice(message,error=false){$('#notice').textContent=message;$('#notice').classList.toggle('error',error);}
function file(b,name){return `/stereotypy-runs/${b.id}/${encodeURIComponent(name)}`;}
function show(b){
 liveView.updateBatch(b);$('#results-step').disabled=false;$('#results-step').classList.add('active');$('#videos-step').classList.remove('active');
 current=b;busy=['queued','running'].includes(b.status);$('#setup').hidden=!viewingSetup;$('#run').hidden=viewingSetup;$('#new-batch').hidden=busy;
 $('#run-title').textContent=busy?'Analyzing videos':b.status==='failed'?'Analysis could not finish':'Results';notice(b.status==='failed'?b.message:'',b.status==='failed');
 $('#progress').max=b.entries.length;$('#progress').value=b.entries.filter(e=>['complete','failed'].includes(e.status)).length;
 if(busy)$('#progress').removeAttribute('value');
 $('#downloads').innerHTML=(b.downloads||[]).map(n=>`<a href="${file(b,n)}">${n==='results.xlsx'?'Download Excel':n==='summary.csv'?'Download scores CSV':'Download behavior intervals'}</a>`).join('');
 const host=$('#results');
 b.entries.forEach((e,i)=>{
  let card=host.querySelector(`[data-entry="${i}"]`);if(!card){card=document.createElement('article');card.className='run-card';card.dataset.entry=i;host.append(card);}
  if(!card.querySelector('h3'))card.innerHTML=`<h3></h3><small data-meta></small><p data-status></p><div class="scores"></div><p data-coverage></p><details hidden><summary>Play annotated video</summary><video controls preload="none"></video></details>`;
  card.querySelector('h3').textContent='Mouse '+e.id;
  card.querySelector('[data-meta]').textContent=[e.sex&&e.sex!=='unknown'?e.sex:'',e.genotype].filter(Boolean).join(' · ');
  card.querySelector('[data-status]').textContent=e.status==='failed'?e.message:'';
  const totals=e.cumulative,rows=e.summary||(totals?['grooming','digging','gnawing_nonfood','rearing','jumping','circling'].map(behavior=>({behavior,candidate_seconds:totals.seconds[behavior]})):[]);
  card.querySelector('.scores').innerHTML=rows.map(s=>`<div class="score">${esc(s.behavior==='gnawing_nonfood'?'Nonfood gnawing':s.behavior)}<strong>${s.candidate_seconds==null?'Unavailable':s.candidate_seconds.toFixed(1)+' s'}</strong></div>`).join('');
  card.querySelector('[data-coverage]').textContent=totals?`Through ${totals.through_s.toFixed(1)} s · Other ${totals.other_seconds.toFixed(1)} s · Unscored ${totals.unknown_seconds.toFixed(1)} s`:e.status==='complete'?'Legacy overlapping scores. Run again for exclusive scoring.':'';
  if(e.status==='complete'){
   const d=card.querySelector('details'),v=card.querySelector('video');d.hidden=false;
   if(!d.dataset.ready){d.dataset.ready='true';v.poster=file(b,e.poster_file);v.setAttribute('aria-label','Annotated video for mouse '+e.id);d.addEventListener('toggle',()=>{if(d.open){if(!v.getAttribute('src'))v.src=file(b,e.video_file);}else v.pause();});}
  }

 });
 if(b.export_notice)notice(b.message+' · '+b.export_notice,true);
 clearTimeout(timer);if(busy)timer=setTimeout(()=>load(b.id),2500);
}
async function load(id){try{show(await api('/api/stereotypy/batches/'+id));localStorage.setItem('stereotypyAutomaticBatch',id);}catch(e){notice(e.message,true);timer=setTimeout(()=>load(id),5000);}}
async function start(){viewingSetup=false;liveView.setVisible(true);busy=true;setup.render();notice('Starting automatic analysis…');try{const b=await api('/api/stereotypy/batches',{entries:queue.entries});$('#results').replaceChildren();await load(b.id);}catch(e){busy=false;setup.render();notice(e.message,true);}}
$('#videos-step').onclick=()=>{viewingSetup=true;$('#run').hidden=true;$('#setup').hidden=false;liveView.setVisible(false);setup.render();};
$('#results-step').onclick=()=>{viewingSetup=false;$('#setup').hidden=true;$('#run').hidden=false;liveView.setVisible(true);};
$('#new-batch').onclick=()=>{viewingSetup=true;liveView.setVisible(false);clearTimeout(timer);localStorage.removeItem('stereotypyAutomaticBatch');current=null;busy=false;$('#run').hidden=true;$('#setup').hidden=false;$('#results').replaceChildren();setup.render();notice('');};
async function init(){try{const [draft,videos,history]=await Promise.all([api('/api/stereotypy/draft'),api('/api/videos'),api('/api/stereotypy/batches')]);queue={entries:draft.entries||[]};setup.render();setup.setAvailable(videos);for(const b of history){const a=document.createElement('a');a.href='#';a.textContent=`${b.entries.map(e=>e.id).join(', ')} · ${b.status.replaceAll('_',' ')}`;a.onclick=ev=>{ev.preventDefault();viewingSetup=false;liveView.setVisible(true);$('#results').replaceChildren();load(b.id);};$('#history-list').append(a);}const active=history.find(b=>['queued','running'].includes(b.status));const saved=localStorage.getItem('stereotypyAutomaticBatch');if(active||history.some(b=>b.id===saved))await load(active?.id||saved);}catch(e){notice(e.message,true);}}
init();
