const $=s=>document.querySelector(s),el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n;};
let draft={entries:[],diameter_px:110,pcutoff:.6},profile,available=[],current=0,img=new Image(),editMode='circles',drag=null,job=null,saveTimer,saveChain=Promise.resolve(),busy=false,frameToken=0,frameReady=false,adding=false;
const entry=()=>draft.entries[current];
const validSampleIds=()=>draft.entries.every(e=>String(e.id||'').trim())&&new Set(draft.entries.map(e=>String(e.id||'').trim())).size===draft.entries.length;
const liveView=new LiveAnalysisView($('#live-analysis'));
let activeJob=null,watchTimer=null,watchGeneration=0,statusRequest=null,resultBatchId=null,resultCards=new Map();
liveView.onViewResult=async(id,run)=>{await watch(id);document.getElementById('result-'+run)?.scrollIntoView({block:'start',behavior:'smooth'});};
function notice(text,error=false){$('#batch-notice').textContent=text;$('#batch-notice').className=error?'error':'';}
async function api(url,body){const r=await fetch(url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||`Request failed (${r.status})`);return data;}
const guard=fn=>async(...args)=>{try{await fn(...args)}catch(e){notice(e.message,true)}};
function step(name,focus=false){
 if(!['tests','videos','regions','results'].includes(name))name='tests';
 if(name==='regions'&&!entry()){notice('Add a recording first.',true);name='videos';}
 for(const n of ['tests','videos','regions','results'])$('#batch-'+n).hidden=n!==name;
 document.querySelectorAll('[data-step]').forEach(b=>{b.classList.toggle('active',b.dataset.step===name);if(b.dataset.step===name)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
 liveView.setVisible(name==='results');if(name!=='results')document.querySelectorAll('.result video').forEach(v=>v.pause());$('#assay-context').textContent=name==='tests'?'Behavioral tests':'Three Chamber';if(name==='regions'&&entry())loadEntry();history.replaceState(null,'','#'+name);window.scrollTo(0,0);if(focus)$('#'+name+'-heading').focus({preventScroll:true});
}
document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>step(b.dataset.step,true));
window.addEventListener('hashchange',()=>{if(location.hash!=='#workspace-main')step(location.hash.slice(1));});
function save(){clearTimeout(saveTimer);const snapshot=JSON.parse(JSON.stringify(draft));saveChain=saveChain.catch(()=>{}).then(()=>api('/api/batch/draft',snapshot));return saveChain;}
function changed(all=false){if(all)draft.entries.forEach(e=>e.reviewed=false);else if(entry())entry().reviewed=false;renderLists();clearTimeout(saveTimer);saveTimer=setTimeout(()=>save().catch(e=>notice(e.message,true)),300);}
function renderLists(){
 const select=$('#batch-recording');select.replaceChildren();draft.entries.forEach((e,i)=>select.add(new Option(`${i+1}. ${e.id} · ${e.reviewed?'reviewed':'needs review'}`,i)));select.value=current;
 for(const name of ['batch-queue','review-queue']){
  const container=$('#'+name);container.replaceChildren();
  if(!draft.entries.length)container.append(el('p','No videos selected. Add videos above or choose from this workspace.'));
  draft.entries.forEach((e,i)=>{
   const row=el('div',null,'batch-row'),span=el('span',`${e.reviewed?'✓':'○'} ${e.id}`,e.reviewed?'completed':'');
   if(name==='batch-queue'){span.replaceChildren();const fields=el('div',null,'sample-fields');for(const [key,label] of [['id','Sample ID'],['sex','Sex'],['genotype','Genotype']]){const wrapper=el('label',label),input=el(key==='sex'?'select':'input');input.setAttribute('aria-label',`${label} for recording ${i+1}`);if(key==='sex'){for(const [v,t] of [['unknown','Not recorded'],['female','Female'],['male','Male']])input.add(new Option(t,v));}else{input.maxLength=key==='genotype'?120:120;if(key==='genotype'){input.setAttribute('list','genotype-options');input.placeholder='Not recorded';}}input.value=e[key]||(key==='sex'?'unknown':'');input.oninput=()=>{e[key]=input.value;metadataChanged();};wrapper.append(input);fields.append(wrapper);}span.append(fields,el('small',e.video.split('/').pop()));}
   const b=el('button',e.reviewed?'Edit':'Review');b.setAttribute('aria-label',`${e.reviewed?'Edit':'Review'} regions for ${e.id}`);
   b.onclick=()=>{current=i;step('regions',true);};row.append(span,b);
   if(name==='batch-queue'){
    const remove=el('button','Remove');remove.setAttribute('aria-label',`Remove ${e.id}`);remove.disabled=busy||adding;
    remove.onclick=guard(async()=>{draft.entries.splice(i,1);current=Math.min(current,Math.max(0,draft.entries.length-1));renderLists();renderAvailable();if(entry())loadEntry();await save();});row.append(remove);
   }
   container.append(row);
  });
 }
 renderMetadataSummary();
 const count=draft.entries.filter(e=>e.reviewed).length,total=draft.entries.length;
 $('#selected-count').textContent=`${total} recording${total===1?'':'s'}`;
 $('#review-count').textContent=`${count} of ${total} recordings reviewed`;
 $('#start-batch').disabled=busy||adding||!total||count!==total||!validSampleIds();
 $('#start-batch').textContent=busy?'Analysis in progress…':`Analyze ${total===1?'recording':total+' recordings'} →`;
 $('#analysis-help').textContent=busy?'Open Results to follow progress.':!validSampleIds()?'Give every mouse a unique, nonempty sample ID.':count===total&&total?'Ready. Tracking and scoring will run automatically.':'Confirm the regions in every recording to continue.';
 $('#review-regions').disabled=adding||!total;
 $('#previous-recording').disabled=current===0;
 $('#confirm-recording').textContent=current<total-1?'Confirm & next →':'Confirm regions ✓';
 $('#entry-status').textContent=entry()?.reviewed?'Regions confirmed for this recording.':'Check both circles and the chamber boundaries, then confirm.';
}
const modeHelp={descriptive:'Group size, mean, standard deviation, and 95% confidence intervals.',genotype:'Two-sided Welch tests between genotypes, pooling sexes. Check that pooling fits your experimental design.',within_sex:'Separate genotype comparisons among female mice and among male mice, using Welch tests.',factorial:'Two-way ANOVA tests genotype, sex, and their interaction. Requires both sexes in each genotype.',all:'Pooled and within-sex Welch comparisons, plus genotype × sex ANOVA. Holm correction covers all estimable tests.'};
function metadataChanged(){renderMetadataSummary();clearTimeout(saveTimer);saveTimer=setTimeout(()=>save().catch(e=>notice(e.message,true)),400);}
function renderMetadataSummary(){
 const es=draft.entries,genotypes=[...new Set(es.map(e=>(e.genotype||'').trim()).filter(Boolean))];
 $('#cohort-summary').textContent=`${es.length} independent mice · ${genotypes.length} genotype${genotypes.length===1?'':'s'} · ${es.filter(e=>e.sex==='female').length} female · ${es.filter(e=>e.sex==='male').length} male`;
 $('#genotype-options').replaceChildren(...genotypes.map(g=>new Option(g,g)));
 $('#resume-draft').hidden=!es.length;$('#resume-description').textContent=`Three Chamber · ${es.length} recording${es.length===1?'':'s'} in your saved setup`;
 $('#start-batch').disabled=busy||adding||!es.length||!validSampleIds()||es.some(e=>!e.reviewed);if(!validSampleIds())$('#cohort-summary').textContent+=' · Enter a unique ID for each mouse';
 const plan=draft.statistics;if(!plan)return;
 $('#statistics-method').textContent=modeHelp[plan.mode];
 const missingG=es.filter(e=>!(e.genotype||'').trim()).length,missingS=es.filter(e=>!e.sex||e.sex==='unknown').length,missingTarget=es.filter(e=>!['left','right'].includes(e.config.target_side)).length;
 const notes=[];if(plan.mode!=='descriptive'&&genotypes.length<2)notes.push('At least two genotypes needed');if(missingG)notes.push(`${missingG} without genotype`);if(missingS)notes.push(`${missingS} without sex`);if(missingTarget&&plan.metrics.some(m=>['target_nose_seconds','other_nose_seconds','preference_index'].includes(m)))notes.push(`${missingTarget} without a social / novel cup side`);
 $('#statistics-readiness').textContent=notes.length?`${notes.join(' · ')}. Relevant comparisons will document these exclusions. Raw measurements remain available.`:'Metadata ready. Excel will check valid group sizes and matching scoring windows before calculating tests.';
 $('#statistics-review-summary').textContent=`Excel: ${$('#statistics-mode').selectedOptions[0]?.textContent||'Group summaries'} · ${plan.metrics.length} outcome${plan.metrics.length===1?'':'s'}. One mouse per sample ID.`;
}
function initStatistics(options){
 const plan=draft.statistics;$('#statistics-mode').value=plan.mode;$('#statistics-alpha').value=plan.alpha;
 for(const key of (options.order||Object.keys(options.metrics))){const label=options.metrics[key],item=el('label'),input=el('input');input.type='checkbox';input.value=key;input.checked=plan.metrics.includes(key);input.onchange=()=>{const selected=[...$('#statistics-metrics').querySelectorAll('input:checked')].map(i=>i.value);if(!selected.length){input.checked=true;notice('Keep at least one statistical outcome selected.',true);return;}plan.metrics=selected;metadataChanged();};item.append(input,document.createTextNode(label));$('#statistics-metrics').append(item);}
 $('#statistics-mode').onchange=e=>{plan.mode=e.target.value;metadataChanged();};$('#statistics-alpha').onchange=e=>{plan.alpha=Number(e.target.value);metadataChanged();};
}
$('#choose-three-chamber').onclick=guard(async()=>{draft.test_id='three_chamber';await save();step('videos',true);});
$('#apply-metadata').onclick=guard(async()=>{const sex=$('#fill-sex').value,genotype=$('#fill-genotype').value.trim();let count=0;for(const e of draft.entries){let updated=false;if(sex&&(!e.sex||e.sex==='unknown')){e.sex=sex;updated=true;}if(genotype&&!(e.genotype||'').trim()){e.genotype=genotype;updated=true;}if(updated)count++;}renderLists();await save();notice(`Updated missing metadata for ${count} recording${count===1?'':'s'}.`);});
function filterResults(){const term=$('#results-filter').value.trim().toLowerCase();for(const c of resultCards.values()){c.hidden=['male','female','unknown'].includes(term)?c.dataset.sex!==term:!c.dataset.search.includes(term);if(c.hidden)c.querySelectorAll('video').forEach(v=>v.pause());}}
$('#results-filter').oninput=filterResults;
document.addEventListener('play',event=>{if(event.target.tagName==='VIDEO')document.querySelectorAll('video').forEach(v=>{if(v!==event.target)v.pause();});},true);
function syncCenters(){const e=entry();if(!e)return;for(const side of ['left','right'])for(const [i,axis] of ['x','y'].entries())$('#batch-'+side+'-'+axis).value=Number(e.config.cup_circles[side][i].toFixed(1));$('#batch-floor-points').value=e.config.arena.map(p=>p.map(x=>Number(x.toFixed(1))).join(', ')).join('\n');}
function draw(){if(!entry()||!frameReady||!img.naturalWidth)return;const c=$('#batch-canvas'),ctx=c.getContext('2d'),[x1,y1,x2,y2]=profile.crop_xyxy;c.width=x2-x1;c.height=y2-y1;ctx.drawImage(img,x1,y1,c.width,c.height,0,0,c.width,c.height);ctx.save();ctx.translate(-x1,-y1);const cfg=entry().config,a=cfg.arena,cc=cfg.cup_circles;ctx.lineWidth=2;ctx.strokeStyle='#7ce0b7';ctx.beginPath();a.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();a.forEach(([x,y],i)=>{ctx.fillStyle='#7ce0b7';ctx.fillRect(x-4,y-4,8,8);ctx.fillText(i+1,x+6,y-6);});for(const d of cfg.dividers_fraction){const p=ChamberGeometry.project(a,d,0),q=ChamberGeometry.project(a,d,1);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();if(editMode==='floor'){const m=ChamberGeometry.project(a,d,.5);ctx.fillStyle='#7ce0b7';ctx.fillRect(m[0]-6,m[1]-6,12,12);}}for(const [side,color] of [['left','#8bd7f2'],['right','#ffbd80']]){const [x,y]=cc[side],r=cc.diameter_px/2;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();ctx.fillRect(x+r-4,y-4,8,8);ctx.fillText(side==='left'?'L':'R',x-5,y-r-7);}ctx.restore();}
function loadEntry(){if(!entry())return;frameReady=false;$('#confirm-recording').disabled=true;const ctx=$('#batch-canvas').getContext('2d');ctx.clearRect(0,0,$('#batch-canvas').width,$('#batch-canvas').height);const e=entry();$('#region-sample').textContent=[e.id,e.sex&&e.sex!=='unknown'?e.sex:'Sex not recorded',e.genotype||'Genotype not recorded'].join(' · ');$('#batch-target').value=e.config.target_side||'unspecified';$('#batch-diameter').value=$('#batch-diameter-slider').value=draft.diameter_px;$('#batch-cutoff').value=draft.pcutoff;$('#batch-divider-left').value=Number((e.config.dividers_fraction[0]*100).toFixed(3));$('#batch-divider-right').value=Number((e.config.dividers_fraction[1]*100).toFixed(3));$('#batch-frame').value=e.reference_frame??1237;syncCenters();renderLists();const token=++frameToken;const image=new Image();image.onload=()=>{if(token===frameToken){img=image;frameReady=true;$('#confirm-recording').disabled=false;draw();}};image.onerror=()=>notice('The reference frame could not be loaded.',true);image.src='/api/frame?video='+encodeURIComponent(e.video)+'&frame='+$('#batch-frame').value;}
function resize(value){if(busy)return;const d=Number(value);if(!Number.isFinite(d)||d<4||d>500||d===draft.diameter_px)return;draft.diameter_px=d;draft.entries.forEach(e=>e.config.cup_circles.diameter_px=d);$('#batch-diameter').value=d;$('#batch-diameter-slider').value=d;changed(true);draw();}
$('#batch-diameter').oninput=e=>resize(e.target.value);$('#batch-diameter-slider').oninput=e=>resize(e.target.value);
$('#batch-cutoff').oninput=e=>{draft.pcutoff=Number(e.target.value);draft.entries.forEach(e=>e.config.pcutoff=draft.pcutoff);changed(true);};
$('#batch-recording').onchange=e=>{current=Number(e.target.value);loadEntry();};$('#review-regions').onclick=()=>step('regions',true);
$('#batch-target').onchange=e=>{entry().config.target_side=e.target.value;changed();};
for(const side of ['left','right'])for(const [i,axis] of ['x','y'].entries())$('#batch-'+side+'-'+axis).oninput=e=>{entry().config.cup_circles[side][i]=Number(e.target.value);changed();draw();};
for(const [i,side] of ['left','right'].entries())$('#batch-divider-'+side).oninput=e=>{entry().config.dividers_fraction[i]=Number(e.target.value)/100;changed();draw();};
$('#apply-batch-floor').onclick=guard(()=>{const a=$('#batch-floor-points').value.trim().split('\n').map(r=>r.trim().split(/[,\s]+/).map(Number));if(a.length!==4||a.some(p=>p.length!==2||!p.every(Number.isFinite)))throw Error('Enter four finite x,y corner pairs.');entry().config.arena=a;changed();draw();});
$('#batch-load-frame').onclick=()=>{if(entry()){entry().reference_frame=Math.max(0,Math.floor(Number($('#batch-frame').value)||0));loadEntry();save().catch(e=>notice(e.message,true));}};
for(const mode of ['circles','floor'])$('#edit-'+mode).onclick=()=>{
 editMode=mode;for(const m of ['circles','floor']){$('#edit-'+m).classList.toggle('active',m===mode);$('#edit-'+m).setAttribute('aria-pressed',String(m===mode));}
 $('#batch-draw-help').textContent=mode==='circles'?'Drag a circle to move it. Drag either square handle to resize both.':'Drag the numbered floor corners or the square handles on the chamber dividers.';draw();
};
const canvas=$('#batch-canvas');function point(event){const b=canvas.getBoundingClientRect();return [(event.clientX-b.left)*canvas.width/b.width+profile.crop_xyxy[0],(event.clientY-b.top)*canvas.height/b.height+profile.crop_xyxy[1]];}
canvas.onpointerdown=e=>{if(busy||!entry()||!frameReady)return;const p=point(e),cfg=entry().config,cc=cfg.cup_circles;drag=null;if(editMode==='floor'){const i=cfg.arena.findIndex(a=>Math.hypot(a[0]-p[0],a[1]-p[1])<14);if(i>=0)drag={kind:'floor',i};else{const j=cfg.dividers_fraction.findIndex(d=>{const q=ChamberGeometry.project(cfg.arena,d,.5);return Math.hypot(q[0]-p[0],q[1]-p[1])<16;});if(j>=0)drag={kind:'divider',i:j};}}else{for(const side of ['left','right']){const a=cc[side];if(Math.hypot(p[0]-a[0]-cc.diameter_px/2,p[1]-a[1])<12){drag={kind:'resize',side};break;}if(Math.hypot(p[0]-a[0],p[1]-a[1])<=cc.diameter_px/2)drag={kind:'move',side,offset:[p[0]-a[0],p[1]-a[1]]};}}if(drag)canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;const p=point(e),cfg=entry().config;if(drag.kind==='resize'){const c=cfg.cup_circles[drag.side];resize(Math.min(500,Math.max(4,Math.round(2*Math.hypot(p[0]-c[0],p[1]-c[1])))));}else{if(drag.kind==='floor')cfg.arena[drag.i]=p;else if(drag.kind==='divider'){const u=ChamberGeometry.unproject(cfg.arena,...p)[0];if(!Number.isFinite(u))return;cfg.dividers_fraction[drag.i]=Math.min(drag.i===0?cfg.dividers_fraction[1]-.01:.99,Math.max(drag.i===0?.01:cfg.dividers_fraction[0]+.01,u));$('#batch-divider-'+(drag.i===0?'left':'right')).value=Number((cfg.dividers_fraction[drag.i]*100).toFixed(3));}else cfg.cup_circles[drag.side]=[p[0]-drag.offset[0],p[1]-drag.offset[1]];changed();draw();syncCenters();}};
canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;save().catch(e=>notice(e.message,true));};
$('#previous-recording').onclick=()=>{current=Math.max(0,current-1);loadEntry();};
$('#confirm-recording').onclick=guard(async()=>{
 if(!frameReady)throw Error('Wait for the recording frame to load.');
 const e=entry(),original=JSON.stringify(e.config),snapshot=JSON.parse(original);snapshot.confirmed=true;
 $('#confirm-recording').disabled=true;
 try{
  await api('/api/batch/review',{config:snapshot});
  if(JSON.stringify(e.config)!==original)throw Error('Regions changed during validation. Confirm the updated placement.');
  e.config.confirmed=true;e.config.zone_geometry_source='User-reviewed linked circles; shared batch diameter in source-video pixels.';e.reviewed=true;await save();
  if(entry()===e&&current<draft.entries.length-1){current++;loadEntry();$('#batch-recording').scrollIntoView({block:'start'});}
  else renderLists();
  if(draft.entries.every(e=>e.reviewed)){$('#start-batch').scrollIntoView({block:'center'});$('#start-batch').focus();notice('All regions confirmed. Ready to analyze.');}
  else notice('Regions confirmed. Review the next recording.');
 }finally{$('#confirm-recording').disabled=!frameReady;}
});
async function addVideo(name){if(draft.entries.some(e=>e.video===name))return;const e=await api('/api/batch/seed',{video:name});let id=e.id,number=2;while(draft.entries.some(e=>e.id===id))id=e.id+'-'+number++;e.id=id;e.sex='unknown';e.genotype='';e.config.cup_circles.diameter_px=draft.diameter_px;e.config.pcutoff=draft.pcutoff;draft.entries.push(e);renderLists();renderAvailable();await save();}
function workspaceChoices(){return available.filter(v=>!v.prepared).map(v=>({source:v,working:available.find(p=>p.prepared&&p.name.split('/').pop().startsWith(v.name.split('/').pop().replace(/\.[^.]+$/,'')+'__'))||v}));}
function renderAvailable(){
 const box=$('#available-videos');box.replaceChildren();const choices=workspaceChoices();
 let remaining=0;for(const {source,working} of choices){
  const added=draft.entries.some(e=>e.video===working.name||e.video===source.name);if(!added)remaining++;
  const row=el('div',null,'batch-row'),button=el('button',added?'Added':'Add');button.disabled=added||adding;
  button.onclick=guard(async()=>{await addVideo(working.name);if(draft.entries.length===1)loadEntry();notice('Recording added.');});row.append(el('span',source.name.split('/').pop()),button);box.append(row);
 }
 if(!choices.length)box.append(el('p','No other recordings in this workspace. Use Add videos above.'));
 $('#add-all').disabled=adding||!remaining;
}
$('#add-all').onclick=guard(async()=>{adding=true;renderLists();renderAvailable();try{for(const {working} of workspaceChoices())await addVideo(working.name);if(entry())loadEntry();notice('Recordings added. Add sample details, then review regions.');}finally{adding=false;renderLists();renderAvailable();}});
$('#batch-upload').onchange=guard(async e=>{
 const files=[...e.target.files];if(!files.length)return;adding=true;e.target.disabled=true;renderLists();renderAvailable();let added=0;const failures=[];
 try{
  for(let i=0;i<files.length;i++){
   const f=files[i];$('#upload-status').textContent=`Adding ${i+1}/${files.length}: ${f.name}`;
   try{const data=new FormData();data.append('file',f);const r=await fetch('/api/videos/upload',{method:'POST',body:data});const d=await r.json();if(!r.ok)throw Error(d.error);await addVideo(d.name);added++;}catch(error){failures.push(`${f.name}: ${error.message}`);}
  }
  available=await api('/api/videos');if(entry())loadEntry();
  $('#upload-status').textContent=`${added} recording${added===1?'':'s'} added. ${failures.length?failures.length+' could not be added.':'Add sample details, then review regions.'}`;
  if(failures.length)notice(failures.join(' · '),true);else notice('Videos added. Add sample details and choose your Excel comparisons.');
 }finally{adding=false;e.target.disabled=false;e.target.value='';renderLists();renderAvailable();}
});
function link(text,url){const a=el('a',text);a.href=url;return a;}
function renderJob(b){
 const running=['queued','running'].includes(b.status),panel=$('#batch-progress');panel.hidden=running;panel.replaceChildren();
 if(liveView.host.hidden||liveView.batch?.id!==b.id)panel.append(el('h2',b.message));
 if(b.workbook){const a=link('Download Excel table',`/batches/${b.id}/results.xlsx`);a.className='export-primary';panel.append(a);if(b.statistics){const names={descriptive:'group summaries',genotype:'pooled genotype comparisons',within_sex:'genotype comparisons within each sex',factorial:'genotype × sex ANOVA',all:'all group comparisons'};panel.append(el('p',`Excel includes ${names[b.statistics.mode]||'group analysis'} · ${b.statistics.metrics.length} selected outcomes. Methods and exclusions are included in the workbook.`));}}
 const box=$('#batch-results-list');if(resultBatchId!==b.id){box.querySelectorAll('video').forEach(v=>v.pause());$('#results-filter').value='';box.replaceChildren();resultCards=new Map();resultBatchId=b.id;}
 const completed=b.entries.filter(e=>['complete','failed'].includes(e.status));
 $('#completed-heading').hidden=!running;$('#completed-heading').textContent=completed.length?'Completed recordings':'Completed recordings will appear here';
 for(const e of completed){
  const key=JSON.stringify([e.id,e.run_id,e.status,e.error]);if(resultCards.has(key))continue;
  const c=el('article',null,'panel result'),heading=el('div',null,'result-heading');heading.append(el('h2','Recording '+e.id),el('span',e.status));c.dataset.sex=e.sex||'unknown';c.dataset.search=[e.id,e.sex,e.genotype].filter(Boolean).join(' ').toLowerCase();c.append(heading,el('p',[e.sex&&e.sex!=='unknown'?e.sex:'Sex not recorded',e.genotype||'Genotype not recorded'].join(' · '),'sample-meta'));const source=el('details',null,'recording-source');source.append(el('summary','Source recording'),el('p',e.video.split('/').pop()));c.append(source);
  if(e.summary){
   const s=e.summary,fmt=n=>Number.isFinite(n)?n.toFixed(2)+' s':'—',table=el('table',null,'measurements'),head=el('thead'),hr=el('tr');
   for(const t of ['Time','Left','Center','Right']){const th=el('th',t);th.scope='col';hr.append(th);}head.append(hr);table.append(head);const body=el('tbody');
   for(const [label,...values] of [['Chamber occupancy',s.left_chamber_seconds,s.center_chamber_seconds,s.right_chamber_seconds],['Nose in cup circle',s.left_nose_seconds,null,s.right_nose_seconds]]){const row=el('tr'),th=el('th',label);th.scope='row';row.append(th,...values.map(v=>el('td',fmt(v))));body.append(row);}table.append(body);c.append(table);
   c.append(el('p',`${fmt(s.analyzed_seconds)} analyzed · ${s.cup_diameter_px} px circles · ${fmt(s.unknown_chamber_seconds)} unknown chamber time`));
   const preview=el('details',null,'review-video'),summary=el('summary'),poster=el('img');poster.alt='';poster.loading='lazy';poster.src='/api/frame?video='+encodeURIComponent(e.video)+'&frame=0&crop=arena';summary.append(poster,el('span','▶  Watch annotated review'),el('small','Click to play · every scored frame'));const video=el('video');video.controls=true;video.preload='none';video.setAttribute('aria-label',`Annotated review for ${e.id}`);preview.append(summary,video);preview.ontoggle=()=>{if(preview.open){document.querySelectorAll('.review-video').forEach(p=>{if(p!==preview)p.open=false;});liveView.setCollapsed(true);if(!video.getAttribute('src'))video.src=`/outputs/${e.run_id}/review.mp4`;video.play().catch(()=>{});}else video.pause();};c.append(preview);
   c.append(el('p',`Nose coverage ${(s.nose_scoreable_fraction*100).toFixed(1)}% · center coverage ${(s.center_valid_fraction*100).toFixed(1)}%. Tracking accuracy has not been validated.`));
   const details=el('details');details.append(el('summary','Detailed exports & scoring information'),el('p','Nose-in-circle time measures proximity. Low-confidence landmarks are left unscored.'));
   const downloads=el('div',null,'download-links');for(const [label,file] of [['Summary CSV','summary.csv'],['Per-frame CSV','frames.csv'],['Bouts CSV','bouts.csv'],['Annotated video','review.mp4'],['Region settings','calibration.json']])downloads.append(link(label,`/outputs/${e.run_id}/${file}?download=1`));details.append(downloads);c.append(details);
  }
  if(e.error)c.append(el('p',e.error));if(e.run_id)c.id='result-'+e.run_id;resultCards.set(key,c);box.append(c);
 }
 $('#results-filter-label').hidden=completed.length<2;filterResults();$('#analysis-history').open=false;
}
async function refreshJobs(generation){
 const controller=new AbortController();statusRequest=controller;
 const ids=[...new Set([job,activeJob].filter(Boolean))];
 const responses=await Promise.allSettled(ids.map(async id=>{const r=await fetch('/api/batches/'+id,{cache:'no-store',signal:controller.signal});if(!r.ok)throw Error('Could not read analysis progress.');return r.json();}));
 if(generation!==watchGeneration)return;
 statusRequest=null;const wasBusy=busy;let retry=false;
 for(const [i,result] of responses.entries()){
  if(result.status==='rejected'){if(result.reason.name!=='AbortError'){retry=true;notice('Progress connection interrupted. Reconnecting…',true);}continue;}
  const b=result.value,running=['queued','running'].includes(b.status);
  if(b.id===activeJob||(!activeJob&&running)){
   liveView.updateBatch(b);activeJob=running?b.id:null;
  }
  if(b.id===job)renderJob(b);
 }
 busy=!!activeJob;if(busy!==wasBusy)renderLists();
 $('#results-heading').textContent=activeJob?'Your analysis is running.':'Review your measurements.';
 $('#results-description').textContent=activeJob?'Watch annotated video while the batch processes.':'Download the combined table and check each recording’s annotated video.';
 if(activeJob||retry)watchTimer=setTimeout(()=>refreshJobs(generation).catch(e=>notice(e.message,true)),retry?3000:1800);
 else await historyList();
}
async function watch(id){
 if(!activeJob&&liveView.batch?.id!==id){liveView.setCollapsed(true);liveView.host.hidden=true;}
 job=id;watchGeneration++;clearTimeout(watchTimer);watchTimer=null;statusRequest?.abort();statusRequest=null;
 await refreshJobs(watchGeneration);
}
$('#start-batch').onclick=guard(async()=>{if(busy||adding)return;busy=true;renderLists();try{await save();const b=await api('/api/batches',draft);activeJob=b.id;job=b.id;liveView.updateBatch(b);renderJob(b);step('results',true);await watch(b.id);}catch(e){busy=!!activeJob;renderLists();throw e;}});
async function historyList(){
 const list=await api('/api/batches'),box=$('#batch-history');box.replaceChildren();
 if(!job)$('#analysis-history').open=true;
 if(!list.length)box.append(el('p','No analyses yet. Add recordings and confirm their regions to create your first results.'));
 for(const b of list){
  const row=el('div',null,'batch-row'),button=el('button','Open results');button.onclick=guard(async()=>{job=b.id;step('results',true);await watch(b.id);});
  const ids=b.entries.map(e=>e.id).join(', '),text=el('span',`${b.entries.length} recording${b.entries.length===1?'':'s'} · ${ids}`);text.append(el('small',`${b.status.replaceAll('_',' ')} · ${b.id}`));row.append(text,button);box.append(row);
 }return list;
}
guard(async()=>{
 let options;[draft,profile,available,options]=await Promise.all([api('/api/batch/draft'),api('/api/profile'),api('/api/videos'),api('/api/statistics/options')]);draft.test_id='three_chamber';draft.statistics={...options.defaults,...draft.statistics};initStatistics(options);
 renderAvailable();renderLists();if(!entry())$('#workspace-library').open=true;
 const old={inspect:'videos',calibrate:'regions',analysis:'regions',tracking:'regions'},requested=location.hash.slice(1);step(old[requested]||requested||'tests');
 const list=await historyList();const active=list.find(b=>['running','queued'].includes(b.status));
 if(active){activeJob=active.id;job=active.id;liveView.updateBatch(active);step('results');await watch(active.id);}
})();
