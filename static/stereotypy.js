'use strict';
const $ = id => document.getElementById(id);
let session = null, rows = [], undo = [], editing = null, dirty = false, exact = null, busy = false;
const player = $('player');
function notice(text, error = false) { $('notice').textContent = text; $('notice').classList.toggle('error', error); }
async function api(url, data) {
  const response = await fetch(url, data === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  if (!response.ok) { const result = await response.json(); throw new Error(result.error || 'Request failed.'); }
  return response.json();
}
function action(id, fn) { $(id).addEventListener('click', async () => { try { await fn(); } catch (e) { notice(e.message, true); } }); }
function option(value, text) { const el = document.createElement('option'); el.value = value; el.textContent = text; return el; }
function fmt(n) { return n == null ? '—' : Number(n).toFixed(3); }
function markDirty() { dirty = true; $('save-state').textContent = 'Unsaved changes'; }
let queue={entries:[]},setupView,currentIndex=0,preparing=false,queueSave=Promise.resolve();
function saveQueue(){const snapshot=structuredClone(queue);queueSave=queueSave.catch(()=>{}).then(()=>api('/api/stereotypy/draft',snapshot));return queueSave;}
function showView(name){if((dirty||busy||preparing)&&name==='setup')throw Error('Save your changes before returning to videos.');for(const id of ['setup','review','stereo-results'])$(id).hidden=id!==name;document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===name));player.pause();window.scrollTo(0,0);}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{try{showView(b.dataset.view);}catch(e){notice(e.message,true);}});
async function inventory(){
 const [videos,saved]=await Promise.all([api('/api/videos'),api('/api/stereotypy/sessions')]);
 setupView?.setAvailable(videos.filter(v=>!v.prepared));
 $('sessions').replaceChildren(option('','Choose a recording'));saved.forEach(s=>$('sessions').append(option(s.id,`${s.animal_id} · ${s.sex||'not recorded'} · ${s.genotype||'genotype not recorded'}`)));
 if(session)$('sessions').value=session.id;
}
async function prepareEntry(index){
 if(dirty||busy||preparing||(typeof cageChanged!=='undefined'&&cageChanged))throw Error('Save annotations and cage changes before changing recordings.');
 if(!queue.entries[index])return;
 const entry=queue.entries[index];preparing=true;setupView.render();$('queue-recording').disabled=true;$('next-recording').disabled=true;
 try{
  await api('/api/stereotypy/setup',{entries:queue.entries});
  await saveQueue();
  if(entry.session_ref){await api(`/api/stereotypy/sessions/${entry.session_ref}/metadata`,{animal_id:entry.id,sex:entry.sex,genotype:entry.genotype});}
  else{
   let job=await api('/api/stereotypy/sessions',{video:entry.video,animal_id:entry.id,sex:entry.sex,genotype:entry.genotype,view_confirmed:$('side-view').checked});
   while(['queued','running'].includes(job.status)){notice(`${entry.id}: ${job.message}`);await new Promise(resolve=>setTimeout(resolve,1000));job=await api(`/api/jobs/${job.id}`);}
   if(job.status!=='complete')throw Error(job.message);entry.session_ref=job.session_id;await saveQueue();
  }
  currentIndex=index;await inventory();await load(entry.session_ref);notice('Ready to review. Unmarked time stays unreviewed.');
 }finally{preparing=false;setupView.render();$('queue-recording').disabled=false;$('next-recording').disabled=currentIndex>=queue.entries.length-1;}
}
$('queue-recording').onchange=async e=>{try{await prepareEntry(Number(e.target.value));}catch(error){e.target.value=currentIndex;notice(error.message,true);}};
action('next-recording',()=>prepareEntry(currentIndex+1));
$('sessions').addEventListener('change', async () => {
  try {
    if (dirty || busy || preparing || (typeof cageChanged!=='undefined'&&cageChanged)) { $('sessions').value = session?.id || ''; throw new Error('Save annotations and cage changes before changing sessions.'); }
    if ($('sessions').value) await load($('sessions').value);
  } catch (e) { notice(e.message, true); }
});
async function load(id) {
  session = await api(`/api/stereotypy/sessions/${id}`); rows = structuredClone(session.annotations); undo = []; dirty = false; editing = null; exact = null;
  $('review').hidden = false; $('sessions').value = id;
  $('session-title').textContent = [session.animal_id,session.sex||'Sex not recorded',session.genotype||'Genotype not recorded'].join(' · ');
  $('queue-recording').replaceChildren(...queue.entries.map((e,i)=>option(i,e.id)));const index=queue.entries.findIndex(e=>e.session_ref===id);if(index>=0){currentIndex=index;$('queue-recording').value=index;}else{$('queue-recording').append(option('saved',session.animal_id));$('queue-recording').value='saved';}
  $('next-recording').disabled=index<0||index>=queue.entries.length-1;document.querySelectorAll('[data-view]').forEach(b=>b.disabled=false);showView('review');
  player.src = `/api/video?video=${encodeURIComponent(session.video)}`; $('exact-frame').hidden = true;
  $('behavior').replaceChildren(...session.summary.map(r => option(r.behavior, r.behavior.replaceAll('_', ' '))));
  $('window-start').value = session.start_s; $('window-end').value = session.end_s; $('merge-gap').value = session.merge_gap_s;
  $('onset').value = session.start_s; $('offset').value = session.start_s;
  $('seek').max = session.video_manifest.frames.length - 1;
  $('timing-info').textContent = `${session.video_manifest.frame_count} source frames · ${fmt(session.video_manifest.duration_s)} s · ${session.video_manifest.source_gaps.length} source gaps · ${session.video_manifest.variable_frame_rate ? 'variable' : 'constant'} frame timing. Frame buttons show the exact source image.`;
  resetEdit(); definition(); render(); savedSummary();
  if(typeof cageRefresh==='function')cageRefresh();
}
function definition() { $('definition').textContent = session?.ethogram[$('behavior').value] || ''; }
$('behavior').addEventListener('change', definition);
function frameIndex() {
  if (exact !== null) return exact;
  const frames = session.video_manifest.frames;
  let low = 0, high = frames.length;
  while (low < high) { const mid = (low + high) >> 1; if (frames[mid].start_s <= player.currentTime) low = mid + 1; else high = mid; }
  return Math.max(0, low - 1);
}
function time() { return exact === null ? Math.min(player.currentTime,session.video_manifest.duration_s) : session.video_manifest.frames[exact].start_s; }
function clock() { if (!session) return; $('clock').textContent = `${fmt(time())} s · frame ${frameIndex()}`; $('seek').value = frameIndex(); }
function showFrame(index) {
  exact = Math.max(0, Math.min(session.video_manifest.frames.length - 1, index)); player.pause();
  const f = session.video_manifest.frames[exact]; player.currentTime = f.start_s;
  $('exact-frame').src = `/api/stereotypy/sessions/${session.id}/frame/${exact}${$('cage-view')?.value==='review'?'?crop=cage':''}`; $('exact-frame').hidden = false; clock();
}
player.addEventListener('timeupdate', clock);
player.addEventListener('play', () => { exact = null; $('exact-frame').hidden = true; });
player.addEventListener('error', () => notice('Browser playback is unavailable for this codec. Exact source-frame buttons remain available.', true));
$('exact-frame').addEventListener('error', () => notice('The exact frame could not be loaded. Check that the source file has not changed.', true));
$('seek').addEventListener('input', () => { if (session) showFrame(Number($('seek').value)); });
action('back', () => showFrame(frameIndex() - 1)); action('forward', () => showFrame(frameIndex() + 1));
action('play', async () => { if (player.paused) { exact = null; $('exact-frame').hidden = true; await player.play(); } else player.pause(); });
$('speed').addEventListener('change', () => { player.playbackRate = Number($('speed').value); });
action('mark-start', () => { $('onset').value = time(); }); action('mark-end', () => { $('offset').value = time(); });
action('include-frame', () => { $('offset').value = session.video_manifest.frames[frameIndex()].end_s; });
document.addEventListener('keydown', event => {
  if (!session || event.ctrlKey || event.metaKey || event.altKey || /INPUT|SELECT|TEXTAREA|BUTTON|VIDEO/.test(event.target.tagName)) return;
  if (event.key.toLowerCase() === 'i') { event.preventDefault(); $('onset').value = time(); }
  if (event.key.toLowerCase() === 'o') { event.preventDefault(); $('offset').value = time(); }
});
function resetEdit() { editing = null; $('add').textContent = 'Add interval'; $('cancel-edit').hidden = true; }
action('cancel-edit', resetEdit);
action('add', () => {
  const row = {behavior:$('behavior').value, label:$('label').value, start_s:Number($('onset').value), end_s:Number($('offset').value), note:$('note').value};
  if (!$('onset').value || !$('offset').value || !Number.isFinite(row.start_s) || !Number.isFinite(row.end_s) || !(row.start_s < row.end_s) || row.start_s < Number($('window-start').value) || row.end_s > Number($('window-end').value)) throw new Error('Enter an onset before the offset, inside the analysis window.');
  if (rows.some((r,i) => i !== editing && r.behavior === row.behavior && r.start_s < row.end_s && row.start_s < r.end_s)) throw new Error('This overlaps an interval for the same behavior. Edit that interval first.');
  undo.push(structuredClone(rows));
  if (editing === null) rows.push(row); else rows[editing] = row;
  resetEdit(); markDirty(); render();
});
action('undo', () => { if (undo.length) { rows = undo.pop(); resetEdit(); markDirty(); render(); } });
for (const id of ['window-start','window-end','merge-gap']) $(id).addEventListener('input', markDirty);
function render() {
  $('annotations').replaceChildren(); $('timeline').replaceChildren(); $('undo').disabled = !undo.length;
  rows.forEach((row, index) => {
    const tr = document.createElement('tr');
    for (const value of [row.behavior,row.label,fmt(row.start_s),fmt(row.end_s),row.note]) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
    const td = document.createElement('td');
    for (const name of ['Edit','Remove']) {
      const button = document.createElement('button'); button.textContent = name;
      button.onclick = () => {
        if (busy) return;
        if (name === 'Remove') { undo.push(structuredClone(rows)); rows.splice(index,1); resetEdit(); markDirty(); render(); }
        else { editing = index; $('behavior').value = row.behavior; $('label').value = row.label; $('onset').value = row.start_s; $('offset').value = row.end_s; $('note').value = row.note; definition(); $('add').textContent = 'Update interval'; $('cancel-edit').hidden = false; }
      }; td.append(button);
    }
    tr.append(td); $('annotations').append(tr);
  });
  for (const {behavior} of session.summary) {
    const line = document.createElement('div'); line.className = 'timeline-row';
    const label = document.createElement('span'); label.textContent = behavior.replaceAll('_',' ');
    const track = document.createElement('div'); track.className = 'timeline-track';
    rows.filter(r => r.behavior === behavior).forEach(r => {
      const b = document.createElement('button'); b.className = `timeline-segment ${r.label}`; b.style.left = `${100*r.start_s/session.video_manifest.duration_s}%`; b.style.width = `${100*(r.end_s-r.start_s)/session.video_manifest.duration_s}%`;
      b.title = `${behavior}: ${r.label}, ${fmt(r.start_s)}–${fmt(r.end_s)} s`; b.setAttribute('aria-label', b.title);
      b.onclick = () => { exact = null; player.currentTime = r.start_s; showFrame(frameIndex()); }; track.append(b);
    }); line.append(label,track); $('timeline').append(line);
  }
}
function savedSummary() {
  $('summary').replaceChildren();
  for (const r of session.summary) { const tr = document.createElement('tr'); for (const value of [r.behavior,fmt(r.active_seconds),fmt(r.scored_seconds),fmt(r.unknown_seconds),fmt(r.percent_scored),`${fmt(r.coverage_percent)}%`,r.observed_segments ?? '—',r.censored_segments ?? '—']) { const td = document.createElement('td'); td.textContent = value; tr.append(td); } $('summary').append(tr); }
  $('save-state').textContent = `Saved revision ${session.revision} · ${session.ethogram_version}`;
}
async function save() {
  if (busy) throw new Error('A save is already in progress.');
  busy = true;
  const controls = [...$('review').querySelectorAll('input,select,button')]; const previous = controls.map(el => el.disabled); controls.forEach(el => { el.disabled = true; });
  try {
    session = await api(`/api/stereotypy/sessions/${session.id}`, {revision:session.revision, annotator_id:$('annotator').value, start_s:$('window-start').value, end_s:$('window-end').value, merge_gap_s:$('merge-gap').value, annotations:rows});
    rows = structuredClone(session.annotations); dirty = false; resetEdit(); render(); savedSummary(); notice('Revision saved. Measurements updated; previous revisions retained.');
  } finally { controls.forEach((el,i) => { el.disabled = previous[i]; }); busy = false; $('undo').disabled = !undo.length; }
}
action('save', save);
action('export', async () => {
  if (dirty || session.revision === 0) await save();
  const response = await fetch(`/api/stereotypy/sessions/${session.id}/export`, {method:'POST'});
  if (!response.ok) { const result = await response.json(); throw new Error(result.error); }
  const url = URL.createObjectURL(await response.blob()); const a = document.createElement('a'); a.href = url; a.download = `stereotypy-${session.id.slice(0,8)}-r${session.revision}.zip`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 30000); notice('Export saved with source timing and revision history.');
});
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
(async()=>{
 queue=await api('/api/stereotypy/draft');$('side-view').checked=queue.view_confirmed===true;
 setupView=new RecordingSetup($('recording-setup'),{entries:()=>queue.entries,disabled:()=>preparing||busy,change:()=>saveQueue(),seed:async video=>({video,id:video.split('/').pop().replace(/^[a-f0-9]{8}_/,'').replace(/\.[^.]+$/,'').split('_')[0]}),continue:async()=>{if(!$('side-view').checked)throw Error('Confirm that each video shows one mouse from the side.');await prepareEntry(0);}});
 $('recording-setup').querySelector('.recording-next').before($('stereo-view-check'));
 $('side-view').onchange=()=>{queue.view_confirmed=$('side-view').checked;saveQueue().catch(e=>notice(e.message,true));};
 await inventory();
 const requestedSession=new URLSearchParams(location.search).get('session');if(requestedSession)await load(requestedSession);
})().catch(e=>notice(e.message,true));
