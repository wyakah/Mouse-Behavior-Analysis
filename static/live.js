/* Analysis status and smooth review playback have separate lifecycles. */
class LiveAnalysisView {
  constructor(host) {
    this.host=host;this.batch=null;this.snapshot=null;this.previewHidden=true;this.sectionHidden=false;this.seenActive=false;
    this.timer=null;this.request=null;this.epoch=0;this.queueKey='';
    host.innerHTML=`
      <div class="live-heading"><div><div class="live-eyebrow"><span class="live-dot"></span><span data-live="label">LIVE ANALYSIS</span></div><h2 data-live="title">Analysis, in motion.</h2><p data-live="subtitle"></p></div><div class="live-tools"><button data-action="hide" aria-expanded="true">Hide preview</button><button data-action="focus" aria-pressed="false">Focus view ↗</button></div></div>
      <div class="live-layout"><div class="live-main">
        <div class="live-stagebar" aria-label="Processing stages"></div>
        <div class="stream-now"><div><span class="stream-caption">NOW WATCHING</span><strong data-live="watching">Preparing recording</strong></div><button data-action="active" hidden>Watch active recording →</button></div>
        <div class="stream-screen"><video class="stream-video" controls muted playsinline preload="auto" aria-label="Buffered annotated analysis video"></video><div class="stream-empty"><div class="live-aperture" aria-hidden="true"><i></i><i></i><i></i><i></i><span>Ⅲ</span></div><strong data-live="empty-title">Preparing annotated footage</strong><p data-live="empty-text">Playback begins after a few seconds of video are ready.</p></div></div>
        <div class="stream-status"><span data-live="playback">Waiting for video</span><span data-live="ready">Video ready through —</span></div>
        <p class="stream-message" data-live="stream-message" role="status"></p>
        <div class="stream-key"><span><i style="background:#f8d85e"></i>Nose</span><span><i style="background:#85e4b6"></i>Body center</span><span><i style="background:#c9a4ff"></i>Tail base</span><span>Dashed marker = uncertain</span></div>
        <div class="live-pass"><div><span data-live="pass-label">Preparing recording</span><b data-live="percent">—</b></div><progress data-live="progress" aria-label="Current processing pass"></progress><small data-live="pass-detail">Processing continues independently of playback.</small></div>
      </div><aside class="live-queue"><div class="live-queue-heading"><h3>Recording queue</h3><span data-live="batch-count"></span></div><div class="live-queue-list"></div><div class="live-queue-note"><p>Watch at your own pace.<br>Analysis keeps moving.</p></div></aside></div>
      <div class="live-footnote"><span data-live="connection" role="status">Buffered video · every frame in order</span><span>Confidence is not validated accuracy.</span></div>`;
    this.q=name=>host.querySelector(`[data-live="${name}"]`);
    this.stages=[['preparing','Prepare'],['localizing','Locate'],['tracking','Track'],['scoring','Score'],['rendering','Review']];
    this.stages.forEach(([key,label],i)=>{const n=document.createElement('div');n.dataset.stage=key;const dot=document.createElement('span');dot.className='stage-mark';dot.textContent=i+1;const t=document.createElement('span');t.textContent=label;n.append(dot,t);host.querySelector('.live-stagebar').append(n);});
    this.player=new AnalysisStreamPlayer(host.querySelector('video'),()=>this.renderPlayback(),()=>{
      if(this.isRunning()&&this.batch.current_index!==this.player.index)this.watchRecording(this.batch.current_index);
    });
    const compact=document.createElement('div');compact.className='live-compact';
    const launch=document.createElement('button');launch.className='preview-launch';launch.innerHTML='<img alt=""><span>▶ Watch analysis</span>';launch.onclick=()=>this.setCollapsed(false);
    compact.append(launch,host.querySelector('.live-pass'));host.querySelector('.live-layout').before(compact);
    host.querySelector('[data-action="hide"]').onclick=()=>this.setCollapsed(!this.previewHidden);
    this.setCollapsed(true);
    host.querySelector('[data-action="focus"]').onclick=()=>{
      const on=host.classList.toggle('focus-view'),b=host.querySelector('[data-action="focus"]');b.textContent=on?'Standard view ↙':'Focus view ↗';b.setAttribute('aria-pressed',String(on));
      host.scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
    };
    host.querySelector('[data-action="active"]').onclick=()=>this.watchRecording(this.batch.current_index);
  }
  setVisible(visible){this.sectionHidden=!visible;this.player.setHidden(!visible||this.previewHidden);}
  setCollapsed(hidden){this.previewHidden=hidden;this.host.classList.toggle('preview-collapsed',hidden);const b=this.host.querySelector('[data-action="hide"]');b.textContent=hidden?'Expand preview':'Collapse preview';b.setAttribute('aria-expanded',String(!hidden));this.host.querySelector('[data-action="focus"]').hidden=hidden;if(hidden){this.host.classList.remove('focus-view');const focus=this.host.querySelector('[data-action="focus"]');focus.textContent='Focus view ↗';focus.setAttribute('aria-pressed','false');}this.player.setHidden(hidden||this.sectionHidden);if(hidden&&this.batch&&!this.isRunning())this.host.hidden=true;if(!hidden)document.querySelectorAll('.review-video').forEach(p=>p.open=false);this.renderPlayback();}
  isRunning(){return this.batch&&['queued','running'].includes(this.batch.status);}
  time(seconds){if(!Number.isFinite(seconds))return '—';const n=Math.max(0,Math.floor(seconds));return `${Math.floor(n/60).toString().padStart(2,'0')}:${(n%60).toString().padStart(2,'0')}`;}
  watchRecording(index,expand=true){if(!Number.isInteger(index))return;if(expand)this.setCollapsed(false);const poster=this.host.querySelector('.preview-launch img');poster.src='/api/frame?video='+encodeURIComponent(this.batch.entries[index].video)+'&frame=0&crop=arena';this.player.select(this.batch.id,index);this.renderPlayback();}
  updateBatch(batch) {
    const running=['queued','running'].includes(batch.status);
    if(!running&&(!this.seenActive||this.batch?.id!==batch.id))return;
    const newBatch=this.batch?.id!==batch.id;
    if(newBatch){this.setCollapsed(true);this.epoch++;this.request?.abort();clearTimeout(this.timer);this.timer=null;this.request=null;this.snapshot=null;this.queueKey='';}
    this.batch=batch;this.seenActive||=running;this.host.hidden=false;
    this.host.classList.toggle('is-running',running);this.host.classList.toggle('processing-finished',!running);
    this.q('label').textContent=running?'LIVE ANALYSIS':'ANALYSIS FINISHED';
    this.q('title').textContent=running?'Analysis, in motion.':batch.status==='interrupted'?'Analysis interrupted':batch.status==='complete'?'Ready for review.':'Review needs attention.';
    const active=batch.entries.find(e=>e.status==='running');
    this.q('subtitle').textContent=active?`Processing ${active.id} · ${batch.current_index+1} of ${batch.entries.length}`:batch.message;
    this.q('batch-count').textContent=`${batch.entries.filter(e=>e.status==='complete').length} / ${batch.entries.length} complete`;
    if(newBatch||this.player.index===null)this.watchRecording(batch.current_index??0,false);
    else if(this.player.video.ended&&running&&!this.previewHidden)this.watchRecording(batch.current_index);
    const key=JSON.stringify(batch.entries.map(e=>[e.id,e.status,e.run_id]));
    if(key!==this.queueKey){this.queueKey=key;const box=this.host.querySelector('.live-queue-list');box.replaceChildren();batch.entries.forEach((e,i)=>{
      const row=document.createElement('div');row.className='live-queue-item '+e.status;
      const marker=document.createElement('span');marker.className='queue-marker';marker.textContent=e.status==='complete'?'✓':e.status==='failed'?'!':String(i+1);
      const text=document.createElement('div'),name=document.createElement('strong'),state=document.createElement('small');name.textContent=e.id;state.textContent=({pending:'Waiting',running:'Processing',complete:'Results ready',failed:'Needs attention'})[e.status]||e.status;text.append(name,state);row.append(marker,text);
      if(e.status!=='pending'){const button=document.createElement('button');button.textContent='Watch';button.setAttribute('aria-label',`Watch recording ${e.id}`);button.onclick=()=>this.watchRecording(i);row.append(button);}
      box.append(row);
    });}
    this.renderProgress();this.renderPlayback();this.host.hidden=!running&&this.previewHidden;
    if(running&&!this.timer&&!this.request)this.poll();
    if(!running){clearTimeout(this.timer);this.timer=null;this.request?.abort();this.request=null;this.epoch++;}
  }
  async poll(){
    this.timer=null;if(!this.isRunning()||this.request)return;
    const epoch=this.epoch,controller=new AbortController();this.request=controller;
    try{
      const r=await fetch(`/api/batches/${encodeURIComponent(this.batch.id)}/live?image=0`,{cache:'no-store',signal:controller.signal});if(!r.ok)throw Error('Preview status unavailable');
      const packet=await r.json();if(epoch!==this.epoch)return;
      if(packet.snapshot?.recording_index===this.batch.current_index)this.snapshot=packet.snapshot;
      this.q('connection').textContent='Buffered video · every frame in order';this.renderProgress();
    }catch(e){if(e.name!=='AbortError')this.q('connection').textContent='Processing status reconnecting · playback continues';}
    finally{if(this.request===controller)this.request=null;if(epoch===this.epoch&&this.isRunning())this.timer=setTimeout(()=>this.poll(),800);}
  }
  renderProgress(){
    const s=this.snapshot?.recording_index===this.batch?.current_index?this.snapshot:null,stage=s?.stage||'preparing',running=this.isRunning();
    const titles={preparing:'Preparing recording',localizing:'Locating the subject',tracking:'Tracking landmarks',cached:'Using saved predictions',scoring:'Calculating measurements',rendering:'Writing final review video',exporting:'Creating Excel table',failed:'Recording failed'};
    const phase=stage==='cached'?2:this.stages.findIndex(([key])=>key===stage);
    this.host.querySelectorAll('.live-stagebar>div').forEach((n,i)=>{n.classList.toggle('current',running&&i===phase);n.classList.toggle('done',i<phase||stage==='exporting'||!running);});
    this.q('pass-label').textContent=running?`${this.batch.entries[this.batch.current_index]?.id||''} · ${titles[stage]||stage}`:this.batch.message;
    const pct=Number.isInteger(s?.frame_index)&&s?.total_frames?Math.min(100,100*s.frames_done/s.total_frames):null;
    this.q('percent').textContent=!running?'':pct===null?'—':`${pct.toFixed(0)}%`;
    const progress=this.q('progress');progress.hidden=!running;
    if(pct===null)progress.removeAttribute('value');else{progress.max=100;progress.value=pct;}
    this.q('pass-detail').textContent=running?(stage==='localizing'?'Finding the free subject before landmark tracking starts.':stage==='tracking'?'Estimating landmarks and encoding annotated footage.':stage==='rendering'?'Saving the final annotated review. Playback continues independently.':s?.message||'Processing continues independently of playback.'):'Playback remains available. Final measurements and downloads are below.';
  }
  renderPlayback(){
    if(!this.player||!this.batch)return;
    const p=this.player,video=p.video,entry=this.batch.entries[p.index],hasVideo=!!p.media;
    this.q('watching').textContent=entry?`Recording ${entry.id}`:'Preparing recording';
    this.host.querySelector('[data-action="active"]').hidden=!this.isRunning()||p.index===this.batch.current_index||!Number.isInteger(this.batch.current_index);
    this.host.querySelector('.stream-empty').hidden=hasVideo&&!this.previewHidden;
    this.q('empty-title').textContent=this.previewHidden?'Preview hidden':p.state==='unavailable'?'Preview unavailable':'Preparing annotated footage';
    this.q('empty-text').textContent=this.previewHidden?'Every frame continues to be analyzed.':p.message||'Playback begins when a few seconds of annotated video are ready.';
    this.q('playback').textContent=hasVideo?`${video.ended?'Finished':video.paused?'Paused':p.buffering?'Buffering':'Watching'} ${this.time(video.currentTime)} / ${this.time(p.manifest?.source_duration_s)}`:'Waiting for video';
    this.q('ready').textContent=`Video ready through ${this.time(p.manifest?.available_until_s)}`;
    this.q('stream-message').textContent=(!hasVideo?'':p.message)||(
!p.autoStart&&p.buffering?'Waiting for processing to produce the next frames.':p.autoStart&&hasVideo?'Building a short buffer for smooth playback.':'');
  }
}
window.LiveAnalysisView=LiveAnalysisView;
