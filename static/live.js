/* Latest-frame monitoring. This view never estimates landmarks or behavior totals. */
class LiveAnalysisView {
  constructor(host) {
    this.host=host;this.batch=null;this.snapshot=null;this.revision='';this.image=null;
    this.timer=null;this.request=null;this.epoch=0;this.imageToken=0;this.previewHidden=false;
    this.seenActive=false;this.connected=true;this.queueKey='';this.frameKey='';
    host.innerHTML=`
      <div class="live-heading"><div><div class="live-eyebrow"><span class="live-dot"></span><span data-live="label">LIVE ANALYSIS</span></div><h2 data-live="title">Preparing your recordings</h2><p data-live="subtitle"></p></div><div class="live-tools"><button data-action="hide" aria-expanded="true">Hide preview</button><button data-action="focus" aria-pressed="false">Focus view <span aria-hidden="true">↗</span></button></div></div>
      <div class="live-layout"><div class="live-main">
        <div class="live-stagebar" aria-label="Processing stages"></div>
        <div class="live-preview-wrap"><div class="live-hud"><span class="live-stage-pill" data-live="stage">Preparing</span><span data-live="frame">Waiting for a processed frame</span></div>
          <canvas class="live-canvas" width="1120" height="800" role="img" aria-label="Latest processed arena frame with tracking overlays"></canvas>
          <div class="live-empty"><div class="live-aperture" aria-hidden="true"><i></i><i></i><i></i><i></i><span>Ⅲ</span></div><strong data-live="empty-title">Getting the arena ready</strong><p data-live="empty-text">The first processed frame will appear here.</p></div>
          <div class="live-view-footer"><span data-live="image-note">Actual processed frames</span><span class="live-overlay-key"><i class="key-left"></i> Left cup <i class="key-right"></i> Right cup</span></div>
        </div>
        <div class="live-timeline"><div class="live-time-labels"><span>Recording position <b data-live="position">—</b></span><span data-live="duration">—</span></div><div class="live-source-track"><div data-live="source-fill"></div></div></div>
        <div class="live-landmarks" aria-label="Latest frame landmark confidence"></div>
        <div class="live-pass"><div><span data-live="pass-label">Preparing recording</span><b data-live="percent">—</b></div><progress data-live="progress" aria-label="Current processing pass"></progress><small data-live="pass-detail">Each processing pass reports its own frame progress.</small></div>
      </div><aside class="live-queue"><div class="live-queue-heading"><h3>Recording queue</h3><span data-live="batch-count"></span></div><div class="live-queue-list"></div><div class="live-queue-note"><span class="live-local-icon" aria-hidden="true">↳</span><p>Processed one at a time.<br>Completed results stay available below.</p></div></aside></div>
      <div class="live-footnote"><span data-live="connection" role="status" aria-live="polite">Connected · up to 2 preview updates/s</span><span>Every frame is analyzed. Confidence is not validated accuracy.</span></div>`;
    this.q=name=>host.querySelector(`[data-live="${name}"]`);
    this.canvas=host.querySelector('canvas');this.empty=host.querySelector('.live-empty');
    host.querySelector('[data-action="hide"]').onclick=()=>{
      this.previewHidden=!this.previewHidden;host.classList.toggle('preview-hidden',this.previewHidden);
      const b=host.querySelector('[data-action="hide"]');b.textContent=this.previewHidden?'Show preview':'Hide preview';b.setAttribute('aria-expanded',String(!this.previewHidden));
      this.revision='';this.renderFrame();
    };
    host.querySelector('[data-action="focus"]').onclick=()=>{
      const on=host.classList.toggle('focus-view'),b=host.querySelector('[data-action="focus"]');b.textContent=on?'Standard view ↙':'Focus view ↗';b.setAttribute('aria-pressed',String(on));host.scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
    };
    this.stages=[['preparing','Prepare'],['localizing','Locate'],['tracking','Track'],['scoring','Score'],['rendering','Review']];
    for(const [key,label] of this.stages){const n=document.createElement('div');n.dataset.stage=key;const dot=document.createElement('span');dot.className='stage-mark';dot.textContent=String(host.querySelectorAll('.live-stagebar>div').length+1);const t=document.createElement('span');t.textContent=label;n.append(dot,t);host.querySelector('.live-stagebar').append(n);}
    this.colors={nose:'#f8d85e',center:'#85e4b6',tail_base:'#c9a4ff'};
    for(const [part,label] of [['nose','Nose'],['center','Body center'],['tail_base','Tail base']]){
      const card=document.createElement('div');card.className='landmark-card';card.dataset.part=part;card.innerHTML='<span class="landmark-name"><i></i><span></span></span><strong>—</strong><small>Waiting</small>';card.querySelector('i').style.background=this.colors[part];card.querySelector('.landmark-name span').textContent=label;host.querySelector('.live-landmarks').append(card);
    }
  }
  isRunning(){return this.batch&&['queued','running'].includes(this.batch.status);}
  time(seconds){if(!Number.isFinite(seconds))return '—';const n=Math.max(0,Math.floor(seconds));return `${Math.floor(n/60).toString().padStart(2,'0')}:${(n%60).toString().padStart(2,'0')}`;}
  updateBatch(batch) {
    const running=['queued','running'].includes(batch.status);
    if(!running&&(!this.seenActive||this.batch?.id!==batch.id))return;
    const changed=this.batch?.id!==batch.id||this.batch?.current_index!==batch.current_index;
    if(this.batch?.id!==batch.id){this.epoch++;this.request?.abort();clearTimeout(this.timer);this.seenActive=running;this.queueKey='';}
    this.batch=batch;this.seenActive||=running;this.host.hidden=false;
    this.host.classList.toggle('is-running',running);
    this.host.classList.toggle('is-finished',!running);
    if(!running){this.host.classList.remove('focus-view');const focus=this.host.querySelector('[data-action="focus"]');focus.textContent='Focus view ↗';focus.setAttribute('aria-pressed','false');}
    if(changed){this.snapshot=null;this.image=null;this.imageToken++;this.revision='';this.frameKey='';this.renderFrame();}
    const done=batch.entries.filter(e=>e.status==='complete').length,failed=batch.entries.filter(e=>e.status==='failed').length;
    const active=batch.entries.find(e=>e.status==='running'),index=batch.entries.indexOf(active);
    this.q('title').textContent=active?`Recording ${active.id}`:running?(batch.status==='queued'?'Your batch is queued':'Finishing your analysis'):(batch.status==='interrupted'?'Analysis interrupted':batch.status==='export_failed'?'Excel export needs attention':failed?'Analysis finished with issues':'Analysis complete');
    this.q('subtitle').textContent=active?`${index+1} of ${batch.entries.length} · ${active.video.split('/').pop()}`:batch.message;
    this.q('label').textContent=running?'LIVE ANALYSIS':batch.status==='interrupted'?'ANALYSIS INTERRUPTED':'ANALYSIS FINISHED';
    this.q('batch-count').textContent=`${done} / ${batch.entries.length} complete`;
    const key=JSON.stringify(batch.entries.map(e=>[e.id,e.status,e.run_id]));
    if(key!==this.queueKey){this.queueKey=key;const box=this.host.querySelector('.live-queue-list');box.replaceChildren();batch.entries.forEach((e,i)=>{
      const row=document.createElement('div');row.className='live-queue-item '+e.status;const marker=document.createElement('span');marker.className='queue-marker';marker.textContent=e.status==='complete'?'✓':e.status==='failed'?'!':String(i+1);
      const text=document.createElement('div'),name=document.createElement('strong'),state=document.createElement('small');name.textContent=e.id;state.textContent=({pending:'Waiting',running:'Processing',complete:'Results ready',failed:'Needs attention'})[e.status]||e.status;text.append(name,state);row.append(marker,text);
      if(e.run_id){const button=document.createElement('button');button.textContent='View';button.setAttribute('aria-label',`View completed recording ${e.id}`);button.onclick=()=>{const target=document.getElementById('result-'+e.run_id);if(target)target.scrollIntoView({block:'start',behavior:'smooth'});else this.onViewResult?.(this.batch.id,e.run_id);};row.append(button);}box.append(row);
    });}
    if(running&&!this.timer&&!this.request)this.poll();
    if(!running){clearTimeout(this.timer);this.timer=null;this.request?.abort();this.request=null;this.epoch++;this.q('connection').textContent=batch.status==='interrupted'?'Processing interrupted · completed outputs retained':'Preview finished · review completed recordings below';this.renderFrame();}
  }
  async poll(){
    this.timer=null;if(!this.isRunning()||this.request)return;
    const id=this.batch.id,epoch=this.epoch,controller=new AbortController();this.request=controller;
    try{
      if(document.visibilityState==='hidden'||this.host.closest('[hidden]'))return;
      const params=new URLSearchParams({since:this.revision,image:this.previewHidden?'0':'1'});
      const r=await fetch(`/api/batches/${encodeURIComponent(id)}/live?${params}`,{cache:'no-store',signal:controller.signal});if(!r.ok)throw Error('Preview unavailable');
      const packet=await r.json();if(epoch!==this.epoch||id!==this.batch?.id)return;
      this.connected=true;this.host.classList.remove('disconnected');this.q('connection').textContent=this.previewHidden?'Preview hidden · analysis continues':'Connected · up to 2 preview updates/s';
      const s=packet.snapshot;
      if(s&&s.recording_index===this.batch.current_index){
        if(s.image&&s.revision!==this.revision){const image=new Image(),token=++this.imageToken;image.onload=()=>{if(token!==this.imageToken||epoch!==this.epoch||s.recording_index!==this.batch.current_index)return;this.image=image;this.snapshot=s;this.revision=s.revision;this.renderFrame();};image.onerror=()=>{this.revision='';};image.src=s.image;}
        else if(!packet.unchanged||!this.snapshot){if(s.stage!==this.snapshot?.stage){this.image=null;this.imageToken++;}this.snapshot=s;this.revision=s.revision;this.renderFrame();}
        if(Date.now()/1000-packet.snapshot.updated_at>8)this.q('connection').textContent='Connected · waiting for the next completed frame';
      }
    }catch(e){if(e.name!=='AbortError'&&epoch===this.epoch){this.connected=false;this.host.classList.add('disconnected');this.q('connection').textContent='Preview connection interrupted · retrying';}}
    finally{if(this.request===controller)this.request=null;if(epoch===this.epoch&&this.isRunning())this.timer=setTimeout(()=>this.poll(),600);}
  }
  renderFrame(){
    const s=this.snapshot,stage=s?.stage||'preparing',running=this.isRunning();
    const titles={preparing:'Preparing recording',localizing:'Locating the subject',tracking:'Tracking landmarks',cached:'Using saved predictions',scoring:'Calculating measurements',rendering:'Writing review video',exporting:'Creating Excel table',complete:'Complete',complete_with_errors:'Finished with issues',failed:'Recording failed',export_failed:'Excel export needs attention'};
    this.q('stage').textContent=running?(titles[stage]||stage):'Last processed frame';
    const phase=stage==='cached'?2:this.stages.findIndex(([key])=>key===stage);
    this.host.querySelectorAll('.live-stagebar>div').forEach((n,i)=>{n.classList.toggle('current',running&&i===phase);n.classList.toggle('done',i<phase||['exporting','complete','complete_with_errors'].includes(stage));n.classList.toggle('reused',stage==='cached'&&i===2);});
    const hasFrame=Number.isInteger(s?.frame_index),total=s?.total_frames;
    this.q('frame').textContent=hasFrame?`Frame ${s.frames_done.toLocaleString()}${total?' / '+total.toLocaleString():''}`:'Waiting for a processed frame';
    this.q('position').textContent=this.time(s?.source_time_s);this.q('duration').textContent=this.time(s?.source_duration_s);
    this.q('source-fill').style.width=`${hasFrame&&s.source_duration_s?Math.min(100,s.source_time_s/s.source_duration_s*100):0}%`;
    this.q('pass-label').textContent=titles[stage]||stage;
    const pct=hasFrame&&total?Math.min(100,100*s.frames_done/total):null;
    this.q('percent').textContent=pct===null?'—':`${pct.toFixed(0)}%`;
    if(pct===null)this.q('progress').removeAttribute('value');else{this.q('progress').max=100;this.q('progress').value=pct;}
    this.q('pass-detail').textContent=stage==='localizing'?'Candidate body regions · landmarks are estimated in the next pass.':stage==='rendering'?'Measurements calculated · preserving every frame in the review video.':stage==='tracking'?'DeepLabCut predictions · source coordinates and timestamps retained.':stage==='cached'?'Saved predictions match this recording’s content hash.':s?.message||'Frame progress appears when this pass begins.';
    this.q('image-note').textContent=!running?'Last available preview':stage==='localizing'?'Candidate box · not anatomical landmarks':stage==='rendering'?'Scored landmarks · review export':'Latest processed frame · sampled preview';
    this.empty.hidden=!!this.image&&!this.previewHidden;
    this.q('empty-title').textContent=this.previewHidden?'Preview hidden':!running?'Processing finished':titles[stage]||'Preparing';
    this.q('empty-text').textContent=this.previewHidden?'All frames continue to be processed.':!running?'Completed results and review videos are available below.':stage==='tracking'?'Loading the model or waiting for the first completed prediction.':'The next processed frame will appear here.';
    for(const card of this.host.querySelectorAll('.landmark-card')){const point=s?.landmarks?.[card.dataset.part],state=point?.state||'waiting';card.dataset.state=state;card.querySelector('strong').textContent=Number.isFinite(point?.likelihood)?point.likelihood.toFixed(2):'—';card.querySelector('small').textContent=state==='accepted'?(card.dataset.part==='nose'&&s.nose_scoreable===false?'Not scoreable':'Above cutoff'):state==='uncertain'?'Below cutoff':state==='missing'?(stage==='localizing'?'Not estimated':'Missing'):'Waiting';}
    this.paint();
  }
  paint(){
    const c=this.canvas,ctx=c.getContext('2d'),s=this.snapshot;ctx.clearRect(0,0,c.width,c.height);
    if(!this.image||!s?.crop||this.previewHidden)return;
    const [x1,y1,x2,y2]=s.crop,w=x2-x1,h=y2-y1;c.height=Math.round(c.width*h/w);ctx.drawImage(this.image,0,0,c.width,c.height);ctx.save();ctx.scale(c.width/w,c.height/h);ctx.translate(-x1,-y1);
    const geometry=s.geometry;if(geometry){
      ctx.lineWidth=1;ctx.strokeStyle='rgba(169,222,211,.75)';ctx.beginPath();geometry.arena.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();
      for(const d of geometry.dividers_fraction){const p=ChamberGeometry.project(geometry.arena,d,0),q=ChamberGeometry.project(geometry.arena,d,1);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();}
      const cc=geometry.cup_circles;for(const [side,color] of [['left','#8bd7f2'],['right','#ffbd80']]){const [x,y]=cc[side];ctx.lineWidth=1.7;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=5;ctx.beginPath();ctx.arc(x,y,cc.diameter_px/2,0,2*Math.PI);ctx.stroke();ctx.shadowBlur=0;ctx.font='600 10px -apple-system, sans-serif';ctx.fillText(side.toUpperCase(),x-13,y-cc.diameter_px/2-8);}
    }
    if(s.bbox){const [x,y,right,bottom]=s.bbox;ctx.setLineDash([5,4]);ctx.strokeStyle='#98ead2';ctx.lineWidth=1.5;ctx.strokeRect(x,y,right-x,bottom-y);ctx.setLineDash([]);ctx.font='10px -apple-system, sans-serif';ctx.fillStyle='#baf4e2';ctx.fillText('SUBJECT CANDIDATE',x,y-8);}
    for(const [a,b] of [['nose','center'],['center','tail_base']]){const p=s.landmarks?.[a],q=s.landmarks?.[b];if(p?.state==='accepted'&&q?.state==='accepted'){ctx.strokeStyle='rgba(243,250,240,.65)';ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();}}
    for(const [part,p] of Object.entries(s.landmarks||{})){if(p.state==='missing')continue;const color=this.colors[part]||'#fff';ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5;
      if(p.state==='accepted'){ctx.shadowColor=color;ctx.shadowBlur=9;ctx.beginPath();ctx.arc(p.x,p.y,3.5,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;ctx.strokeStyle='rgba(10,26,25,.95)';ctx.stroke();ctx.strokeStyle=color;ctx.globalAlpha=.45;ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1;}
      else{ctx.setLineDash([2,3]);ctx.beginPath();ctx.arc(p.x,p.y,6,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(p.x-2,p.y-2);ctx.lineTo(p.x+2,p.y+2);ctx.moveTo(p.x+2,p.y-2);ctx.lineTo(p.x-2,p.y+2);ctx.stroke();}
    }ctx.restore();
  }
}
window.LiveAnalysisView=LiveAnalysisView;
