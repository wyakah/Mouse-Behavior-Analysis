/* Buffered, all-frame playback. Processing never waits for the viewer. */
class AnalysisStreamPlayer {
  constructor(video, changed, ended) {
    this.video=video;this.changed=changed;this.ended=ended;this.epoch=0;this.timer=null;
    this.hidden=false;this.state='waiting';this.manifest=null;this.index=null;this.batchId=null;
    for(const event of ['timeupdate','waiting','playing','pause','seeking','seeked','loadedmetadata','ended']) {
      video.addEventListener(event,()=>{
        if(event==='waiting')this.buffering=true;
        if(event==='playing')this.buffering=false;
        this.changed();
        if(event==='ended'&&this.manifest?.status==='complete')this.ended();
      });
    }
  }
  select(batchId,index) {
    if(this.batchId===batchId&&this.index===index)return;
    this.release();this.batchId=batchId;this.index=index;this.manifest=null;this.next=0;
    this.generation=null;this.state='waiting';this.message='Preparing annotated footage';this.autoStart=true;
    this.buffering=false;this.poll();this.changed();
  }
  release() {
    this.epoch++;clearTimeout(this.timer);this.controller?.abort();this.controller=null;
    this.video.pause();this.video.removeAttribute('src');this.video.load();
    if(this.url)URL.revokeObjectURL(this.url);
    this.url=null;this.media=null;this.buffer=null;
  }
  resetMedia() {
    this.video.pause();this.video.removeAttribute('src');this.video.load();
    if(this.url)URL.revokeObjectURL(this.url);
    this.url=null;this.media=null;this.buffer=null;this.next=0;this.autoStart=true;
  }
  async open(mime,signal) {
    const Media=window.MediaSource||window.ManagedMediaSource;
    if(!Media||!Media.isTypeSupported(mime))throw new Error('This browser cannot play the live stream. The final review video will still be available.');
    const media=new Media();this.media=media;
    this.video.disableRemotePlayback=true;
    const ready=this.event(media,'sourceopen',signal);
    this.url=URL.createObjectURL(media);this.video.src=this.url;
    await ready;
    try{this.buffer=media.addSourceBuffer(mime);this.buffer.mode='segments';}
    catch{throw new Error('This browser cannot play the live stream. The final review video will still be available.');}
  }
  async maybePlay(manifest) {
    const buffered=this.video.buffered.length?this.video.buffered.end(this.video.buffered.length-1)-this.video.currentTime:0;
    if(!this.hidden&&this.autoStart&&(buffered>=3||(buffered>0&&manifest.status!=='streaming'))){
      this.autoStart=false;
      try{await this.video.play();}catch{this.message='Press play to watch the annotated footage.';}
    }
  }
  event(target,type,signal) {
    return new Promise((resolve,reject)=>{
      const cleanup=()=>{target.removeEventListener(type,done);target.removeEventListener('error',error);signal.removeEventListener('abort',abort);};
      const done=()=>{cleanup();resolve();};
      const error=()=>{cleanup();reject(new Error('Video segment could not be decoded.'));};
      const abort=()=>{cleanup();reject(new DOMException('Aborted','AbortError'));};
      target.addEventListener(type,done,{once:true});target.addEventListener('error',error,{once:true});signal.addEventListener('abort',abort,{once:true});
      if(signal.aborted)abort();
    });
  }
  async poll() {
    if(this.controller)return;
    const epoch=this.epoch;clearTimeout(this.timer);
    if(this.hidden)return;
    const controller=new AbortController();this.controller=controller;let again=true;
    try {
      const response=await fetch(`${this.apiBase||'/api/batches'}/${encodeURIComponent(this.batchId)}/video/${this.index}`,{cache:'no-store',signal:controller.signal});
      if(!response.ok)throw new Error('Preview connection interrupted. Reconnecting…');
      const packet=await response.json();if(epoch!==this.epoch)return;
      const manifest=packet.manifest;
      if(!manifest) {
        this.state='waiting';this.message=packet.entry_status==='pending'?'This recording is waiting in the queue.':'Preparing the model and annotated footage';
        if(!['queued','running'].includes(packet.status)||['failed','complete'].includes(packet.entry_status)){
          this.state='unavailable';this.message='Live footage is unavailable. Check the recording’s results below.';again=false;
        }
      } else {
        if(this.generation&&this.generation!==manifest.generation)this.resetMedia();
        this.manifest=manifest;this.generation=manifest.generation;
        if(manifest.segments.length&&!this.media)await this.open(manifest.mime,controller.signal);
        for(;this.next<manifest.segments.length;) {
          if(this.hidden)break;
          const segment=manifest.segments[this.next];
          const r=await fetch(manifest.base_url+segment.file,{signal:controller.signal});
          if(!r.ok)throw new Error('Waiting for the next video segment. Reconnecting…');
          const data=await r.arrayBuffer();if(epoch!==this.epoch)return;
          this.buffer.timestampOffset=segment.start_s;
          const appended=this.event(this.buffer,'updateend',controller.signal);
          try{this.buffer.appendBuffer(data);}catch(error){controller.abort();await appended.catch(()=>{});throw error;}
          await appended;
          this.next++;
          this.state='ready';this.message='';
          await this.maybePlay(manifest);
          this.changed();
        }
        await this.maybePlay(manifest);
        if(manifest.status!=='streaming'&&this.next===manifest.segments.length) {
          if(this.media?.readyState==='open'&&!this.buffer.updating)this.media.endOfStream();
          again=false;
          if(manifest.status==='failed'){this.state='partial';this.message=manifest.error||'Preview stopped early; analysis results remain available.';}
          else if(!manifest.segments.length){this.state='unavailable';this.message='No preview frames were produced.';}
        }
      }
    } catch(error) {
      if(epoch!==this.epoch||error.name==='AbortError')return;
      this.message=error.message;
      if(error.message.includes('cannot play')||error.message.includes('decoded')){this.state='unavailable';again=false;}
      else this.state='reconnecting';
    } finally {
      if(this.controller===controller)this.controller=null;
      if(epoch===this.epoch){this.changed();if(again&&!this.hidden)this.timer=setTimeout(()=>this.poll(),800);}
    }
  }
  setHidden(hidden) {
    if(this.hidden===hidden)return;
    this.hidden=hidden;
    if(hidden){this.resumeOnShow=!this.video.paused;this.video.pause();clearTimeout(this.timer);}
    else {if(this.resumeOnShow)this.video.play().catch(()=>{});if(!this.controller)this.poll();}
  }
}
window.AnalysisStreamPlayer=AnalysisStreamPlayer;
