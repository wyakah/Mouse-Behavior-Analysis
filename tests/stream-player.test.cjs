const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function setup() {
  const video=new EventTarget();Object.assign(video,{paused:true,currentTime:0,ended:false,playCount:0,bufferEnd:0});
  video.buffered={get length(){return video.bufferEnd?1:0;},end(){return video.bufferEnd;}};
  video.play=async()=>{video.paused=false;video.playCount++;};video.pause=()=>{video.paused=true;};
  video.load=()=>{};video.removeAttribute=()=>{video.bufferEnd=0;};
  const requests=[];let afterAppend=()=>{};
  class Buffer extends EventTarget {
    constructor(){super();this.updating=false;this.timestampOffset=0;this.appends=0;}
    appendBuffer(data){this.appends++;this.updating=true;video.bufferEnd=this.timestampOffset+2;afterAppend();queueMicrotask(()=>{this.updating=false;this.dispatchEvent(new Event('updateend'));});}
  }
  class Media extends EventTarget {
    static isTypeSupported(){return true;}
    constructor(){super();this.readyState='closed';}
    addSourceBuffer(){this.buffer=new Buffer();return this.buffer;}
    endOfStream(){this.readyState='ended';}
  }
  let manifest={generation:'a',status:'streaming',mime:'video/mp4',base_url:'/segments/',segments:[{file:'one',start_s:0}],available_until_s:2};
  const sandbox={window:{MediaSource:Media},URL:{createObjectURL(media){queueMicrotask(()=>{media.readyState='open';media.dispatchEvent(new Event('sourceopen'));});return 'blob:test';},revokeObjectURL(){}},setTimeout,clearTimeout,AbortController,DOMException,
    fetch:async(url)=>{requests.push(url);return {ok:true,json:async()=>({manifest,status:'running',entry_status:'running'}),arrayBuffer:async()=>new ArrayBuffer(1)};}};
  vm.runInNewContext(fs.readFileSync('static/stream-player.js','utf8'),sandbox);
  const player=new sandbox.window.AnalysisStreamPlayer(video,()=>{},()=>{});
  player.batchId='batch';player.index=0;player.autoStart=true;player.next=0;
  return {player,video,requests,setManifest(m){manifest={...manifest,...m};},onAppend(fn){afterAppend=fn;}};
}

test('a short completed clip starts even when its segment arrived before completion',async()=>{
  const h=setup();try {
    await h.player.poll();assert.equal(h.video.playCount,0);assert.equal(h.player.next,1);
    h.setManifest({status:'complete'});await h.player.poll();
    assert.equal(h.video.playCount,1);assert.equal(h.player.buffer.appends,1);assert.equal(h.player.media.readyState,'ended');
  }finally{h.player.release();}
});

test('hiding during append commits it once and resumes without duplicate segments',async()=>{
  const h=setup();try {
    h.onAppend(()=>h.player.setHidden(true));await h.player.poll();
    assert.equal(h.player.next,1);assert.equal(h.video.playCount,0);
    h.onAppend(()=>{});h.setManifest({status:'complete'});h.player.hidden=false;await h.player.poll();
    assert.equal(h.player.buffer.appends,1);assert.equal(h.video.playCount,1);
  }finally{h.player.release();}
});

test('status refreshes preserve a manually paused player and buffered footage',async()=>{
  const h=setup();try {
    h.setManifest({status:'complete',segments:[{file:'one',start_s:0},{file:'two',start_s:2}],available_until_s:4});
    await h.player.poll();const media=h.player.media;h.video.pause();h.video.currentTime=1.25;
    await h.player.poll();assert.equal(h.player.media,media);assert.equal(h.video.currentTime,1.25);
    assert.equal(h.video.paused,true);assert.equal(h.video.playCount,1);assert.equal(h.player.buffer.appends,2);
  }finally{h.player.release();}
});
