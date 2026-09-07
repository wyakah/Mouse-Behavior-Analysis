const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
function harness(count){
 const context={window:{}};vm.runInNewContext(fs.readFileSync('static/recording-setup.js','utf8'),context);
 const view=Object.create(context.window.RecordingSetup.prototype),entries=Array.from({length:count},(_,i)=>({id:`M${i}`,video:`${i}.mp4`}));let seedCalls=0,saves=0;
 Object.assign(view,{busy:false,input:{value:'files'},status(message){this.message=message;},options:{entries:()=>entries,seed:async video=>{seedCalls++;return {id:'New',video};},change:async()=>{saves++;}}});
 return {view,entries,counts:()=>({seedCalls,saves})};
}
test('over-limit file selection is rejected before uploading any file',async()=>{
 const h=harness(149);await h.view.upload([{},{}]);assert.match(h.view.message,/at most 1 more/);assert.equal(h.entries.length,149);assert.equal(h.view.input.value,'');assert.equal(h.view.busy,false);assert.equal(h.counts().seedCalls,0);
});
test('150th video is saved; 151st is rejected before seeding',async()=>{
 const h=harness(149);await h.view.add('new.mp4');assert.equal(h.entries.length,150);assert.equal(h.entries[149].sex,'unknown');assert.equal(h.entries[149].genotype,'');assert.equal(h.counts().saves,1);await assert.rejects(h.view.add('extra.mp4'),/150 videos/);assert.equal(h.counts().seedCalls,1);
});
test('Continue needs unique nonempty mouse IDs and no more than 150 rows',()=>{
 const h=harness(150);assert.equal(h.view.valid(),true);h.entries[0].id=' ';assert.equal(h.view.valid(),false);h.entries[0].id='M1';assert.equal(h.view.valid(),false);h.entries[0].id='M0';h.entries.push({id:'M151'});assert.equal(h.view.valid(),false);
});
test('re-adding a selected workspace video is a no-op',async()=>{
 const h=harness(1);await h.view.add('0.mp4');assert.equal(h.entries.length,1);assert.equal(h.counts().saves,0);
});
test('Three Chamber requires stranger position; stereotypy does not',()=>{
 const h=harness(1);assert.equal(h.view.valid(),true);
 h.view.options.strangerPosition=true;assert.equal(h.view.valid(),false);
 h.entries[0].config={target_side:'right'};assert.equal(h.view.valid(),true);
 h.entries[0].stranger_side='';assert.equal(h.view.valid(),false);
 h.entries[0].stranger_side='left';assert.equal(h.view.valid(),true);
});
test('eight videos upload as bounded binary chunks and all receive metadata rows',async()=>{
 const context={window:{}};let starts=0,chunks=0;
 context.fetch=async(url,options)=>({ok:true,status:200,json:async()=>{
  if(url.endsWith('/start'))return {id:String(++starts),chunk_bytes:3};
  if(url.includes('/chunk')){chunks++;assert.ok(options.body.length<=3);return {offset:Number(new URL('http://local'+url).searchParams.get('offset'))+options.body.length};}
  return {name:`video${starts}.mp4`};
 }});
 vm.runInNewContext(fs.readFileSync('static/recording-setup.js','utf8'),context);
 const entries=[],view=Object.create(context.window.RecordingSetup.prototype);
 Object.assign(view,{busy:false,input:{value:''},render(){},status(){},options:{entries:()=>entries,seed:async video=>({video,id:video}),change:async()=>{}}});
 await view.upload(Array.from({length:8},(_,i)=>({name:`${i}.mp4`,size:7,slice:(a,b)=>Buffer.alloc(b-a)})));
 assert.equal(starts,8);assert.equal(chunks,24);assert.equal(entries.length,8);assert.equal(view.busy,false);
});
