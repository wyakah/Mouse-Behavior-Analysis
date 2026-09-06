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
 const h=harness(19);await h.view.upload([{},{}]);assert.match(h.view.message,/at most 1 more/);assert.equal(h.entries.length,19);assert.equal(h.view.input.value,'');assert.equal(h.view.busy,false);assert.equal(h.counts().seedCalls,0);
});
test('twentieth video is saved; twenty-first is rejected before seeding',async()=>{
 const h=harness(19);await h.view.add('new.mp4');assert.equal(h.entries.length,20);assert.equal(h.entries[19].sex,'unknown');assert.equal(h.entries[19].genotype,'');assert.equal(h.counts().saves,1);await assert.rejects(h.view.add('extra.mp4'),/20 videos/);assert.equal(h.counts().seedCalls,1);
});
test('Continue needs unique nonempty mouse IDs and no more than twenty rows',()=>{
 const h=harness(20);assert.equal(h.view.valid(),true);h.entries[0].id=' ';assert.equal(h.view.valid(),false);h.entries[0].id='M1';assert.equal(h.view.valid(),false);h.entries[0].id='M0';h.entries.push({id:'M21'});assert.equal(h.view.valid(),false);
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
