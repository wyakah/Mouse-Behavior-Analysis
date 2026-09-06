const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const context={};vm.createContext(context);vm.runInContext(fs.readFileSync('static/social-metrics.js','utf8'),context);
test('eight ordered metrics distinguish chamber and zone denominators',()=>{
 const s={stranger_side:'right',left_chamber_seconds:200,center_chamber_seconds:100,right_chamber_seconds:100,left_nose_seconds:8,right_nose_seconds:20,nose_scoreable_fraction:.8};
 const cells=context.threeChamberMetrics(s);
 assert.equal(cells.length,8);assert.equal(cells[0].label,'Left Chamber (Object) Time (s)');assert.equal(cells[3].label,'Chamber Stranger Interaction %');assert.equal(cells[3].value,'25.00%');assert.equal(cells[5].value,'N/A');assert.equal(cells[7].value,'5.00%');
 const missing=context.threeChamberMetrics({...s,nose_scoreable_fraction:0});assert.equal(missing[3].value,'25.00%');assert.equal(missing[7].value,'—');
 const empty=context.threeChamberMetrics({...s,left_chamber_seconds:0,center_chamber_seconds:0,right_chamber_seconds:0});assert.equal(empty[3].value,'—');assert.equal(empty[7].value,'—');
});
