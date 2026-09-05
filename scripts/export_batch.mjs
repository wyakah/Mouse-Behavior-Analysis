import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const [input, destination] = process.argv.slice(2);
const batch = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
const results = wb.worksheets.add('Results');
const setup = wb.worksheets.add('Setup');
const events = wb.worksheets.add('Bouts');
const safe = v => typeof v === 'string' && /^[=+@-]/.test(v) ? "'" + v : v ?? null;
const col = i => {let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
const priority = ['analyzed_seconds','left_chamber_seconds','center_chamber_seconds','right_chamber_seconds','left_nose_seconds','right_nose_seconds','unknown_chamber_seconds','outside_chamber_seconds','nose_unscoreable_seconds','nose_scoreable_fraction','center_valid_fraction','preference_index','target_side','cup_diameter_px'];
const metadata = new Set(['video','tracking_file','geometry_source','zone_geometry_source','occupancy_bounds_note']);
const keys = [...new Set([...priority,...batch.entries.flatMap(e=>Object.keys(e.summary||{}))])].filter(k=>!metadata.has(k));
const label = k => k.replace(/_seconds$/,' (s)').replace(/_fraction$/,' (fraction)').replace(/_px$/,' (px)').replace(/_/g,' ');
const rows = batch.entries.map(e=>[e.id,path.basename(e.video),e.status,...keys.map(k=>e.summary?.[k]),e.error||null]);
function table(sheet,headers,data,name){
  const count=Math.max(1,data.length),end=col(headers.length-1);
  sheet.showGridLines=false;
  sheet.getRange(`A1:${end}${count+1}`).format.font={name:'Arial',size:10,color:'#22313D'};
  sheet.getRange(`A1:${end}1`).values=[headers];
  if(data.length)sheet.getRange(`A2:${end}${data.length+1}`).values=data.map(row=>row.map(safe));
  sheet.getRange(`A1:${end}1`).format={fill:'#34495E',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:46,verticalAlignment:'center'};
  sheet.getRange(`A2:${end}${count+1}`).format.rowHeight=36;
  sheet.getRange(`A2:${end}${count+1}`).format.verticalAlignment='center';
  sheet.getRange(`A1:${end}${count+1}`).format.columnWidth=20;
  sheet.getRange(`A2:${end}${count+1}`).setNumberFormat('0.00');
  sheet.getRange(`A1:A${count+1}`).format.columnWidth=16;
  if(data.length)sheet.tables.add(`A1:${end}${data.length+1}`,true,name);
  sheet.freezePanes.freezeRows(1);
  return count;
}
table(results,['ID','Recording','Status',...keys.map(label),'Error'],rows,'BatchResults');
results.getRange(`B1:B${rows.length+1}`).format.columnWidth=49;
results.getRange(`B2:B${rows.length+1}`).format.wrapText=true;
results.getRange(`A2:C${rows.length+1}`).setNumberFormat('@');
results.getRange(`${col(keys.length+3)}1:${col(keys.length+3)}${rows.length+1}`).format.columnWidth=52;
results.getRange(`${col(keys.length+3)}2:${col(keys.length+3)}${rows.length+1}`).format.wrapText=true;
for(const [i,k] of keys.entries()){
  if(k.endsWith('_fraction')||k==='center_occupancy_coverage')results.getRange(`${col(i+3)}2:${col(i+3)}${rows.length+1}`).setNumberFormat('0.0%');
  if(['decoded_frames','scored_frames','intervals_over_twice_median'].includes(k))results.getRange(`${col(i+3)}2:${col(i+3)}${rows.length+1}`).setNumberFormat('0');
}
const setupRows=batch.entries.map(e=>{const c=e.config,cc=c.cup_circles;return [e.id,cc.diameter_px,...cc.left,...cc.right,...c.dividers_fraction,c.pcutoff,JSON.stringify(c.arena),e.summary?.geometry_source||c.geometry_source||'User-reviewed floor',e.summary?.zone_geometry_source||c.zone_geometry_source||'Linked circles',e.manifest?.video_sha256,e.manifest?.tracks_sha256,e.run_id?`http://127.0.0.1:8765/outputs/${e.run_id}/review.mp4`:null];});
table(setup,['ID','Shared diameter (px)','Left center x (px)','Left center y (px)','Right center x (px)','Right center y (px)','Left divider fraction','Right divider fraction','Likelihood cutoff','Floor corners (source pixels)','Floor provenance','Zone provenance','Video SHA256','Tracks SHA256','Annotated video'],setupRows,'BatchSetup');
setup.getRange(`J1:O${setupRows.length+1}`).format.columnWidth=55;
setup.getRange(`J2:O${setupRows.length+1}`).format.wrapText=true;
setup.getRange(`A2:A${setupRows.length+1}`).setNumberFormat('@');
setup.getRange(`A2:O${setupRows.length+1}`).format.rowHeight=70;
const notes=[
 ['Scoring rule','Accepted free-subject nose inside either circle, boundary included. Both circles share one diameter in source-video pixels.'],
 ['Consistency','The batch uses one diameter and likelihood cutoff. Centers, floor corners and dividers are reviewed per recording. Equal pixel diameters assume consistent camera framing; they do not establish equal physical distance.'],
 ['Time window','First 600 seconds, or the available recording duration if shorter. Actual presentation timestamps weight each frame.'],
 ['Missing observations','Low-confidence and missing landmarks are not interpolated. Failed-video measurements and unavailable physical distances are blank, never zero.'],
 ['Coverage','Confidence coverage is not validated anatomical accuracy. Nose scoreability also requires the nose to lie inside the arena.'],
 ['Preference index','(Target cup nose time − other cup nose time) / (sum of cup nose times). Blank if target is unspecified or the denominator is zero.'],
 ['Occupancy bounds','Lower/upper chamber times allocate unknown tracking time only; they do not include identity, geometry or landmark errors.'],
 ['Exports','Each completed video also has per-frame CSV, summary JSON/CSV, bouts, geometry, run provenance and annotated video in the local Results view.'],
 ['Behavior interpretation','Nose-in-zone time is a proximity proxy. It does not independently establish sniffing or social investigation.'],
 ['Assay reference','https://doi.org/10.1016/j.heliyon.2024.e36352'],
 ['Method distinction','These linked pixel circles are a user-defined region method, not the publication’s calibrated 1 cm external cup ring.']
];
const nr=3;
setup.getRange('Q1').values=[['Scoring notes']];
setup.getRange('Q1').format.font={name:'Arial',size:12,bold:true};
setup.getRange(`Q${nr}:R${nr+notes.length-1}`).values=notes;
setup.getRange(`Q${nr}:R${nr+notes.length-1}`).format={font:{name:'Arial',size:10},wrapText:true,rowHeight:68,verticalAlignment:'center'};
setup.getRange(`Q1:Q${nr+notes.length-1}`).format.columnWidth=24;
setup.getRange(`R1:R${nr+notes.length-1}`).format.columnWidth=72;
const boutRows=batch.entries.flatMap(e=>(e.bouts||[]).map(b=>[e.id,b.behavior,b.start_s,b.duration_s,b.start_frame,b.end_frame_exclusive]));
table(events,['ID','Zone event','Start (s)','Duration (s)','Start frame','End frame (exclusive)'],boutRows,'BatchBouts');
events.getRange(`A2:B${Math.max(2,boutRows.length+1)}`).setNumberFormat('@');
events.getRange(`E2:F${Math.max(2,boutRows.length+1)}`).setNumberFormat('0');
if(!boutRows.length)events.getRange('A3').values=[['No detected cup-zone bouts.']];
console.log((await wb.inspect({kind:'table',range:`Results!A1:K${Math.min(6,rows.length+1)}`,tableMaxRows:6,tableMaxCols:11,maxChars:5000})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},summary:'Formula error scan',maxChars:1500})).ndjson);
for(const [sheetName,range,file] of [['Results',`A1:J${Math.min(8,rows.length+1)}`,'results-preview'],['Results',`K1:V${Math.min(8,rows.length+1)}`,'metrics-preview'],['Results',`W1:${col(keys.length+3)}${Math.min(8,rows.length+1)}`,'other-metrics-preview'],['Setup',`A1:I${Math.min(8,rows.length+1)}`,'setup-preview'],['Setup',`J1:O${Math.min(8,rows.length+1)}`,'provenance-preview'],['Setup',`Q1:R${nr+notes.length-1}`,'notes-preview'],['Bouts',`A1:F${Math.min(12,Math.max(3,boutRows.length+1))}`,'bouts-preview']]){
  const png=await wb.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(path.join(path.dirname(destination),file+'.png'),new Uint8Array(await png.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(destination);
console.log(destination);
