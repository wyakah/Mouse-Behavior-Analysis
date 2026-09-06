import fs from 'node:fs/promises';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const [report, out, comparisonPath] = process.argv.slice(2);
const data = JSON.parse(await fs.readFile(comparisonPath || `${report}/comparison.json`, 'utf8'));
const videos = await Promise.all(data.videos.map(v => fs.readFile(`${report}/${v.mouse_id}.json`, 'utf8').then(JSON.parse)));
const wb = Workbook.create();
const col = i => String.fromCharCode(65+i);
function table(name, headers, rows, widths, note) {
  const s = wb.worksheets.add(name), last = col(headers.length-1), end = rows.length+5;
  s.showGridLines = false;
  s.getRange(`A1:${last}${end}`).format.font = {name:'Arial',size:10,color:'#263a31'};
  s.getRange('A1').values = [[name]];
  s.getRange('A1').format.font = {name:'Arial',size:16,bold:true};
  s.getRange('A2').values = [[note]];
  s.getRange(`A5:${last}5`).values = [headers];
  s.getRange(`A5:${last}5`).format = {fill:'#344b40',font:{name:'Arial',size:10,bold:true,color:'#ffffff'},wrapText:true,rowHeight:44};
  s.getRange(`A6:${last}${end}`).values = rows;
  s.getRange(`A6:${last}${end}`).format.rowHeight = 28;
  widths.forEach((w,i) => s.getRange(`${col(i)}1:${col(i)}${end}`).format.columnWidth=w);
  s.tables.add(`A5:${last}${end}`,true,name.replaceAll(' ','')+'Table');
  return s;
}
const coverage = table('Coverage', ['Mouse ID','Window (s)','Old body proposal (s)','New body eligible (s)','Nose available (s)','Axis available (s)','Scene flagged (s)','Nose / window','Axis / window'],
  data.videos.map(v => [v.mouse_id,1200,v.baseline_body_proposed_seconds,v.appearance_available_seconds,v.nose_available_seconds,v.axis_available_seconds,v.scene_flag_seconds,null,null]),
  [12,14,20,20,20,20,20,19,19], 'First 1,200 seconds. Availability is not accuracy. Old and new body criteria differ.');
coverage.getRange('B6:G10').setNumberFormat('0.0');
coverage.getRange('H6:I6').formulas = [['=E6/B6','=F6/B6']];
coverage.getRange('H6:I10').fillDown();
coverage.getRange('H6:I10').setNumberFormat('0.0%');
const behaviors = table('Behavior diagnostics', ['Mouse ID','Behavior','Raw baseline candidate (s)','Scene/body eligible candidate (s)','Pose conflict (s)','Accepted time (s)','Interpretation'],
  videos.flatMap(v => v.behavior.map(b => [v.mouse_id,b.behavior,b.baseline_raw_candidate_seconds,b.scene_eligible_candidate_seconds,b.behavior==='rearing'?b.pose_conflict_candidate_seconds:null,null,v.mouse_id==='743'?'Withheld under prior visual audit':'Unvalidated diagnostic candidates'])),
  [12,23,23,26,20,20,42], 'Fewer candidate seconds is not evidence of improved accuracy. Zero does not establish absence.');
behaviors.getRange('C6:F35').setNumberFormat('0.0');
behaviors.getRange('G6:G35').format.wrapText=true;
behaviors.getRange('A6:G35').format.rowHeight=34;
behaviors.freezePanes.freezeRows(5);
const lighting = table('Lighting comparison', ['Mouse ID','Input','Paired frames','Nose above cutoff','Axis available','Landmark error (px)'],
  videos.flatMap(v=>v.variants.map(r=>[v.mouse_id,r.mode,r.sampled_frames,r.nose_available,r.axis_available,null])),
  [12,18,19,23,20,24], 'Paired systematic frames. No labeled coordinate reference. Raw remains the default.');
lighting.getRange('C6:E20').setNumberFormat('0');
const methods = wb.worksheets.add('Methods');methods.showGridLines=false;
const notes = [
  ['Window','The first 1,200 seconds per mouse. Five recordings. No group statistics.'],
  ['Behavior models','Frozen video classifier scores reproduced from cached features. No new fitted model or accuracy gain.'],
  ['Pose','Side-view SuperAnimal Quadruped HRNet-W32 at 5 Hz. Nose, neck, mid-back, tail base and four paws. Confidence cutoff 0.6 is not accuracy.'],
  ['Body proposals','Untrained dark-body appearance proposals plus conservative scene and ambiguity checks. Can fail near fixtures and when the view changes.'],
  ['Scene flags','Static cage-edge agreement below 0.45. May flag a valid but reframed cage. These are review flags, not independently verified unusable time.'],
  ['Motion','Every source frame contributes optical flow. Pose speeds require consecutive available points. No interpolated hidden points.'],
  ['Rearing conflict','A high video rearing score with an apparently horizontal reliable pose. This is a review cue, not an accepted correction.'],
  ['Mouse 743','All experimental behavior totals remain withheld under the prior visual audit. Diagnostic numbers are retained for investigating failures.'],
  ['Missing values','Accepted behavior times and landmark errors are blank because no independent human references are available. Blank is not zero.'],
  ['Training','Corrected landmarks and explicit behavior intervals are needed for supervised pose fine-tuning and fitting the fusion model. No pseudo-labels were accepted.'],
  ['Comparison report','http://127.0.0.1:8765/reports/stereotypy-robustness/index.html'],
  ['Original report','http://127.0.0.1:8765/reports/stereotypy-five-video/index.html'],
];
methods.getRange('A1').values=[['Methods and interpretation']];
methods.getRange('A1').format.font={name:'Arial',size:16,bold:true};
methods.getRange(`A4:B${notes.length+3}`).values=notes;
methods.getRange(`A4:B${notes.length+3}`).format={font:{name:'Arial',size:10},wrapText:true,rowHeight:50};
methods.getRange(`A4:A${notes.length+3}`).format.columnWidth=24;
methods.getRange(`B4:B${notes.length+3}`).format.columnWidth=100;
await fs.mkdir(out,{recursive:true});
console.log((await wb.inspect({kind:'table',range:'Coverage!A5:I10',include:'values,formulas',tableMaxRows:6,tableMaxCols:9,maxChars:2200})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:20},summary:'Formula error scan'})).ndjson);
for (const [sheetName,range] of [['Coverage','A1:I10'],['Behavior diagnostics','A1:G12'],['Lighting comparison','A1:F20'],['Methods','A1:B15']]) {
  const png=await wb.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(`${out}/${sheetName.replaceAll(' ','-')}.png`,new Uint8Array(await png.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(`${out}/robustness-comparison.xlsx`);
