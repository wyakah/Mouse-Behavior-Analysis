import fs from 'node:fs/promises';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const [input, outputDir] = process.argv.slice(2);
const data = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
const durations = wb.worksheets.add('Candidate durations');
const segments = wb.worksheets.add('Candidate segments');
const methods = wb.worksheets.add('Methods');
const safe = v => typeof v === 'string' && /^[=+@-]/.test(v) ? "'" + v : v ?? null;
const column = i => String.fromCharCode(65 + i);
function table(sheet, headers, rows, widths) {
  const last = column(headers.length - 1), end = rows.length + 5;
  sheet.showGridLines = false;
  sheet.getRange(`A1:${last}${end}`).format.font = {name:'Arial', size:10, color:'#263a31'};
  sheet.getRange('A1').values = [[sheet.name]];
  sheet.getRange('A1').format.font = {name:'Arial', size:16, bold:true};
  sheet.getRange('A2').values = [['Unreviewed model candidates. First 1,200 seconds per mouse. No statistical tests.']];
  sheet.getRange(`A5:${last}5`).values = [headers];
  if (rows.length) sheet.getRange(`A6:${last}${end}`).values = rows.map(r => r.map(safe));
  sheet.getRange(`A5:${last}5`).format = {fill:'#344b40',font:{name:'Arial',size:10,bold:true,color:'#ffffff'},wrapText:true,rowHeight:44,verticalAlignment:'center'};
  sheet.getRange(`A6:${last}${end}`).format.rowHeight = 24;
  sheet.getRange(`A6:${last}${end}`).format.verticalAlignment = 'center';
  widths.forEach((width,i) => sheet.getRange(`${column(i)}1:${column(i)}${end}`).format.columnWidth = width);
  if(rows.length) sheet.tables.add(`A5:${last}${end}`,true,sheet===durations?'CandidateDurations':'CandidateSegments');
  sheet.freezePanes.freezeRows(5);
  return end;
}
const rows = data.videos.flatMap(v => v.summary.map(s => [v.mouse_id,s.behavior,s.candidate_seconds,s.candidate_segments,s.sampled_coverage_s,null,s.threshold,s.seconds_at_lower_threshold,s.seconds_at_higher_threshold,s.accepted_seconds,v.localization_audit_passed?s.evidence:'Withheld: failed localization audit']));
const end = table(durations,['Mouse ID','Behavior','Candidate time (s)','Candidate segments','Eligible time (s)','Candidate / eligible time','Threshold','Time at threshold −0.10 (s)','Time at threshold +0.10 (s)','Accepted time (s)','Evidence limitation'],rows,[12,22,18,18,14,18,14,22,22,18,58]);
durations.getRange(`C6:C${end}`).setNumberFormat('0.0');
durations.getRange(`D6:D${end}`).setNumberFormat('0');
durations.getRange(`E6:E${end}`).setNumberFormat('0.0');
durations.getRange(`G6:G${end}`).setNumberFormat('0.00');
durations.getRange(`H6:J${end}`).setNumberFormat('0.0');
durations.getRange(`K6:K${end}`).format.wrapText = true;
durations.getRange(`A6:K${end}`).format.rowHeight = 34;
durations.getRange('F6').formulas = [['=IF(ISNUMBER(C6),C6/E6,"")']];
durations.getRange(`F6:F${end}`).fillDown();
durations.getRange(`F6:F${end}`).setNumberFormat('0.0%');
const events = data.videos.flatMap(v => v.bouts.map(b => [v.mouse_id,b.behavior,b.start_s,b.end_s,null,'Unreviewed candidate']));
const eventEnd = table(segments,['Mouse ID','Behavior','Start (s)','End (s)','Duration (s)','Status'],events,[12,24,18,18,18,30]);
if(events.length){segments.getRange(`C6:E${eventEnd}`).setNumberFormat('0.000');segments.getRange('E6').formulas=[['=D6-C6']];segments.getRange(`E6:E${eventEnd}`).fillDown();}
methods.showGridLines = false;
const notes = [
  ['Analysis window','First 1,200 seconds of source time. All five sources contain the full window.'],
  ['Interpretation','Candidate seconds are time above a fixed model threshold. No accepted behavioral labels or local accuracy estimates are available.'],
  ['Quality exclusions','Ambiguous, missing, edge or camera-motion body proposals are excluded. A failed whole-recording localization audit withholds all six candidate totals. Blank candidate times mean rejected, not absent.'],
  ['Overlapping behaviors','Each behavior is evaluated separately. Their candidate times may sum to more than 1,200 seconds.'],
  ['Segments','Contiguous threshold crossings. These are not validated behavioral bouts, jump counts or completed circles. No gap merging or minimum duration filter.'],
  ['Priority heads','Grooming, digging and rearing: frozen DINOv2-small, native optical flow and temporal classifiers. Sampled at 2 Hz.'],
  ['Weak heads','Gnawing, jumping and circling: earlier MobileNet six-head model. Weak video-bag supervision, no independent positive validation. Default threshold 0.50.'],
  ['Temporal resolution','Priority scores held until the next 0.5-second sample. Weak scores held across their 2-second window. Every source frame is rendered for review.'],
  ['Sensitivity','Thresholds shifted by ±0.10. This is not a confidence interval.'],
  ['Missing values','Blank accepted times mean unreviewed. Blank candidate times mean a failed localization audit. Neither means zero. Zero candidate time does not establish absence.'],
  ['Validation','External validation target not met. Local accuracy requires separately reviewed reference labels.'],
  ['Training and statistics','No new fitting, pseudo-label training, hypothesis tests or group statistics were performed.'],
  ['Mouse 745','Previously used for unlabeled development. Do not treat this mouse as an independent validation animal.'],
  ['Mouse 743 rejection','Camera-away footage generated false behavior scores. Body proposals often capture only a small part of the nearly stationary mouse. All six candidate totals are withheld.'],
  ['Source dimensions','Files are approximately 536–572 × 286–324 pixels despite the 540p filename. Fine paw/mouth movements may be obscured.'],
  ['Annotated results','http://127.0.0.1:8765/reports/stereotypy-five-video/index.html'],
  ['External validation report','http://127.0.0.1:8765/reports/stereotypy-priority/index.html'],
  ['CBAS source','https://github.com/jones-lab-tamu/CBAS/tree/v2-stable'],
  ['LabGym source','https://github.com/umyelab/LabGym/blob/master/LabGym_Zoo.md'],
  ...data.videos.flatMap(v=>[[`Mouse ${v.mouse_id} recording`,v.source],[`Mouse ${v.mouse_id} source SHA-256`,v.source_sha256],[`Mouse ${v.mouse_id} cage bounds`,JSON.stringify(v.mapping.crop_xyxy)],[`Mouse ${v.mouse_id} body proposals`,`${v.tracking.proposal_percent.toFixed(1)}% availability; not accuracy.`]]),
  ['Priority selection SHA-256',data.videos[0].models.priority_selection_sha256],
  ['Weak checkpoint SHA-256',data.videos[0].models.weak_checkpoint_sha256],
];
methods.getRange('A1').values=[['Stereotypy methods']];methods.getRange('A1').format.font={name:'Arial',size:16,bold:true};
methods.getRange(`A4:B${notes.length+3}`).values=notes;
methods.getRange(`A4:B${notes.length+3}`).format.font={name:'Arial',size:10,color:'#263a31'};
methods.getRange(`A4:A${notes.length+3}`).format.columnWidth=32;
methods.getRange(`B4:B${notes.length+3}`).format.columnWidth=105;
methods.getRange(`A4:B${notes.length+3}`).format.wrapText=true;
methods.getRange(`A4:B${notes.length+3}`).format.rowHeight=42;
await fs.mkdir(outputDir,{recursive:true});
console.log((await wb.inspect({kind:'table',range:'Candidate durations!A5:J9',include:'values,formulas',tableMaxRows:5,tableMaxCols:10,maxChars:2500})).ndjson);
console.log((await wb.inspect({kind:'table',range:'Candidate durations!A24:K25',include:'values,formulas',tableMaxRows:2,tableMaxCols:11,maxChars:1600})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:20},summary:'Formula error scan'})).ndjson);
for(const [sheetName,range,name] of [['Candidate durations','A1:K10','durations'],['Candidate segments','A1:F10','segments'],['Methods','A1:B16','methods']]){
 const png=await wb.render({sheetName,range,scale:1,format:'png'});
 await fs.writeFile(`${outputDir}/${name}-preview.png`,new Uint8Array(await png.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(`${outputDir}/candidate-summary.xlsx`);
