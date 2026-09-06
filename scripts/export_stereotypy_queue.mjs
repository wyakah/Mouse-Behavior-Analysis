import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const [source,dest]=process.argv.slice(2),data=JSON.parse(await fs.readFile(source,'utf8'));
const wb=Workbook.create(),s=wb.worksheets.add('Automated scores');
s.showGridLines=false;
s.getRange('A1').values=[['Provisional automatic stereotypy scores · first 20 minutes maximum']];
s.getRange('A2').values=[['Candidate durations are not validated accuracy. One behavior per interval. Other activity and unknown time remain separate. No statistics.']];
s.getRange('A4:L4').values=[['Mouse ID','Recorded sex','Genotype','Behavior','Candidate time (s)','Candidate bouts','Coverage (s)','Unknown (s)','Window (s)','Status','Evidence','Other activity (s)']];
s.getRange('A4:L4').format={fill:'#344b40',font:{bold:true,color:'#ffffff'},wrapText:true,rowHeight:40};
if(data.rows.length)s.getRange(`A5:L${data.rows.length+4}`).values=data.rows.map(row=>row.map((v,i)=>i===9&&v==='provisional_automated'?'Provisional':typeof v==='string'&&v.startsWith('=')?"'"+v:v));
s.getRange(`A1:L${data.rows.length+4}`).format.columnWidth=20;
s.getRange(`K1:K${data.rows.length+4}`).format.columnWidth=64;
s.getRange(`A1:L${data.rows.length+4}`).format.font.name='Arial';
if(data.rows.length){
 s.getRange(`A5:L${data.rows.length+4}`).format.rowHeight=34;
 s.getRange(`D5:D${data.rows.length+4}`).format.wrapText=true;
 s.getRange(`J5:K${data.rows.length+4}`).format.wrapText=true;
 for(const c of ['E','G','H','I','L'])s.getRange(`${c}5:${c}${data.rows.length+4}`).setNumberFormat('0.0');
}
s.freezePanes.freezeRows(4);
console.log((await wb.inspect({kind:'table',range:'Automated scores!A4:L10',include:'values',tableMaxRows:7,tableMaxCols:12,maxChars:2000})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:10},summary:'Export error check'})).ndjson);
const preview=await wb.render({sheetName:'Automated scores',range:'A1:L10',scale:1});
await fs.writeFile(dest+'.png',new Uint8Array(await preview.arrayBuffer()));
await (await SpreadsheetFile.exportXlsx(wb)).save(dest);
