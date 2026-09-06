"""Portable Excel exports, shared by web and desktop installations."""
import json
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

def cell_value(value):
    if isinstance(value,(dict,list,tuple)):value=json.dumps(value,ensure_ascii=False)
    # Force untrusted strings to text: never allow sample metadata to become formulas.
    return value

def sheet(book,name,headers,rows):
    ws=book.create_sheet(name);ws.append(headers)
    for row in rows:
        ws.append([cell_value(v) for v in row])
        for cell in ws[ws.max_row]:
            if isinstance(cell.value,str):cell.data_type='s'
            elif isinstance(cell.value,float):cell.number_format='0.000'
    ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
    for cell in ws[1]:
        cell.font=Font(name='Calibri',bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='245C4C');cell.alignment=Alignment(wrap_text=True)
    ws.row_dimensions[1].height=32
    for i,h in enumerate(headers,1):ws.column_dimensions[get_column_letter(i)].width=min(48,max(16,len(str(h))+2))
    return ws

def records(book,name,rows,default=('Status',)):
    keys=list(dict.fromkeys(k for row in rows for k in row)) or list(default)
    return sheet(book,name,keys,[[row.get(k) for k in keys] for row in rows])

def save(book,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.tmp.xlsx');book.save(temp);temp.replace(destination)

def export_stereotypy(data,destination):
    book=Workbook();book.remove(book.active)
    sheet(book,'Automated scores',data['headers'],data['rows'])
    save(book,destination)

def export_three_chamber(batch,destination):
    book=Workbook();book.remove(book.active);rows=[];setup=[];bouts=[]
    for e in batch['entries']:
        from threechamber.social import stranger_metrics
        summary=dict(e.get('summary',{}));summary.update(stranger_metrics(summary));target=summary.get('target_side')
        for name,side in [('target_nose_seconds',target),('other_nose_seconds','right' if target=='left' else 'left')]:
            summary[name]=summary.get(side+'_nose_seconds') if target in ('left','right') and summary.get('nose_scoreable_fraction',0)>0 else None
        rows.append(dict(sample_id=e['id'],sex=e.get('sex','unknown'),genotype=e.get('genotype',''),recording=e['video'],status=e['status'],**summary,error=e.get('error')))
        setup.append(dict(sample_id=e['id'],config=e.get('config'),provenance=e.get('manifest'),review_video=f"outputs/{e['run_id']}/review.mp4" if e.get('run_id') else None))
        bouts.extend(dict(sample_id=e['id'],**b) for b in e.get('bouts',[]))
    ws=records(book,'Results',rows)
    for cell in ws[1]:
        if cell.value=='stranger_interaction_percent':cell.value='Stranger Interaction %'
        elif cell.value=='stranger_side':cell.value='Stranger mouse position'
    records(book,'Setup',setup);records(book,'Bouts',bouts)
    report=batch.get('statistics_report')
    if report:
        for name,key in [('Groups','groups'),('Comparisons','comparisons'),('ANOVA','anova'),('Exclusions','exclusions')]:
            if report.get(key):records(book,name,report[key])
        records(book,'Statistics notes',[dict(setting=k,value=report.get(k)) for k in ('settings','versions','notes','sources','test_count')])
    sheet(book,'Scoring notes',['Setting','Value'],[
        ['Scoring window','First 600 seconds, or available duration if shorter'],
        ['Cup measurement','Nose within user-defined linked pixel circles'],
        ['Missing observations','Missing and low-confidence landmarks remain unscored; failed values are blank'],
        ['Timing','Source timestamps; cumulative durations'],
        ['Stranger Interaction %','100 × stranger nose-in-zone seconds / (left + center + right chamber seconds). Unknown and outside chamber time are excluded. Missing inputs or zero denominator are blank.'],
        ['Application',batch.get('application_version','Source checkout')]])
    save(book,destination)
