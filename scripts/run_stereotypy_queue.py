"""Run uploaded recordings sequentially; no annotation session is created."""
import csv,json,os,sys,traceback,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,allow_nan=False));temp.replace(path)

def csv_value(value):
    return "'"+value if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')) else value

def automatic_mapping(path):
    # Preserve the entire field of view: fixed side-view recordings already contain
    # the cage. A guessed tight crop can remove jumping/rearing headroom.
    import av
    with av.open(str(path)) as c:
        f=next(c.decode(video=0));w,h=f.width,f.height
    return [0,0,w-w%2,h-h%2],round(h*.8)

def export_results(dest,state):
    rows=[];bouts=[]
    for e in state['entries']:
        for b in e.get('summary',[]):
            rows.append(dict(mouse_id=e['id'],sex=e['sex'],genotype=e['genotype'],behavior=b['behavior'],candidate_seconds=b['candidate_seconds'],candidate_segments=b['candidate_segments'],coverage_seconds=b['sampled_coverage_s'],unknown_seconds=b['unknown_seconds'],window_seconds=e['duration_s'],status='provisional_automated',evidence=b['evidence'],other_activity_seconds=(e.get('cumulative') or {}).get('other_seconds')))
        if e['status']=='failed':rows.append(dict(mouse_id=e['id'],sex=e['sex'],genotype=e['genotype'],status='failed',evidence=e['message']))
        if e.get('result_file'):
            r=json.loads((dest/'published'/e['result_file']).read_text())
            bouts.extend(dict(mouse_id=e['id'],**b) for b in r['bouts'])
    pub=dest/'published';pub.mkdir(exist_ok=True)
    fields=['mouse_id','sex','genotype','behavior','candidate_seconds','candidate_segments','coverage_seconds','unknown_seconds','window_seconds','status','evidence','other_activity_seconds']
    for name,data,columns in [('summary.csv',rows,fields),('candidates.csv',bouts,['mouse_id','behavior','start_s','end_s','duration_s'])]:
        with (pub/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows({k:csv_value(v) for k,v in row.items()} for row in data)
    state['downloads']=['summary.csv','candidates.csv']
    save(dest/'workbook.json',dict(headers=fields,rows=[[r.get(k) for k in fields] for r in rows]))
    try:
        from threechamber.workbooks import export_stereotypy
        export_stereotypy(json.loads((dest/'workbook.json').read_text()),pub/'results.xlsx')
        state['downloads'].insert(0,'results.xlsx')
    except Exception as exc:state['export_notice']='Excel export failed; complete CSV results are available. '+str(exc)

def run(dest,analyzer=None):
    if analyzer is None:
        import analyze_stereotypy_batch as analyzer
    state=json.loads((dest/'batch.json').read_text())
    state.update(status='running',pid=os.getpid(),message='Preparing automatic analysis');save(dest/'batch.json',state)
    analyzer.RUN=dest/'work';analyzer.REPORT=dest/'published'
    (dest/'sources').mkdir(exist_ok=True);analyzer.REPORT.mkdir(exist_ok=True)
    device='mps' if analyzer.torch.backends.mps.is_available() else 'cpu'
    print('Automatic batch device:',device,flush=True)
    for i,e in enumerate(state['entries']):
        key=f'{i+1:03}';e.update(status='running',message='Reading video',stage='preparing');state['current_index']=i;state['message']=f'Analyzing mouse {e["id"]} ({i+1}/{len(state["entries"])})';save(dest/'batch.json',state)
        try:
            source=(ROOT/e['video']).resolve();path=dest/'sources'/(key+'_source'+source.suffix)
            if not path.exists():
                if os.name=='nt':shutil.copy2(source,path)
                else:path.symlink_to(source)
            analyzer.MAPPINGS[key]=automatic_mapping(source)
            mouse,out,index,mapping=analyzer.inputs(path)
            e.update(duration_s=index['duration_s'],message='Detecting behaviors and motion',stage='scoring');save(dest/'batch.json',state)
            prediction_stream=None
            if hasattr(analyzer,'incremental_result'):
                result,prediction_stream=analyzer.incremental_result(path,out,index,mapping,device)
            else:
                analyzer.features(path,mouse,out,index,mapping,device)
                result=analyzer.predict(path,mouse,out,index,mapping,device)
            result.update(display_id=e['id'],mouse_id=e['id'],source=e['video'],queue_index=i,mapping_method='full field of view; estimated bedding reference',manual_scoring_required=False)
            e.update(message='Loading models' if prediction_stream is not None else 'Creating video',stage='preparing' if prediction_stream is not None else 'rendering');save(dest/'batch.json',state)
            state['current_index']=i
            def progress(snapshot):
                e.update(cumulative=snapshot,summary=None,message='Analyzing',stage='scoring' if prediction_stream is not None else 'rendering')
                save(dest/'batch.json',state)
            analyzer.render(path,mouse,out,index,mapping,result,progress=progress,stream_folder=dest/'live'/str(i),prediction_stream=prediction_stream)
            result.update(status='provisional_automated',quality_method='Automatic body proposal gate; no human approval or accuracy validation')
            save(analyzer.REPORT/(key+'.json'),result)
            e.update(status='complete',message='Automatic scoring complete',summary=result['summary'],cumulative=result['cumulative'],scoring_policy=result['scoring_policy'],video_file=key+'.mp4',poster_file=key+'.jpg',result_file=key+'.json')
        except Exception as exc:
            traceback.print_exc();e.update(status='failed',message=str(exc))
        save(dest/'batch.json',state)
    state['message']='Exporting results'
    state['entries'][state.get('current_index',0)]['stage']='exporting'
    save(dest/'batch.json',state)
    export_results(dest,state)
    completed=sum(e['status']=='complete' for e in state['entries'])
    state.update(status='complete' if completed==len(state['entries']) else 'complete_with_errors' if completed else 'failed',message=f'{completed} of {len(state["entries"])} videos analyzed')
    save(dest/'batch.json',state)

if __name__=='__main__':run(Path(sys.argv[1]).resolve())
