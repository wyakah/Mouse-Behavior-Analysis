"""Persistent reviewed batch queue and sequential full-video analysis."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import json, os, subprocess, uuid, re, shutil
from flask import request,jsonify,send_from_directory
from threechamber.core import analysis_geometry,metadata,analyze,sha256,propose_circles
from threechamber.preparation import prepare_trial


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def inside(root,name):
    path=(Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('Choose a workspace file.')
    return path


def validate_batch(root,draft):
    entries=draft.get('entries',[])
    if not isinstance(entries,list) or not entries:raise ValueError('Add recordings to the batch first.')
    try:diameter=float(draft['diameter_px']);cutoff=float(draft.get('pcutoff',.6))
    except (KeyError,TypeError,ValueError) as e:raise ValueError('Set the shared diameter and likelihood cutoff.') from e
    if not 0<=cutoff<=1:raise ValueError('Likelihood cutoff must be between 0 and 1.')
    ids=set();videos=set();prepared=[]
    profile=json.loads((Path(root)/'profiles/ethovision_three_chamber.json').read_text())
    for entry in entries:
        e=deepcopy(entry);identifier=str(e.get('id','')).strip()
        if not identifier or identifier in ids:raise ValueError('Each recording needs a unique, nonempty ID.')
        ids.add(identifier);e['id']=identifier
        p=inside(root,e['video'])
        if not p.is_file() or p.suffix.lower() not in ('.mp4','.avi','.mov','.mkv','.m4v'):raise ValueError(f'{identifier}: video missing.')
        if p in videos:raise ValueError('The same recording appears more than once.')
        videos.add(p)
        info=metadata(p)
        if [info['width'],info['height']]!=profile['source_size']:raise ValueError(f'{identifier}: this batch requires the saved EthoVision framing ({profile["source_size"][0]} × {profile["source_size"][1]}).')
        if e.get('reviewed') is not True:raise ValueError(f'{identifier}: review the cup centers, diameter, floor and dividers first.')
        cfg=e.get('config',{})
        if cfg.get('analysis_mode')!='circle_zones' or cfg.get('cup_circles',{}).get('diameter_px')!=diameter:raise ValueError(f'{identifier}: cup diameter differs from the shared batch diameter. Review this recording again.')
        if cfg.get('pcutoff')!=cutoff:raise ValueError(f'{identifier}: likelihood cutoff differs from the batch setting.')
        analysis_geometry(cfg)
        e['status']='pending';e.pop('error',None);e.pop('summary',None);e.pop('run_id',None)
        prepared.append(e)
    return dict(diameter_px=diameter,pcutoff=cutoff,entries=prepared)


def get_tracks(root,working,dest,progress):
    digest=sha256(working)
    for manifest in sorted((root/'outputs/dlc').glob('*/selection_manifest.json')):
        try:m=json.loads(manifest.read_text())
        except (ValueError,OSError):continue
        tracks=manifest.parent/'selected_tracks.csv'
        if m.get('source_sha256')==digest and m.get('coordinate_space')=='full_source_pixels' and tracks.is_file():
            progress('Reusing predictions verified against this video’s content hash')
            return tracks
    interpreter=Path(os.environ.get('DLC_PYTHON',str(root/'.dlc-env/bin/python')))
    if not interpreter.is_file():raise ValueError('DeepLabCut runtime is missing.')
    dest.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key,value in {'MPLCONFIGDIR':'.cache/matplotlib','TORCH_HOME':'.cache/torch','HF_HOME':'.cache/huggingface','XDG_CACHE_HOME':'.cache'}.items():env[key]=str(root/value)
    env['DLC_LIGHT']='True'
    with (dest/'inference.log').open('w') as log:
        progress('Locating the free subject')
        subprocess.run([str(root/'.venv/bin/python'),str(root/'scripts/localize_video.py'),'--video',str(working),'--output',str(dest/'localization')],check=True,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env)
        progress('DeepLabCut nose and body-center inference')
        subprocess.run([str(interpreter),str(root/'scripts/roi_inference.py'),'--video',str(working),'--proposals',str(dest/'localization/proposals.json'),'--output',str(dest)],check=True,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env)
    return dest/'selected_tracks.csv'


def export_workbook(root,batch,folder):
    import pandas as pd
    payload=deepcopy(batch)
    for e in payload['entries']:
        if e.get('status')=='complete':
            out=root/'outputs'/e['run_id']
            e['bouts']=pd.read_csv(out/'bouts.csv').to_dict('records')
            e['manifest']=json.loads((out/'manifest.json').read_text())
    save_json(folder/'workbook-data.json',payload)
    deps=Path(os.environ.get('CODEX_WORKSPACE_DEPENDENCIES',str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies')))
    runtime=root/'.cache/xlsx-runtime';runtime.mkdir(parents=True,exist_ok=True)
    modules=runtime/'node_modules'
    if not modules.exists():modules.symlink_to(deps/'node/node_modules',target_is_directory=True)
    shutil.copyfile(root/'scripts/export_batch.mjs',runtime/'export_batch.mjs')
    artifact_folder=root/'outputs'/batch['id'];artifact_folder.mkdir(parents=True,exist_ok=True)
    with (folder/'excel-export.log').open('w') as log:
        subprocess.run([str(deps/'node/bin/node'),str(runtime/'export_batch.mjs'),str(folder/'workbook-data.json'),str(artifact_folder/'results.xlsx')],check=True,cwd=root,stdout=log,stderr=subprocess.STDOUT)


def process_batch(root,batch,tracker=get_tracks,exporter=export_workbook,analyzer=analyze):
    root=Path(root);folder=root/'batches'/batch['id'];folder.mkdir(parents=True,exist_ok=True)
    def persist():save_json(folder/'batch.json',batch)
    batch.update(status='running',message='Starting batch');persist()
    for i,e in enumerate(batch['entries']):
        e['status']='running';batch['current_index']=i
        def progress(message):
            batch['message']=f'{i+1}/{len(batch["entries"])} · {e["id"]}: {message}';persist()
        try:
            progress('Preparing the first 600 seconds')
            working=prepare_trial(root,inside(root,e['video']))
            tr=tracker(root,working,root/'outputs/dlc'/f'{batch["id"]}-{i+1:03d}',progress)
            cfg=deepcopy(e['config']);cfg.update(start_s=0,end_s=min(600,metadata(working)['duration_seconds']),review_crop_xyxy=json.loads((root/'profiles/ethovision_three_chamber.json').read_text())['crop_xyxy'])
            runid=f'{batch["id"]}-{i+1:03d}'
            summary=analyzer(working,tr,cfg,root/'outputs'/runid,progress)
            e.update(status='complete',summary=summary,run_id=runid,working_video=str(working.relative_to(root)),tracks=str(tr.relative_to(root)))
        except Exception as error:e.update(status='failed',error=str(error))
        persist()
    batch.update(message='Creating the combined Excel workbook');persist()
    try:
        exporter(root,batch,folder)
        batch.update(status='complete' if all(e['status']=='complete' for e in batch['entries']) else 'complete_with_errors',message='Batch complete. Review the results and Excel workbook.',workbook='results.xlsx')
    except Exception as error:batch.update(status='export_failed',message=f'Video results retained; Excel export failed: {error}')
    persist();return batch


def register_batch(app,root_fn,pool):
    active=set()
    @app.get('/batch')
    def batch_page():return app.send_static_file('batch.html')
    @app.route('/api/batch/draft',methods=['GET','POST'])
    def batch_draft():
        p=root_fn()/'batches/draft.json'
        if request.method=='POST':
            data=request.get_json()
            if not isinstance(data,dict) or not isinstance(data.get('entries'),list):raise ValueError('Invalid batch draft.')
            save_json(p,data);return jsonify(saved=True)
        return jsonify(json.loads(p.read_text()) if p.exists() else dict(diameter_px=110,pcutoff=.6,entries=[]))
    @app.post('/api/batch/seed')
    def batch_seed():
        name=request.get_json()['video'];root=root_fn();p=inside(root,name);info=metadata(p)
        saved=root/'configs'/f'{p.stem}.json'
        cfg=json.loads(saved.read_text()) if saved.exists() else {}
        profile=json.loads((root/'profiles/ethovision_three_chamber.json').read_text())
        if [info['width'],info['height']]!=profile['source_size']:raise ValueError('Use recordings with the same resolution as the saved EthoVision camera profile. Mixed framing cannot share a pixel diameter.')
        if not cfg.get('arena'):
            x1,y1,x2,y2=profile['expected_arena_xyxy'] if 'expected_arena_xyxy' in profile else [193,190,735,558]
            cfg['arena']=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        if not cfg.get('cup_circles'):cfg.update(propose_circles(cfg))
        width=cfg.get('width_cm',0)
        cfg.update(analysis_mode='circle_zones',dividers_fraction=cfg.get('dividers_fraction') or ([v/width for v in cfg['dividers_cm']] if width and cfg.get('dividers_cm') else [1/3,2/3]),confirmed=False)
        namebase=re.sub(r'^[a-f0-9]{8}_','',p.stem)
        return jsonify(video=name,id=namebase.split('_')[0],config=cfg,reviewed=False,width=info['width'],height=info['height'])
    @app.post('/api/batch/review')
    def batch_review():
        analysis_geometry(request.get_json()['config']);return jsonify(valid=True)
    @app.post('/api/batches')
    def batch_start():
        root=root_fn();batch=validate_batch(root,request.get_json())
        batch.update(id='batch-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6],status='queued',message='Batch queued')
        save_json(root/'batches'/batch['id']/'batch.json',batch);active.add(batch['id'])
        def work():
            try:process_batch(root,batch)
            finally:active.discard(batch['id'])
        pool.submit(work);return jsonify(batch)
    @app.get('/api/batches')
    def batch_list():
        return jsonify([json.loads(p.read_text()) for p in sorted((root_fn()/'batches').glob('*/batch.json'),reverse=True)])
    @app.get('/api/batches/<identifier>')
    def batch_status(identifier):
        p=inside(root_fn(),'batches/'+identifier+'/batch.json')
        if not p.exists():raise ValueError('Batch not found.')
        batch=json.loads(p.read_text())
        if batch['status'] in ('queued','running') and identifier not in active:batch.update(status='interrupted',message='The app restarted during processing. Completed video outputs are preserved; start a new batch to retry.')
        return jsonify(batch)
    @app.get('/batches/<identifier>/results.xlsx')
    def batch_download(identifier):
        return send_from_directory(inside(root_fn(),'outputs/'+identifier),'results.xlsx',as_attachment=True)
