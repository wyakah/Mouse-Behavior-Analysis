"""Persistent reviewed batch queue and sequential full-video analysis."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import json, os, subprocess, uuid, re, shutil, time
from flask import request,jsonify,send_from_directory
from threechamber.core import analysis_geometry,metadata,analyze,sha256,propose_circles
from threechamber.preparation import prepare_trial
from threechamber.recordings import validate_recordings
from threechamber.live import LivePublisher,atomic_json
from threechamber.statistics import settings as statistics_settings, sample_metadata, analyze_statistics, METRICS


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def inside(root,name):
    path=(Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('Choose a workspace file.')
    return path


def validate_batch(root,draft):
    test_id=draft.get('test_id','three_chamber')
    if test_id!='three_chamber':raise ValueError('Automatic batch analysis is not available for this test. Select Three Chamber, or open the Stereotypy manual-scoring workspace.')
    plan=statistics_settings(draft.get('statistics'))
    entries=validate_recordings(draft.get('entries',[]),require_ids=True)
    if not isinstance(entries,list) or not entries:raise ValueError('Add recordings to the batch first.')
    try:diameter=float(draft['diameter_px']);cutoff=float(draft.get('pcutoff',.6))
    except (KeyError,TypeError,ValueError) as e:raise ValueError('Set the shared diameter and likelihood cutoff.') from e
    if not 0<=cutoff<=1:raise ValueError('Likelihood cutoff must be between 0 and 1.')
    ids=set();videos=set();prepared=[]
    profile=json.loads((Path(root)/'profiles/ethovision_three_chamber.json').read_text())
    for entry in entries:
        e=deepcopy(entry);identifier=str(e.get('id','')).strip()
        if not identifier or identifier in ids:raise ValueError('Each recording needs a unique, nonempty ID.')
        ids.add(identifier);e['id']=identifier;e.update(sample_metadata(e))
        p=inside(root,e['video'])
        if not p.is_file() or p.suffix.lower() not in ('.mp4','.avi','.mov','.mkv','.m4v'):raise ValueError(f'{identifier}: video missing.')
        if p in videos:raise ValueError('The same recording appears more than once.')
        videos.add(p)
        info=metadata(p)
        if [info['width'],info['height']]!=profile['source_size']:raise ValueError(f'{identifier}: this batch requires the saved EthoVision framing ({profile["source_size"][0]} × {profile["source_size"][1]}).')
        if e.get('reviewed') is not True:raise ValueError(f'{identifier}: review the cup centers, diameter, floor and dividers first.')
        cfg=e.get('config',{})
        from threechamber.social import stranger_side
        side=stranger_side({'stranger_side':e['stranger_side']}) if 'stranger_side' in e else stranger_side(cfg)
        if side not in ('left','right'):raise ValueError(f'{identifier}: choose the stranger mouse position (left or right).')
        e['stranger_side']=side;cfg.update(stranger_side=side,target_side=side)
        if cfg.get('analysis_mode')!='circle_zones' or cfg.get('cup_circles',{}).get('diameter_px')!=diameter:raise ValueError(f'{identifier}: cup diameter differs from the shared batch diameter. Review this recording again.')
        if cfg.get('pcutoff')!=cutoff:raise ValueError(f'{identifier}: likelihood cutoff differs from the batch setting.')
        analysis_geometry(cfg)
        e['status']='pending';e.pop('error',None);e.pop('summary',None);e.pop('run_id',None)
        prepared.append(e)
    return dict(test_id=test_id,statistics=plan,diameter_px=diameter,pcutoff=cutoff,entries=prepared)


def get_tracks(root,working,dest,progress):
    live=getattr(progress,'live',None)
    digest=sha256(working)
    for manifest in sorted((root/'outputs/dlc').glob('*/selection_manifest.json')):
        try:m=json.loads(manifest.read_text())
        except (ValueError,OSError):continue
        tracks=manifest.parent/'selected_tracks.csv'
        if m.get('source_sha256')==digest and m.get('coordinate_space')=='full_source_pixels' and tracks.is_file():
            if live:live.phase('cached','Using predictions verified against this recording')
            progress('Reusing predictions verified against this video’s content hash')
            return tracks
    interpreter=Path(os.environ.get('DLC_PYTHON',str(root/'.dlc-env/bin/python')))
    if not interpreter.is_file():raise ValueError('DeepLabCut runtime is missing.')
    dest.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key,value in {'MPLCONFIGDIR':'.cache/matplotlib','TORCH_HOME':'.cache/torch','HF_HOME':'.cache/huggingface','XDG_CACHE_HOME':'.cache'}.items():env[key]=str(root/value)
    env['DLC_LIGHT']='True'
    if live and not live.disabled:env['THREECHAMBER_LIVE_DIR']=str(live.folder)
    with (dest/'inference.log').open('w') as log:
        progress('Locating the free subject')
        subprocess.run([str(interpreter),str(root/'scripts/localize_video.py'),'--video',str(working),'--output',str(dest/'localization')],check=True,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env)
        progress('DeepLabCut nose and body-center inference')
        if live:live.phase('tracking','Starting DeepLabCut and loading the pose model')
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
    report=analyze_statistics(payload)
    payload['statistics_report']=report
    save_json(folder/'statistics.json',report)
    save_json(folder/'workbook-data.json',payload)
    from threechamber.workbooks import export_three_chamber
    artifact_folder=root/'outputs'/batch['id'];artifact_folder.mkdir(parents=True,exist_ok=True)
    version=root/'desktop-version.json'
    if version.exists():payload['application_version']=json.loads(version.read_text())['version']
    export_three_chamber(payload,artifact_folder/'results.xlsx')


def process_batch(root,batch,tracker=get_tracks,exporter=export_workbook,analyzer=analyze):
    root=Path(root);folder=root/'batches'/batch['id'];folder.mkdir(parents=True,exist_ok=True)
    def persist():save_json(folder/'batch.json',batch)
    batch.update(status='running',message='Starting batch',started_at=time.time());persist()
    live=None
    for i,e in enumerate(batch['entries']):
        live=None
        e['status']='running';batch['current_index']=i
        def progress(message):
            batch['message']=f'{i+1}/{len(batch["entries"])} · {e["id"]}: {message}';persist()
        try:
            info=metadata(inside(root,e['video']))
            e['config']['recording_metadata']={k:e.get(k,'') for k in ('id','sex','genotype')}
            context=dict(batch_id=batch['id'],recording_index=i,recording_id=e['id'],recording_count=len(batch['entries']),
                         source_duration_s=min(600,info['duration_seconds']),source_size=[info['width'],info['height']],cutoff=e['config'].get('pcutoff',.6),
                         crop=json.loads((root/'profiles/ethovision_three_chamber.json').read_text())['crop_xyxy'],
                         geometry={k:e['config'][k] for k in ('arena','dividers_fraction','cup_circles')},
                         config=dict(e['config'],review_crop_xyxy=json.loads((root/'profiles/ethovision_three_chamber.json').read_text())['crop_xyxy']))
            live=LivePublisher(folder/'live',context)
            try:atomic_json(live.folder/'context.json',context)
            except OSError:live.disabled=True
            progress.live=live
            live.phase('preparing','Preparing the first 10 minutes')
            progress('Preparing the first 600 seconds')
            working=prepare_trial(root,inside(root,e['video']))
            tr=tracker(root,working,root/'outputs/dlc'/f'{batch["id"]}-{i+1:03d}',progress)
            cfg=deepcopy(e['config']);cfg.update(start_s=0,end_s=min(600,metadata(working)['duration_seconds']),review_crop_xyxy=json.loads((root/'profiles/ethovision_three_chamber.json').read_text())['crop_xyxy'])
            runid=f'{batch["id"]}-{i+1:03d}'
            summary=analyzer(working,tr,cfg,root/'outputs'/runid,progress)
            e.update(status='complete',summary=summary,run_id=runid,working_video=str(working.relative_to(root)),tracks=str(tr.relative_to(root)))
        except Exception as error:
            e.update(status='failed',error=str(error))
            if live:live.phase('failed','This recording could not be completed')
        persist()
    batch.update(message='Creating Excel summaries and group statistics');persist()
    if live:live.phase('exporting','Creating Excel summaries and group statistics')
    try:
        exporter(root,batch,folder)
        batch.update(status='complete' if all(e['status']=='complete' for e in batch['entries']) else 'complete_with_errors',message='Batch complete. Review the results and Excel workbook.',workbook='results.xlsx')
    except Exception as error:batch.update(status='export_failed',message=f'Video results retained; Excel export failed: {error}')
    batch['finished_at']=time.time()
    if live:live.phase(batch['status'],batch['message'])
    persist();return batch


def register_batch(app,root_fn,pool):
    active=set()
    def current_status(batch):
        if batch['status'] in ('queued','running') and batch['id'] not in active:
            batch.update(status='interrupted',message='The app restarted during processing. Completed video outputs are preserved; start a new batch to retry.')
        return batch
    @app.get('/batch')
    def batch_page():return app.send_static_file('batch.html')
    @app.get('/api/statistics/options')
    def statistics_options():
        return jsonify(metrics=METRICS,order=list(METRICS),defaults=statistics_settings())

    @app.route('/api/batch/draft',methods=['GET','POST'])
    def batch_draft():
        p=root_fn()/'batches/draft.json'
        if request.method=='POST':
            data=request.get_json()
            if not isinstance(data,dict) or not isinstance(data.get('entries'),list):raise ValueError('Invalid batch draft.')
            validate_recordings(data['entries']);save_json(p,data);return jsonify(saved=True)
        return jsonify(json.loads(p.read_text()) if p.exists() else dict(test_id='three_chamber',statistics=statistics_settings(),diameter_px=110,pcutoff=.6,entries=[]))
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
        files=sorted((root_fn()/'batches').glob('*/batch.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        return jsonify([current_status(json.loads(p.read_text())) for p in files])
    @app.get('/api/batches/<identifier>')
    def batch_status(identifier):
        p=inside(root_fn(),'batches/'+identifier+'/batch.json')
        if not p.exists():raise ValueError('Batch not found.')
        response=jsonify(current_status(json.loads(p.read_text())));response.headers['Cache-Control']='no-store';return response
    @app.get('/api/batches/<identifier>/live')
    def batch_live(identifier):
        folder=inside(root_fn(),'batches/'+identifier)
        if not (folder/'batch.json').is_file():raise ValueError('Batch not found.')
        batch=current_status(json.loads((folder/'batch.json').read_text()))
        try:snapshot=json.loads((folder/'live/snapshot.json').read_text())
        except (OSError,ValueError):snapshot=None
        if snapshot and (snapshot.get('batch_id')!=identifier or snapshot.get('recording_index')!=batch.get('current_index')):snapshot=None
        unchanged=bool(snapshot and request.args.get('since')==snapshot['revision'])
        if snapshot and (unchanged or request.args.get('image','1')=='0'):snapshot.pop('image',None)
        response=jsonify(batch_id=identifier,status=batch['status'],snapshot=snapshot,unchanged=unchanged,server_time=time.time())
        response.headers['Cache-Control']='no-store';return response
    @app.get('/api/batches/<identifier>/video/<int:index>')
    def batch_video(identifier,index):
        folder=inside(root_fn(),'batches/'+identifier)
        if not (folder/'batch.json').is_file():raise ValueError('Batch not found.')
        batch=current_status(json.loads((folder/'batch.json').read_text()))
        if not 0<=index<len(batch['entries']):raise ValueError('Recording not found.')
        try:manifest=json.loads((folder/'live/video'/str(index)/'manifest.json').read_text())
        except (OSError,ValueError):manifest=None
        if manifest:
            manifest['base_url']=f'/batches/{identifier}/video/{index}/'
            if manifest['status']=='streaming' and batch['status']=='interrupted':
                manifest.update(status='failed',error='Processing was interrupted. Available preview footage is retained.')
        response=jsonify(manifest=manifest,status=batch['status'],entry_status=batch['entries'][index]['status'])
        response.headers['Cache-Control']='no-store';return response
    @app.get('/batches/<identifier>/video/<int:index>/<filename>')
    def batch_video_segment(identifier,index,filename):
        if not re.fullmatch(r'[a-f0-9]{32}-[0-9]{5}\.mp4',filename):raise ValueError('Invalid preview segment.')
        response=send_from_directory(inside(root_fn(),'batches/'+identifier+'/live/video/'+str(index)),filename,mimetype='video/mp4')
        response.headers['Cache-Control']='private, max-age=31536000, immutable';return response
    @app.get('/batches/<identifier>/results.xlsx')
    def batch_download(identifier):
        return send_from_directory(inside(root_fn(),'outputs/'+identifier),'results.xlsx',as_attachment=True)
