"""Local-only UI. Run: .venv/bin/python app.py"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlparse
import json, os, subprocess, sys, uuid
import cv2
from flask import Flask,request,jsonify,send_file,send_from_directory,abort
from threechamber.core import metadata,analyze,calibration,analysis_geometry,propose_interaction_zones,propose_circles
ROOT=Path(__file__).resolve().parent
app=Flask(__name__,static_folder='static'); app.config['MAX_CONTENT_LENGTH']=4*1024*1024*1024
pool=ThreadPoolExecutor(max_workers=1); jobs={}
from threechamber.labeling import register_labeling
register_labeling(app,ROOT)
from threechamber.batch import register_batch
register_batch(app,lambda:ROOT,pool)
from stereotypy.api import register_stereotypy
register_stereotypy(app,lambda:ROOT,pool,jobs)

def local(value):
    p=(ROOT/value).resolve()
    if not p.is_relative_to(ROOT): raise ValueError('Use a file inside the project folder.')
    return p

def video_path(name):
    p=local(name)
    if p.suffix.lower() not in ('.mp4','.avi','.mov','.mkv','.m4v') or not p.is_file(): raise ValueError('Video not found.')
    return p

@app.before_request
def origin_guard():
    if request.host.split(':')[0] not in ('127.0.0.1','localhost'): abort(403)
    if request.method=='POST':
        origin=request.headers.get('Origin')
        if origin and urlparse(origin).netloc!=request.host: abort(403)

@app.errorhandler(ValueError)
def value_error(e): return jsonify(error=str(e)),400

@app.get('/')
def index(): return app.send_static_file('batch.html')

@app.get('/advanced')
def advanced(): return app.send_static_file('index.html')

@app.get('/api/videos')
def videos():
    found=[]
    for p in sorted(list(ROOT.iterdir())+list((ROOT/'prepared').glob('*.mp4'))+list((ROOT/'videos').glob('*'))+list((ROOT/'stereotypy_videos').glob('*'))):
        if p.suffix.lower() in ('.mp4','.avi','.mov','.mkv','.m4v') and p.is_file():
            found.append(dict(name=str(p.relative_to(ROOT)),prepared=p.parent.name=='prepared',**metadata(p)))
    return jsonify(found)

@app.post('/api/videos/upload')
def upload_video():
    from werkzeug.utils import secure_filename
    f=request.files.get('file')
    if f is None or Path(f.filename or '').suffix.lower() not in ('.mp4','.avi','.mov','.mkv','.m4v'):raise ValueError('Choose an MP4, AVI, MOV, or MKV video.')
    folder=ROOT/'videos';folder.mkdir(exist_ok=True)
    name=uuid.uuid4().hex[:8]+'_'+secure_filename(f.filename);p=folder/name;f.save(p)
    try:info=metadata(p)
    except Exception as e:
        p.rename(p.with_suffix(p.suffix+'.unreadable'))
        raise ValueError('Uploaded file cannot be read as video; a copy was retained as .unreadable.') from e
    return jsonify(name=str(p.relative_to(ROOT)),prepared=False,**info)

@app.get('/api/readiness')
def readiness():
    video=video_path(request.args['video']);info=metadata(video);p=ROOT/'configs'/f'{video.stem}.json';cfg=json.loads(p.read_text()) if p.exists() else {}
    issues=[];arena_ready=scale_ready=cups_ready=zones_ready=False
    try:analysis_geometry(dict(cfg,analysis_mode='chambers_only'));arena_ready=True
    except (ValueError,KeyError,TypeError):issues.append(dict(code='arena',message='Confirm four floor corners and chamber dividers.',blocking_for='all'))
    try:scale_ready=float(cfg.get('width_cm',0))>0 and float(cfg.get('depth_cm',0))>0
    except (TypeError,ValueError):pass
    try:calibration(cfg);cups_ready=True
    except (ValueError,KeyError,TypeError):pass
    try:analysis_geometry(dict(cfg,analysis_mode='drawn_zones'));zones_ready=True
    except (ValueError,KeyError,TypeError):pass
    if cfg.get('analysis_mode')=='drawn_zones' and not zones_ready:issues.append(dict(code='zones',message='Draw separate left and right interaction zones inside the arena.',blocking_for='cup_metrics'))
    if cfg.get('analysis_mode')=='calibrated':
        if not scale_ready:issues.append(dict(code='scale',message='Measured scale is required only for the optional 1 cm metric. Selected-zone scoring does not need it.',blocking_for='cup_metrics'))
        elif not cups_ready:issues.append(dict(code='cups',message='Confirm non-crossing cup floor footprints.',blocking_for='cup_metrics'))
    ann=ROOT/'labeling/annotations.json';reviewed=sum(r.get('reviewed') is True for r in json.loads(ann.read_text()).values()) if ann.exists() else 0
    issues.append(dict(code='validation',message='95–99% correct tracking has not been established against reviewed ground truth.',blocking_for='validated_claim'))
    profile=json.loads((ROOT/'profiles/ethovision_three_chamber.json').read_text())
    files=[str(p.relative_to(ROOT)) for d in [ROOT/'tracks',ROOT/'outputs/dlc'] if d.exists() for p in d.rglob('*.csv')]
    return jsonify(video=str(video.relative_to(ROOT)),duration_s=info['duration_seconds'],scoring_end_s=min(600,info['duration_seconds']),profile_matches=profile['source_size']==[info['width'],info['height']],calibration=dict(arena_ready=arena_ready,scale_ready=scale_ready,cups_ready=cups_ready,zones_ready=zones_ready),tracking=dict(files=files,human_reviewed_frames=reviewed,accuracy_validated=False),issues=issues)

@app.post('/api/zones/suggest')
def suggest_zones():
    return jsonify(propose_interaction_zones(request.get_json()))

@app.post('/api/zones/circles')
def suggest_circles():
    return jsonify(propose_circles(request.get_json()))

@app.get('/api/frame')
def frame():
    p=video_path(request.args['video']); cap=cv2.VideoCapture(str(p))
    idx=max(0,min(int(request.args.get('frame',0)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1))
    cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,im=cap.read(); cap.release()
    if not ok: raise ValueError('Cannot decode this frame.')
    if request.args.get('crop')=='arena':
        profile=json.loads((ROOT/'profiles/ethovision_three_chamber.json').read_text())
        from threechamber.subject import validate_profile
        validate_profile(profile,im.shape[1],im.shape[0])
        x1,y1,x2,y2=profile['crop_xyxy'];im=im[y1:y2,x1:x2]
    from io import BytesIO
    return send_file(BytesIO(cv2.imencode('.jpg',im)[1].tobytes()),mimetype='image/jpeg')

@app.get('/api/video')
def source(): return send_file(video_path(request.args['video']),conditional=True)

@app.route('/api/config',methods=['GET','POST'])
def config():
    name=request.args['video']; video_path(name)
    p=ROOT/'configs'/f'{Path(name).stem}.json'
    if request.method=='POST':
        cfg=request.get_json(); p.write_text(json.dumps(cfg,indent=2)); return jsonify(saved=True)
    return jsonify(json.loads(p.read_text()) if p.exists() else None)

@app.get('/api/inspection')
def inspection():
    p=ROOT/'reports'/'video_inspection.json'
    return jsonify(json.loads(p.read_text()) if p.exists() else [])

@app.get('/reports/<path:name>')
def reports(name): return send_from_directory(ROOT/'reports',name)

@app.get('/api/tracks')
def tracks():
    dirs=[ROOT/'tracks',ROOT/'outputs'/'dlc']
    return jsonify([str(p.relative_to(ROOT)) for d in dirs if d.exists() for p in d.rglob('*') if p.suffix in ('.csv','.h5')])

@app.post('/api/upload')
def upload():
    f=request.files['file']
    if Path(f.filename).suffix.lower() not in ('.csv','.h5','.hdf5'): raise ValueError('Import a DLC CSV or HDF5 file.')
    folder=ROOT/'tracks'; folder.mkdir(exist_ok=True)
    from werkzeug.utils import secure_filename
    p=folder/(uuid.uuid4().hex[:8]+'_'+secure_filename(f.filename)); f.save(p)
    return jsonify(path=str(p.relative_to(ROOT)))

@app.post('/api/analyze')
def run():
    data=request.get_json(); video=video_path(data['video']); tr=local(data['tracks']); cfg=data['config']; analysis_geometry(cfg)
    if not tr.is_file(): raise ValueError('Choose a tracking file first.')
    jobid=datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
    dest=ROOT/'outputs'/jobid; jobs[jobid]={'id':jobid,'status':'queued','message':'Queued'}
    def work():
        try:
            jobs[jobid].update(status='running')
            result=analyze(video,tr,cfg,dest,lambda msg:jobs[jobid].update(message=msg))
            jobs[jobid].update(status='complete',message='Review package ready',summary=result)
        except Exception as e: jobs[jobid].update(status='failed',message=str(e))
    pool.submit(work); return jsonify(jobs[jobid])

@app.post('/api/dlc')
def dlc():
    data=request.get_json(); video=video_path(data['video']); config=local(data['config'])
    if not config.is_file() or config.suffix not in ('.yaml','.yml'): raise ValueError('Provide the trained DLC project config.yaml inside this project folder.')
    interpreter=Path(os.environ.get('DLC_PYTHON',str(ROOT/'.dlc-env/bin/python')))
    if not interpreter.is_file(): raise ValueError('DeepLabCut runtime missing. Follow README setup and set DLC_PYTHON, or import a DLC CSV.')
    jobid='dlc-'+uuid.uuid4().hex[:10]; dest=ROOT/'outputs'/'dlc'/jobid; dest.mkdir(parents=True)
    jobs[jobid]={'id':jobid,'status':'queued','message':'DLC inference queued'}
    def work():
        jobs[jobid].update(status='running',message='DeepLabCut inference running; log saved in output folder')
        with (dest/'inference.log').open('w') as log:
            try:
                env=os.environ.copy(); env.update(MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),DLC_LIGHT='True',TORCH_HOME=str(ROOT/'.cache/torch'),HF_HOME=str(ROOT/'.cache/huggingface'))
                subprocess.run([str(interpreter),str(ROOT/'scripts'/'dlc_workflow.py'),'infer','--config',str(config),'--video',str(video),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT,env=env)
                csvs=list(dest.glob('*.csv'))
                if not csvs: raise ValueError('DLC produced no CSV; inspect inference.log.')
                jobs[jobid].update(status='complete',message='DeepLabCut tracks ready; select CSV and analyze',tracks=str(csvs[0].relative_to(ROOT)))
            except Exception as e: jobs[jobid].update(status='failed',message=f'{e}. See outputs/dlc/{jobid}/inference.log')
    pool.submit(work); return jsonify(jobs[jobid])

@app.get('/api/profile')
def arena_profile():
    return jsonify(json.loads((ROOT/'profiles/ethovision_three_chamber.json').read_text()))

@app.post('/api/specialized')
def specialized():
    data=request.get_json();video=video_path(data['video'])
    if data.get('crop_confirmed') is not True:raise ValueError('Check that the saved crop contains the complete arena in this recording.')
    from threechamber.subject import validate_profile
    profile=json.loads((ROOT/'profiles/ethovision_three_chamber.json').read_text());cap=cv2.VideoCapture(str(video))
    try:validate_profile(profile,int(cap.get(3)),int(cap.get(4)))
    finally:cap.release()
    interpreter=Path(os.environ.get('DLC_PYTHON',str(ROOT/'.dlc-env/bin/python')))
    if not interpreter.is_file():raise ValueError('DeepLabCut runtime missing.')
    jobid='specialized-'+uuid.uuid4().hex[:10];dest=ROOT/'outputs/dlc'/jobid;dest.mkdir(parents=True)
    jobs[jobid]={'id':jobid,'status':'queued','message':'Cropped multi-candidate DLC inference queued'}
    def work():
        jobs[jobid].update(status='running',message='Tracking the focused arena. Experimental predictions require review.')
        try:
            from threechamber.preparation import prepare_trial
            jobs[jobid].update(message='Preparing and checking the first 600-second trial; source is preserved.')
            working=prepare_trial(ROOT,video)
            previous_config=ROOT/'configs'/f'{video.stem}.json';working_config=ROOT/'configs'/f'{working.stem}.json'
            if previous_config.exists() and not working_config.exists():working_config.write_bytes(previous_config.read_bytes())
            jobs[jobid].update(message='Locating the free subject, then estimating nose and center with DeepLabCut.')
            with (dest/'inference.log').open('w') as log:
                subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/localize_video.py'),'--video',str(working),'--output',str(dest/'localization')],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                subprocess.run([str(interpreter),str(ROOT/'scripts/roi_inference.py'),'--video',str(working),'--proposals',str(dest/'localization/proposals.json'),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            jobs[jobid].update(status='complete',message='Experimental tracks ready. Review subject identity and cup contacts before scoring.',working_video=str(working.relative_to(ROOT)),tracks=str((dest/'selected_tracks.csv').relative_to(ROOT)))
        except Exception as e:jobs[jobid].update(status='failed',message=f'{e}. See {dest.relative_to(ROOT)}/inference.log')
    pool.submit(work);return jsonify(jobs[jobid])

@app.get('/api/jobs/<jobid>')
def job(jobid):
    if jobid not in jobs: abort(404)
    result=dict(jobs[jobid])
    if result['status']=='running':
        import re
        log=ROOT/'outputs/dlc'/jobid/'inference.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-8192));tail=f.read().decode('utf-8',errors='replace')
            matches=re.findall(r'(\d+)%[^\r\n]*?(\d+)/(\d+)',tail)
            if matches:
                pct,frame,total=matches[-1];result.update(progress_percent=int(pct),message=f'DeepLabCut pose inference: {frame} / {total} frames ({pct}%).')
            elif 'decoded |' in tail:result['message']='Building and checking body localization proposals for the complete trial.'
    return jsonify(result)

@app.get('/api/results')
def results():
    found=[]
    for p in sorted((ROOT/'outputs').glob('*/manifest.json'),reverse=True):
        if (p.parent/'summary.json').is_file() and json.loads(p.read_text()).get('status')=='complete':
            found.append({'id':p.parent.name,'summary':json.loads((p.parent/'summary.json').read_text())})
    return jsonify(found)

@app.get('/outputs/<jobid>/<filename>')
def output(jobid,filename):
    if filename not in ('summary.csv','summary.json','frames.csv','bouts.csv','review.mp4','manifest.json','calibration.json','review_intervals.json'): abort(404)
    return send_from_directory(local('outputs/'+jobid),filename,conditional=True,as_attachment=request.args.get('download')=='1')

if __name__=='__main__':
    for folder in ['configs','outputs','reports','tracks']: (ROOT/folder).mkdir(exist_ok=True)
    app.run(host='127.0.0.1',port=int(os.environ.get('PORT','8765')),debug=False,threaded=True)
