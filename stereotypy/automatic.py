"""Persistent autonomous stereotypy queues. Human annotation is never a prerequisite."""
import csv,json,os,re,subprocess,uuid,threading
from pathlib import Path
from datetime import datetime,timezone
from flask import request,jsonify,send_from_directory
from threechamber.recordings import validate_recordings


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,allow_nan=False));temp.replace(path)


def model_readiness(root):
    required=['.dlc-env/bin/python','scripts/run_stereotypy_queue.py','outputs/stereotypy-training/v2/selection.json','outputs/stereotypy-training/v2/preprocessing.joblib','outputs/stereotypy-training/v2/temporal.pt','outputs/stereotypy-training/v2/binary-temporal.pt','outputs/stereotypy-training/v1/temporal-model.pt']
    missing=[p for p in required if not (root/p).is_file()]
    return dict(ready=not missing,missing=missing,validated=False)

def reconcile(path):
    state=json.loads(path.read_text())
    def alive(pid):
        if not pid:return False
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
        except PermissionError:return True
    if state['status'] in ('queued','running') and not alive(state.get('pid')) and not alive(state.get('server_pid')):
        state.update(status='failed',message='Analysis was interrupted. Start the batch again to retry automatically.')
        for entry in state['entries']:
            if entry['status'] in ('queued','running'):entry.update(status='failed',message='Interrupted before completion')
        save(path,state)
    return state


def register_automatic(app,root_getter,pool):
    lock=threading.Lock()
    def root():return Path(root_getter()).resolve()
    def folder(identifier):
        if not re.fullmatch(r'stereo-[0-9a-f]{12}',identifier):raise ValueError('Unknown analysis')
        return root()/'outputs'/identifier
    def load(identifier):
        p=folder(identifier)/'batch.json'
        if not p.exists():raise ValueError('Analysis not found')
        return reconcile(p)
    def launch(identifier):
        dest=folder(identifier)
        try:
            with (dest/'worker.log').open('a') as log:
                process=subprocess.Popen([str(root()/'.dlc-env/bin/python'),str(root()/'scripts/run_stereotypy_queue.py'),str(dest)],cwd=root(),stdout=log,stderr=subprocess.STDOUT)
                process.wait()
            state=load(identifier)
            if state['status'] in ('queued','running'):
                state.update(status='failed',message='Analysis stopped unexpectedly. The run log contains details.');save(dest/'batch.json',state)
        except Exception as exc:
            state=load(identifier);state.update(status='failed',message=str(exc));save(dest/'batch.json',state)
    @app.get('/api/stereotypy/automatic/readiness')
    def automatic_readiness():return jsonify(model_readiness(root()))
    @app.get('/api/stereotypy/batches')
    def automatic_history():
        records=[]
        for p in sorted((root()/'outputs').glob('stereo-*/batch.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            b=reconcile(p);records.append(b)
        return jsonify(records)
    @app.get('/api/stereotypy/batches/<identifier>')
    def automatic_status(identifier):return jsonify(load(identifier))
    @app.get('/api/stereotypy/batches/<identifier>/live')
    def automatic_live(identifier):
        b=load(identifier);i=b.get('current_index',0);e=b['entries'][i]
        response=jsonify(snapshot=dict(e.get('cumulative') or {},recording_index=i,stage=e.get('stage','preparing'),message=e['message']))
        response.headers['Cache-Control']='no-store';return response
    @app.get('/api/stereotypy/batches/<identifier>/video/<int:index>')
    def automatic_video(identifier,index):
        b=load(identifier)
        if not 0<=index<len(b['entries']):raise ValueError('Unknown recording')
        p=folder(identifier)/'live'/str(index)/'manifest.json'
        manifest=json.loads(p.read_text()) if p.exists() else None
        if manifest:
            manifest['base_url']=f'/stereotypy-runs/{identifier}/video/{index}/'
            if b['entries'][index]['status']=='failed' and manifest['status']=='streaming':manifest.update(status='failed',error=b['entries'][index]['message'])
        response=jsonify(manifest=manifest,status=b['status'],entry_status=b['entries'][index]['status'])
        response.headers['Cache-Control']='no-store';return response
    @app.get('/stereotypy-runs/<identifier>/video/<int:index>/<name>')
    def automatic_segment(identifier,index,name):
        b=load(identifier)
        if not 0<=index<len(b['entries']) or not re.fullmatch(r'[a-f0-9]{32}-[0-9]{5}\.mp4',name):raise ValueError('Unknown preview segment')
        return send_from_directory(folder(identifier)/'live'/str(index),name,mimetype='video/mp4',conditional=True)
    @app.post('/api/stereotypy/batches')
    def automatic_start():
        data=request.get_json(silent=True)
        if not isinstance(data,dict):raise ValueError('Send a video queue')
        entries=validate_recordings(data.get('entries'),require_ids=True)
        cleaned=[]
        for e in entries:
            source=(root()/e['video']).resolve()
            if not source.is_relative_to(root()) or not source.is_file() or source.suffix.lower() not in ('.mov','.mp4','.avi','.mkv','.m4v'):raise ValueError('Choose an uploaded local video')
            cleaned.append(dict(video=e['video'],id=e['id'].strip(),sex=e.get('sex') or 'unknown',genotype=e.get('genotype') or '',status='queued',message='Queued'))
        if not model_readiness(root())['ready']:raise ValueError('Automatic model files are missing on this installation. Restore the trained model bundle; manual annotation is not required to run it.')
        with lock:
            for p in (root()/'outputs').glob('stereo-*/batch.json'):
                old=reconcile(p)
                if old['status'] in ('queued','running'):
                    keys=('video','id','sex','genotype')
                    if [{k:e[k] for k in keys} for e in old['entries']]==[{k:e[k] for k in keys} for e in cleaned]:return jsonify(old)
                    return jsonify(error='Another stereotypy batch is running. Open its progress before starting another.',active_batch_id=old['id']),409
            identifier='stereo-'+uuid.uuid4().hex[:12]
            state=dict(id=identifier,status='queued',server_pid=os.getpid(),message='Waiting for the analysis worker',entries=cleaned,created_at=datetime.now(timezone.utc).isoformat(),window_seconds=1200,validated=False,statistics_performed=False)
            save(folder(identifier)/'batch.json',state);pool.submit(launch,identifier)
        return jsonify(state),202
    @app.get('/stereotypy-runs/<identifier>/<path:name>')
    def automatic_file(identifier,name):
        # Do not expose staged source symlinks or worker/internal state through export URLs.
        if not re.fullmatch(r'(results\.xlsx|summary\.csv|candidates\.csv|worker\.log|[0-9]{3}\.(mp4|jpg|json))',name):raise ValueError('Unknown result file')
        return send_from_directory(folder(identifier)/'published' if name!='worker.log' else folder(identifier),name,conditional=True,as_attachment=name.endswith(('.csv','.xlsx')))
