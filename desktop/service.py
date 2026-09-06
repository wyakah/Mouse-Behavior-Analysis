"""Private loopback service for the desktop shell; no system Python required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import threading
import time

DATA_DIRS = ('videos', 'configs', 'outputs', 'reports', 'tracks', 'batches', 'labeling', 'prepared', '.cache')

def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def install_workspace(bundle, workspace):
    """Update only manifest-owned program files; never replace experiment data."""
    workspace.mkdir(parents=True, exist_ok=True)
    for name in DATA_DIRS:
        (workspace/name).mkdir(exist_ok=True)
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name, expected in manifest.get('model_files',{}).items():
        source=(bundle/name).resolve()
        if not source.is_relative_to(bundle.resolve()) or digest(source)!=expected:
            raise ValueError(f'Model package is damaged: {name}')
    for name, expected in manifest['program_files'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in ('app.py','threechamber','stereotypy','scripts','static','profiles','desktop'):
            raise ValueError('Invalid program manifest path')
        source=bundle/'payload'/rel; target=workspace/rel
        if digest(source)!=expected:raise ValueError(f'Application file is damaged: {name}')
        if target.is_symlink():raise ValueError(f'Unexpected program link: {name}')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or digest(target)!=expected:
            temp=target.with_suffix(target.suffix+'.installing');shutil.copyfile(source,temp);temp.replace(target)
    for name, source in {'.venv':bundle/'runtime','.dlc-env':bundle/'runtime',
                         'outputs/stereotypy-training':bundle/'models/stereotypy-training',
                         '.cache/huggingface':bundle/'models/huggingface',
                         '.cache/torch':bundle/'models/torch'}.items():
        target=workspace/name
        if target.is_symlink():target.unlink()
        elif hasattr(target,'is_junction') and target.is_junction():target.rmdir()
        elif target.exists():raise ValueError(f'Expected a managed runtime link: {name}')
        if os.name=='nt':
            import _winapi
            _winapi.CreateJunction(str(source.resolve()),str(target))
        else:target.symlink_to(source,target_is_directory=True)
    (workspace/'desktop-version.json').write_text(json.dumps({k:manifest[k] for k in ('version','model_files')}))
    return manifest

def secure_app(app, token):
    from flask import request, abort, redirect, jsonify
    @app.before_request
    def desktop_auth():
        if request.path=='/desktop/connect':
            if not secrets.compare_digest(request.args.get('token',''), token):abort(403)
            response=redirect('/')
            response.set_cookie('behavior_session',token,httponly=True,samesite='Strict')
            response.headers['Cache-Control']='no-store'
            response.headers['Referrer-Policy']='no-referrer'
            return response
        if not secrets.compare_digest(request.cookies.get('behavior_session',''), token):abort(403)
    @app.get('/api/desktop')
    def desktop_info():
        return jsonify(version='0.1.5',workspace=str(app.config['DESKTOP_WORKSPACE']),offline=True)

def active_jobs(workspace):
    for pattern in ('batches/*/batch.json','outputs/stereo-*/batch.json'):
        for path in workspace.glob(pattern):
            try:
                if json.loads(path.read_text()).get('status') in ('queued','running'):return True
            except (OSError,ValueError):pass
    return False

def mark_interrupted(workspace):
    for pattern in ('batches/*/batch.json','outputs/stereo-*/batch.json'):
        for path in workspace.glob(pattern):
            try:
                state=json.loads(path.read_text())
                if state.get('status') not in ('queued','running'):continue
                state.update(status='interrupted' if pattern.startswith('batches') else 'failed',message='Analysis was interrupted. Completed results are saved; run the batch again to retry.')
                for entry in state.get('entries',[]):
                    if entry.get('status') in ('running','queued','pending'):entry.update(status='failed',message='Interrupted')
                temp=path.with_suffix('.recovery');temp.write_text(json.dumps(state));temp.replace(path)
            except (ValueError,OSError):pass
    (workspace/'.desktop-active').unlink(missing_ok=True)

def run(bundle, workspace, ready):
    workspace.mkdir(parents=True,exist_ok=True)
    lock=(workspace/'.desktop.lock').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:raise RuntimeError('Behavior Studio is already running. Open the existing window.')
    install_workspace(bundle,workspace)
    mark_interrupted(workspace)
    os.chdir(workspace)
    python=bundle/('runtime/python.exe' if os.name=='nt' else 'runtime/bin/python3')
    if os.name=='nt':
        import subprocess
        class QuietProcess(subprocess.Popen):
            def __init__(self,*args,**kwargs):
                kwargs['creationflags']=kwargs.get('creationflags',0)|subprocess.CREATE_NO_WINDOW
                super().__init__(*args,**kwargs)
        subprocess.Popen=QuietProcess
    os.environ.update(DLC_PYTHON=str(python),DLC_LIGHT='True',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0,str(workspace))
    import app as application
    application.app.config['DESKTOP_WORKSPACE']=str(workspace)
    token=secrets.token_urlsafe(32)
    secure_app(application.app,token)
    from werkzeug.serving import make_server, WSGIRequestHandler
    class QuietHandler(WSGIRequestHandler):
        def log_request(self, *args, **kwargs):pass
    server=make_server('127.0.0.1',0,application.app,threaded=True,request_handler=QuietHandler)
    def watch_jobs():
        marker=workspace/'.desktop-active'
        while True:
            if active_jobs(workspace):marker.touch()
            else:marker.unlink(missing_ok=True)
            time.sleep(.5)
    threading.Thread(target=watch_jobs,daemon=True).start()
    ready.parent.mkdir(parents=True,exist_ok=True)
    temp=ready.with_suffix('.tmp')
    with open(temp,'w',opener=lambda path,flags:os.open(path,flags,0o600)) as output:
        json.dump(dict(url=f'http://127.0.0.1:{server.server_port}/desktop/connect?token={token}',pid=os.getpid()),output)
    temp.replace(ready)
    print(f'Behavior Studio ready on loopback port {server.server_port}',flush=True)
    server.serve_forever()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('bundle','workspace','ready'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle.resolve(),args.workspace.resolve(),args.ready.resolve())
