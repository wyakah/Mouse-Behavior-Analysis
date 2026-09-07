"""Persistent, switchable experiment workspaces; engine files stay separate from recordings."""
import hashlib,json,os,shutil,tempfile,threading
from pathlib import Path
from flask import request,jsonify

DATA_DIRS=('videos','prepared','configs','outputs','tracks','reports','batches','labeling','.cache/upload-temp')
MARKER='.behavior-workspace.json'

def identity(path):return hashlib.sha256(str(path).encode()).hexdigest()[:20]

def prepare_workspace(engine,target):
    engine,target=Path(engine).resolve(),Path(target).resolve()
    if target==engine:return target
    if target.is_relative_to(engine) or engine.is_relative_to(target):raise ValueError("Choose a folder outside the application folder.")
    if not target.is_dir():raise ValueError('Choose an existing folder on a connected drive.')
    if not (target/MARKER).exists() and any(target.iterdir()):raise ValueError('Choose an empty folder, or an existing Behavior Analysis workspace.')
    # Probe the actual destination, not a cached permissions flag.
    with tempfile.TemporaryFile(dir=target) as probe:probe.write(b'workspace check')
    (target/MARKER).write_text(json.dumps({'schema':1}))
    for name in DATA_DIRS:(target/name).mkdir(parents=True,exist_ok=True)
    for name in ('scripts','profiles','static','threechamber','stereotypy'):
        shutil.copytree(engine/name,target/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Model links avoid duplicate weights. exFAT drives use a local copy instead.
    for name in ('outputs/stereotypy-training','.cache/huggingface','.cache/torch'):
        source=engine/name;dest=target/name
        if not source.exists():continue
        if dest.is_symlink():dest.unlink()
        elif hasattr(dest,'is_junction') and dest.is_junction():dest.rmdir()
        elif dest.exists():
            shutil.copytree(source,dest,dirs_exist_ok=True);continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        try:
            if os.name=='nt':
                import _winapi
                _winapi.CreateJunction(str(source.resolve()),str(dest))
            else:dest.symlink_to(source.resolve(),target_is_directory=True)
        except OSError:shutil.copytree(source,dest)
    return target


def register_storage(app,engine,get_root,set_root,busy):
    engine=Path(engine);prefs=engine/'.storage-preferences.json';gate=threading.Lock()
    def status():
        root=get_root();available=root.is_dir();free=None
        if available:
            try:free=shutil.disk_usage(root).free
            except OSError:available=False
        return dict(path=str(root),workspace_id=identity(root),available=available,
                    free_bytes=free)
    @app.before_request
    def workspace_guard():
        if request.method!='POST':return
        if not gate.acquire(blocking=False):return jsonify(error='An upload or workspace change is in progress. Try again when it finishes.'),409
        from flask import g
        g.workspace_locked=True
        expected=request.headers.get('X-Workspace')
        if expected and expected!=identity(get_root()):return jsonify(error='The storage location changed. Refresh this page before continuing.'),409
        if request.path!='/api/storage' and not get_root().is_dir():return jsonify(error='The storage drive is disconnected. Reconnect it or choose another location.'),409
    @app.teardown_request
    def release_gate(error):
        from flask import g
        if getattr(g,'workspace_locked',False):g.workspace_locked=False;gate.release()
    @app.get('/api/storage')
    def storage_info():return jsonify(status())
    @app.get('/api/storage/folders')
    def folders():
        p=Path(request.args.get('path') or Path.home()).expanduser()
        if not p.is_absolute() or not p.is_dir():raise ValueError('Folder is unavailable. Reconnect the drive or choose another folder.')
        drives=list(os.listdrives()) if hasattr(os,'listdrives') else [str(Path.home()),'/']
        if Path('/Volumes').is_dir():drives.extend(str(x) for x in Path('/Volumes').iterdir() if x.is_dir())
        try:children=sorted((x for x in p.iterdir() if x.is_dir() and not x.name.startswith('.')),key=lambda x:x.name.lower())
        except PermissionError:raise ValueError('This folder is not accessible. Choose another location.')
        return jsonify(path=str(p),parent=str(p.parent),workspace=(p/MARKER).exists() or p.resolve()==engine.resolve(),folders=[dict(name=x.name,path=str(x)) for x in children],drives=drives)
    @app.post('/api/storage')
    def change_storage():
        if busy():return jsonify(error='Wait for the current analysis to finish before changing storage.'),409
        data=request.get_json();raw=data.get('path') if isinstance(data,dict) else None
        if not isinstance(raw,str) or not raw.strip():raise ValueError('Choose a storage folder.')
        target=Path(raw).expanduser()
        if not target.is_absolute():raise ValueError('Choose an absolute folder path.')
        if data.get('create_workspace'):
            if not target.is_dir():raise ValueError('Connect the drive and choose an existing parent folder.')
            target=target/'Mouse Behavior Analysis';target.mkdir(exist_ok=True)
        target=prepare_workspace(engine,target)
        temp=prefs.with_suffix('.tmp');temp.write_text(json.dumps({'path':str(target)}));temp.replace(prefs)
        set_root(target)
        return jsonify(status())
