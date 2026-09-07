"""Bounded binary uploads avoid materializing entire videos in the webview."""
import json,os,uuid,time
from pathlib import Path
from flask import request,jsonify
from werkzeug.utils import secure_filename
from threechamber.core import metadata
CHUNK_BYTES=16*1024*1024

def register_uploads(app,root):
    active=app.extensions.setdefault("active_uploads",{})
    def paths(identifier):
        if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Invalid upload identifier.')
        folder=root()/'.cache/upload-temp'
        return folder/(identifier+'.json'),folder/(identifier+'.part')
    @app.post('/api/videos/upload/start')
    def start():
        data=request.get_json();name=data.get('name');size=data.get('size')
        if not isinstance(name,str) or Path(name).suffix.lower() not in ('.mp4','.mov','.avi','.mkv','.m4v'):raise ValueError('Choose a video file.')
        if type(size) is not int or size<=0:raise ValueError('The selected file is empty.')
        identifier=uuid.uuid4().hex;info,part=paths(identifier);info.parent.mkdir(parents=True,exist_ok=True)
        safe=secure_filename(Path(name).stem)[:180] or 'video'
        info.write_text(json.dumps(dict(name=safe+Path(name).suffix.lower(),size=size)));part.touch();active[identifier]=time.monotonic()
        return jsonify(id=identifier,chunk_bytes=CHUNK_BYTES)
    @app.post('/api/videos/upload/chunk')
    def chunk():
        identifier=request.args.get('id','');active[identifier]=time.monotonic();info,part=paths(identifier);data=json.loads(info.read_text());offset=int(request.args.get('offset','-1'))
        if offset!=part.stat().st_size:return jsonify(error='Upload offset changed. Retry this file.'),409
        received=0
        try:
            with part.open('ab') as output:
                while True:
                    block=request.stream.read(min(1024*1024,CHUNK_BYTES-received+1))
                    if not block:break
                    received+=len(block)
                    if received>CHUNK_BYTES or offset+received>data['size']:raise ValueError('Upload chunk is larger than expected.')
                    output.write(block)
        except Exception:
            with part.open('r+b') as output:output.truncate(offset)
            raise
        return jsonify(offset=offset+received)
    @app.post('/api/videos/upload/finish')
    def finish():
        identifier=request.get_json().get('id','');info,part=paths(identifier);data=json.loads(info.read_text())
        if part.stat().st_size!=data['size']:raise ValueError('Upload is incomplete. Retry this file.')
        try:details=metadata(part)
        except Exception as error:raise ValueError('The uploaded file cannot be decoded as a video.') from error
        folder=root()/'videos';folder.mkdir(exist_ok=True);dest=folder/(identifier[:8]+'_'+data['name']);part.replace(dest);info.unlink();active.pop(identifier,None)
        return jsonify(name=str(dest.relative_to(root())),prepared=False,**details)
    @app.post('/api/videos/upload/cancel')
    def cancel():
        identifier=request.get_json().get('id','');info,part=paths(identifier);info.unlink(missing_ok=True);part.unlink(missing_ok=True);active.pop(identifier,None)
        return jsonify(cancelled=True)
