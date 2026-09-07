import json
from pathlib import Path
import pytest
from flask import Flask
from threechamber.storage import register_storage,prepare_workspace,identity
from threechamber.uploads import register_uploads

@pytest.fixture
def storage(tmp_path,monkeypatch):
    engine=tmp_path/'engine';engine.mkdir()
    for name in ('scripts','profiles','static','threechamber','stereotypy'):
        (engine/name).mkdir();(engine/name/'example.py').write_text('# engine')
    state={'root':engine,'busy':False};app=Flask(__name__);app.testing=True
    @app.errorhandler(ValueError)
    def invalid(e):return {'error':str(e)},400
    register_storage(app,engine,lambda:state['root'],lambda p:state.update(root=p),lambda:state['busy'])
    register_uploads(app,lambda:state['root'])
    monkeypatch.setattr('threechamber.uploads.metadata',lambda p:{'duration_seconds':1,'width':100,'height':100})
    return app.test_client(),state,engine

def test_external_workspace_preserves_old_data_and_rejects_stale_requests(storage,tmp_path):
    c,state,engine=storage;(engine/'original.mp4').write_bytes(b'original')
    drive=tmp_path/'External Drive';drive.mkdir()
    r=c.post('/api/storage',json={'path':str(drive),'create_workspace':True});assert r.status_code==200
    target=drive/'Mouse Behavior Analysis';assert state['root']==target
    assert (target/'outputs').is_dir() and (target/'scripts/example.py').is_file()
    assert (engine/'original.mp4').read_bytes()==b'original'
    assert json.loads((engine/'.storage-preferences.json').read_text())['path']==str(target)
    stale=c.post('/api/videos/upload/start',headers={'X-Workspace':identity(engine)},json={'name':'a.mp4','size':1});assert stale.status_code==409
    state['busy']=True
    assert c.post('/api/storage',json={'path':str(engine)}).status_code==409
    state['busy']=False
    assert c.post('/api/storage',json={'path':str(engine)}).status_code==200

def test_seven_plus_uploads_use_binary_chunks_and_offsets(storage):
    c,state,engine=storage
    for i in range(8):
        upload=c.post('/api/videos/upload/start',json={'name':f'mouse{i}.mp4','size':7}).json['id']
        route='/api/videos/upload/chunk?id='+upload
        assert c.post(route+'&offset=0',data=b'abc',content_type='application/octet-stream').json['offset']==3
        assert c.post(route+'&offset=0',data=b'abc').status_code==409
        assert c.post('/api/videos/upload/finish',json={'id':upload}).status_code==400
        assert c.post(route+'&offset=3',data=b'defg').json['offset']==7
        result=c.post('/api/videos/upload/finish',json={'id':upload});assert result.status_code==200
        assert (state['root']/result.json['name']).read_bytes()==b'abcdefg'
    assert len(list((engine/'videos').iterdir()))==8
    assert not list((engine/'.cache/upload-temp').iterdir())

def test_oversized_chunk_rolls_back_and_cancel_removes_partial(storage,monkeypatch):
    c,_,engine=storage;monkeypatch.setattr('threechamber.uploads.CHUNK_BYTES',3)
    upload=c.post('/api/videos/upload/start',json={'name':'a.mp4','size':8}).json['id']
    assert c.post('/api/videos/upload/chunk?id='+upload+'&offset=0',data=b'1234').status_code==400
    assert (engine/'.cache/upload-temp'/f'{upload}.part').stat().st_size==0
    assert c.post('/api/videos/upload/cancel',json={'id':upload}).status_code==200

def test_nonempty_unrelated_folder_never_overwritten(storage,tmp_path):
    c,_,engine=storage;target=tmp_path/'unrelated';target.mkdir();(target/'keep.txt').write_text('keep')
    assert c.post('/api/storage',json={'path':str(target)}).status_code==400
    assert (target/'keep.txt').read_text()=='keep'
    assert c.post('/api/storage',json={'path':str(engine/'scripts'),'create_workspace':True}).status_code==400

def test_real_video_decodes_after_chunk_upload(storage,tmp_path,monkeypatch):
    import cv2,numpy as np
    from threechamber.core import metadata
    monkeypatch.setattr('threechamber.uploads.metadata',metadata)
    c,state,_=storage
    source=tmp_path/'mouse.avi';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    for i in range(10):writer.write(np.full((48,64,3),i*20,np.uint8))
    writer.release();data=source.read_bytes()
    upload=c.post('/api/videos/upload/start',json={'name':'mouse.avi','size':len(data)}).json['id']
    assert c.post('/api/videos/upload/chunk?id='+upload+'&offset=0',data=data).status_code==200
    result=c.post('/api/videos/upload/finish',json={'id':upload})
    assert result.status_code==200 and result.json['width']==64
    assert (state['root']/result.json['name']).read_bytes()==data

def test_start_accepts_files_over_four_gb_without_buffering(storage):
    c,_,engine=storage
    r=c.post('/api/videos/upload/start',json={'name':'large.mp4','size':5*1024**3})
    assert r.status_code==200
    assert (engine/'.cache/upload-temp'/(r.json['id']+'.part')).stat().st_size==0
    c.post('/api/videos/upload/cancel',json={'id':r.json['id']})

def test_external_drive_without_symlink_support_copies_models(storage,tmp_path,monkeypatch):
    _,_,engine=storage;model=engine/'outputs/stereotypy-training/v2';model.mkdir(parents=True);(model/'weights.pt').write_bytes(b'weights')
    target=tmp_path/'exfat';target.mkdir()
    def unsupported(*args,**kwargs):raise OSError('Links unsupported')
    monkeypatch.setattr(Path,'symlink_to',unsupported)
    prepare_workspace(engine,target)
    assert (target/'outputs/stereotypy-training/v2/weights.pt').read_bytes()==b'weights'

def test_cancel_releases_upload_lock_even_when_drive_cannot_be_cleaned(storage,monkeypatch):
    c,_,engine=storage
    identifier=c.post('/api/videos/upload/start',json={'name':'a.mp4','size':8}).json['id']
    original=Path.unlink
    def disconnected(self,*args,**kwargs):
        if self.name==identifier+'.json':raise OSError('Drive disconnected')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',disconnected)
    with pytest.raises(OSError):c.post('/api/videos/upload/cancel',json={'id':identifier})
    assert identifier not in c.application.extensions['active_uploads']
