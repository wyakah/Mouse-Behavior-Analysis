from pathlib import Path
import io,json
import av,numpy as np,pytest
from app import app
from threechamber.preparation import prepare_trial

def tiny_video(path,width=64,height=64):
 with av.open(str(path),'w') as out:
  s=out.add_stream('libx264',rate=10);s.width=width;s.height=height;s.pix_fmt='yuv420p'
  for _ in range(4):
   for p in s.encode(av.VideoFrame.from_ndarray(np.zeros((height,width,3),np.uint8),format='rgb24')):out.mux(p)
  for p in s.encode():out.mux(p)

def test_local_video_upload_preserves_bytes_and_lists_file(tmp_path,monkeypatch):
 import app as server
 tiny=tmp_path/'fixture.mp4';tiny_video(tiny);root=tmp_path/'workspace';root.mkdir();(root/'prepared').mkdir()
 monkeypatch.setattr(server,'ROOT',root)
 with app.test_client() as c:
  r=c.post('/api/videos/upload',data={'file':(io.BytesIO(tiny.read_bytes()),'trial.mp4')},content_type='multipart/form-data')
  assert r.status_code==200
  assert (root/r.json['name']).read_bytes()==tiny.read_bytes()
  assert any(v['name']==r.json['name'] for v in c.get('/api/videos').json)
  assert c.post('/api/videos/upload',data={'file':(io.BytesIO(b'not video'),'bad.txt')},content_type='multipart/form-data').status_code==400

def test_partial_analysis_and_non_analysis_manifests_are_not_results(tmp_path,monkeypatch):
 import app as server
 monkeypatch.setattr(server,'ROOT',tmp_path)
 d=tmp_path/'outputs'/'localization';d.mkdir(parents=True);(d/'manifest.json').write_text('{"status":"complete"}')
 d=tmp_path/'outputs'/'pending';d.mkdir();(d/'summary.json').write_text('{}')
 with app.test_client() as c:assert c.get('/api/results').json==[]

def test_short_recording_requires_no_trim(tmp_path):
 p=tmp_path/'tiny.mp4';tiny_video(p);original=p.read_bytes();assert prepare_trial(tmp_path,p)==p;assert p.read_bytes()==original
