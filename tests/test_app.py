from app import app
from test_flow import tiny_video
import pytest


@pytest.fixture
def sample_workspace(tmp_path,monkeypatch):
 import app as server
 (tmp_path/'prepared').mkdir()
 for i in range(3):
  tiny_video(tmp_path/f'{i}.mp4')
  tiny_video(tmp_path/'prepared'/f'{i}__first_600s.mp4')
 monkeypatch.setattr(server,'ROOT',tmp_path)
 return tmp_path


def test_video_inventory_and_frames(sample_workspace):
 with app.test_client() as c:
  r=c.get('/api/videos');assert r.status_code==200
  videos=r.json;assert sum(not v['prepared'] for v in videos)==3
  assert sum(v['prepared'] for v in videos)==3
  r=c.get('/api/frame',query_string={'video':videos[0]['name'],'frame':1237})
  assert r.status_code==200 and r.mimetype=='image/jpeg'

def test_origin_and_path_protection():
 with app.test_client() as c:
  assert c.post('/api/analyze',json={},headers={'Origin':'https://example.org'}).status_code==403
  assert c.get('/api/video',query_string={'video':'../secret.mp4'}).status_code==400

def test_missing_dlc_model_error(sample_workspace):
 with app.test_client() as c:
  name=c.get('/api/videos').json[0]['name']
  r=c.post('/api/dlc',json={'video':name,'config':'missing.yaml'})
  assert r.status_code==400 and 'trained' in r.json['error']
