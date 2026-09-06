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

def test_stereotypy_folder_is_available_without_duplicate_upload(sample_workspace):
 folder=sample_workspace/'stereotypy_videos';folder.mkdir()
 tiny_video(folder/'mouse.mov')
 with app.test_client() as c:
  entries=c.get('/api/videos').json
  match=[v for v in entries if v['name']=='stereotypy_videos/mouse.mov']
  assert len(match)==1 and match[0]['prepared'] is False

def test_origin_and_path_protection():
 with app.test_client() as c:
  assert c.post('/api/analyze',json={},headers={'Origin':'https://example.org'}).status_code==403
  assert c.get('/api/video',query_string={'video':'../secret.mp4'}).status_code==400

def test_missing_dlc_model_error(sample_workspace):
 with app.test_client() as c:
  name=c.get('/api/videos').json[0]['name']
  r=c.post('/api/dlc',json={'video':name,'config':'missing.yaml'})
  assert r.status_code==400 and 'trained' in r.json['error']


def test_side_view_session_keeps_chamber_draft_and_statistics(sample_workspace, monkeypatch):
 import app as server
 monkeypatch.setattr(server.pool,'submit',lambda fn:fn())
 with app.test_client() as c:
  draft=dict(test_id='three_chamber',diameter_px=110,pcutoff=.6,
             statistics={'mode':'within_sex','alpha':.05,'metrics':['left_nose_seconds']},
             entries=[{'id':'existing-mouse','video':'0.mp4','sex':'female','genotype':'WT','reviewed':True}])
  assert c.post('/api/batch/draft',json=draft).status_code==200
  result=c.post('/api/stereotypy/sessions',json=dict(video='1.mp4',animal_id='side-mouse',session_id='day1',apparatus_id='side-cage',view_confirmed=True))
  assert result.status_code==202 and result.json['status']=='complete'
  url='/api/stereotypy/sessions/'+result.json['session_id']
  saved=c.post(url,json=dict(revision=0,annotator_id='TEST',start_s=0,end_s=.4,
                            annotations=[dict(behavior='grooming',label='present',start_s=0,end_s=.2)]))
  assert saved.status_code==200
  assert saved.json['summary'][0]['active_seconds']==pytest.approx(.2)
  assert c.get('/api/batch/draft').json==draft
  assert c.get('/api/statistics/options').status_code==200
  assert c.post('/api/batches',json=dict(draft,test_id='stereotypy')).status_code==400
  assert c.get(url).json['revision']==1
  assert c.get('/').status_code==200 and c.get('/stereotypy').status_code==200
