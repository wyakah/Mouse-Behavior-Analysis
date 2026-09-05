import json
from pathlib import Path
import cv2,numpy as np,pandas as pd,pytest
from flask import Flask
from threechamber.subject import SubjectSelector,validate_profile,ANCHORS
from threechamber.labeling import validate_annotation,export_labels,register_labeling,PARTS
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def profile():return json.loads((ROOT/'profiles/ethovision_three_chamber.json').read_text())
def pose(x=100,y=100,p=.9):
 return np.array([[x+20,y,p],[x+12,y-5,p],[x+12,y+5,p],[x,y,p],[x-20,y,p]])
def test_profile_rejects_resized_or_invalid_crop(profile):
 assert validate_profile(profile,1024,768)==[190,750,175,575]
 with pytest.raises(ValueError,match='resolution'):validate_profile(profile,640,480)
 profile['crop_xyxy']=[-1,0,750,575]
 with pytest.raises(ValueError,match='within'):validate_profile(profile,1024,768)
def test_second_candidate_selected_without_identity_by_rank(profile):
 bad=pose();bad[4,:2]=bad[0,:2]
 r=SubjectSelector(profile).select([bad,pose()],ANCHORS,0,'a')
 assert r['candidate_index']==1

def test_ambiguous_mice_are_unknown(profile):
 r=SubjectSelector(profile).select([pose(),pose(250,p=.86)],ANCHORS,0,'a')
 assert r['candidate_index'] is None and r['reason']=='ambiguous_candidates'

def test_continuity_rejects_jump_without_filling_and_resets(profile):
 s=SubjectSelector(profile);s.select([pose()],ANCHORS,0,'a')
 r=s.select([pose(400)],ANCHORS,.04,'a');assert r['candidate_index'] is None and r['temporal_used']
 assert s.previous_time==0
 r=s.select([pose(400)],ANCHORS,1,'a');assert r['candidate_index']==0 and not r['temporal_used']
 r=s.select([pose()],ANCHORS,1.04,'b');assert r['candidate_index']==0 and not r['temporal_used']

def test_temporal_preference_uses_real_candidate_not_interpolated_pose(profile):
 s=SubjectSelector(profile);s.select([pose()],ANCHORS,0,'a')
 r=s.select([pose(150,p=.91),pose(101,p=.9)],ANCHORS,.04,'a')
 assert r['candidate_index']==1 and s.previous_center.tolist()==[101,100]

def test_missing_and_unconfident_candidates_remain_unknown(profile):
 s=SubjectSelector(profile)
 assert s.select([pose(p=.1)],ANCHORS,0,'a')['candidate_index'] is None
 assert s.select(np.full((5,5,3),-1),ANCHORS,1,'a')['candidate_index'] is None

def test_human_label_validation(profile):
 e={'source':'675.mp4','source_frame':1};d={'points':{p:None for p in PARTS},'annotator':'Reviewer','reviewed':True}
 assert validate_annotation(d,e,profile)['points']['nose'] is None
 for changes in [{'reviewed':False},{'annotator':''},{'points':{'nose':[200,200]}},{'points':dict(d['points'],nose=[900,400])}]:
  with pytest.raises(ValueError):validate_annotation(dict(d,**changes),e,profile)

def test_export_only_reviewed_labels_with_crop_offsets_and_video_split(tmp_path,profile):
 (tmp_path/'labeling/images').mkdir(parents=True);(tmp_path/'profiles').mkdir()
 (tmp_path/'profiles/ethovision_three_chamber.json').write_text(json.dumps(profile))
 entries=[];anns={}
 for i,source in enumerate(['675_a.mp4','685_b.mp4','678_c.mp4']):
  e={'id':str(i),'source':source,'source_frame':4,'image':f'labeling/images/{i}.png'};entries.append(e);cv2.imwrite(str(tmp_path/e['image']),np.zeros((768,1024,3),np.uint8))
  if i<2:
   points={p:([200,200] if p=='nose' else None) for p in PARTS}
   if i==0:points.update(left_ear=[210,210],right_ear=[220,210])
   anns[str(i)]=validate_annotation(dict(points=points,reviewed=True,annotator='Human'),e,profile)
 (tmp_path/'labeling/queue.json').write_text(json.dumps(entries));(tmp_path/'labeling/annotations.json').write_text(json.dumps(anns))
 out,manifest=export_labels(tmp_path)
 assert manifest['reviewed_frames']==2
 assert manifest['parts']==['nose','center','tail_base']
 canonical=json.loads((out/'annotations.json').read_text())
 assert set(canonical['0']['points'])==set(PARTS)
 assert canonical['0']['reviewed_at']==anns['0']['reviewed_at']
 assert json.loads((out/'original_annotations.json').read_text())==anns
 assert json.loads((tmp_path/'labeling/annotations.json').read_text())==anns
 from scripts.finetune_specialized import validate_export,validate_label_values
 split=json.loads((out/'split.json').read_text())
 validate_export(manifest,split,canonical)
 merged=pd.concat([pd.read_hdf(p) for p in sorted((out/'labeled-data').glob('*/CollectedData_Researcher.h5'))])
 validate_label_values(merged,manifest,split,canonical)
 assert {e['split'] for e in json.loads((out/'split.json').read_text())}=={'train','validation'}
 df=pd.read_hdf(out/'labeled-data/675_a/CollectedData_Researcher.h5')
 assert list(df.columns.get_level_values('bodyparts').unique())==PARTS
 assert df.iloc[0][('Researcher','nose','x')]==10
 assert df.iloc[0][('Researcher','nose','y')]==25
 assert np.isnan(df.iloc[0][('Researcher','center','x')])
 assert cv2.imread(str(out/'labeled-data/675_a/img000004.png')).shape[:2]==(400,560)
 assert not (out/'labeled-data/678_c').exists()
 app=Flask(__name__);register_labeling(app,tmp_path)
 @app.errorhandler(ValueError)
 def err(e):return {'error':str(e)},400
 with app.test_client() as c:
  assert c.get('/api/labeling').json['parts']==PARTS
  assert c.post('/api/labeling/2',json=dict(anns['0'],reviewed=False)).status_code==400
  assert c.post('/api/labeling/2',json=anns['0']).status_code==200
  assert c.post('/api/labeling/2',json=anns['0']).status_code==200
  assert len(list((tmp_path/'labeling/history').glob('*.json')))==1

def test_empty_review_queue_cannot_export(tmp_path):
 (tmp_path/'labeling').mkdir();(tmp_path/'labeling/queue.json').write_text('[]')
 with pytest.raises(ValueError,match='No manually reviewed'):export_labels(tmp_path)

def test_fresh_checkout_has_empty_labeling_queue(tmp_path,profile):
 (tmp_path/'profiles').mkdir()
 (tmp_path/'profiles/ethovision_three_chamber.json').write_text(json.dumps(profile))
 app=Flask(__name__);register_labeling(app,tmp_path)
 with app.test_client() as c:
  response=c.get('/api/labeling')
  assert response.status_code==200
  assert response.json['entries']==[] and response.json['annotations']=={}
  assert response.json['parts']==['nose','center','tail_base']
 with pytest.raises(ValueError,match='No manually reviewed'):export_labels(tmp_path)

def test_validation_counts_misses_in_tolerance_denominator():
 from scripts.evaluate_reviewed import evaluate
 q=[{'id':str(i),'source':'685.mp4','source_frame':i} for i in range(3)]
 ann={str(i):{'reviewed':True,'points':{'nose':[0,0],'center':None}} for i in range(3)}
 d=pd.DataFrame([dict(source='685.mp4',source_frame=i,nose_x=x,nose_y=0,nose_likelihood=p,center_x=0,center_y=0,center_likelihood=0) for i,x,p in [(0,3,.9),(1,10,.9),(2,0,.1)]])
 r=evaluate(d,q,ann)['summary'][0]
 assert r['coverage']==pytest.approx(2/3)
 assert r['median_error_px']==6.5
 assert r['fraction_within_tolerance_including_misses']==pytest.approx(1/3)

def test_cropped_preview_profile_and_specialized_guards(tmp_path,monkeypatch,profile):
 import app as server
 from test_flow import tiny_video
 (tmp_path/'profiles').mkdir()
 (tmp_path/'profiles/ethovision_three_chamber.json').write_text(json.dumps(profile))
 tiny_video(tmp_path/'trial.mp4',1024,768)
 monkeypatch.setattr(server,'ROOT',tmp_path)
 from app import app
 with app.test_client() as c:
  assert c.get('/api/profile').json['crop_xyxy']==[190,175,750,575]
  name=c.get('/api/videos').json[0]['name']
  r=c.get('/api/frame',query_string={'video':name,'frame':10,'crop':'arena'})
  assert cv2.imdecode(np.frombuffer(r.data,np.uint8),cv2.IMREAD_COLOR).shape[:2]==(400,560)
  r=c.post('/api/specialized',json={'video':name})
  assert r.status_code==400 and 'crop' in r.json['error']
