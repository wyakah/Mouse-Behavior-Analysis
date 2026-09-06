from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from threechamber.core import score,analysis_geometry,propose_circles
from threechamber.batch import validate_batch,process_batch
from test_analysis import tracks
from test_flow import tiny_video


def circle_cfg():
 return dict(analysis_mode='circle_zones',confirmed=True,arena=[[0,0],[300,0],[300,200],[0,200]],cup_circles={'diameter_px':40,'left':[50,100],'right':[250,100]},dividers_fraction=[1/3,2/3],pcutoff=.6)

def test_exact_circles_share_radius_boundary_and_move_independently():
 cfg=circle_cfg();xy=[[70,100],[70.001,100],[250,120],[250,120.001]]
 out,s=score(tracks(xy),np.arange(4.),np.ones(4),cfg)
 assert out.left_interaction.tolist()==[True,False,False,False]
 assert out.right_interaction.tolist()==[False,False,True,False]
 assert s['cup_diameter_px']==40 and s['interaction_threshold_cm'] is None
 assert 'nose_x_cm' not in out
 cfg['cup_circles']['diameter_px']=42
 out,_=score(tracks(xy),np.arange(4.),np.ones(4),cfg)
 assert out.left_interaction.tolist()==[True,True,False,False]
 assert out.right_interaction.tolist()==[False,False,True,True]
 cfg['cup_circles']['left']=[90,100]
 out,_=score(tracks([[50,100],[250,100]]),np.arange(2.),np.ones(2),cfg)
 assert out.left_interaction.tolist()==[False,False] and out.right_interaction.tolist()==[False,True]

def test_circle_geometry_rejects_outside_overlap_and_nonfinite():
 cfg=circle_cfg()
 for circles in ({'diameter_px':-1,'left':[50,100],'right':[250,100]}, {'diameter_px':40,'left':[10,100],'right':[250,100]}, {'diameter_px':40,'left':[50,100],'right':[60,100]}, {'diameter_px':40,'left':[float('nan'),100],'right':[250,100]}):
  with pytest.raises(ValueError):analysis_geometry(dict(cfg,cup_circles=circles))

def batch_fixture(tmp_path):
 (tmp_path/'profiles').mkdir();(tmp_path/'outputs').mkdir()
 (tmp_path/'profiles/ethovision_three_chamber.json').write_text(json.dumps({'source_size':[64,64],'crop_xyxy':[0,0,64,64]}))
 cfg=dict(target_side='left',analysis_mode='circle_zones',confirmed=True,arena=[[0,0],[63,0],[63,63],[0,63]],cup_circles={'diameter_px':12,'left':[16,32],'right':[48,32]},dividers_fraction=[1/3,2/3],pcutoff=.6)
 entries=[]
 for i in [1,2]:
  p=tmp_path/f'{i}.mp4';tiny_video(p)
  tracks([[16,32]]*4).to_csv(tmp_path/f'{i}.csv',index_label='frame')
  entries.append(dict(id=str(i),video=p.name,config=deepcopy(cfg),reviewed=True))
 return dict(diameter_px=12,pcutoff=.6,entries=entries)

def test_batch_requires_all_reviews_and_consistent_settings(tmp_path):
 draft=batch_fixture(tmp_path);assert len(validate_batch(tmp_path,draft)['entries'])==2
 for edit in ['review','size','cutoff','id','video']:
  bad=deepcopy(draft)
  if edit=='review':bad['entries'][1]['reviewed']=False
  if edit=='size':bad['entries'][1]['config']['cup_circles']['diameter_px']=14
  if edit=='cutoff':bad['entries'][1]['config']['pcutoff']=.7
  if edit=='id':bad['entries'][1]['id']='1'
  if edit=='video':bad['entries'][1]['video']='1.mp4'
  with pytest.raises(ValueError):validate_batch(tmp_path,bad)

def test_real_batch_analysis_retains_failures_and_continues(tmp_path):
 draft=batch_fixture(tmp_path);batch=validate_batch(tmp_path,draft);batch['id']='batch-fixture';captured=[]
 def tracker(root,working,dest,progress):
  if working.stem=='1':raise ValueError('fixture tracking failure')
  return root/(working.stem+'.csv')
 def exporter(root,b,folder):captured.append(deepcopy(b))
 result=process_batch(tmp_path,batch,tracker=tracker,exporter=exporter)
 assert result['status']=='complete_with_errors'
 assert result['entries'][0]['status']=='failed' and 'summary' not in result['entries'][0]
 assert result['entries'][1]['summary']['left_nose_seconds']==pytest.approx(.4)
 assert (tmp_path/'outputs/batch-fixture-002/review.mp4').exists()
 assert len(captured[0]['entries'])==2
 assert json.loads((tmp_path/'batches/batch-fixture/batch.json').read_text())['status']=='complete_with_errors'

def test_two_video_batch_runs_each_and_retains_one_row_per_id(tmp_path):
 batch=validate_batch(tmp_path,batch_fixture(tmp_path));batch['id']='batch-both'
 result=process_batch(tmp_path,batch,tracker=lambda root,working,dest,progress:root/(working.stem+'.csv'),exporter=lambda *args:None)
 assert result['status']=='complete' and [e['id'] for e in result['entries']]==['1','2']
 assert all(e['summary']['cup_diameter_px']==12 and e['summary']['decoded_frames']==4 for e in result['entries'])
