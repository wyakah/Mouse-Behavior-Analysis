import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import av
from threechamber.core import score,load_tracks,bouts,analyze,timeline,calibration

@pytest.fixture
def cfg():
 return dict(confirmed=True,width_cm=30,depth_cm=20,arena=[[0,0],[300,0],[300,200],[0,200]],cups={'left':[[30,80],[50,80],[50,120],[30,120]],'right':[[250,80],[270,80],[270,120],[250,120]]},pcutoff=.6,target_side='left')

def tracks(nose,center=None,p=None):
 a=np.asarray(nose); c=np.asarray(center if center is not None else nose)
 return pd.DataFrame(dict(nose_x=a[:,0],nose_y=a[:,1],nose_likelihood=p if p is not None else np.ones(len(a)),center_x=c[:,0],center_y=c[:,1],center_likelihood=np.ones(len(a))))

def test_one_cm_boundary_and_cup_interior(cfg):
 df=tracks([[60,100],[60.1,100],[40,100],[240,100]])
 out,s=score(df,np.array([0.,.1,.3,.6]),np.array([.1,.2,.3,.4]),cfg)
 assert out.left_interaction.tolist()==[True,False,False,False]
 assert s['left_nose_seconds']==pytest.approx(.1)
 assert s['right_nose_seconds']==pytest.approx(.4)
 assert s['nose_unscoreable_seconds']==pytest.approx(.3)
 assert s['preference_index']==pytest.approx(-.6)

def test_chambers_partition_and_uncertainty(cfg):
 df=tracks([[10,10],[100,10],[200,10],[301,10],[50,10]],p=[1,1,.1,1,1]);df.loc[4,'center_likelihood']=.1
 out,s=score(df,np.arange(5.),np.ones(5),cfg)
 assert out.chamber.tolist()==['left','center','right','outside','unknown']
 assert sum(s[k+'_chamber_seconds'] for k in ['left','center','right','outside','unknown'])==5
 assert not out.nose_valid[2]

def test_trim_partial_frames_and_bouts(cfg):
 cfg.update(start_s=.15,end_s=.35)
 out,s=score(tracks([[60,100]]*4),np.arange(4)/10,np.ones(4)/10,cfg)
 out.attrs['start_s']=.15
 assert s['left_nose_seconds']==pytest.approx(.2)
 b=bouts(out);assert len(b)==1 and b.iloc[0].duration_s==pytest.approx(.2) and b.iloc[0].start_s==.15

def test_no_interpolation_and_empty_index(cfg):
 df=tracks([[150,100]]*3,p=[1,.1,1]);out,s=score(df,np.arange(3.),np.ones(3),cfg)
 assert s['preference_index'] is None and s['nose_unscoreable_seconds']==1
 assert not out.nose_valid[1]

def test_degenerate_and_unconfirmed(cfg):
 cfg['confirmed']=False
 with pytest.raises(ValueError,match='Confirm'):calibration(cfg)
 cfg['confirmed']=True;cfg['arena']=[[0,0]]*4
 with pytest.raises(ValueError):calibration(cfg)

def test_mismatched_frames(cfg):
 with pytest.raises(ValueError,match='mismatch'):score(tracks([[0,0]]),np.arange(2.),np.ones(2),cfg)

def test_dlc_three_row_csv(tmp_path):
 cols=pd.MultiIndex.from_product([['DLC_model'],['nose','center'],['x','y','likelihood']],names=['scorer','bodyparts','coords'])
 p=tmp_path/'dlc.csv';pd.DataFrame([[1,2,.9,3,4,.8]],columns=cols).to_csv(p)
 result=load_tracks(p);assert result.nose_x[0]==1 and result.center_likelihood[0]==.8

def test_multi_animal_requires_subject(tmp_path):
 cols=pd.MultiIndex.from_product([['DLC'],['subject','stimulus'],['nose','center'],['x','y','likelihood']],names=['scorer','individuals','bodyparts','coords'])
 p=tmp_path/'dlc.csv';pd.DataFrame([[.9]*12],columns=cols).to_csv(p)
 with pytest.raises(ValueError,match='subject'):load_tracks(p)
 assert len(load_tracks(p,individual='subject'))==1

def test_missing_frames_rejected(tmp_path):
 p=tmp_path/'flat.csv';df=tracks([[1,2],[3,4]]);df.index=[0,2];df.to_csv(p,index_label='frame')
 with pytest.raises(ValueError,match='every video frame'):load_tracks(p)

def test_perspective_calibration(cfg):
 # Double source resolution; physical behavior should be unchanged.
 df=tracks([[60,100],[240,100]]);a,s=score(df,np.arange(2.),np.ones(2),cfg)
 cfg['arena']=(np.array(cfg['arena'])*2).tolist();cfg['cups']={k:(np.array(v)*2).tolist() for k,v in cfg['cups'].items()}
 df[['nose_x','nose_y','center_x','center_y']]*=2
 b,z=score(df,np.arange(2.),np.ones(2),cfg)
 assert s['left_nose_seconds']==z['left_nose_seconds'] and s['right_nose_seconds']==z['right_nose_seconds']

@pytest.mark.parametrize('zone_mode',[False,True])
def test_end_to_end_export(tmp_path,cfg,zone_mode):
 if zone_mode:
  cfg.update(analysis_mode='drawn_zones',width_cm=0,depth_cm=0,interaction_zones={'left':[[20,60],[70,60],[70,140],[20,140]],'right':[[230,60],[280,60],[280,140],[230,140]]})
 from fractions import Fraction
 video=tmp_path/'synthetic.mp4'
 with av.open(str(video),'w') as c:
  stream=c.add_stream('libx264',rate=10);stream.width=320;stream.height=240;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,1000)
  for i in range(10):
   im=np.full((240,320,3),170,np.uint8);f=av.VideoFrame.from_ndarray(im,format='rgb24');f.pts=i*100;f.time_base=Fraction(1,1000)
   for packet in stream.encode(f):c.mux(packet)
  for packet in stream.encode():c.mux(packet)
 df=tracks([[60,100]]*5+[[240,100]]*5);p=tmp_path/'tracks.csv';df.to_csv(p,index_label='frame')
 out=tmp_path/'result';s=analyze(video,p,cfg,out)
 assert s['left_nose_seconds']==pytest.approx(.5) and s['right_nose_seconds']==pytest.approx(.5)
 assert (out/'manifest.json').exists() and (out/'review.mp4').stat().st_size>1000
 t,dt=timeline(out/'review.mp4');assert len(t)==10 and t[-1]==pytest.approx(.9)
 assert json.loads((out/'manifest.json').read_text())['status']=='complete'
 if zone_mode:
  assert s['interaction_threshold_cm'] is None
  assert s['interaction_measure']=='nose_in_selected_zone'
  assert pd.read_csv(out/'frames.csv').left_cup_distance_cm.isna().all()

def test_hdf5_import(tmp_path):
 cols=pd.MultiIndex.from_product([['DLC'],['nose','center'],['x','y','likelihood']])
 p=tmp_path/'tracks.h5';pd.DataFrame([[1,2,.9,3,4,.8]],columns=cols).to_hdf(p,key='df_with_missing')
 assert load_tracks(p).nose_x[0]==1

def test_overlapping_rings_do_not_double_count(cfg):
 cfg['cups']['right']=[[65,80],[85,80],[85,120],[65,120]]
 out,s=score(tracks([[57.5,100],[60,100],[67,100]]),np.arange(3.),np.ones(3),cfg)
 assert s['left_nose_seconds']==0 and s['right_nose_seconds']==0
 assert s['nose_unscoreable_seconds']==3

def test_chambers_without_scale_preserve_missingness_and_never_invent_cup_times(cfg):
 cfg.update(analysis_mode='chambers_only',width_cm=0,depth_cm=0,cups={'left':[[1,1]],'right':[]},dividers_fraction=[1/3,2/3])
 df=tracks([[10,10],[100,10],[200,10],[150,50]],p=[1,1,1,.1]);df.loc[3,'center_likelihood']=.1
 out,s=score(df,np.arange(4.),np.ones(4),cfg)
 assert out.chamber.tolist()==['left','center','right','unknown']
 assert sum(s[k+'_chamber_seconds'] for k in ['left','center','right','unknown','outside'])==4
 assert s['left_nose_seconds'] is None and s['preference_index'] is None
 assert 'nose_x_cm' not in out and 'nose_x_fraction' in out
 assert not out.cup_metrics_available.any()
 assert s['left_chamber_upper_seconds']==2

def test_chamber_mode_invalid_dividers_and_metric_mode_still_need_scale(cfg):
 cfg.update(analysis_mode='chambers_only',width_cm=0,depth_cm=0,dividers_fraction=[.7,.3])
 with pytest.raises(ValueError,match='fractions'):score(tracks([[10,10]]),np.array([0.]),np.array([1.]),cfg)
 cfg['analysis_mode']='calibrated'
 with pytest.raises(ValueError,match='dimensions'):score(tracks([[10,10]]),np.array([0.]),np.array([1.]),cfg)

def test_selected_zones_use_nose_boundary_timestamps_and_missingness(cfg):
 cfg.update(analysis_mode='drawn_zones',width_cm=0,depth_cm=0,interaction_zones=cfg['cups'])
 # Nose inside the left region counts even if body center is in the center chamber.
 df=tracks([[40,100],[50,100],[50.1,100],[260,100],[40,100],[-1,100]],center=[[150,100]]*6,p=[1,1,1,1,.1,1])
 out,s=score(df,np.array([0.,.1,.3,.6,1.,1.5]),np.array([.1,.2,.3,.4,.5,.5]),cfg)
 assert out.left_interaction.tolist()==[True,True,False,False,False,False]
 assert s['left_nose_seconds']==pytest.approx(.3) and s['right_nose_seconds']==pytest.approx(.4)
 assert s['nose_unscoreable_seconds']==pytest.approx(1.)
 assert s['center_chamber_seconds']==pytest.approx(2.)
 assert s['preference_index']==pytest.approx(-1/7)
 assert s['interaction_threshold_cm'] is None and 'nose_x_cm' not in out
 assert out.left_cup_distance_cm.isna().all() and out.cup_metrics_available.all()
 assert len(bouts(out))==2

def test_selected_zones_reject_missing_crossing_overlap_and_outside(cfg):
 from copy import deepcopy
 cfg.update(analysis_mode='drawn_zones',interaction_zones=deepcopy(cfg['cups']))
 for zones in ({},dict(left=cfg['cups']['left'],right=cfg['cups']['left']),dict(left=[[-1,0],[20,0],[20,20]],right=cfg['cups']['right']),dict(left=[[0,0],[20,20],[0,20],[20,0]],right=cfg['cups']['right'])):
  cfg['interaction_zones']=zones
  with pytest.raises(ValueError):score(tracks([[40,100]]),np.array([0.]),np.array([1.]),cfg)

def test_zone_suggestions_are_separate_and_do_not_infer_scale(cfg):
 from copy import deepcopy
 from threechamber.core import propose_interaction_zones,analysis_geometry
 cfg.update(width_cm=0,depth_cm=0);before=deepcopy(cfg)
 proposal=propose_interaction_zones(cfg)
 assert cfg==before and '25%' in proposal['zone_geometry_source']
 assert min(x for x,y in proposal['interaction_zones']['left'])<30
 cfg.update(proposal,analysis_mode='drawn_zones');analysis_geometry(cfg)
