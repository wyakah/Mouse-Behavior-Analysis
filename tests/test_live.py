"""Preview telemetry must stay aligned, bounded, optional, and scientifically inert."""
from copy import deepcopy
from pathlib import Path
import base64
import json

import cv2
import numpy as np
import pandas as pd
import pytest
from flask import Flask

from threechamber.live import LivePublisher,FrameCache,PredictionTap,landmark_states,atomic_json
from threechamber.batch import register_batch,process_batch,validate_batch
from threechamber.core import analyze,timeline
from test_batch import batch_fixture


def context():
    return dict(batch_id='test',recording_index=0,recording_id='a',crop=[2,2,18,18],cutoff=.6)


def test_snapshot_pairs_crop_and_landmarks_with_exact_frame_and_throttles(tmp_path):
    now=[10.]
    live=LivePublisher(tmp_path,context(),clock=lambda:now[0],wall=lambda:123.,interval=.5)
    live.phase('tracking',total_frames=100)
    image=np.zeros((20,20,3),np.uint8);image[2:18,2:18]=(0,0,255)
    row=dict(nose_x=6.,nose_y=7.,nose_likelihood=.8,center_x=12.,center_y=11.,center_likelihood=.2)
    assert live.frame(image,31,1.376,row)
    snapshot=json.loads((tmp_path/'snapshot.json').read_text())
    assert snapshot['frame_index']==31 and snapshot['frames_done']==32 and snapshot['source_time_s']==1.376
    assert snapshot['landmarks']['nose']['state']=='accepted'
    assert snapshot['landmarks']['center']['state']=='uncertain'
    assert snapshot['landmarks']['tail_base']['state']=='missing'
    raw=np.frombuffer(base64.b64decode(snapshot['image'].split(',')[1]),np.uint8)
    decoded=cv2.imdecode(raw,cv2.IMREAD_COLOR)
    assert decoded.shape==(16,16,3) and decoded[8,8,2]>245
    first=(tmp_path/'snapshot.json').read_bytes()
    assert not live.frame(image,32,1.42,row)
    assert (tmp_path/'snapshot.json').read_bytes()==first
    now[0]+=.5
    assert live.frame(image,45,2.1,row) and live.published_frames==2
    assert not list(tmp_path.glob('*.tmp'))


def test_stage_change_removes_old_frame_and_counts(tmp_path):
    live=LivePublisher(tmp_path,context());live.phase('localizing',total_frames=20)
    live.frame(np.zeros((20,20,3),np.uint8),5,.2,bbox=[3,3,10,10])
    live.phase('tracking','Loading model',20)
    snap=json.loads((tmp_path/'snapshot.json').read_text())
    assert snap['stage']=='tracking' and snap['image'] is None
    assert snap['frame_index'] is None and snap['landmarks']=={} and snap['bbox'] is None


def test_invalid_or_unavailable_points_never_become_confident_landmarks():
    row=dict(nose_x=float('nan'),nose_y=2,nose_likelihood=.99,center_x=3,center_y=4,center_likelihood=1.5,tail_base_x=3,tail_base_y=4,tail_base_likelihood=.6)
    points=landmark_states(row,.6,20,20)
    assert points['nose']['state']==points['center']['state']=='missing'
    assert points['tail_base']['state']=='accepted'
    row.update(tail_base_x=25)
    assert landmark_states(row,.6,20,20)['tail_base']['state']=='missing'


def test_preview_disk_failure_is_isolated_from_processing(tmp_path,monkeypatch):
    import threechamber.live as module
    monkeypatch.setattr(module,'atomic_json',lambda *a:(_ for _ in ()).throw(OSError('disk error')))
    live=LivePublisher(tmp_path,context())
    with pytest.warns(RuntimeWarning,match='analysis continues'):assert live.phase('tracking') is False
    assert live.disabled and not live.frame(np.zeros((20,20,3),np.uint8),0,0)


def test_frame_cache_is_bounded_and_cannot_pair_a_different_frame():
    cache=FrameCache(3)
    for i in range(6):cache.remember(i,np.array([i]))
    assert len(cache.frames)==3 and cache.take(1) is None
    assert cache.take(4).tolist()==[4] and list(cache.frames)==[5]
    assert cache.take(4) is None


def test_dlc_writer_retains_all_predictions_even_when_preview_fails():
    seen=[]
    def observe(index,prediction):
        seen.append(index)
        if index==1:raise RuntimeError('preview only')
    tap=PredictionTap(observe);tap.open()
    poses=[np.full((1,3,3),i,dtype=float) for i in range(4)]
    tap.add_prediction(poses[0],features=None)
    with pytest.warns(RuntimeWarning):tap.add_prediction(poses[1])
    for p in poses[2:]:tap.add_prediction(p)
    tap.close()
    assert seen==[0,1] and len(tap.predictions)==4
    for expected,actual in zip(poses,tap.predictions):np.testing.assert_array_equal(actual['bodyparts'],expected)


def test_preview_on_off_preserves_measurements_and_review_timestamps(tmp_path):
    batch=batch_fixture(tmp_path);cfg=batch['entries'][0]['config'];video=tmp_path/'1.mp4';tracks=tmp_path/'1.csv'
    baseline=analyze(video,tracks,cfg,tmp_path/'baseline')
    live=LivePublisher(tmp_path/'live',dict(context(),crop=[0,0,64,64]),interval=0)
    def progress(message):pass
    progress.live=live
    actual=analyze(video,tracks,cfg,tmp_path/'preview',progress)
    assert actual==baseline
    for name in ['frames.csv','summary.csv','bouts.csv']:
        assert (tmp_path/'baseline'/name).read_bytes()==(tmp_path/'preview'/name).read_bytes()
    a,_=timeline(tmp_path/'baseline/review.mp4');b,_=timeline(tmp_path/'preview/review.mp4')
    np.testing.assert_array_equal(a,b)
    snapshot=json.loads((tmp_path/'live/snapshot.json').read_text())
    assert snapshot['stage']=='rendering' and snapshot['frame_index']==3
    assert live.published_frames==4


def test_next_recording_gets_its_own_live_context_and_failure_continues(tmp_path):
    batch=validate_batch(tmp_path,batch_fixture(tmp_path));batch['id']='live-batch';seen=[]
    def tracker(root,video,dest,progress):
        seen.append(deepcopy(progress.live.context))
        if len(seen)==1:raise RuntimeError('first recording fails')
        return root/'2.csv'
    result=process_batch(tmp_path,batch,tracker=tracker,exporter=lambda *a:None)
    assert result['status']=='complete_with_errors'
    assert [s['recording_id'] for s in seen]==['1','2'] and [s['recording_index'] for s in seen]==[0,1]
    snap=json.loads((tmp_path/'batches/live-batch/live/snapshot.json').read_text())
    assert snap['recording_id']=='2' and snap['stage']=='complete_with_errors'


class HoldExecutor:
    def submit(self,fn):self.work=fn


def test_live_api_is_no_store_omits_repeated_images_and_rejects_stale_recording(tmp_path):
    draft=batch_fixture(tmp_path);app=Flask(__name__);executor=HoldExecutor();register_batch(app,lambda:tmp_path,executor)
    with app.test_client() as client:
        batch=client.post('/api/batches',json=draft).json;identifier=batch['id'];url=f'/api/batches/{identifier}/live'
        assert client.get(url).json['snapshot'] is None
        batch['current_index']=0;atomic_json(tmp_path/'batches'/identifier/'batch.json',batch)
        live=LivePublisher(tmp_path/'batches'/identifier/'live',dict(context(),batch_id=identifier))
        live.phase('tracking',total_frames=4);live.frame(np.zeros((20,20,3),np.uint8),0,0)
        response=client.get(url);snapshot=response.json['snapshot'];revision=snapshot['revision']
        assert response.headers['Cache-Control']=='no-store' and snapshot['image'].startswith('data:image/jpeg;')
        repeat=client.get(url,query_string={'since':revision}).json
        assert repeat['unchanged'] and 'image' not in repeat['snapshot']
        assert 'image' not in client.get(url,query_string={'image':0}).json['snapshot']
        batch['current_index']=1;atomic_json(tmp_path/'batches'/identifier/'batch.json',batch)
        assert client.get(url).json['snapshot'] is None
        (tmp_path/'batches'/identifier/'live/snapshot.json').write_text('invalid')
        assert client.get(url).json['snapshot'] is None
    restarted=Flask('restart');register_batch(restarted,lambda:tmp_path,HoldExecutor())
    with restarted.test_client() as client:
        assert client.get('/api/batches').json[0]['status']=='interrupted'
        assert client.get(url).json['status']=='interrupted'


def test_prefetched_cropped_frames_remain_paired_at_dlc_queue_depth(tmp_path):
    # DLC prefetches four batches plus a producer/active batch ahead of results.
    cache=FrameCache(8*(4+3))
    for i in range(48):cache.remember(i,np.full((16,16,3),i,dtype=np.uint8))
    for index in range(8):
        frame=cache.take(index)
        assert frame[0,0,0]==index
    live=LivePublisher(tmp_path,dict(context(),source_size=[20,20]),interval=0)
    live.phase('tracking',total_frames=100)
    assert live.frame(cache.take(8),8,.32,row={'nose_x':7,'nose_y':8,'nose_likelihood':.9},image_is_crop=True)
    snapshot=json.loads((tmp_path/'snapshot.json').read_text())
    assert snapshot['frame_index']==8 and snapshot['landmarks']['nose']['x']==7
    assert snapshot['crop']==[2,2,18,18]
