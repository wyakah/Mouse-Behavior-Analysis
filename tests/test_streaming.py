"""All-frame streaming must preserve ordering, timestamps and scientific output."""
import json
from pathlib import Path
import cv2
import av
import numpy as np
import pandas as pd
import pytest
from flask import Flask
from threechamber.streaming import SegmentWriter,VideoPublisher
from threechamber.annotation import AnnotationRenderer
from threechamber.live import LivePublisher
from threechamber.core import analyze,timeline
from threechamber.batch import register_batch
from test_batch import circle_cfg,batch_fixture


def test_segments_preserve_every_variable_rate_frame_and_duration(tmp_path):
    cfg=circle_cfg();durations=np.resize([.04,.07,.03],103);times=np.r_[0,np.cumsum(durations)[:-1]]
    context=dict(recording_id='mouse',recording_index=0,source_duration_s=float(durations.sum()))
    writer=SegmentWriter(tmp_path,cfg,[300,200],25,context,segment_seconds=.5)
    decoded=[];decoded_durations=[];counts=[]
    for i,(t,dt) in enumerate(zip(times,durations)):
        writer.write(np.full((200,300,3),i,np.uint8),i,t,dt,{})
        if i==30:
            m=json.loads((tmp_path/'manifest.json').read_text())
            assert m['status']=='streaming' and 0<m['frame_count']<31
            assert all((tmp_path/s['file']).is_file() for s in m['segments'])
    writer.close();m=json.loads((tmp_path/'manifest.json').read_text())
    assert m['status']=='complete' and m['frame_count']==len(times)
    for segment in m['segments']:
        with av.open(str(tmp_path/segment['file'])) as container:
            assert container.streams.video[0].codec_context.extradata[:4].hex()=='0142c01f'
            frames=list(container.decode(video=0));counts.append(len(frames))
            assert frames[0].key_frame and frames[0].pts==0
            decoded.extend(float(f.pts*f.time_base)+segment['start_s'] for f in frames)
            decoded_durations.extend(float(f.duration*f.time_base) for f in frames)
        assert len(frames)==segment['frames']==segment['last_frame']-segment['first_frame']+1
    np.testing.assert_allclose(decoded,times,atol=1e-6,rtol=0)
    np.testing.assert_allclose(decoded_durations,durations,atol=1e-6,rtol=0)
    assert sum(counts)==103 and m['available_until_s']==pytest.approx(durations.sum())
    assert not list(tmp_path.glob('*.part'))


def test_renderer_cumulative_seconds_count_each_scored_frame_once():
    renderer=AnnotationRenderer(circle_cfg(),[300,200]);image=np.zeros((200,300,3),np.uint8)
    row=dict(chamber='left',duration_s=.2,nose_scoreable=True,left_interaction=True,right_interaction=False)
    renderer.draw(image,0,0,row);renderer.draw(image,0,0,row)
    renderer.draw(image,1,.2,dict(row,chamber='center',duration_s=.3,left_interaction=False))
    assert renderer.totals['left']==.2 and renderer.totals['center']==.3
    assert renderer.totals['left_nose']==.2 and renderer.totals['right_nose']==0

def test_renderer_uses_identical_coordinates_for_source_and_crop():
    cfg=dict(circle_cfg(),review_crop_xyxy=[10,10,290,190]);renderer=AnnotationRenderer(cfg,[300,200])
    image=np.full((200,300,3),80,np.uint8)
    row=dict(nose_x=55,nose_y=110,nose_likelihood=.99,center_x=70,center_y=120,center_likelihood=.2,tail_base_x=90,tail_base_y=130,tail_base_likelihood=float('nan'))
    full=renderer.draw(image,4,.2,row);crop=renderer.draw(image[10:190,10:290],4,.2,row,True)
    np.testing.assert_array_equal(full,crop)
    np.testing.assert_array_equal(full[110-10+106,55-10],renderer.COLORS['nose'])
    assert full.shape==(286,280,3)
    # Missing points are absent; a low-confidence center never gets a solid marker.
    assert not np.array_equal(full[120-10+106,70-10+2],renderer.COLORS['center'])


def test_stream_rejects_gaps_instead_of_silently_skipping_frames(tmp_path):
    writer=SegmentWriter(tmp_path,circle_cfg(),[300,200],25,dict(recording_id='a',recording_index=0,source_duration_s=1))
    with pytest.raises(ValueError,match='consecutive'):writer.write(np.zeros((200,300,3),np.uint8),1,0,.04,{})
    writer.close('alignment lost')
    assert json.loads((tmp_path/'manifest.json').read_text())['status']=='failed'


def test_video_worker_encoder_failure_does_not_escape_to_analysis(tmp_path,monkeypatch):
    import threechamber.streaming as module
    live=LivePublisher(tmp_path,dict(recording_id='a',recording_index=0,source_duration_s=1,source_size=[300,200]))
    monkeypatch.setattr(module.SegmentWriter,'write',lambda *a,**k:(_ for _ in ()).throw(OSError('encoder failed')))
    with pytest.warns(RuntimeWarning,match='analysis continues'):
        publisher=VideoPublisher(live,circle_cfg(),25)
        publisher.submit(np.zeros((200,300,3),np.uint8),0,0,.04,{})
        publisher.close()
    assert publisher.failure and not publisher.worker.is_alive()
    assert json.loads((publisher.folder/'manifest.json').read_text())['status']=='failed'


def test_streamed_scoring_and_export_are_equivalent_to_preview_off(tmp_path):
    batch=batch_fixture(tmp_path);cfg=batch['entries'][0]['config'];video=tmp_path/'1.mp4';tracks=tmp_path/'1.csv'
    expected=analyze(video,tracks,cfg,tmp_path/'off')
    live=LivePublisher(tmp_path/'live',dict(batch_id='a',recording_id='a',recording_index=0,source_duration_s=.4,source_size=[64,64],crop=[0,0,64,64]))
    def progress(message):pass
    progress.live=live
    assert analyze(video,tracks,cfg,tmp_path/'on',progress)==expected
    for file in ['frames.csv','summary.csv','bouts.csv']:
        assert (tmp_path/'on'/file).read_bytes()==(tmp_path/'off'/file).read_bytes()
    np.testing.assert_array_equal(timeline(tmp_path/'on/review.mp4')[0],timeline(video)[0])
    m=json.loads((tmp_path/'live/video/0/manifest.json').read_text())
    assert m['frame_count']==4 and m['status']=='complete'


def test_stream_api_retains_previous_recording_and_rejects_bad_paths(tmp_path):
    folder=tmp_path/'batches/job';folder.mkdir(parents=True)
    (folder/'batch.json').write_text(json.dumps(dict(id='job',status='running',current_index=1,entries=[dict(status='complete'),dict(status='running')])))
    video=folder/'live/video/0';video.mkdir(parents=True)
    (video/'manifest.json').write_text(json.dumps(dict(status='complete',segments=[])))
    app=Flask(__name__);register_batch(app,lambda:tmp_path,None);client=app.test_client()
    r=client.get('/api/batches/job/video/0');assert r.status_code==200 and r.headers['Cache-Control']=='no-store'
    assert r.json['manifest']['status']=='complete'
    # A restarted job cannot promise more segments from an interrupted encoder.
    (video/'manifest.json').write_text(json.dumps(dict(status='streaming',segments=[])))
    assert client.get('/api/batches/job/video/0').json['manifest']['status']=='failed'
    app.testing=True
    with pytest.raises(ValueError):client.get('/batches/job/video/0/context.json')
    with pytest.raises(ValueError):client.get('/api/batches/job/video/9')


def test_early_preview_end_is_reported_as_partial_not_complete(tmp_path):
    writer=SegmentWriter(tmp_path,circle_cfg(),[300,200],25,dict(recording_id='a',recording_index=0,source_duration_s=1,expected_frames=25))
    writer.write(np.zeros((200,300,3),np.uint8),0,0,.04,{})
    writer.close()
    m=json.loads((tmp_path/'manifest.json').read_text())
    assert m['status']=='failed' and '1 of 25' in m['error']
    assert not list(tmp_path.glob('*.part'))


def test_encoder_backpressure_disables_preview_without_blocking_predictions(tmp_path,monkeypatch):
    from threading import Event
    import threechamber.streaming as module
    entered=Event();release=Event();original=module.SegmentWriter.write
    def blocked(self,*args):
        entered.set();assert release.wait(3);return original(self,*args)
    monkeypatch.setattr(module.SegmentWriter,'write',blocked)
    live=LivePublisher(tmp_path,dict(recording_id='a',recording_index=0,source_duration_s=1,source_size=[300,200]))
    publisher=VideoPublisher(live,circle_cfg(),25,capacity=1);image=np.zeros((200,300,3),np.uint8)
    try:
        assert publisher.submit(image,0,0,.04,{})
        assert entered.wait(3)
        assert publisher.submit(image,1,.04,.04,{})
        with pytest.warns(RuntimeWarning,match='could not keep up'):
            assert not publisher.submit(image,2,.08,.04,{})
    finally:release.set();publisher.close()
    assert publisher.failure and not publisher.worker.is_alive()
    assert json.loads((publisher.folder/'manifest.json').read_text())['status']=='failed'
