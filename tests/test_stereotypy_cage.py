from fractions import Fraction
from types import SimpleNamespace
import json
import av
import cv2
import numpy as np
import pytest
from stereotypy.cage import validate_mapping, crop_frame, source_box, CageProposer, review_windows
from stereotypy.pilot import run_pilot
from stereotypy.video import index_video
from test_stereotypy import client, create


@pytest.mark.parametrize('roi', [[0,0,20,100],[-1,0,100,100],[0,0,101,100],[0,0,float('nan'),100],[True,0,100,100]])
def test_invalid_roi(roi):
    with pytest.raises(ValueError):validate_mapping(dict(crop_xyxy=roi),100,100)


def test_coordinate_round_trip_preserves_nonzero_crop_origin():
    mapping=validate_mapping(dict(crop_xyxy=[120,30,320,130],floor_y=100),400,200)
    crop=crop_frame(np.zeros((200,400,3),np.uint8),mapping,target_width=100)
    assert crop.shape==(50,100,3)
    assert source_box([0,0,100,50],mapping,crop.shape)==[120,30,320,130]


def test_still_mouse_survives_and_competing_blobs_abstain():
    mapping=validate_mapping(dict(crop_xyxy=[0,0,300,200],floor_y=160),300,200)
    p=CageProposer(np.full((200,300),180,np.uint8),mapping)
    image=np.full((200,300,3),180,np.uint8);image[70:150,80:110]=40
    for _ in range(3):
        result=p.frame(image,.03)
        assert result['status']=='proposed'
        assert result['upright_candidate']
    image[70:150,180:210]=40
    assert p.frame(image,.03)['status']=='ambiguous'


def test_review_sampling_includes_missing_and_does_not_label():
    rows=[dict(start_s=2,status='missing',upright_candidate=False,local_motion=None)]
    windows=review_windows(rows,20)
    assert any(r['reason']=='systematic sample' for r in windows)
    assert any(r['reason']=='localization uncertainty' for r in windows)
    assert all('label' not in r and 0<=r['start_s']<r['end_s']<=20 for r in windows)


def test_full_pilot_preserves_pts_and_never_produces_behavior_scores(tmp_path):
    path=tmp_path/'source.mp4'
    with av.open(str(path),'w') as out:
        s=out.add_stream('libx264',rate=25);s.width=160;s.height=100;s.pix_fmt='yuv420p';s.time_base=Fraction(1,1000)
        for i,pts in enumerate([0,40,80,160]):
            image=np.full((100,160,3),160,np.uint8);image[40:80,20+i*20:40+i*20]=40
            f=av.VideoFrame.from_ndarray(image,format='bgr24');f.pts=pts;f.time_base=Fraction(1,1000)
            for packet in s.encode(f):out.mux(packet)
        for packet in s.encode():out.mux(packet)
    info=index_video(path)
    result=run_pilot(path,dict(crop_xyxy=[0,0,160,100],floor_y=80),tmp_path/'pilot',info)
    assert result['frame_count']==4 and result['behavior_scores'] is None
    assert result['source_gaps']==info['source_gaps']
    with av.open(str(tmp_path/'pilot/review.mp4')) as c:
        times=[float(f.pts*f.time_base) for f in c.decode(video=0)]
        assert float(c.streams.video[0].duration*c.streams.video[0].time_base)==pytest.approx(info['duration_s'],abs=1/60000)
    assert times==pytest.approx([0,.04,.08,.16],abs=1/60000)
    assert sum(result['proposal_seconds'].values())==pytest.approx(sum(r['end_s']-r['start_s'] for r in info['frames']))


def test_mapping_revision_does_not_change_behavior_revision(client):
    c,root=client;url=create(c)
    original=c.get(url).json
    w=original['video_manifest']['width'];h=original['video_manifest']['height']
    response=c.post(url+'/cage',json=dict(revision=0,mapping=dict(crop_xyxy=[0,0,w,h],floor_y=h*.8)))
    assert response.status_code==200
    assert c.post(url+'/cage',json=dict(revision=0,mapping=response.json['mapping'])).status_code==409
    assert c.get(url).json['revision']==original['revision']
    assert c.get(url).json['annotations']==original['annotations']


def test_cage_job_artifacts_and_scores_stay_separate(client):
    c,root=client;url=create(c)
    c.post(url+'/cage',json=dict(revision=0,mapping=dict(crop_xyxy=[0,0,64,64],floor_y=50)))
    response=c.post(url+'/cage/analyze',json={})
    assert response.status_code==202 and response.json['status']=='complete',response.json
    run=response.json['run'];result=c.get(url).json
    assert result['revision']==0 and result['annotations']==[]
    assert len(result['summary'])==6
    assert all(r['active_seconds'] is None for r in result['summary'])
    assert c.get(url+'/cage/'+run['id']+'/review.mp4').status_code==200
    assert c.get(url+'/cage/'+run['id']+'/source-index.json').status_code==404
    assert c.get(url+'/cage/not-a-run/review.mp4').status_code==404


def test_legacy_ethogram_extends_without_relabeling(client):
    c,root=client;url=create(c);sid=url.split('/')[-1]
    path=root/'labeling/stereotypy'/f'{sid}.json';record=json.loads(path.read_text())
    record['ethogram_version']='side-view-draft-1'
    record['definitions']={k:v for k,v in record['definitions'].items() if k not in ('jumping','circling')}
    path.write_text(json.dumps(record))
    updated=c.get(url).json
    assert updated['ethogram_version']=='side-view-draft-2'
    assert updated['revision']==0 and updated['annotations']==[]
    assert len(updated['ethogram_history'])==1
    assert all(r['active_seconds'] is None for r in updated['summary'])


@pytest.mark.parametrize('overlap,accepted',[(1,True),(2,True),(3,False)])
def test_bounded_rounding_is_audited_larger_overlap_rejected(tmp_path,monkeypatch,overlap,accepted):
    path=tmp_path/'fake.mov';path.write_bytes(b'fake')
    frames=[SimpleNamespace(pts=0,duration=20+overlap,time_base=Fraction(1,600),rotation=0),
            SimpleNamespace(pts=20,duration=19,time_base=Fraction(1,600),rotation=0)]
    class Container:
        streams=SimpleNamespace(video=[SimpleNamespace(width=100,height=100,time_base=Fraction(1,600),average_rate=30,codec_context=SimpleNamespace(name='test'))],audio=[])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def decode(self,*args):return iter(frames)
    monkeypatch.setattr(av,'open',lambda _:Container())
    if accepted:
        info=index_video(path)
        assert info['frames'][0]['end_s']==pytest.approx(20/600)
        assert len(info['duration_corrections'])==1
        assert info['frames'][1]['pts']==20
    else:
        with pytest.raises(ValueError,match='tolerance'):index_video(path)
