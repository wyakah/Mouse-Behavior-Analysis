import cv2
import numpy as np
import pytest
from threechamber.localization import ForegroundLocalizer

PROFILE={'source_size':[200,120],'crop_xyxy':[10,10,190,110],'expected_arena_xyxy':[15,15,185,105]}

def scene():
    frame=np.full((120,200,3),130,np.uint8)
    # Static cup is much larger/darker than the free mouse, but should subtract.
    cv2.circle(frame,(145,60),21,(12,12,12),-1)
    return frame

def localizer():
    background=cv2.cvtColor(scene()[10:110,10:190],cv2.COLOR_BGR2GRAY)
    return ForegroundLocalizer(PROFILE,background,np.ones_like(background))

def mouse(frame,position):
    cv2.ellipse(frame,position,(12,6),0,0,360,(10,10,10),-1)
    return frame

def test_static_cup_not_proposed_but_mouse_box_contains_body():
    result=localizer().propose(mouse(scene(),(55,60)),0,'a')
    assert result['status']=='proposal'
    x1,y1,x2,y2=result['bbox_xyxy']
    assert x1<43 and x2>67 and y1<54 and y2>66
    assert x2<125 # Cup has not become the proposed subject.
    assert 'nose' not in result and 'center' not in result

def test_missing_foreground_is_not_carried_forward():
    detector=localizer()
    detector.propose(mouse(scene(),(55,60)),0,'a')
    result=detector.propose(scene(),.04,'a')
    assert result['status']=='unresolved' and result['bbox_xyxy'] is None

def test_equal_disconnected_candidates_abstain():
    frame=mouse(mouse(scene(),(45,40)),(95,80))
    result=localizer().propose(frame,0,'a')
    assert result['status']=='ambiguous' and result['bbox_xyxy'] is None

def test_source_and_timestamp_gaps_do_not_impose_continuity():
    detector=localizer();frame=mouse(scene(),(55,60))
    detector.propose(frame,0,'a')
    assert detector.propose(frame,.04,'a')['temporal_used']
    assert not detector.propose(frame,25,'a')['temporal_used']
    assert not detector.propose(frame,25.04,'b')['temporal_used']

def test_resolution_and_background_guards():
    with pytest.raises(ValueError):localizer().propose(np.zeros((60,100,3),np.uint8),0)
    with pytest.raises(ValueError):ForegroundLocalizer(PROFILE,np.zeros((1,1)),np.zeros((1,1)))

def test_global_exposure_shift_does_not_create_subject():
    result=localizer().propose(np.maximum(scene().astype(np.int16)-25,0).astype(np.uint8),0)
    assert result['status']=='unresolved'

def test_full_video_preserves_decoded_frame_identity_and_irregular_pts(tmp_path):
    import av
    import json
    from fractions import Fraction
    from scripts.localize_video import localize_video
    video=tmp_path/'source.mp4';profile=tmp_path/'profile.json'
    profile.write_text(json.dumps(PROFILE))
    expected=[0,.04,.08,.12,.20,.24,.28,.32,.36,.40,.44,.48]
    with av.open(str(video),'w') as container:
        stream=container.add_stream('libx264',rate=25)
        stream.width=200;stream.height=120;stream.pix_fmt='yuv420p'
        stream.time_base=Fraction(1,1000)
        for i,t in enumerate(expected):
            frame=av.VideoFrame.from_ndarray(mouse(scene(),(35+i*5,60)),format='bgr24')
            frame.pts=round(t*1000);frame.time_base=Fraction(1,1000)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    out=tmp_path/'result'
    manifest=localize_video(video,out,profile,samples=12)
    rows=json.loads((out/'proposals.json').read_text())
    assert manifest['frames']==12
    assert [r['source_frame'] for r in rows]==list(range(12))
    assert [r['source_time_s'] for r in rows]==pytest.approx(expected)
    assert rows[3]['duration_s']==pytest.approx(.08)
    assert 'center_x' not in rows[0] and 'nose_x' not in rows[0]
    assert manifest['source_sha256']

def test_optional_floor_polygon_excludes_wall_without_changing_default():
    profile={**PROFILE,'floor_polygon':[[25,20],[125,20],[125,100],[25,100]],'floor_polygon_source':'test fixture floor'}
    base=localizer()
    restricted=ForegroundLocalizer(profile,base.background,base.noise)
    # A narrow dark component at the far right is visible in the broad arena,
    # but outside the explicit optional floor polygon.
    frame=mouse(scene(),(175,35))
    assert base.propose(frame,0)['status']=='proposal'
    assert restricted.propose(frame,0)['status']=='unresolved'
    del profile['floor_polygon_source']
    with pytest.raises(ValueError):ForegroundLocalizer(profile,base.background,base.noise)

def test_boundary_review_cues_do_not_remove_or_move_proposal():
    import copy
    from threechamber.localization import localization_review_flags,build_review_queue
    proposal={'status':'proposal','support_xy':[181.,60.],'foreground_bbox_xyxy':[177,47,185,73],
               'bbox_xyxy':[155,25,190,95],'source':'a','source_frame':10,'source_time_s':1.,'duration_s':.04}
    original=copy.deepcopy(proposal)
    flags=localization_review_flags(proposal,PROFILE)
    assert 'boundary_support_review' in flags
    assert 'narrow_boundary_support_reflection_suspected' in flags
    queue=build_review_queue([proposal,{**proposal,'source_frame':11,'source_time_s':1.04}],PROFILE)
    assert len(queue['intervals'])==1
    assert queue['intervals'][0]['flagged_frames']==2
    assert queue['intervals'][0]['end_s']==pytest.approx(1.08)
    assert proposal==original
