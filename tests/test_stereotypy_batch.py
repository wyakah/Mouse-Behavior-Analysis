import pytest
from stereotypy.window import analysis_end,clip_manifest
from stereotypy.batch import summarize_predictions,apply_observability,select_sources
from stereotypy.training import CLASSES


def test_twenty_minute_boundary_clips_frame_and_gap_without_changing_source():
    original=dict(duration_s=1300,frames=[dict(start_s=0,end_s=1),dict(start_s=1199.99,end_s=1200.02),dict(start_s=1200.02,end_s=1200.05)],source_gaps=[[1199,1201],[1250,1260]])
    result=clip_manifest(original)
    assert result['duration_s']==1200 and result['frame_count']==2
    assert result['frames'][-1]['end_s']==1200
    assert result['source_gaps']==[[1199,1200]]
    assert original['frames'][-2]['end_s']==1200.02
    assert analysis_end(1198)==1198
    with pytest.raises(ValueError):analysis_end(float('nan'))


def test_candidate_totals_exclude_gaps_and_do_not_invent_accepted_labels():
    rows=[dict(start_s=0,end_s=1,scores=dict.fromkeys(CLASSES,.9)),dict(start_s=1,end_s=2,scores=dict.fromkeys(CLASSES,.1)),dict(start_s=2,end_s=3,scores=dict.fromkeys(CLASSES,.9))]
    summary,bouts=summarize_predictions(rows,dict.fromkeys(CLASSES,.5),3,[(.4,.6)])
    for result in summary:
        assert result['candidate_seconds']==pytest.approx(1.8)
        assert result['unknown_seconds']==pytest.approx(.2)
        assert result['candidate_segments']==3
        assert result['accepted_seconds'] is None and result['accuracy'] is None
    assert len(bouts)==18
    with pytest.raises(ValueError):summarize_predictions(rows,dict.fromkeys(CLASSES,.5),2)


def test_threshold_sensitivity_and_overlapping_behaviors():
    rows=[dict(start_s=0,end_s=.5,scores=dict.fromkeys(CLASSES,.55))]
    summary,_=summarize_predictions(rows,dict.fromkeys(CLASSES,.5),.5)
    assert all(r['candidate_seconds']==.5 and r['seconds_at_higher_threshold']==0 for r in summary)


def test_camera_away_and_failed_localization_never_become_zero_behavior():
    rows=[dict(start_s=0,end_s=1,scores=dict.fromkeys(CLASSES,.9))]
    thresholds=dict.fromkeys(CLASSES,.5)
    summary,bouts=summarize_predictions(rows,thresholds,1)
    result=dict(windows=rows,thresholds=thresholds,duration_s=1,source_gaps=[],summary=summary,bouts=bouts)
    tracking=[dict(start_s=0,end_s=.4,status='proposed'),dict(start_s=.4,end_s=1,status='camera_motion')]
    gated=apply_observability(result,tracking,True)
    assert gated['summary'][0]['candidate_seconds']==pytest.approx(.4)
    assert gated['summary'][0]['unknown_seconds']==pytest.approx(.6)
    rejected=apply_observability(result,tracking,False)
    assert rejected['summary'][0]['candidate_seconds'] is None
    assert rejected['summary'][0]['unknown_seconds']==1
    assert rejected['raw_summary'][0]['candidate_seconds']==1


def test_new_arrivals_cannot_silently_expand_a_selected_batch(tmp_path):
    for mouse in ['710','711','712','743','745','756']:(tmp_path/(mouse+'_test.mov')).touch()
    selected={'710','711','712','743','745'}
    assert {p.stem.split('_')[0] for p in select_sources(tmp_path,selected)}==selected
    (tmp_path/'710_duplicate.mov').touch()
    with pytest.raises(ValueError):select_sources(tmp_path,selected)
