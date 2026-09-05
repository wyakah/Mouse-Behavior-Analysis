"""Validation must measure failures and preserve reviewed-label provenance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scripts.evaluate_reviewed import evaluate, wilson_interval
from scripts.finetune_specialized import validate_export, validate_label_values, require_detector_snapshot, write_status


def examples():
    queue=[dict(id=str(i),source='685.mp4',source_frame=i) for i in range(3)]
    annotations={str(i):dict(reviewed=True,points={'nose':[10,20],'center':None}) for i in range(3)}
    tracks=pd.DataFrame([dict(source='685.mp4',source_frame=0,nose_x=10,nose_y=20,nose_likelihood=.99)])
    return tracks,queue,annotations


def test_absent_prediction_rows_count_as_misses():
    result=evaluate(*examples()); s=result['summary'][0]
    assert s['visible_labeled_frames']==3 and s['missing_prediction_rows']==2
    assert s['coverage']==pytest.approx(1/3)
    assert s['fraction_within_tolerance_including_misses']==pytest.approx(1/3)
    assert result['by_source'][0]['source']=='685.mp4'
    assert result['accuracy_validated'] is False


def test_human_wrong_subject_cannot_pass_with_nearby_landmark():
    tracks,queue,ann=examples()
    ann['0']['prediction_review']={'reviewed':True,'subject_correct':False}
    s=evaluate(tracks,queue,ann)['summary'][0]
    assert s['wrong_subject_frames']==1 and s['identity_reviewed_frames']==1
    assert s['correct_within_tolerance']==0


def test_unreviewed_identity_is_not_ground_truth():
    tracks,queue,ann=examples()
    ann['0']['prediction_review']={'subject_correct':False}
    assert evaluate(tracks,queue,ann)['summary'][0]['correct_within_tolerance']==1


def test_perfect_observed_results_are_not_certified():
    tracks,q,a=examples();r=evaluate(tracks,q[:1],a);s=r['summary'][0]
    assert s['target_99_observed'] is True and s['accuracy_target_validated'] is False
    assert s['within_tolerance_wilson_95'][0]<.95
    assert 'correlated' in r['interval_method']


def test_evaluation_rejects_duplicates_and_bad_provenance():
    tracks,q,a=examples()
    with pytest.raises(ValueError,match='Duplicate'):evaluate(pd.concat([tracks,tracks]),q,a)
    a['0']['source']='another.mp4'
    with pytest.raises(ValueError,match='provenance'):evaluate(tracks,q,a)


def test_empty_predictions_fail_all_visible_labels():
    _,q,a=examples()
    r=evaluate(pd.DataFrame(columns=['source','source_frame']),q,a)
    assert r['summary'][0]['fraction_within_tolerance_including_misses']==0


def test_wilson_is_bounded_with_no_or_all_success():
    assert wilson_interval(0,0) is None
    assert wilson_interval(0,10)[0]==0
    assert wilson_interval(10,10)[1]==pytest.approx(1)


def export_fixture():
    parts=['nose','left_ear','right_ear','center','tail_base']
    m=dict(label_source='Explicitly reviewed annotations only; no automatic acceptance of model suggestions.',reviewed_frames=2,parts=parts,profile={'crop_xyxy':[190,175,750,575]},source_videos=['675_trial.mp4','685_trial.mp4'])
    split=[];annotations={}
    for i,source in enumerate(m['source_videos']):
        split.append(dict(index=['labeled-data',Path(source).stem,'img000001.png'],split='train' if i==0 else 'validation',annotation_id=str(i),source=source,source_frame=1))
        annotations[str(i)]=dict(points={p:[200,200] for p in parts},annotator='Reviewer',reviewed=True,reviewed_at='2026-09-05T12:00:00Z',source=source,source_frame=1)
    return m,split,annotations


def test_known_export_requires_actual_review_and_subject_split():
    m,s,a=export_fixture()
    assert validate_export(m,s,a)['subjects']=={'train':['675'],'validation':['685']}
    a['0']['reviewed']=False
    with pytest.raises(ValueError,match='human review'):validate_export(m,s,a)
    a['0']['reviewed']=True;s[1]['split']='train'
    with pytest.raises(ValueError,match='split violation'):validate_export(m,s,a)


def test_export_rejects_automatic_source_and_empty_labels():
    m,s,a=export_fixture();m['label_source']='automatic predictions'
    with pytest.raises(ValueError,match='label source'):validate_export(m,s,a)
    m,s,a=export_fixture();m['reviewed_frames']=0
    with pytest.raises(ValueError,match='count'):validate_export(m,[],{})


def test_hdf_cannot_override_human_labels():
    m,s,a=export_fixture()
    columns=pd.MultiIndex.from_product([['Researcher'],m['parts'],['x','y']],names=['scorer','bodyparts','coords'])
    df=pd.DataFrame([[10,25]*5]*2,index=pd.MultiIndex.from_tuples([tuple(e['index']) for e in s]),columns=columns)
    validate_label_values(df,m,s,a)
    df.iloc[0,0]=11
    with pytest.raises(ValueError,match='differ from explicitly reviewed'):validate_label_values(df,m,s,a)


def test_detector_gate_rejects_missing_snapshot(tmp_path):
    with pytest.raises(RuntimeError,match='ground-truth-box'):require_detector_snapshot([])
    snapshot=SimpleNamespace(path=tmp_path/'detector.pt')
    with pytest.raises(RuntimeError):require_detector_snapshot([snapshot])
    snapshot.path.write_bytes(b'test snapshot existence gate only')
    assert require_detector_snapshot([snapshot]) is snapshot


def test_status_is_atomic_and_records_failure(tmp_path):
    status={'status':'failed','failed_stage':'training','accuracy_validated':False}
    write_status(tmp_path,status)
    result=json.loads((tmp_path/'fine_tuning_status.json').read_text())
    assert result['failed_stage']=='training' and result['updated_at']
    assert not (tmp_path/'fine_tuning_status.tmp').exists()


def test_evaluation_calls_diagnostic_and_real_detector_separately(tmp_path,monkeypatch):
    import sys
    from scripts.finetune_specialized import evaluate_training
    detector=tmp_path/'snapshot-detector-30.pt';detector.write_bytes(b'fixture')
    snapshot=SimpleNamespace(path=detector)
    monkeypatch.setitem(sys.modules,'deeplabcut.pose_estimation_pytorch.data',SimpleNamespace(DLCLoader=lambda **kw:SimpleNamespace(model_folder=tmp_path)))
    monkeypatch.setitem(sys.modules,'deeplabcut.pose_estimation_pytorch.task',SimpleNamespace(Task=SimpleNamespace(DETECT='detect')))
    monkeypatch.setitem(sys.modules,'deeplabcut.pose_estimation_pytorch.apis.utils',SimpleNamespace(get_model_snapshots=lambda *args:[snapshot]))
    calls=[];writes=[]
    dlc=SimpleNamespace(Engine=SimpleNamespace(PYTORCH='pytorch'),evaluate_network=lambda *args,**kwargs:calls.append((args,kwargs)))
    af=SimpleNamespace(read_config=lambda path:{'detector_snapshotindex':-1},write_config=lambda path,cfg:writes.append((path,dict(cfg))))
    config=tmp_path/'config.yaml'
    result=evaluate_training(config,dlc,af,'cpu')
    assert writes[0][1]['detector_snapshotindex'] is None
    assert calls[0][0][0].name=='pose_only_diagnostic_config.yaml'
    assert calls[0][1]['detector_snapshot_index'] is None
    assert calls[1][0][0]==config and calls[1][1]['detector_snapshot_index']==-1
    assert result['accuracy_validated'] is False
    calls.clear();detector.unlink()
    with pytest.raises(RuntimeError):evaluate_training(config,dlc,af,'cpu')
    assert calls==[]
