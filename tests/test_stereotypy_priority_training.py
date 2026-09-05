"""Acceptance and split integrity for the optional priority training pipeline."""
import numpy as np
import pytest
pytest.importorskip('sklearn')
from stereotypy.priority_training import (binary_metrics, context_indices, labgym_labels,
                                          select_threshold, validate_catalog)


def row(id, split='train', dataset='cbas', group=None, fingerprint=None):
    return dict(id=id,split=split,dataset=dataset,group=group or id,sha256=fingerprint or id)


def test_high_accuracy_from_absence_does_not_pass_acceptance():
    metrics=binary_metrics([0]*99+[1],[.01]*100,.5)
    assert metrics['accuracy']==.99
    assert metrics['balanced_accuracy']==.5
    assert metrics['recall']==0
    assert metrics['target_met'] is False


def test_unknown_labels_never_become_negative_evidence():
    metrics=binary_metrics([-1,0,1],[1,.1,.9],.5)
    assert metrics['n']==2 and metrics['confusion_matrix']==[[1,0],[0,1]]
    assert binary_metrics([-1,0],[1,0],.5)['status']=='insufficient_support'


def test_temporal_context_cannot_cross_missing_time_or_cuts():
    indices=context_indices([True,False,False,True,False],radius=2)
    assert np.all(indices[:3]<3) and np.all(indices[3:]>=3)
    assert indices[0].tolist()==[0,0,0,1,2]
    assert indices[-1].tolist()==[3,3,4,4,4]


def test_same_source_or_mouse_group_cannot_cross_splits():
    with pytest.raises(ValueError,match='crosses'):
        validate_catalog([row('a',fingerprint='same'),row('b','test',fingerprint='same')])
    with pytest.raises(ValueError,match='crosses'):
        validate_catalog([row('a',group='mouse'),row('b','validation',group='mouse')])


def test_local_unlabeled_and_labgym_unidentifiable_sources_cannot_validate():
    with pytest.raises(ValueError,match='Unlabeled'):
        validate_catalog([row('a','test','local')])
    with pytest.raises(ValueError,match='parent recording'):
        validate_catalog([row('a','test','labgym')])


def test_ethogram_mapping_preserves_unknown_and_does_not_relabel_foraging():
    assert labgym_labels('foraging sv')==[0,-1,0]
    assert labgym_labels('body grooming fv')==[1,-1,-1]
    assert labgym_labels('standing sv')==[0,-1,1]
    assert labgym_labels('standing on the wheel')==[0,-1,-1]
    assert labgym_labels('hind paw grooming')==[-1,-1,-1]


def test_threshold_selection_gives_each_camera_dataset_equal_weight():
    y=np.array([0,1,0,1]);p=np.array([.1,.3,.7,.9]);d=np.array(['a','a','b','b'])
    one=select_threshold(y,p,d)
    repeated=select_threshold(np.r_[np.tile(y[:2],100),y[2:]],np.r_[np.tile(p[:2],100),p[2:]],np.r_[np.tile(d[:2],100),d[2:]])
    assert one['threshold']==repeated['threshold']
    assert one['selection_score']==repeated['selection_score']


def test_native_motion_alignment_edges_and_cache_invalidation(tmp_path):
    cv2=pytest.importorskip('cv2')
    from scripts.stereotypy_motion_features import extract
    source=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(80,60))
    if not writer.isOpened():pytest.skip('MJPEG encoder unavailable')
    for i in range(5):
        frame=np.full((60,80,3),220,np.uint8)
        frame[20:40,20+i*3:35+i*3]=30
        writer.write(frame)
    writer.release()
    cache=tmp_path/'visual.npz'
    np.savez_compressed(cache,frame_ids=[0,2,4],geometry=np.zeros((3,7)),fps=10)
    source_row=dict(id='synthetic',path=str(source),feature_path=str(cache),dataset='cbas')
    extract(source_row)
    result=np.load(tmp_path/'visual-motion.npz')
    assert result['frame_ids'].tolist()==[0,2,4]
    assert result['x'][:,-1].tolist()==[0,1,0]
    assert np.abs(result['x'][1,:-1]).sum()>0
    old_hash=str(result['feature_sha256']);result.close()
    np.savez_compressed(cache,frame_ids=[1,2,3],geometry=np.zeros((3,7)),fps=10)
    extract(source_row)
    updated=np.load(tmp_path/'visual-motion.npz')
    assert str(updated['feature_sha256'])!=old_hash
    assert updated['frame_ids'].tolist()==[1,2,3]
    assert updated['x'][:,-1].tolist()==[1,1,1]
