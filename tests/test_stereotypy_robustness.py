import numpy as np
import pytest
from stereotypy.robustness import enhance, pose_evidence, disagreements, temporal_features, LANDMARKS


def test_missing_or_outside_pose_is_unknown_not_horizontal():
    p=np.zeros((8,3)); p[:,2]=.99; p[:,:2]=-1
    e=pose_evidence(p,LANDMARKS,(200,400,3))
    assert not e['axial_available'] and e['horizontal_support'] is None
    assert disagreements({'rearing':.9},{'rearing':.5},e,True)==['pose_unavailable']
    assert disagreements({'rearing':.9},{'rearing':.5},e,False)==['scene_unreliable']


def test_rearing_disagreement_requires_valid_horizontal_axis():
    p=np.array([[90,80,.9],[100,100,.9],[150,100,.9],[200,100,.9]]+[[0,0,0]]*4)
    e=pose_evidence(p,LANDMARKS,(250,500,3))
    assert e['horizontal_support'] and not e['upright_support']
    assert disagreements({'rearing':.9},{'rearing':.5},e,True)==['rearing_pose_disagreement']
    p[1,2]=.1
    assert not pose_evidence(p,LANDMARKS,(250,500,3))['axial_available']


def test_no_motion_interpolation_across_missing_time():
    p=dict(center=[0,0],body_length=100,angle=3.13)
    c=dict(center=[0,-10],body_length=100,angle=-3.13)
    assert temporal_features(p,c,1)['ascent_body_lengths_s'] is None
    assert temporal_features(p,c,.1)['ascent_body_lengths_s']==pytest.approx(1)
    assert abs(temporal_features(p,c,.1)['angular_speed_rad_s'])<1


def test_enhancement_preserves_source_and_shape():
    a=np.full((30,40,3),50,np.uint8); b=enhance(a,'gamma')
    assert np.all(a==50) and b.mean()>a.mean() and b.shape==a.shape
    assert enhance(a,'clahe').dtype==np.uint8
    with pytest.raises(ValueError):enhance(a,'bad')


def test_reference_never_uses_unlabeled_or_partial_intervals():
    from stereotypy.fusion import reference_samples,binary_metrics
    rows=[dict(start_s=0,end_s=.2,body_valid=True),dict(start_s=.2,end_s=.4,body_valid=True),dict(start_s=.4,end_s=.6,body_valid=False)]
    labels=[dict(start_s=.1,end_s=.6,behavior='rearing',label='present')]
    ids,y,w=reference_samples(rows,labels,'rearing')
    assert ids.tolist()==[1] and y.tolist()==[1]
    metrics=binary_metrics([1,0],[.9,.9],[1,9],.5)
    assert metrics['precision']==pytest.approx(.1) and metrics['duration_error_seconds']==9
    assert binary_metrics([],[],[],.5)['f1'] is None


def test_fusion_matrix_preserves_missingness_and_feature_order():
    from stereotypy.fusion import feature_matrix,FEATURES
    from stereotypy.training import CLASSES
    e=dict(nose_available=False,axial_available=False,head_paw_distance=None,body_length=None,upright_support=None,horizontal_support=None)
    m=dict(center_speed_body_lengths_s=None,angular_speed_rad_s=None,ascent_body_lengths_s=None)
    r=dict(scores=dict.fromkeys(CLASSES,.2),evidence=e,motion=m,scene_similarity=None,body_valid=False)
    x=feature_matrix([r]);assert x.shape==(1,len(FEATURES))
    assert np.isnan(x[0,FEATURES.index('head_paw_distance')])
    assert x[0,FEATURES.index('body_valid')]==0
    assert np.allclose(x[0,:6],.2)


def test_pose_benchmark_separates_coverage_from_error(tmp_path,monkeypatch):
    import json
    import scripts.evaluate_stereotypy_pose_reference as evaluator
    monkeypatch.setattr(evaluator,'ROOT',tmp_path)
    report=tmp_path/'reports/stereotypy-robustness';report.mkdir(parents=True)
    folder=tmp_path/'outputs/stereotypy-robustness/A';folder.mkdir(parents=True)
    frame=dict(mouse_id='A',frame_id=0,source_sha256='hash',width=100,height=100,landmarks={p:None for p in LANDMARKS})
    (report/'reference-template.json').write_text(json.dumps(dict(frames=[frame])))
    (folder/'variants.json').write_text(json.dumps([dict(sample=0,mode=m) for m in ['raw','gamma','clahe']]))
    poses=np.zeros((3,8,3));poses[0,0]=[11,12,.9];poses[1,0]=[10,10,.1];poses[2,0]=[13,14,.9]
    np.savez(folder/'pose.npz',parts=LANDMARKS,records=json.dumps([dict(frame_id=0)]),variant_poses=poses)
    ref=tmp_path/'reference.json';ref.write_text(json.dumps(dict(annotator='Reviewer',frames=[frame])))
    with pytest.raises(ValueError,match='No visible'):evaluator.evaluate(ref)
    frame['landmarks']['nose']=dict(visibility='visible',x=10,y=10)
    ref.write_text(json.dumps(dict(annotator='Reviewer',frames=[frame])))
    result=evaluator.evaluate(ref)['results']
    assert result['raw']['nose']['median_error_native_pixels']==pytest.approx(5**.5)
    assert result['gamma']['nose']['available_fraction']==0
    assert result['gamma']['nose']['median_error_native_pixels'] is None
    assert result['clahe']['nose']['median_error_native_pixels']==5
