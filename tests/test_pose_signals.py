import numpy as np
from stereotypy.pose_signals import PoseSignals,summarize_pose
from stereotypy.robustness import LANDMARKS

def pose():
    return np.array([[100,50,.9],[100,70,.9],[100,90,.9],[100,130,.9],[98,52,.9],[102,53,.9],[90,140,.9],[110,140,.9]])

def test_upright_and_paws_near_face_do_not_force_a_behavior():
    r=PoseSignals().add(0,pose(),list(LANDMARKS),(200,300,3))
    assert r['upright_support'] and r['head_paw_distance']<.1
    assert 'behavior' not in r and r['contact_status']=='not_measured'

def test_missing_paw_is_not_filled_and_jumps_are_rejected():
    s=PoseSignals(max_speed=2);s.add(0,pose(),list(LANDMARKS),(200,300,3));p=pose();p[0,0]=250;p[4,2]=.1
    r=s.add(.1,p,list(LANDMARKS),(200,300,3))
    assert r['points']['nose'] is None and r['points']['front_left_paw'] is None
    assert r['rejected_jumps']==['nose']
    p[0,0]=240;r=s.add(1,p,list(LANDMARKS),(200,300,3))
    assert r['points']['nose'] is not None and r['forepaw_speed'] is None

def test_empty_summary_does_not_assert_posture_or_contact():
    r=summarize_pose([]);assert r['upright_fraction'] is None and r['contact_status']=='not_measured'


def test_optional_pose_failure_is_explicit_and_returns_no_fabricated_points():
    from stereotypy.pose_signals import infer_or_error
    class Broken:
        def infer(self,images,times):raise RuntimeError('inference failed')
    samples,error=infer_or_error(Broken(),[],[])
    assert samples==[] and error=='inference failed'
