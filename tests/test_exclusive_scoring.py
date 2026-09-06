import pytest
from stereotypy.exclusive import ExclusiveScorer
from stereotypy.training import CLASSES

def test_exclusive_cumulative_partition_ties_gaps_and_quality():
    s=ExclusiveScorer(dict.fromkeys(CLASSES,.5));p=dict.fromkeys(CLASSES,.1)
    p.update(grooming=.8,digging=.9)
    assert s.add(0,1,p)=='digging'
    assert s.add(1,2,p)=='digging'
    assert s.seconds['digging']==2 and s.seconds['grooming']==0
    p['grooming']=.9
    assert s.add(2,3,p)=='unknown'
    assert s.add(3,4,dict.fromkeys(CLASSES,.1))=='other'
    assert s.add(5,6,p,False)=='unknown'
    assert sum(s.seconds.values())+s.other+s.unknown==s.end==6
    assert s.unknown==3 and len([r for r in s.intervals if r['behavior']=='digging'])==1
    with pytest.raises(ValueError):s.add(0,1,p)

def test_no_qualified_behavior_is_not_forced_into_a_target_class():
    s=ExclusiveScorer(dict.fromkeys(CLASSES,.5))
    assert s.add(0,.25,dict.fromkeys(CLASSES,.49))=='other'
    assert sum(s.seconds.values())==0
