from copy import deepcopy
import json
import math
import pytest
from threechamber.statistics import analyze_statistics, settings, sample_metadata
from threechamber.batch import validate_batch
from test_batch import batch_fixture


def fixture():
    entries=[]
    # Balanced 2x2, cell means 10,14,20,28; within-cell residuals -1,+1.
    for genotype,sex,mean in [('A','female',10),('A','male',14),('B','female',20),('B','male',28)]:
        for delta in [-1,1]:
            value=mean+delta
            entries.append(dict(id=f'{genotype}-{sex}-{delta}',genotype=genotype,sex=sex,status='complete',summary=dict(analyzed_seconds=600,target_side='left',left_nose_seconds=value,right_nose_seconds=5,nose_scoreable_fraction=.95,center_valid_fraction=.99,preference_index=(value-5)/(value+5))))
    return dict(entries=entries,statistics=dict(mode='all',metrics=['target_nose_seconds']))


def test_balanced_factorial_matches_hand_calculated_sums_of_squares():
    batch=fixture();before=deepcopy(batch);r=analyze_statistics(batch)
    assert batch==before
    effects={e['effect']:e for e in r['anova']}
    # Residual SS=8, residual df=4, MS=2; SS genotype=288, sex=72, interaction=8.
    for effect,ss in [('Genotype',288),('Sex',72),('Genotype × sex',8)]:
        row=effects[effect]
        assert row['sum_squares']==pytest.approx(ss)
        assert row['statistic']==pytest.approx(ss/2)
        assert row['partial_eta_squared']==pytest.approx(ss/(ss+8))
        assert row['df_residual']==4
    female=next(t for t in r['comparisons'] if t['sex']=='female')
    assert female['difference']==-10
    assert female['statistic']==pytest.approx(-10/math.sqrt(2))
    assert female['df']==2
    assert female['p']==pytest.approx(0.019419324309079847)
    rows=sorted(r['comparisons']+r['anova'],key=lambda e:e['p'])
    previous=0
    for i,row in enumerate(rows):
        previous=min(1,max(previous,(len(rows)-i)*row['p']))
        assert row['p_adjusted']==pytest.approx(previous)
    json.dumps(r,allow_nan=False)


def test_unknown_metadata_retains_descriptives_without_fabricated_inference():
    batch=fixture()
    for e in batch['entries']:e.pop('sex');e.pop('genotype')
    r=analyze_statistics(batch)
    assert r['test_count']==0
    assert all(x['status']=='Not calculated' and 'p' not in x for x in r['comparisons']+r['anova'])
    assert r['groups'][0]['n']==8
    assert len(r['exclusions'])==16


def test_target_side_counterbalancing_and_missing_are_not_zero():
    batch=fixture();e=batch['entries'][0]
    e['summary']['target_side']='right'
    assert analyze_statistics(batch)['observations'][0]['values']['target_nose_seconds']==5
    e['summary']['left_nose_seconds']=0;e['summary']['target_side']='left'
    assert analyze_statistics(batch)['observations'][0]['values']['target_nose_seconds']==0
    e['summary']['nose_scoreable_fraction']=0
    assert analyze_statistics(batch)['observations'][0]['values']['target_nose_seconds'] is None
    e['status']='failed';e.pop('summary')
    assert analyze_statistics(batch)['observations'][0]['values']['target_nose_seconds'] is None


def test_durations_and_degenerate_variance_block_tests():
    batch=fixture();batch['entries'][0]['summary']['analyzed_seconds']=590
    r=analyze_statistics(batch)
    assert any('durations differ' in (t.get('reason') or '') for t in r['comparisons'])
    assert 'durations differ' in r['anova'][0]['reason']
    for e in batch['entries']:e['summary'].update(analyzed_seconds=600,left_nose_seconds=4)
    r=analyze_statistics(batch)
    assert r['test_count']==0


@pytest.mark.parametrize('bad',[[],{'mode':'bad'},{'metrics':[]},{'metrics':[{}]},{'metrics':['bad']},{'alpha':.1}])
def test_invalid_settings(bad):
    with pytest.raises(ValueError):settings(bad)


def test_metadata_and_assay_validation(tmp_path):
    draft=batch_fixture(tmp_path)
    draft['entries'][0].update(sex=' FEMALE ',genotype=' WT ')
    b=validate_batch(tmp_path,draft)
    assert b['entries'][0]['sex']=='female' and b['entries'][0]['genotype']=='WT'
    assert b['entries'][1]['sex']=='unknown' and b['statistics']['mode']=='descriptive'
    assert draft['entries'][0]['reviewed'] is True
    draft['test_id']='stereotypy'
    with pytest.raises(ValueError,match='not available'):validate_batch(tmp_path,draft)
    with pytest.raises(ValueError):sample_metadata({'genotype':0})
