from stereotypy.distinction import *

def test_unsupported_labels_do_not_become_gnawing_or_digging():
    assert native_label('chewing sv')=='other'
    assert native_label('foraging fv')=='other'
    assert native_label('face grooming fv')=='grooming'
    assert native_label('unobservable')=='unknown'

def test_missing_features_are_explicit_and_unknowns_count_as_misses():
    v=feature_vector(None,{},[.2,.3],True)
    assert len(v)==31 and v[20]==0 and (v[-4:]==0).all()
    m=metrics(['grooming','rearing'],['unknown','rearing'])
    assert m['per_class']['grooming']['recall']==0 and m['coverage']==.5
