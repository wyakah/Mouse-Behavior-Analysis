"""Joint, exclusive grooming/rearing/other classifier features.

Chewing is deliberately not mapped to nonfood gnawing. Unsupported six-assay
classes remain outside this model's vocabulary, not synthetic negatives.
"""
import numpy as np
LABELS=('grooming','rearing','other')
POSE_FEATURES=('nose_usable_fraction','head_paw_distance','forepaw_speed','upright_fraction')

def native_label(name):
    if name is None or name=='unobservable':return 'unknown'
    if 'grooming' in name:return 'grooming'
    if name in ('rearing up','standing'):return 'rearing'
    return 'other'

def feature_vector(scores,pose,baseline,use_pose):
    valid=scores is not None
    values=list(scores) if valid else [0.]*20
    if len(values)!=20:raise ValueError('Expected published LabGym20 class order')
    values.extend([float(valid),float(baseline[0]),float(baseline[1])])
    if use_pose:
        values.extend([float(pose[k]) if pose.get(k) is not None else 0. for k in POSE_FEATURES])
        values.extend([float(pose.get(k) is not None) for k in POSE_FEATURES])
    return np.array(values,float)

def metrics(truth,predicted):
    truth=np.asarray(truth);predicted=np.asarray(predicted);classes=LABELS+('unknown',)
    confusion=[[int(np.sum((truth==a)&(predicted==b))) for b in classes] for a in LABELS]
    per={}
    for c in LABELS:
        tp=int(np.sum((truth==c)&(predicted==c)));fp=int(np.sum((truth!=c)&(predicted==c)));fn=int(np.sum((truth==c)&(predicted!=c)))
        per[c]=dict(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,support=int(np.sum(truth==c)))
    return dict(classes=list(classes),confusion=confusion,per_class=per,accuracy=float(np.mean(truth==predicted)),coverage=float(np.mean(predicted!='unknown')),priority_macro_f1=float(np.mean([per[c]['f1'] for c in LABELS[:2]])),windows=len(truth))
