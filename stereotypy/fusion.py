"""Feature matrix and duration-weighted reference evaluation for supervised fusion.

Missing pose has its own indicators; it is never synthesized from video scores.
"""
import numpy as np
from .training import CLASSES

FEATURES = tuple('video_'+c for c in CLASSES) + (
    'nose_available','axial_available','head_paw_distance','body_length',
    'upright_support','horizontal_support','center_speed_body_lengths_s',
    'angular_speed_rad_s','ascent_body_lengths_s','scene_similarity','body_valid',
    'flow_peak_cage_width_s','flow_upward_cage_width_s','forepaw_bedding_distance','nose_bedding_distance','box_speed_cage_width_s')


def feature_matrix(rows):
    values=[]
    for r in rows:
        e=r['evidence'];m=r['motion']
        values.append([r['scores'][c] for c in CLASSES]+[
            e['nose_available'],e['axial_available'],e['head_paw_distance'],e['body_length'],
            e['upright_support'],e['horizontal_support'],m['center_speed_body_lengths_s'],
            m['angular_speed_rad_s'],m['ascent_body_lengths_s'],r['scene_similarity'],r['body_valid'],
            m.get('flow_peak_cage_width_s'),m.get('flow_upward_cage_width_s'),e.get('forepaw_bedding_distance'),e.get('nose_bedding_distance'),m.get('box_speed_cage_width_s')])
    return np.array([[np.nan if v is None else float(v) for v in row] for row in values])


def reference_samples(rows, labels, behavior):
    """Only fully, explicitly labeled sample intervals can train; boundary samples omitted."""
    labels=sorted([r for r in labels if r['behavior']==behavior],key=lambda r:r['start_s'])
    for a,b in zip(labels,labels[1:]):
        if a['end_s']>b['start_s']:raise ValueError('Overlapping reference labels')
    ids=[];ys=[];weights=[]
    for i,r in enumerate(rows):
        if not r['body_valid']:continue
        for label in labels:
            if label['start_s']<=r['start_s'] and label['end_s']>=r['end_s'] and label['label'] in ('present','absent'):
                ids.append(i);ys.append(int(label['label']=='present'));weights.append(r['end_s']-r['start_s']);break
    return np.array(ids,int),np.array(ys,int),np.array(weights,float)


def binary_metrics(y,p,weights,threshold):
    y=np.asarray(y);p=np.asarray(p);weights=np.asarray(weights);positive=p>=threshold
    tp=float(weights[(y==1)&positive].sum());fp=float(weights[(y==0)&positive].sum())
    fn=float(weights[(y==1)&~positive].sum());tn=float(weights[(y==0)&~positive].sum())
    div=lambda a,b:a/b if b else None
    precision=div(tp,tp+fp);recall=div(tp,tp+fn)
    return dict(precision=precision,recall=recall,f1=div(2*tp,2*tp+fp+fn),
                balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if (tp+fn)*(tn+fp) else None,
                true_positive_seconds=tp,false_positive_seconds=fp,false_negative_seconds=fn,true_negative_seconds=tn,
                predicted_seconds=tp+fp,reference_seconds=tp+fn,duration_error_seconds=fp-fn,scored_seconds=tp+fp+fn+tn)
