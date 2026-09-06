"""Observable pose/motion evidence, independent of behavior predictions.

A likelihood threshold is not a visibility detector. Coordinates passing these
checks are usable estimates, never proof of mouth contact or behavioral truth.
"""
import numpy as np
from .robustness import LANDMARKS,pose_evidence

class PoseSignals:
    def __init__(self,cutoff=.6,max_speed=12.):
        self.cutoff=cutoff;self.max_speed=max_speed;self.previous=None;self.last_t=None
    def add(self,t,pose,parts,shape):
        pose=np.asarray(pose,float).copy();h,w=shape[:2];dt=t-self.last_t if self.last_t is not None else None
        invalid=~np.isfinite(pose).all(axis=1)|(pose[:,2]<self.cutoff)|(pose[:,0]<0)|(pose[:,0]>=w)|(pose[:,1]<0)|(pose[:,1]>=h)
        raw_e=pose_evidence(pose,parts,shape,self.cutoff);length=raw_e['body_length'] or w*.15
        rejected=[]
        if self.previous is not None and dt is not None and 0<dt<=.25:
            valid_previous=np.isfinite(self.previous).all(axis=1)
            jumps=np.linalg.norm(pose[:,:2]-self.previous[:,:2],axis=1)>self.max_speed*length*dt
            reject=jumps&valid_previous&~invalid;invalid|=reject;rejected=[parts[i] for i in np.flatnonzero(reject)]
        pose[invalid]=np.nan
        e=pose_evidence(pose,parts,shape,self.cutoff)
        coordinates={name:([float(v[0]),float(v[1]),float(v[2])] if np.isfinite(v).all() else None) for name,v in zip(parts,pose) if name in LANDMARKS}
        paw_speeds=[]
        if self.previous is not None and dt is not None and 0<dt<=.25 and e['body_length']:
            for name in ('front_left_paw','front_right_paw'):
                if name in parts:
                    i=parts.index(name)
                    if np.isfinite(pose[i]).all() and np.isfinite(self.previous[i]).all():paw_speeds.append(float(np.linalg.norm(pose[i,:2]-self.previous[i,:2])/e['body_length']/dt))
        # Reset rejected points instead of propagating an old coordinate through occlusion.
        self.previous=pose;self.last_t=t
        return dict(start_s=float(t),points=coordinates,body_length=e['body_length'],upright_support=e['upright_support'],head_paw_distance=e['head_paw_distance'],forepaw_speed=max(paw_speeds) if paw_speeds else None,usable_landmarks=sum(v is not None for v in coordinates.values()),rejected_jumps=rejected,contact_status='not_measured',visibility_status='not_calibrated')

def summarize_pose(samples):
    def median(key):
        values=[s[key] for s in samples if s.get(key) is not None]
        return float(np.median(values)) if values else None
    return dict(sample_count=len(samples),nose_usable_fraction=sum(s['points'].get('nose') is not None for s in samples)/len(samples) if samples else 0.,head_paw_distance=median('head_paw_distance'),forepaw_speed=median('forepaw_speed'),upright_fraction=median('upright_support'),contact_status='not_measured')

class PoseRuntime:
    def __init__(self):
        from scripts.stereotypy_robustness_run import setup
        self.runner,self.parts,self.metadata=setup();self.signals=PoseSignals()
    def infer(self,images,times):
        from scripts.stereotypy_robustness_run import infer
        from .robustness import appearance_box
        boxes=[appearance_box(im)[0] for im in images];poses=[np.full((len(self.parts),3),np.nan) for _ in images]
        ids=[i for i,b in enumerate(boxes) if b is not None]
        for first in range(0,len(ids),16):
            ii=ids[first:first+16];pred=infer(self.runner,self.parts,[images[i] for i in ii],[boxes[i] for i in ii])
            for i,p in zip(ii,pred):poses[i]=p
        return [self.signals.add(t,p,self.parts,im.shape) for t,p,im in zip(times,poses,images)]


def infer_or_error(runtime,images,times):
    """Optional anatomy must not abort an otherwise usable video analysis."""
    try:
        return runtime.infer(images,times),None
    except Exception as exc:
        return [],str(exc)
