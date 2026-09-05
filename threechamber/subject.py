"""Select the free subject from actual DLC pose candidates. Never invent missing poses."""
from dataclasses import dataclass
import numpy as np

ANCHORS=('nose','left_ear','right_ear','mouse_center','tail_base')


def validate_profile(profile,width,height):
    if profile['source_size']!=[width,height]:
        raise ValueError('Source resolution differs from saved EthoVision profile; calibrate a new crop.')
    x1,y1,x2,y2=profile['crop_xyxy']
    if not 0<=x1<x2<=width or not 0<=y1<y2<=height:
        raise ValueError('Crop must lie within source image.')
    return [x1,x2,y1,y2] # DLC order

@dataclass
class SubjectSelector:
    profile:dict
    previous_center:object=None
    previous_time:float=None
    previous_source:str=None

    def reset(self):
        self.previous_center=self.previous_time=self.previous_source=None

    def select(self,poses,bodyparts,time_s,source):
        poses=np.asarray(poses,dtype=float)
        index={p:i for i,p in enumerate(bodyparts)}
        if not set(ANCHORS)<=index.keys():raise ValueError('Required nose, ear, center, tail-base anchors are missing.')
        dt=None if self.previous_time is None else time_s-self.previous_time
        temporal=(source==self.previous_source and dt is not None and 0<dt<=self.profile['continuity_reset_seconds'])
        if not temporal:self.reset()
        candidates=[]
        for i,pose in enumerate(poses):
            a=pose[[index[p] for p in ANCHORS]]
            if a.shape!=(5,3) or not np.isfinite(a).all() or (a[:,2]<0).any():continue
            # Confidence aggregation and broad anatomical constraints, not probability calibration.
            quality=float(np.median(a[:,2]))
            nose,le,re,center,tail=a[:,:2]
            bodylength=float(np.linalg.norm(nose-tail))
            ears=float(np.linalg.norm(le-re))
            plausible=(self.profile['minimum_body_length_px']<=bodylength<=self.profile['maximum_body_length_px'] and 1<=ears<=60)
            if not plausible or a[3,2]<.3:continue
            x1,y1,x2,y2=self.profile['crop_xyxy']
            if not (0<=center[0]<x2-x1 and 0<=center[1]<y2-y1):continue
            score=quality
            if temporal:
                distance=float(np.linalg.norm(center-self.previous_center))
                allowed=max(15,self.profile['maximum_speed_px_s']*dt)
                # Extreme jumps are unresolved, not smoothed into a plausible path.
                if distance>allowed:continue
                score+=.12*(1-min(distance/allowed,1))
            candidates.append({'index':i,'quality':quality,'selection_score':score,'center':center})
        candidates.sort(key=lambda x:x['selection_score'],reverse=True)
        eligible=[c for c in candidates if c['quality']>=self.profile['minimum_pose_quality']]
        reason='no_plausible_candidate';chosen=None
        if eligible:
            best=eligible[0]
            competitors=[c for c in eligible[1:] if np.linalg.norm(c['center']-best['center'])>12]
            margin=best['selection_score']-competitors[0]['selection_score'] if competitors else 1.
            if margin>=self.profile['minimum_candidate_margin']:
                chosen=best;reason='selected'
            else:reason='ambiguous_candidates'
        if chosen is not None:
            self.previous_center=chosen['center'];self.previous_time=float(time_s);self.previous_source=source
        # Missing frames cannot carry a pose or refresh its timestamp.
        return {'candidate_index':None if chosen is None else chosen['index'],'reason':reason,
                'candidate_count':len(candidates),'quality':None if chosen is None else chosen['quality'],
                'temporal_used':temporal}
