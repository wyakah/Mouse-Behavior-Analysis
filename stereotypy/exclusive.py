"""Exclusive, threshold-qualified decisions integrated over source time.

Independent head scores are not calibrated probabilities. Select the largest
eligible score, with exact ties unknown. Do not force ordinary activity into
one of the six target behaviors.
"""
from .training import CLASSES

class ExclusiveScorer:
    def __init__(self,thresholds):
        self.thresholds=thresholds;self.seconds=dict.fromkeys(CLASSES,0.)
        self.other=0.;self.unknown=0.;self.end=0.;self.bouts=[];self.intervals=[]
    def add(self,start,end,scores,observable=True):
        if start<self.end-1e-7 or end<=start:raise ValueError('Scoring intervals must be ordered and nonoverlapping')
        if start>self.end+1e-7:self._record(self.end,start,'unknown')
        eligible=sorted((scores[c],c) for c in CLASSES if scores[c]>=self.thresholds[c])
        label='unknown' if not observable else 'other'
        if observable and eligible:
            label='unknown' if len(eligible)>1 and abs(eligible[-1][0]-eligible[-2][0])<1e-9 else eligible[-1][1]
        self._record(start,end,label);return label
    def _record(self,start,end,label):
        dt=end-start
        if label=='unknown':self.unknown+=dt
        elif label=='other':self.other+=dt
        else:self.seconds[label]+=dt
        if self.intervals and self.intervals[-1]['behavior']==label and abs(self.intervals[-1]['end_s']-start)<1e-7:
            self.intervals[-1]['end_s']=end;self.intervals[-1]['duration_s']+=dt
        else:self.intervals.append(dict(behavior=label,start_s=start,end_s=end,duration_s=dt))
        self.end=end
    def summary(self):
        return [dict(behavior=c,candidate_seconds=self.seconds[c] if self.end-self.unknown>1e-7 else None,
          candidate_segments=sum(r['behavior']==c for r in self.intervals) if self.end-self.unknown>1e-7 else None,sampled_coverage_s=self.end-self.unknown,
          unknown_seconds=self.unknown,threshold=self.thresholds[c],accepted_seconds=None,accuracy=None,
          evidence='Exclusive highest threshold-qualified score; unvalidated',status='provisional_exclusive') for c in CLASSES]
    def snapshot(self):
        return dict(through_s=self.end,seconds=self.seconds.copy(),other_seconds=self.other,unknown_seconds=self.unknown)
