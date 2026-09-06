"""Exploratory model candidates, deliberately separate from accepted scoring."""
import math
from .training import CLASSES


def select_sources(folder, selected):
    paths=[p for p in folder.glob('*.mov') if p.stem.split('_')[0] in selected]
    ids=[p.stem.split('_')[0] for p in paths]
    if len(ids)!=len(set(ids)) or set(ids)!=set(selected):
        raise ValueError('Expected exactly one recording for each selected mouse ID.')
    return sorted(paths)


def summarize_predictions(rows,thresholds,duration,gaps=()):
    if not 0 < duration <= 1200 or set(thresholds)!=set(CLASSES):raise ValueError('Invalid assay window or thresholds')
    previous=0
    for row in rows:
        a,b=row['start_s'],row['end_s']
        if not 0<=a<b<=duration or a<previous-1e-7:raise ValueError('Prediction intervals overlap or exceed assay window')
        if set(row['scores'])!=set(CLASSES) or any(not math.isfinite(v) or not 0<=v<=1 for v in row['scores'].values()):raise ValueError('Invalid model scores')
        previous=b
    summary=[];all_bouts=[]
    for c in CLASSES:
        threshold=thresholds[c]
        if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid threshold')
        pieces=[]
        for row in rows:
            spans=[(row['start_s'],row['end_s'])]
            for a,b in gaps:
                spans=[part for x,y in spans for part in ([(x,min(a,y))] if x<a else [])+([(max(b,x),y)] if b<y else []) if part[0]<part[1]]
            pieces.extend((a,b,row['scores'][c]) for a,b in spans)
        bouts=[]
        for a,b,p in pieces:
            if p<threshold:continue
            if bouts and abs(bouts[-1]['end_s']-a)<1e-7:bouts[-1]['end_s']=b
            else:bouts.append(dict(behavior=c,start_s=a,end_s=b))
        for bout in bouts:bout['duration_s']=bout['end_s']-bout['start_s']
        covered=sum(b-a for a,b,p in pieces)
        summary.append(dict(behavior=c,candidate_seconds=sum(v['duration_s'] for v in bouts) if covered else None,candidate_segments=len(bouts) if covered else None,
            sampled_coverage_s=covered,unknown_seconds=max(0,duration-covered),threshold=threshold,
            seconds_at_lower_threshold=sum(b-a for a,b,p in pieces if p>=max(0,threshold-.1)) if covered else None,
            seconds_at_higher_threshold=sum(b-a for a,b,p in pieces if p>=min(1,threshold+.1)) if covered else None,
            accepted_seconds=None,accuracy=None,status='unreviewed_candidates',
            evidence='weak video-bag head; no independent positive validation' if c in ('gnawing_nonfood','jumping','circling') else 'priority temporal head; external validation target not met'))
        all_bouts.extend(bouts)
    return summary,all_bouts


def apply_observability(result, tracking, localization_approved):
    """Exclude failed body proposals; a failed whole-run audit rejects all totals.

    This is a quality gate, not evidence that retained behavior labels are correct.
    Raw model scores remain available for inspecting false detections.
    """
    excluded=[]
    for row in tracking:
        if not localization_approved or row['status']!='proposed':
            a,b=float(row['start_s']),float(row['end_s'])
            if excluded and abs(excluded[-1][1]-a)<1e-7:excluded[-1][1]=b
            else:excluded.append([a,b])
    summary,bouts=summarize_predictions(result['windows'],result['thresholds'],result['duration_s'],[*result['source_gaps'],*excluded])
    if not localization_approved:
        for item in summary:item['status']='withheld_failed_localization_audit'
    return dict(result,raw_summary=result['summary'],raw_bouts=result['bouts'],summary=summary,bouts=bouts,
                excluded_intervals=excluded,localization_audit_passed=localization_approved,
                status='unreviewed_model_candidates' if localization_approved else 'withheld_failed_localization_audit')


def candidate_windows(rows,thresholds,duration):
    import numpy as np
    chosen=[]
    for c in CLASSES:
        scores=np.array([r['scores'][c] for r in rows]);used=[]
        for reason,order,count in [('high score',np.argsort(-scores),2),('threshold boundary',np.argsort(abs(scores-thresholds[c])),1)]:
            added=0
            for i in order:
                t=rows[i]['start_s'];start=max(0,min(duration-6,t-2))
                if any(abs(t-u)<12 for u in used):continue
                used.append(t);chosen.append(dict(behavior=c,start_s=start,end_s=min(duration,start+6),reason=reason,score=float(scores[i])));added+=1
                if added==count:break
    for t in np.linspace(10,max(10,duration-16),8):chosen.append(dict(behavior='all',start_s=float(t),end_s=min(duration,float(t)+6),reason='systematic sample',score=None))
    return chosen
