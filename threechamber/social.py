"""Stranger interaction, expressed against observed time in all three chambers."""
import math

def stranger_side(config):
    side=config.get('stranger_side',config.get('target_side','unspecified'))
    if side not in ('left','right','unspecified',None,''):
        raise ValueError('Stranger mouse position must be left or right.')
    return side if side in ('left','right') else 'unspecified'

def stranger_metrics(summary):
    side=stranger_side(summary)
    times=[summary.get(k+'_chamber_seconds') for k in ('left','center','right')]
    valid=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
    total=sum(times) if all(valid(v) for v in times) else None
    nose=summary.get(side+'_nose_seconds') if side in ('left','right') else None
    coverage=summary.get('nose_scoreable_fraction')
    if not valid(nose) or not valid(coverage) or coverage<=0:nose=None
    return dict(stranger_side=side,stranger_interaction_seconds=nose,
                total_chamber_seconds=total,
                stranger_interaction_percent=100*nose/total if nose is not None and total is not None and total>0 else None)
