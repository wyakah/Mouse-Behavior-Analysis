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
    chamber=summary.get(side+'_chamber_seconds') if side in ('left','right') else None
    percent=lambda value:100*value/total if valid(value) and total is not None and total>0 else None
    zones={k+'_zone_seconds':summary.get(k+'_nose_seconds') if valid(coverage) and coverage>0 and valid(summary.get(k+'_nose_seconds')) else None for k in ('left','right')}
    return dict(stranger_side=side,stranger_interaction_seconds=nose,
                total_chamber_seconds=total,**zones,center_zone_seconds=None,
                chamber_stranger_interaction_percent=percent(chamber),
                zone_stranger_interaction_percent=percent(nose),
                stranger_interaction_percent=percent(nose))  # Legacy zone-based alias.


METRIC_KEYS = ('left_chamber_seconds','center_chamber_seconds','right_chamber_seconds',
               'chamber_stranger_interaction_percent','left_zone_seconds','center_zone_seconds',
               'right_zone_seconds','zone_stranger_interaction_percent')



def chamber_label(side, config):
    target=stranger_side(config)
    role=('Stranger' if side==target else 'Object') if side in ('left','right') and target in ('left','right') else None
    return side.title()+(f' ({role})' if role else '')
