"""Automatic missing-nose diagnostics. Proximity is context, never imputed interaction."""
import numpy as np


def nose_gap_diagnostics(rows, cfg):
    weights=rows.duration_s.to_numpy(dtype=float)
    missing=(~rows.nose_valid.to_numpy(dtype=bool))&(weights>0)
    near=np.zeros(len(rows),dtype=bool)
    circles=cfg.get('cup_circles') if cfg.get('analysis_mode')=='circle_zones' else None
    if circles:
        xy=rows[['center_x','center_y']].to_numpy(dtype=float)
        for side in ('left','right'):
            near|=np.linalg.norm(xy-np.asarray(circles[side]),axis=1)<=float(circles['diameter_px'])/2
        near&=rows.center_valid.to_numpy(dtype=bool)
    edges=np.diff(np.r_[False,missing,False].astype(int))
    gaps=[]
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        gaps.append(dict(start_frame=int(a),end_frame_exclusive=int(b),start_s=float(rows.iloc[a].time_s),
                         duration_s=float(weights[a:b].sum()),center_in_cup_zone_seconds=float(weights[a:b][near[a:b]].sum())))
    total=float(weights.sum())
    return dict(nose_missing_seconds=float(weights[missing].sum()),
                nose_valid_fraction=float(weights[~missing].sum()/total) if total else None,
                center_in_zone_nose_missing_seconds=float(weights[near&missing].sum()) if circles else None,
                longest_nose_gap_seconds=max((g['duration_s'] for g in gaps),default=0.),gaps=gaps,
                interpretation='Center-in-zone is a diagnostic context only, not nose interaction. Missing noses are not interpolated. Coverage is not anatomical accuracy.')
