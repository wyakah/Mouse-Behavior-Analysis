"""Fixed-camera subject *box proposals*, never anatomical landmark estimates.

Foreground support is a geometric cue, not calibrated pose confidence. Occlusion,
moving stimulus mice, camera movement and dark floor marks require review.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


def build_background(video: str | Path, crop_xyxy, samples: int = 61):
    """Estimate static grayscale scene and robust temporal noise from a full recording."""
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x1, y1, x2, y2 = map(int, crop_xyxy)
    frames = []
    for index in np.unique(np.linspace(0, max(count - 1, 0), samples).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < min(10, samples):
        raise ValueError('Insufficient readable frames for a static background.')
    stack = np.asarray(frames, dtype=np.int16)
    background = np.median(stack, axis=0).astype(np.uint8)
    noise = np.median(np.abs(stack - background), axis=0).astype(np.uint8)
    return background, noise


@dataclass
class ForegroundLocalizer:
    profile: dict
    background: np.ndarray
    noise: np.ndarray
    previous: tuple | None = None

    def __post_init__(self):
        x1, y1, x2, y2 = self.profile['crop_xyxy']
        if self.background.shape != (y2-y1, x2-x1) or self.noise.shape != self.background.shape:
            raise ValueError('Background shape differs from the saved camera crop.')
        # Interior includes doorways and cup contacts; no cup exclusion mask.
        ax1, ay1, ax2, ay2 = self.profile.get('expected_arena_xyxy', [x1,y1,x2,y2])
        self.arena_mask = np.zeros_like(self.background, dtype=np.uint8)
        self.arena_mask[max(0,ay1-y1):min(y2-y1,ay2-y1),max(0,ax1-x1):min(x2-x1,ax2-x1)] = 1
        if self.profile.get('floor_polygon') is not None:
            if not str(self.profile.get('floor_polygon_source','')).strip():
                raise ValueError('An optional floor polygon requires an explicit source label.')
            points=np.asarray(self.profile['floor_polygon'],dtype=np.float32)
            if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all() or not cv2.isContourConvex(points) or abs(cv2.contourArea(points))<1:
                raise ValueError('Floor polygon must be finite, non-crossing and convex.')
            polygon_mask=np.zeros_like(self.arena_mask)
            cv2.fillConvexPoly(polygon_mask,np.rint(points-[x1,y1]).astype(np.int32),1)
            self.arena_mask &= polygon_mask
            if not np.any(self.arena_mask):
                raise ValueError('Floor polygon does not intersect the saved arena crop.')

    def propose(self, frame: np.ndarray, time_s: float, source: str = '', return_mask=False):
        x1, y1, x2, y2 = self.profile['crop_xyxy']
        if frame.shape[:2] != tuple(reversed(self.profile['source_size'])):
            raise ValueError('Source resolution differs from the saved camera profile.')
        gray = cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
        # Median residual absorbs small whole-frame exposure shifts. A strong
        # negative residual identifies the dark free mouse against a lighter floor.
        difference = self.background.astype(np.float32) - gray.astype(np.float32)
        exposure = float(np.median(difference[self.arena_mask > 0]))
        difference -= exposure
        threshold = np.maximum(22., 5. * self.noise.astype(np.float32) + 8.)
        foreground = ((difference > threshold) & (gray < 0.72*self.background) & (self.arena_mask > 0)).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground)
        candidates = []
        rejected_components = []
        temporal = bool(self.previous and self.previous[0] == source and 0 < time_s-self.previous[1] <= 0.25)
        for label in range(1,n):
            bx, by, bw, bh, area = map(int,stats[label])
            if not 35 <= area <= 2200 or max(bw,bh) > 115 or min(bw,bh) < 4:
                if area >= 35:
                    rejected_components.append({'foreground_area_px':area, 'foreground_bbox_xyxy':[bx+x1,by+y1,bx+x1+bw,by+y1+bh], 'reason':'component_outside_development_shape_limits'})
                continue
            cx, cy = centroids[label]
            pixels = labels == label
            contrast = float(np.median(difference[pixels]))
            score = area * min(contrast,100.) / 100.
            distance = None
            if temporal:
                distance = float(np.hypot(cx-self.previous[2],cy-self.previous[3]))
                # Only rank plausible proposals; never propagate a missing blob.
                score *= 1.0 + 0.5*np.exp(-distance/25.)
            # Padding is context for DLC. The center below is only a blob support
            # location and is deliberately not named/returned as mouse_center.
            pad = 22
            box = [max(x1, bx+x1-pad),max(y1, by+y1-pad),min(x2,bx+x1+bw+pad),min(y2,by+y1+bh+pad)]
            candidates.append({'bbox_xyxy':box,'foreground_bbox_xyxy':[bx+x1,by+y1,bx+x1+bw,by+y1+bh],'foreground_area_px':area,'foreground_contrast':contrast,'ranking_score':float(score),'support_xy':[float(cx+x1),float(cy+y1)],'distance_from_previous_px':distance})
        candidates.sort(key=lambda c:c['ranking_score'],reverse=True)
        result = {'status':'unresolved','bbox_xyxy':None,'candidate_count':len(candidates),'temporal_used':temporal,'quality_kind':'foreground_support_not_pose_probability','candidates':candidates,'rejected_components':rejected_components,'foreground_fraction':float(np.count_nonzero(foreground)/np.count_nonzero(self.arena_mask))}
        if candidates:
            best=candidates[0]
            ratio = best['ranking_score']/max(candidates[1]['ranking_score'],1.) if len(candidates)>1 else None
            if ratio is not None and ratio < 1.5:
                result['status']='ambiguous'
            else:
                result.update(best);result['status']='proposal';result['runner_up_ratio']=ratio
                self.previous=(source,float(time_s),best['support_xy'][0]-x1,best['support_xy'][1]-y1)
        if return_mask:
            return result,foreground*255
        return result


def localization_review_flags(result: dict, profile: dict) -> list[str]:
    """Priority cues only: never remove a proposal or change a coordinate.

    Outer-wall proximity can describe the actual mouse or its reflection. Cup
    height is only a broad camera-layout prior, not a measured cup boundary.
    """
    flags=[]
    if result.get('status')!='proposal':
        return ['ambiguous_foreground_candidates' if result.get('status')=='ambiguous' else 'missing_foreground_proposal']
    xy=result.get('support_xy');box=result.get('foreground_bbox_xyxy')
    if xy is None or box is None:return flags
    x1,y1,x2,y2=profile.get('expected_arena_xyxy',profile['crop_xyxy'])
    x,y=xy
    near_vertical=min(abs(x-x1),abs(x-x2))<=20
    near_horizontal=min(abs(y-y1),abs(y-y2))<=20
    if near_vertical or near_horizontal:
        flags.append('boundary_support_review')
        width=box[2]-box[0];height=box[3]-box[1]
        if min(width,height)<=14 or max(width,height)/max(min(width,height),1)>=2.5:
            flags.append('narrow_boundary_support_reflection_suspected')
        if near_vertical and .25 <= (y-y1)/(y2-y1) <= .75:
            flags.append('boundary_support_at_approximate_cup_height')
    return flags


def build_review_queue(records: list[dict], profile: dict) -> dict:
    """Collect geometry/motion concerns into timestamped review intervals."""
    events=[]
    high_flags={'large_foreground_support_step','narrow_boundary_support_reflection_suspected','boundary_support_at_approximate_cup_height'}
    for r in records:
        flags=sorted(set(r.get('proposal_qc_flags',[])+localization_review_flags(r,profile)))
        if not flags:continue
        events.append({'frame':r.get('frame',r['source_frame']),'source':r['source'],
            'source_frame':r['source_frame'],'source_time_s':r['source_time_s'],
            'duration_s':r.get('duration_s',0.),'flags':flags,
            'priority':'high' if high_flags.intersection(flags) else 'review',
            'status':r['status'],'bbox_xyxy':r.get('bbox_xyxy'),'support_xy':r.get('support_xy')})
    intervals=[]
    for event in events:
        if intervals and intervals[-1]['source']==event['source'] and event['source_time_s']-intervals[-1]['last_flag_time_s']<=1.:
            interval=intervals[-1]
            interval['last_flag_time_s']=event['source_time_s']
            interval['end_frame']=event['source_frame']
            interval['end_s']=event['source_time_s']+event['duration_s']
            interval['flagged_frames']+=1
            interval['flags']=sorted(set(interval['flags']+event['flags']))
            if event['priority']=='high':interval['priority']='high'
        else:
            intervals.append({'source':event['source'],'start_frame':event['source_frame'],
                'end_frame':event['source_frame'],'start_s':event['source_time_s'],
                'end_s':event['source_time_s']+event['duration_s'],
                'last_flag_time_s':event['source_time_s'],'flagged_frames':1,
                'flags':event['flags'].copy(),'priority':event['priority']})
    return {'status':'review_priority_only_not_identity_verdict',
        'method':'Support within 20 px of provisional outer arena edges, narrow shape, approximate central cup-height band, missing/ambiguous proposals and large steps. These cues can flag a real mouse and do not modify proposals.',
        'geometry_source':'Provisional expected_arena_xyxy; no measured cup or physical dimensions assumed.',
        'events':events,'intervals':intervals}
