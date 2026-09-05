"""Explicit side-view geometry and conservative, untrained body proposals.

These image features prioritize review. They are not anatomical landmarks,
behavior labels, calibrated distances, or estimates of tracking accuracy.
"""
import math
import cv2
import numpy as np

VERSION = 'side-cage-proposals-1'


def validate_mapping(data, width, height):
    if not isinstance(data, dict):
        raise ValueError('Draw the cage rectangle first.')
    roi = data.get('crop_xyxy')
    if (not isinstance(roi, list) or len(roi) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in roi)):
        raise ValueError('Cage bounds must contain four finite pixel coordinates.')
    x1, y1, x2, y2 = [round(v) for v in roi]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2-x1 >= 64 and y2-y1 >= 64):
        raise ValueError('Draw a cage rectangle at least 64 pixels wide and high, inside the source image.')
    floor = data.get('floor_y', y1 + .8*(y2-y1))
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or not y1 < floor < y2:
        raise ValueError('The bedding reference must be inside the cage.')
    return dict(version=1, coordinate_space='unrotated_source_pixels', crop_xyxy=[x1,y1,x2,y2],
                floor_y=float(floor), note='Bedding is a visual reference, not a calibrated floor plane.')


def crop_frame(image, mapping, target_width=1280):
    x1,y1,x2,y2 = mapping['crop_xyxy']
    crop = image[y1:y2,x1:x2]
    width = min(target_width, x2-x1)
    width -= width % 2
    height = max(2, round((y2-y1)*width/(x2-x1)/2)*2)
    return cv2.resize(crop, (width,height), interpolation=cv2.INTER_AREA)


def source_box(box, mapping, shape):
    x1,y1,x2,y2 = mapping['crop_xyxy']
    height,width = shape[:2]
    return [x1+box[0]*(x2-x1)/width, y1+box[1]*(y2-y1)/height,
            x1+box[2]*(x2-x1)/width, y1+box[3]*(y2-y1)/height]


class CageProposer:
    def __init__(self, background, mapping):
        self.background = background.astype(np.float32)
        self.mapping = mapping
        self.previous = None
        self.previous_center = None
        band=max(16,background.shape[0]//5)
        self.reference_band=self.background[:band].copy()

    def frame(self, image, dt):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Correct a global brightness shift; never adapt the background onto a still mouse.
        shift = float(np.median(gray.astype(np.float32)-self.background))
        difference = self.background + shift - gray
        mask = ((difference > 22) & (gray < 115)).astype(np.uint8)*255
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n,labels,stats,centers = cv2.connectedComponentsWithStats(mask)
        candidates = sorted([i for i in range(1,n) if .0015*gray.size < stats[i,4] < .20*gray.size],
                            key=lambda i: stats[i,4], reverse=True)
        result = dict(status='missing', bbox_source=None, upright_candidate=False,
                      aspect_ratio=None, centroid_speed_cage_width_s=None, local_motion=None)
        shift_xy,response=cv2.phaseCorrelate(self.reference_band,gray[:self.reference_band.shape[0]].astype(np.float32))
        camera_shift=float(math.hypot(*shift_xy))
        result.update(camera_shift_pixels=camera_shift,camera_shift_response=float(response))
        camera_moved=response>.15 and camera_shift>.005*gray.shape[1]
        if candidates:
            i = candidates[0]
            x,y,w,h,area = [int(v) for v in stats[i]]
            ambiguous = len(candidates)>1 and stats[candidates[1],4] > .45*area
            edge = x<=2 or y<=2 or x+w>=gray.shape[1]-2 or y+h>=gray.shape[0]-2
            result.update(status='camera_motion' if camera_moved else 'ambiguous' if ambiguous else 'edge' if edge else 'proposed',
                          bbox_source=source_box([x,y,x+w,y+h],self.mapping,image.shape),
                          aspect_ratio=h/w)
            center = centers[i]
            if self.previous_center is not None and 0 < dt < .2:
                result['centroid_speed_cage_width_s'] = float(np.linalg.norm(center-self.previous_center)/gray.shape[1]/dt)
            if self.previous is not None and 0 < dt < .2:
                region = labels==i
                result['local_motion'] = float(np.mean(cv2.absdiff(gray,self.previous)[region]))
            x1,y1,x2,y2 = self.mapping['crop_xyxy']
            floor = (self.mapping['floor_y']-y1)/(y2-y1)*gray.shape[0]
            result['upright_candidate'] = bool(result['status']=='proposed' and h/w>1.35 and h>.28*gray.shape[0] and y<floor-.25*gray.shape[0])
            self.previous_center = center if result['status']=='proposed' else None
        else:
            self.previous_center = None
        self.previous = gray
        return result


def review_windows(rows, duration):
    """Diverse, bounded review windows; never silently turn proposals into labels."""
    windows=[]
    # Systematic windows also sample negatives and detector failures, reducing selection bias.
    for t in np.linspace(0, max(0,duration-6), 8):
        windows.append(dict(start_s=float(t), end_s=min(duration,float(t)+6), reason='systematic sample'))
    for kind, predicate in [
        ('upright posture', lambda r:r['upright_candidate']),
        ('localization uncertainty', lambda r:r['status']!='proposed'),
        ('stationary movement', lambda r:r['status']=='proposed' and r['local_motion'] is not None and r['local_motion']>3 and r['centroid_speed_cage_width_s'] is not None and r['centroid_speed_cage_width_s']<.08),
    ]:
        count=0; last=-10
        for row in rows:
            t=row['start_s']
            if predicate(row) and t-last>=12:
                windows.append(dict(start_s=max(0,t-2),end_s=min(duration,t+4),reason=kind))
                last=t;count+=1
                if count==8:break
    return sorted(windows,key=lambda r:r['start_s'])
