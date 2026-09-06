"""Independent evidence checks; never substitutes confidence for ground truth.

Coordinates are cage pixels. Distances are normalized to pose body length, not cm.
Thresholds are exploratory review rules, not trained behavior decisions.
"""
import cv2
import numpy as np

VERSION = 'pose-motion-review-1'
LANDMARKS = ('nose', 'neck_base', 'back_middle', 'tail_base',
             'front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw')


def enhance(image, mode='raw'):
    if mode == 'raw':
        return image.copy()
    if mode == 'gamma':
        lut = np.round(255 * (np.arange(256) / 255.) ** .8).astype(np.uint8)
        return cv2.LUT(image, lut)
    if mode == 'clahe':
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    raise ValueError('Unknown enhancement')


def scene_similarity(reference, image):
    """Static cage edge agreement. Low agreement is a review flag, not truth."""
    def edges(im):
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(cv2.resize(gray, (256, 128)), 50, 130) > 0
    a, b = edges(reference), edges(image)
    # Ignore the lower moving-animal/bedding region; retain apparatus structure.
    a[80:] = False; b[80:] = False
    if min(a.sum(), b.sum()) < 30:
        return None
    ad = cv2.dilate(a.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    bd = cv2.dilate(b.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return float(min((a & bd).sum()/a.sum(), (b & ad).sum()/b.sum()))


def pose_evidence(pose, parts, shape, cutoff=.6):
    p = dict(zip(parts, np.asarray(pose)))
    height, width = shape[:2]
    def point(name):
        v = p.get(name)
        if v is None or not np.isfinite(v).all() or v[2] < cutoff:
            return None
        if not (0 <= v[0] < width and 0 <= v[1] < height):
            return None
        return v[:2]
    valid = {name: point(name) for name in LANDMARKS}
    nose, neck, back, tail = [valid[n] for n in LANDMARKS[:4]]
    result = dict(visible_landmarks=sum(v is not None for v in valid.values()),
                  nose_available=nose is not None, axial_available=False,
                  upright_support=None, horizontal_support=None, body_length=None,
                  head_paw_distance=None, center=None, angle=None)
    if back is not None:
        result['center'] = back.tolist()  # mid-back proxy, not center of mass
    if neck is not None and tail is not None:
        length = float(np.linalg.norm(neck-tail))
        if .04*width <= length <= .7*width:
            dy = float(tail[1]-neck[1]); dx = float(neck[0]-tail[0])
            result.update(axial_available=True, body_length=length,
                          angle=float(np.arctan2(-dy, dx)),
                          upright_support=dy/length > .7,
                          horizontal_support=abs(dy)/length < .35)
            paws = [valid[n] for n in ('front_left_paw','front_right_paw') if valid[n] is not None]
            if nose is not None and paws:
                result['head_paw_distance'] = float(min(np.linalg.norm(nose-v) for v in paws)/length)
    return result


def disagreements(scores, thresholds, evidence, scene_valid):
    """Return reasons for review. Never turn missing evidence into a negative label."""
    flags = []
    if not scene_valid:
        return ['scene_unreliable']
    if not evidence['axial_available']:
        flags.append('pose_unavailable')
    elif scores['rearing'] >= thresholds['rearing'] and evidence['horizontal_support']:
        flags.append('rearing_pose_disagreement')
    elif scores['rearing'] < thresholds['rearing'] and evidence['upright_support']:
        flags.append('upright_video_disagreement')
    return flags


def temporal_features(previous, current, dt):
    result = dict(center_speed_body_lengths_s=None, angular_speed_rad_s=None,
                  ascent_body_lengths_s=None)
    if previous is None or not 0 < dt <= .25:
        return result
    length = current.get('body_length')
    if length and previous.get('center') is not None and current.get('center') is not None:
        delta = np.array(current['center']) - previous['center']
        result.update(center_speed_body_lengths_s=float(np.linalg.norm(delta)/length/dt),
                      ascent_body_lengths_s=float(-delta[1]/length/dt))
    if previous.get('angle') is not None and current.get('angle') is not None:
        d = current['angle'] - previous['angle']
        result['angular_speed_rad_s'] = float(np.arctan2(np.sin(d),np.cos(d))/dt)
    return result


def appearance_box(image):
    """Dark-body silhouette proposal independent of motion; explicitly untrained.

Rejects cage borders and very thin fixture/tail components. Does not claim an
anatomical segmentation and can still confuse a dark fixture with a mouse.
"""
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);h,w=gray.shape
    smooth=cv2.GaussianBlur(gray,(5,5),0)
    local=cv2.GaussianBlur(gray,(0,0),max(5,w*.05))
    threshold=min(95,float(np.percentile(smooth,5))+3)
    mask=(smooth<threshold).astype(np.uint8)
    mask[:round(.18*h)]=0;mask[round(.91*h):]=0
    mask[:,:round(.12*w)]=0;mask[:,round(.95*w):]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    candidates=[]
    for x,y,bw,bh,area in stats[1:]:
        if .0015*h*w<area<.15*h*w and min(bw,bh)>8 and .25<bh/bw<5:
            candidates.append((int(area),[int(x),int(y),int(x+bw),int(y+bh)]))
    candidates.sort(reverse=True)
    if not candidates:return None,False
    return candidates[0][1],len(candidates)>1 and candidates[1][0]>.65*candidates[0][0]


def pad_box(box, shape, fraction=.25):
    h,w=shape[:2];x1,y1,x2,y2=box;pad=fraction*max(x2-x1,y2-y1)
    return [max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)]
