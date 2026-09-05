"""Best-effort latest-frame telemetry. Never a source of scientific measurements."""
from collections import OrderedDict
from pathlib import Path
import base64
import json
import os
import time
import threading
import uuid
import warnings

import cv2
import numpy as np

PARTS = ('nose', 'center', 'tail_base')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def landmark_states(row, cutoff, width, height):
    """Finite low-confidence predictions remain distinct from missing observations."""
    points = {}
    for part in PARTS:
        values = [row.get(part + '_' + c) for c in ('x', 'y', 'likelihood')]
        try:
            x, y, q = map(float, values)
            finite = np.isfinite([x, y, q]).all() and 0 <= q <= 1 and 0 <= x < width and 0 <= y < height
        except (TypeError, ValueError):
            finite = False
        if not finite:
            points[part] = dict(state='missing', x=None, y=None, likelihood=None)
        else:
            accepted = q >= cutoff and bool(row.get(part + '_valid', True))
            points[part] = dict(state='accepted' if accepted else 'uncertain', x=x, y=y, likelihood=q)
    return points


class LivePublisher:
    """One atomic, bounded snapshot at most every interval; errors disable preview only."""
    def __init__(self, folder, context, interval=.5, clock=time.monotonic, wall=time.time):
        self.folder = Path(folder)
        self.context = context
        self.interval = max(0., interval)
        self.clock, self.wall = clock, wall
        self.last = -float('inf')
        self.disabled = False
        self.stage = 'preparing'
        self.stage_started_at = wall()
        self.total_frames = None
        self.published_frames = 0

    @classmethod
    def from_environment(cls):
        folder = os.environ.get('THREECHAMBER_LIVE_DIR')
        if not folder:
            return None
        try:
            return cls(folder, json.loads((Path(folder) / 'context.json').read_text()))
        except (OSError, ValueError):
            return None

    def _write(self, payload):
        if self.disabled:
            return False
        try:
            atomic_json(self.folder / 'snapshot.json', payload)
            return True
        except Exception as exc:
            self.disabled = True
            warnings.warn(f'Live preview disabled; analysis continues: {type(exc).__name__}', RuntimeWarning)
            return False

    def phase(self, stage, message='', total_frames=None):
        self.stage = stage
        self.total_frames = int(total_frames) if total_frames else None
        self.stage_started_at = self.wall()
        self.last = -float('inf')
        return self._write(self._base(message))

    def _base(self, message=''):
        return dict(schema=1, revision=uuid.uuid4().hex, **self.context,
                    stage=self.stage, stage_started_at=self.stage_started_at,
                    updated_at=self.wall(), message=message, frame_index=None,
                    frames_done=0, total_frames=self.total_frames, source_time_s=None,
                    image=None, landmarks={}, bbox=None)

    def due(self):
        return not self.disabled and self.clock() - self.last >= self.interval

    def frame(self, image, frame_index, source_time_s, row=None, bbox=None, force=False, image_is_crop=False, encode_image=True):
        if self.disabled or (not force and not self.due()):
            return False
        # Throttle before rendering/encoding so skipped preview frames cost no JPEG work.
        self.last = self.clock()
        try:
            height, width = image.shape[:2]
            if image_is_crop:width,height=self.context['source_size']
            crop = self.context.get('crop', [0, 0, width, height])
            x1, y1, x2, y2 = map(int, crop)
            if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                raise ValueError('Preview crop outside frame')
            if image_is_crop:
                if image.shape[:2]!=(y2-y1,x2-x1):raise ValueError('Cropped preview dimensions do not match source crop')
                tile=image
            else:tile = image[y1:y2, x1:x2]
            encoded_image=None
            if encode_image:
                scale = min(1., 960 / max(tile.shape[:2]))
                if scale < 1:
                    tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.jpg', tile, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if not ok:
                    raise ValueError('Preview encoding failed')
                encoded_image='data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii')
            payload = self._base()
            payload.update(frame_index=int(frame_index), frames_done=int(frame_index)+1,
                           source_time_s=float(source_time_s), crop=crop,
                           image=encoded_image,
                           landmarks=landmark_states(row or {}, self.context.get('cutoff', .6), width, height),
                           bbox=None if bbox is None else [float(v) for v in bbox])
            if row is not None:
                payload['chamber'] = str(row.get('chamber', '')) or None
                payload['nose_scoreable'] = bool(row['nose_scoreable']) if 'nose_scoreable' in row else None
            saved = self._write(payload)
            if saved:
                self.published_frames += 1
            return saved
        except Exception as exc:
            self.disabled = True
            warnings.warn(f'Live frame preview disabled; analysis continues: {type(exc).__name__}', RuntimeWarning)
            return False


class FrameCache:
    """Bound memory even if a future DLC runner prefetches more frames than expected."""
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.frames = OrderedDict()
        self.lock = threading.Lock()

    def remember(self, index, image):
        with self.lock:
            self.frames[index] = image
            while len(self.frames) > self.capacity:
                self.frames.popitem(last=False)

    def take(self, index):
        with self.lock:
            image = self.frames.get(index)
            while self.frames and next(iter(self.frames)) <= index:
                self.frames.popitem(last=False)
            return image


class PredictionTap:
    """DLC 3 per-frame writer protocol; collect predictions unchanged and observe copies."""
    def __init__(self, observer):
        self.predictions = []
        self.observer = observer
        self.observer_failed = False

    def open(self):
        pass

    def close(self):
        pass

    def add_prediction(self, bodyparts, **extra):
        prediction = {'bodyparts': bodyparts, **{k: v for k, v in extra.items() if v is not None}}
        self.predictions.append(prediction)
        if not self.observer_failed:
            try:
                self.observer(len(self.predictions)-1, prediction)
            except Exception as exc:
                self.observer_failed = True
                warnings.warn(f'Prediction preview disabled; inference continues: {type(exc).__name__}', RuntimeWarning)
