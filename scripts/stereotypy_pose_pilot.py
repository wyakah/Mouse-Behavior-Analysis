"""Bounded side-view SuperAnimal trial on timestamped, cage-cropped samples."""
import os
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key, value in {'MPLCONFIGDIR': '.cache/matplotlib', 'TORCH_HOME': '.cache/torch',
                   'HF_HOME': '.cache/huggingface', 'XDG_CACHE_HOME': '.cache'}.items():
    os.environ[key] = str(ROOT / value)
os.environ['DLC_LIGHT'] = 'True'


def main():
    import cv2
    import av
    import torch
    import deeplabcut as dlc
    out = ROOT / 'outputs/stereotypy-pilot/pose'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    video = out / 'sampled-cages.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 2, (1280, 560))
    for name, roi in [('745_stereotypy.MOV', [1200, 245, 3120, 1090]),
                      ('729_stereotypy.mov', [300, 340, 3210, 1590])]:
        with av.open(str(ROOT / name)) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base)
            for t in [i * (duration - 1) / 23 for i in range(24)]:
                container.seek(int(t / stream.time_base), stream=stream)
                for frame in container.decode(stream):
                    if float(frame.pts * frame.time_base) >= t:
                        break
                x1, y1, x2, y2 = roi
                crop = frame.to_ndarray(format='bgr24')[y1:y2, x1:x2]
                scale = min(1280 / crop.shape[1], 560 / crop.shape[0])
                import numpy as np
                resized = cv2.resize(crop, (round(crop.shape[1]*scale), round(crop.shape[0]*scale)))
                canvas = np.zeros((560, 1280, 3), np.uint8)
                canvas[:resized.shape[0], :resized.shape[1]] = resized
                writer.write(canvas)
                rows.append(dict(pilot_frame=len(rows), video=name, source_pts=frame.pts,
                                 time_base=str(frame.time_base), source_time_s=float(frame.pts*frame.time_base),
                                 crop_xyxy=roi, scale_x=resized.shape[1]/crop.shape[1],
                                 scale_y=resized.shape[0]/crop.shape[0]))
    writer.release()
    (out / 'sample-manifest.json').write_text(json.dumps(rows, indent=2))
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    started = time.time()
    print('Starting 48-frame side-view trial on', device, flush=True)
    dlc.video_inference_superanimal(
        videos=[str(video)], superanimal_name='superanimal_quadruped', model_name='hrnet_w32',
        detector_name='fasterrcnn_resnet50_fpn_v2', dest_folder=out, video_adapt=False,
        max_individuals=1, device=device, pcutoff=.6, create_labeled_video=True,
        batch_size=4, detector_batch_size=4)
    (out / 'runtime.json').write_text(json.dumps(dict(
        model='SuperAnimal-Quadruped HRNet-W32', detector='FasterRCNN ResNet50 FPN v2',
        deeplabcut_version=dlc.__version__, device=device, runtime_s=time.time()-started,
        status='experimental_unvalidated', sampled_frames=len(rows),
        purpose='Pose feasibility only. Noncontiguous frames must not be used to score behavior.'), indent=2))


if __name__ == '__main__':
    main()
