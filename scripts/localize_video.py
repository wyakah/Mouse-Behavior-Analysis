"""Strict full-video foreground proposals with original frame identity and PTS.

Run with .venv/bin/python. Boxes condition a pose model; foreground support is
never exported as the anatomical mouse center and is not a calibrated confidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import av
import numpy as np
import pandas as pd
from threechamber.core import strict_frames,sha256
from threechamber.localization import build_background,ForegroundLocalizer,localization_review_flags,build_review_queue
from threechamber.subject import validate_profile
from threechamber.live import LivePublisher


def localize_video(video, output, profile_path, samples=61, live=None):
    video=Path(video).resolve();output=Path(output).resolve();profile_path=Path(profile_path).resolve()
    output.mkdir(parents=True,exist_ok=True)
    profile=json.loads(profile_path.read_text());started=time.time()
    with av.open(str(video)) as container:
        stream=container.streams.video[0]
        validate_profile(profile,stream.width,stream.height)
        expected_count=stream.frames
    if live:live.phase('localizing','Building the recording background',expected_count)
    background,noise=build_background(video,profile['crop_xyxy'],samples)
    np.savez_compressed(output/'background.npz',background=background,noise=noise)
    localizer=ForegroundLocalizer(profile,background,noise)
    records=[];timestamps=[];durations=[];previous=None
    with av.open(str(video)) as container:
        stream=container.streams.video[0]
        first_pts=None
        for frame_index,frame in enumerate(strict_frames(container,stream)):
            if frame.pts is None:
                raise ValueError(f'Frame {frame_index} lacks a presentation timestamp.')
            source_pts=float(frame.pts*frame.time_base)
            if first_pts is None:first_pts=source_pts
            timestamp=source_pts-first_pts
            if timestamps and timestamp<=timestamps[-1]:
                raise ValueError(f'Non-increasing source timestamp at frame {frame_index}.')
            image=frame.to_ndarray(format='bgr24')
            result=localizer.propose(image,timestamp,video.name)
            if live:live.frame(image,frame_index,timestamp,bbox=result.get('bbox_xyxy'),force=frame_index+1==expected_count)
            qc=localization_review_flags(result,profile);step=None;speed=None
            if result['status']=='proposal':
                xy=np.array(result['support_xy'])
                if previous is not None and 0<timestamp-previous[0]<=.25:
                    step=float(np.linalg.norm(xy-previous[1]));speed=step/(timestamp-previous[0])
                    if step>max(25.,800.*(timestamp-previous[0])):
                        qc.append('large_foreground_support_step')
                previous=(timestamp,xy)
            if result['foreground_fraction']>.08:
                qc.append('large_scene_change_or_obstruction')
            records.append({'frame':frame_index,'source':video.name,'source_frame':frame_index,
                'source_time_s':timestamp,'source_pts_s':source_pts,**result,
                'proposal_qc_flags':qc,'support_step_px':step,'support_speed_px_s':speed})
            timestamps.append(timestamp)
            durations.append(float(frame.duration*frame.time_base) if frame.duration else 0.)
            if (frame_index+1)%1000==0:
                print(f'{frame_index+1}/{expected_count or "?"} decoded | {timestamp:.2f}s source | {time.time()-started:.1f}s elapsed',flush=True)
    if len(records)<2:
        raise ValueError('At least two decoded frames are needed.')
    if expected_count and len(records)!=expected_count:
        raise ValueError(f'Container/decode frame mismatch: {expected_count} vs {len(records)}.')
    weights=np.r_[np.diff(timestamps),durations[-1] or np.median(np.diff(timestamps))]
    for row,duration in zip(records,weights):row['duration_s']=float(duration)
    status_counts=Counter(r['status'] for r in records)
    source_hash=sha256(video)
    manifest={'status':'complete_experimental_unvalidated','source_video':str(video),'source_sha256':source_hash,
        'profile':profile,'profile_sha256':sha256(profile_path),'frames':len(records),
        'first_source_pts_s':first_pts,'last_normalized_timestamp_s':timestamps[-1],
        'decoded_duration_s':float(weights.sum()),'status_counts':dict(status_counts),
        'proposal_coverage_fraction':status_counts['proposal']/len(records),
        'proposal_duration_fraction':float(sum(r['duration_s'] for r in records if r['status']=='proposal')/weights.sum()),
        'qc_flag_counts':dict(Counter(flag for r in records for flag in r['proposal_qc_flags'])),
        'background':{'method':'Temporal median and median absolute deviation of evenly spaced source frames.',
            'requested_sample_count':samples,'file':'background.npz','sha256':sha256(output/'background.npz')},
        'seconds':time.time()-started,'method':'Dark foreground against fixed per-recording background; heuristic shape and temporal candidate ranking. No landmark estimation or gap filling.',
        'validation':'Proposal availability is not identification or landmark accuracy. Human validation remains required; large-step QC flags mark review points without changing or filling tracks.'}
    # Only publish aligned outputs once the entire original stream has decoded.
    (output/'proposals.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    flat=[{k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in r.items() if k not in ('candidates','rejected_components')} for r in records]
    pd.DataFrame(flat).to_csv(output/'localization.csv',index=False)
    queue=build_review_queue(records,profile)
    (output/'review_qc.json').write_text(json.dumps(queue,indent=2,allow_nan=False))
    (output/'review_intervals.json').write_text(json.dumps(queue['intervals'],indent=2,allow_nan=False))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--output',required=True)
    p.add_argument('--profile',default=str(ROOT/'profiles/ethovision_three_chamber.json'));p.add_argument('--background-samples',type=int,default=61)
    args=p.parse_args()
    if args.background_samples<10:p.error('Use at least 10 background samples.')
    localize_video(args.video,args.output,args.profile,args.background_samples,live=LivePublisher.from_environment())

if __name__=='__main__':main()
