"""Full-source cage review pass; experimental proposals stay separate from scores."""
from pathlib import Path
from fractions import Fraction
import csv
import json
import hashlib
import av
import cv2
import numpy as np
from .video import index_video
from .window import clip_manifest
from .cage import VERSION, validate_mapping, crop_frame, CageProposer, review_windows
IMPLEMENTATION_HASHES={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['pilot.py','cage.py','video.py','window.py']}


def run_pilot(path, mapping, out, manifest=None, progress=None):
    path,out=Path(path),Path(out)
    manifest=clip_manifest(manifest or index_video(path))
    if manifest['rotation_degrees'] != 0:
        raise ValueError('Normalize and verify rotated footage before cage analysis; originals are unchanged.')
    mapping=validate_mapping(mapping,manifest['width'],manifest['height'])
    signature=path.stat()
    if (signature.st_size,signature.st_mtime_ns)!=(manifest['source_size'],manifest['source_mtime_ns']):
        raise ValueError('Source changed since indexing.')
    with path.open('rb') as handle:
        if hashlib.file_digest(handle,'sha256').hexdigest()!=manifest['source_sha256']:
            raise ValueError('Source content hash changed since indexing.')
    out.mkdir(parents=True,exist_ok=True)
    samples=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        for t in np.linspace(0,max(0,manifest['duration_s']-.1),32):
            target=t+manifest['source_origin_s']
            container.seek(int(target/stream.time_base),stream=stream)
            for frame in container.decode(stream):
                if float(frame.pts*frame.time_base)>=target:break
            samples.append(cv2.cvtColor(crop_frame(frame.to_ndarray(format='bgr24'),mapping),cv2.COLOR_BGR2GRAY))
    background=np.percentile(samples,90,axis=0).astype(np.uint8)
    cv2.imwrite(str(out/'background.jpg'),background)
    proposer=CageProposer(background,mapping)
    rows=[];height,width=background.shape
    destination=out/'review.mp4';temporary=out/'review.partial.mp4'
    with av.open(str(path)) as container, av.open(str(temporary),'w',options={'movflags':'+faststart','movie_timescale':'60000'}) as output:
        stream=container.streams.video[0]
        encoded=output.add_stream('libx264',rate=Fraction(str(manifest['nominal_fps'] or 30)).limit_denominator(100000))
        encoded.width=width;encoded.height=height;encoded.pix_fmt='yuv420p'
        encoded.time_base=Fraction(1,60000)
        encoded.codec_context.time_base=Fraction(1,60000)
        encoded.codec_context.max_b_frames=0
        encoded.options={'crf':'21','preset':'veryfast'}
        durations={round(r['start_s']*60000):r['end_s']-r['start_s'] for r in manifest['frames']}
        def mux(packet):
            key=round(float(packet.pts*packet.time_base)*60000)
            packet.duration=round(durations[key]/float(packet.time_base))
            output.mux(packet)
        previous=None
        for index,frame in enumerate(container.decode(stream)):
            if index >= len(manifest['frames']):break
            timing=manifest['frames'][index]
            if frame.pts!=timing['pts']:raise ValueError('Source timing changed during analysis.')
            crop=crop_frame(frame.to_ndarray(format='bgr24'),mapping)
            dt=timing['start_s']-previous if previous is not None else 0
            features=proposer.frame(crop,dt)
            row=dict(timing,**features);rows.append(row);previous=timing['start_s']
            x1,y1,x2,y2=mapping['crop_xyxy']
            floor=round((mapping['floor_y']-y1)/(y2-y1)*height)
            cv2.line(crop,(0,floor),(width,floor),(170,135,30),1)
            if features['bbox_source']:
                b=features['bbox_source'];a=(round((b[0]-x1)/(x2-x1)*width),round((b[1]-y1)/(y2-y1)*height));z=(round((b[2]-x1)/(x2-x1)*width),round((b[3]-y1)/(y2-y1)*height))
                cv2.rectangle(crop,a,z,(80,220,170) if features['status']=='proposed' else (40,180,250),2)
            cv2.rectangle(crop,(0,0),(width,56),(25,30,35),-1)
            title=f"{timing['start_s']:.2f}s | Body: {features['status']}"
            if features['upright_candidate']:title+=' | Upright candidate'
            cv2.putText(crop,title,(14,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(crop,'EXPERIMENTAL REVIEW - behavior not scored',(14,45),cv2.FONT_HERSHEY_SIMPLEX,.45,(120,205,245),1,cv2.LINE_AA)
            if index==0:cv2.imwrite(str(out/'poster.jpg'),crop)
            f=av.VideoFrame.from_ndarray(crop,format='bgr24');f.pts=round(timing['start_s']*60000);f.time_base=Fraction(1,60000)
            f.duration=round((timing['end_s']-timing['start_s'])*60000)
            for packet in encoded.encode(f):mux(packet)
            if progress and index%150==0:progress(index,len(manifest['frames']))
        for packet in encoded.encode():mux(packet)
    if len(rows)!=manifest['frame_count']:raise ValueError('Review is incomplete; frame count differs from source.')
    if (path.stat().st_size,path.stat().st_mtime_ns)!=(signature.st_size,signature.st_mtime_ns):raise ValueError('Source changed while processing.')
    temporary.replace(destination)
    with (out/'frame-features.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    windows=review_windows(rows,manifest['duration_s'])
    (out/'review-windows.json').write_text(json.dumps(windows,indent=2))
    seconds={status:sum(r['end_s']-r['start_s'] for r in rows if r['status']==status) for status in ['proposed','ambiguous','edge','missing','camera_motion']}
    result=dict(version=VERSION,status='experimental_unvalidated',source=path.name,
        source_sha256=manifest['source_sha256'],mapping=mapping,frame_count=len(rows),duration_s=manifest['duration_s'],
        source_gaps=manifest['source_gaps'],proposal_seconds=seconds,
        proposal_coverage_percent=100*seconds['proposed']/manifest['duration_s'],
        behavior_scores=None,review_windows=windows,
        method='90th-percentile cage background, dark foreground components, ambiguity rejection; untrained image heuristics.',
        limitations=['Proposal coverage is not accuracy. No anatomical keypoints are inferred.',
                    'Upright candidates are not confirmed rearing. Grooming, digging and gnawing are not scored.',
                    'Fixed camera, one dark mouse and visible contrasting background required. Bedding, hopper and glare can hide body parts.',
                    'Review video is a timestamped derivative. Exact source frames and source intervals remain authoritative.'],
        parameters=dict(background_samples=32,background_percentile=90,dark_threshold=115,difference_threshold=22,
            maximum_crop_width=1280,morphology_kernel=5,component_area_fraction=[.0015,.20],
            competing_component_ratio=.45,camera_band_fraction=.2,camera_response_cutoff=.15,
            camera_shift_cage_width=.005,upright_aspect_ratio=1.35,upright_height_fraction=.28,
            upright_above_bedding_fraction=.25,maximum_motion_dt_s=.2),
        review_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        implementation_sha256=IMPLEMENTATION_HASHES)
    (out/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
