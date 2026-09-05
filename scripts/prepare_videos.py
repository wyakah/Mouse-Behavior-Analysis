"""Make lossless first-N-second working copies, then require error-free full decoding.
Original recordings are untouched. Train/infer on the prepared file, never reuse original tracks.
"""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av
import imageio_ffmpeg
p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=600);a=p.parse_args()
if not 0<a.seconds<=600: p.error('For these damaged sample tails, use an interval ending at or before 600 s.')
root=Path(__file__).resolve().parents[1];out=root/'prepared';out.mkdir(exist_ok=True)
for src in sorted(root.glob('*.mp4')):
 dst=out/f'{src.stem}__first_{a.seconds:g}s.mp4'
 if dst.exists(): print('Already exists:',dst.name,flush=True);continue
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-i',str(src),'-t',str(a.seconds),'-map','0:v:0','-an','-c:v','copy','-movflags','+faststart',str(dst)],check=True)
 times=[]
 with av.open(str(dst)) as c:
  s=c.streams.video[0]
  for f in c.decode(s): times.append(float(f.pts*f.time_base))
 if any(b<=a for a,b in zip(times,times[1:])): raise ValueError('Prepared timestamps are not strictly increasing; do not analyze this file.')
 report={'original':src.name,'prepared':str(dst.relative_to(root)),'requested_interval_s':[0,a.seconds],'decoded_frames':len(times),'first_pts_s':times[0],'last_pts_s':times[-1],'max_frame_interval_s':max(b-a for a,b in zip(times,times[1:])),'method':'Lossless stream copy; video only; no rescaling or frame-rate conversion.','validation':'Complete strict decode passed. Infer DLC on THIS file to maintain frame alignment.','caution':'Confirm the chosen interval matches the experiment. Original final seconds are excluded.'}
 dst.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
