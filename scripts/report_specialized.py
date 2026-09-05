"""Generate focused, explicitly preliminary reviews from actual DLC predictions."""
from pathlib import Path
from fractions import Fraction
import json,html,cv2,av,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];report=ROOT/'reports';summary={};alltiles=[]
for name,video,stills in [('specialized-pilot-recall',ROOT/'outputs/pretrained-pilot/pilot.mp4',True),('specialized-continuous',ROOT/'outputs/specialized-continuous/contact_clips.mp4',False)]:
 out=ROOT/'outputs'/name;df=pd.read_csv(out/'selected_tracks.csv');cap=cv2.VideoCapture(str(video));tiles=[];times=[];offset=0;previous=None
 with av.open(str(report/(name+'_review.mp4')),'w') as target:
  stream=target.add_stream('libx264',rate=1 if stills else Fraction(224686,10000));stream.width=560;stream.height=480;stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
  for i,r in df.iterrows():
   ok,im=cap.read()
   if not ok:raise ValueError('Review input frame missing')
   im=im[175:575,190:750].copy();canvas=np.zeros((480,560,3),np.uint8);canvas[80:]=im
   for part,color in [('nose',(40,240,250)),('center',(100,245,90))]:
    x,y,p=r[part+'_x'],r[part+'_y'],r[part+'_likelihood']
    if np.isfinite([x,y,p]).all():
     xy=(round(x-190),round(y-175+80))
     if p>=.6:cv2.circle(canvas,xy,4,color,-1)
     else:cv2.drawMarker(canvas,xy,(30,130,255),cv2.MARKER_TILTED_CROSS,10,1)
   status='Unresolved: no accepted subject pose' if pd.isna(r.candidate_index) else f'Nose p={r.nose_likelihood:.2f} | center p={r.center_likelihood:.2f}'
   lines=[f'Subject {r.source[:3]} | source {r.source_time_s:.3f}s | frame {r.source_frame}',status,'Yellow: nose | green: center | orange X: p < 0.6','PRELIMINARY | '+('Sampled stills, 1 second each' if stills else '3 separate contact clips; source time shown')]
   for j,line in enumerate(lines):cv2.putText(canvas,line,(10,17+j*18),0,.42,(230,240,240),1,cv2.LINE_AA)
   if stills:pts=float(i)
   else:
    if previous is None or previous.source!=r.source:
     start=r.source_time_s;offset=0 if previous is None else times[-1]+1/22.4686
    pts=offset+r.source_time_s-start
   times.append(pts);previous=r
   f=av.VideoFrame.from_ndarray(canvas,format='bgr24');f.pts=round(pts*1000000);f.time_base=Fraction(1,1000000)
   for packet in stream.encode(f):target.mux(packet)
   if stills or i%15==0:tiles.append(canvas)
  for packet in stream.encode():target.mux(packet)
 cap.release()
 records=[]
 for source,g in df.groupby('source',sort=False):records.append({'subject':source[:3],'frames':len(g),'selected':int(g.candidate_index.notna().sum()),'nose_above_cutoff':int((g.nose_likelihood>=.6).sum()),'center_above_cutoff':int((g.center_likelihood>=.6).sum())})
 summary[name]={'rows':records,'frames':len(df),'selected':int(df.candidate_index.notna().sum()),'nose_above_cutoff':int((df.nose_likelihood>=.6).sum()),'temporal_frames':int(df.temporal_used.sum()),'video':name+'_review.mp4'}
 if stills:
  for j in range(3):cv2.imwrite(str(report/f'specialized_subject_{records[j]["subject"]}.jpg'),np.vstack([np.hstack(tiles[k:k+3]) for k in range(j*12,(j+1)*12,3)]))
 else:cv2.imwrite(str(report/'specialized_contact_sheet.jpg'),np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
 df.to_csv(report/(name+'_tracks.csv'),index=False)
# Continuity ablation on identical pose detections: tests selection behavior, never an accuracy score.
from sys import path
path.insert(0,str(ROOT))
from threechamber.subject import SubjectSelector
out=ROOT/'outputs/specialized-continuous';manifest=json.loads((out/'selection_manifest.json').read_text());raw=json.loads(next(out.glob('*before_adapt.json')).read_text());mapping=json.loads((out/'frame_map.json').read_text());p=manifest['profile'].copy();p['continuity_reset_seconds']=0;sel=SubjectSelector(p);without=[sel.select(r['bodyparts'],manifest['bodyparts'],m['source_time_s'],m['source'])['candidate_index'] for r,m in zip(raw,mapping)];actual=pd.read_csv(out/'selected_tracks.csv').candidate_index
summary['continuity_ablation']={'selected_without_continuity':sum(x is not None for x in without),'selected_with_continuity':int(actual.notna().sum()),'different_selected_index_or_missing':sum((None if pd.isna(v) else int(v))!=w for v,w in zip(actual,without)),'note':'No ground truth. Different selections are not evidence of better accuracy.'}
summary['status']='experimental_unvalidated';summary['weights_finetuned']=False;summary['behavior_times_calculated']=False
(report/'specialized_results.json').write_text(json.dumps(summary,indent=2))
rows=''.join(f'<tr><td>{r["subject"]}</td><td>{r["frames"]}</td><td>{r["selected"]}</td><td>{r["nose_above_cutoff"]}</td><td>{r["center_above_cutoff"]}</td></tr>' for r in summary['specialized-continuous']['rows'])
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-chamber tracking improvements</title><style>*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#273b42;font:15px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1150px;margin:auto;padding:35px}h1{font-size:40px;letter-spacing:-1.2px;line-height:1.2}h2{font-size:24px}a{color:#13746d}.tag{font-size:12px;color:#13746d;font-weight:600;letter-spacing:1.2px}section{padding:26px;background:white;border:1px solid #dce5e9;border-radius:12px;margin:24px 0}p{color:#637680}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}video{display:block;width:100%;background:#111;border-radius:7px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #e0e8eb;padding:12px}.note{background:#fff3d9;border-left:3px solid #e6b458;padding:16px}.links{display:flex;gap:22px;flex-wrap:wrap}.metric{font-size:38px;color:#176e64;line-height:1.3}.muted{font-size:13px;color:#788b94}.button{display:inline-block;padding:10px 16px;background:#176e64;color:white;text-decoration:none;border-radius:6px}details{margin:18px 0}details img{width:100%;margin-top:15px}@media(max-width:750px){main{padding:18px}.grid{grid-template-columns:1fr}h1{font-size:30px}section{padding:18px}}</style></head><body><main><a href="/">← Three Chamber</a><h1>A closer view. A more cautious tracker.</h1><p>Specialized for your 1024 × 768 EthoVision recordings. Tested with actual pretrained DeepLabCut predictions on all three samples.</p><div class="tag">EXPERIMENTAL · HUMAN VALIDATION PENDING</div><section><h2>What changed</h2><div class="grid"><div><b>Focused arena</b><p>A saved 560 × 400 crop removes 71.5% of the source pixels from tracking and review. The cups remain visible. Calibration clicks and exported tracks retain original image coordinates.</p><b>Several candidates, one subject</b><p>The detector considers up to five candidates. Broad anatomy and pose-quality checks reject implausible poses; adjacent timestamps can favor continuity. Missing poses stay missing.</p></div><div><b>Measured coverage</b><div class="metric">27 / 36</div><p>Sampled frames with a selected nose at likelihood ≥ 0.6, versus 26 / 36 for the original fast pretrained pilot. Both pilots used the same arena crop.</p><p class="note">This is not an accuracy improvement claim. Some accepted landmarks still drift onto cup edges or place the center away from the body. Increasing detection confidence alone cannot resolve that.</p></div></div></section><section><h2>Review the evidence</h2><div class="grid"><div><h3>36 sampled frames</h3><video controls preload="metadata" src="specialized-pilot-recall_review.mp4"></video><p class="muted">Discontinuous still montage: 12 frames per recording, each held for one second. Source frame and time are displayed.</p></div><div><h3>Three short cup-contact clips</h3><video controls preload="metadata" src="specialized-continuous_review.mp4"></video><p class="muted">270 frames, approximately 12 seconds total. Each recording contributes one continuous 90-frame segment; continuity resets between recordings.</p></div></div><h3>Contact-clip detection coverage</h3><table><thead><tr><th>Subject</th><th>Frames</th><th>Pose selected</th><th>Nose ≥ 0.6</th><th>Center ≥ 0.6</th></tr></thead><tbody>ROWS</tbody></table><p>197 / 270 frames had a selected pose; 73 were unresolved. Only 179 / 270 had a selected nose above the scoring cutoff. These difficult clips reinforce the need for arena-specific labels.</p><p class="muted">The median pose-quality selector threshold is 0.55; each landmark must separately pass the 0.6 scoring cutoff. Neither number is a probability of correct tracking.</p><div class="links"><a href="specialized-pilot-recall_tracks.csv">Sampled predictions CSV</a><a href="specialized-continuous_tracks.csv">Contact-clip predictions CSV</a><a href="specialized_results.json">Run summary JSON</a><a href="pretrained_results.html">Original two-model comparison</a></div><details><summary>All 36 cropped source frames with selected landmarks</summary>IMAGES</details></section><section><h2>Build accuracy from reviewed examples</h2><p>126 source frames are ready to label: 36 pilot frames and 90 diverse frames from the recordings. Mark the free mouse’s nose, left ear, right ear, body center, and tail base. Hidden landmarks can be marked not visible.</p><a class="button" href="/labeling">Open review & label →</a><p>Only explicitly reviewed frames enter a DLC training export. Recordings 675 and 678 are assigned to training; 685 is reserved for development validation. An additional unseen recording will be needed for a final independent assessment.</p><p>The local fine-tuning script initializes from SuperAnimal weights and uses the reviewed labels. No model weights have been fine-tuned yet. Evaluation reports nose error in pixels, missing detections, and the fraction within a chosen tolerance, including misses.</p><p class="note">Behavior-duration results remain pending. We still need measured floor dimensions, confirmed cup footprints, and validated subject tracking to report the requested 1 cm interaction and chamber times.</p></section></main></body></html>'''
page=page.replace('ROWS',rows).replace('IMAGES',''.join(f'<img src="specialized_subject_{i}.jpg" alt="Subject {i}, 12 annotated sampled frames">' for i in ['675','678','685']))
(report/'specialized_results.html').write_text(page);print(json.dumps(summary,indent=2))
