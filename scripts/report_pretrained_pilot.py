"""Summarize real pretrained predictions without claiming calibrated behavior times."""
from pathlib import Path
import json,cv2,numpy as np,pandas as pd,av
from fractions import Fraction
root=Path(__file__).resolve().parents[1];mapping=json.loads((root/'outputs/pretrained-pilot/frame_map.json').read_text());records=[];tables={}
for folder,label in [('pretrained-pilot','ResNet50 + MobileNet detector'),('pretrained-pilot-strong','HRNet32 + ResNet50 detector')]:
 out=root/'outputs'/folder;files=list(out.glob('*.h5'))
 if not files:continue
 df=pd.read_hdf(files[0]);d=df[df.columns.get_level_values(0)[0]][df.columns.get_level_values(1)[0]]
 flat=pd.DataFrame(mapping)
 for part in ['nose','mouse_center']:
  p=d[(part,'likelihood')].to_numpy();valid=np.isfinite(p)&(p>=0)
  flat[part+'_x']=np.where(valid,d[(part,'x')]+190,np.nan)
  flat[part+'_y']=np.where(valid,d[(part,'y')]+175,np.nan)
  flat[part+'_likelihood']=np.where(valid,p,np.nan)
  flat[part+'_above_0_6']=valid&(p>=.6)
 flat.to_csv(out/'pilot_predictions.csv',index=False);tables[folder]=flat
 rows=[]
 for src,g in flat.groupby('source',sort=False):
  rows.append({'subject':src[:3],'sampled_frames':len(g),'nose_above_cutoff':int(g.nose_above_0_6.sum()),'center_above_cutoff':int(g.mouse_center_above_0_6.sum()),'no_prediction':int(g.nose_likelihood.isna().sum())})
 records.append({'id':folder,'model':label,'rows':rows,'total_nose_above_cutoff':int(flat.nose_above_0_6.sum()),'total_center_above_cutoff':int(flat.mouse_center_above_0_6.sum()),'sampled_frames':len(flat)})
 # Each sampled frame shown for 1 second. This is a discontinuous inspection montage.
 cap=cv2.VideoCapture(str(root/'outputs/pretrained-pilot/pilot.mp4'));cells=[]
 with av.open(str(out/'review.mp4'),'w') as target:
  stream=target.add_stream('libx264',rate=1);stream.width=1024;stream.height=768;stream.pix_fmt='yuv420p'
  for i,r in flat.iterrows():
   ok,im=cap.read()
   if not ok:raise ValueError('Pilot decode failed')
   for part,color in [('nose',(20,245,245)),('mouse_center',(90,240,90))]:
    p=r[part+'_likelihood'];x,y=r[part+'_x'],r[part+'_y']
    if np.isfinite([p,x,y]).all():
     xy=(int(round(x)),int(round(y)))
     if 0<=xy[0]<1024 and 0<=xy[1]<768:
      if p>=.6:cv2.circle(im,xy,5,color,-1)
      else:cv2.drawMarker(im,xy,(60,120,255),cv2.MARKER_TILTED_CROSS,10,2)
   cv2.rectangle(im,(0,0),(1024,95),(20,26,29),-1)
   for y,text in [(24,f'PRELIMINARY TRACKING PILOT | Subject {r.source[:3]} | source time {r.source_time_s:.2f}s'),(48,f'Source frame {r.source_frame} | nose p={r.nose_likelihood:.3f} | center p={r.mouse_center_likelihood:.3f}'),(72,'Yellow: nose | Green: center | Orange X: below 0.6 | Missing: no marker'),(91,'Sampled stills, NOT continuous footage. No calibrated interaction times.')]:
    cv2.putText(im,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(238,242,242),1,cv2.LINE_AA)
   f=av.VideoFrame.from_ndarray(im,format='bgr24');f.pts=i;f.time_base=Fraction(1,1)
   for pkt in stream.encode(f):target.mux(pkt)
   crop=im[170:590,185:760];crop=cv2.resize(crop,(460,336));crop=cv2.copyMakeBorder(crop,30,0,0,0,cv2.BORDER_CONSTANT,value=(20,25,25))
   cv2.putText(crop,f'{r.source[:3]} | {r.source_time_s:.1f}s | nose p={r.nose_likelihood:.2f}',(7,20),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1);cells.append(crop)
  for pkt in stream.encode():target.mux(pkt)
 cap.release()
 for j in range(3):cv2.imwrite(str(root/'reports'/f'{folder}_{j+1}.jpg'),np.vstack([np.hstack(cells[k:k+3]) for k in range(j*12,(j+1)*12,3)]))
result={'status':'pilot_only','cutoff':.6,'models':records,'sampled_frames_per_video':12,'full_video_scoring_completed':False,'cup_times':None,'chamber_times':None,'reason':'Physical scale missing; pretrained predictions require manual validation, especially at cup contacts.','video_source':'First 600 seconds of each verified prepared recording; 12 sampled frames per video.','warning':'Confidence coverage is not tracking accuracy. The montage is discontinuous and cannot be used to calculate behavior durations.'}
if len(tables)==2:
 a,b=tables.values();both=a.nose_above_0_6&b.nose_above_0_6;dist=np.hypot(a.nose_x-b.nose_x,a.nose_y-b.nose_y)
 result['model_comparison']={'both_above_cutoff':int(both.sum()),'median_nose_disagreement_px':float(dist[both].median()),'max_nose_disagreement_px':float(dist[both].max())}
(root/'reports/pretrained_results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
