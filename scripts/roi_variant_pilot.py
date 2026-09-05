"""Bounded pretrained pose/bounding-box variant pilot; does not change production inference."""
from pathlib import Path
import sys,os,json,argparse,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k,v in {'MPLCONFIGDIR':'.cache/matplotlib','TORCH_HOME':'.cache/torch','HF_HOME':'.cache/huggingface','XDG_CACHE_HOME':'.cache'}.items():os.environ[k]=str(ROOT/v)
os.environ['DLC_LIGHT']='True'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True,type=Path);p.add_argument('--proposals',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--dataset');p.add_argument('--model',choices=['resnet_50','hrnet_w32'],default='resnet_50');p.add_argument('--bbox-scale',type=float,default=1);a=p.parse_args()
 if not .5<=a.bbox_scale<=2:p.error('Box scale must be between0.5 and2.')
 import numpy as np,pandas as pd,torch,deeplabcut as dlc
 import deeplabcut.pose_estimation_pytorch as pep
 from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
 torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
 records=json.loads(a.proposals.read_text());records=records['frames'] if isinstance(records,dict) else records
 if a.dataset:records=[r for r in records if r.get('dataset')==a.dataset]
 if len(records)>100:raise ValueError('This pilot is limited to100frames; do not run full videos.')
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 cfg=PoseConfig.build_for_superanimal_inference('superanimal_topviewmouse',model_name=a.model,detector_name='fasterrcnn_mobilenet_v3_large_fpn',max_individuals=1,device=device)
 snapshot=Path(dlc.__file__).parent/f'modelzoo/checkpoints/superanimal_topviewmouse_{a.model}.pt'
 cfg.to_yaml(out/'model_config.yaml',overwrite=True)
 it=pep.VideoIterator(a.video)
 if it.get_n_frames()!=len(records):raise ValueError(f'Proposal/video frame mismatch: {len(records)} / {it.get_n_frames()}')
 contexts=[]
 for r in records:
  b=r.get('bbox_xyxy')
  if b is None:boxes=np.zeros((0,4),float)
  else:
   x1,y1,x2,y2=b;cx=(x1+x2)/2;cy=(y1+y2)/2;w=(x2-x1)*a.bbox_scale;h=(y2-y1)*a.bbox_scale
   x1=max(0,cx-w/2);y1=max(0,cy-h/2);x2=min(1024,cx+w/2);y2=min(768,cy+h/2);boxes=np.array([[x1,y1,x2-x1,y2-y1]],float)
  contexts.append({'bboxes':boxes,'bbox_scores':np.ones(len(boxes))})
 it.set_context(contexts)
 runner=pep.get_pose_inference_runner(cfg,snapshot,batch_size=8,device=device,max_individuals=1)
 started=time.time();print('Running DLC subject ROI on',device,'frames',len(records),flush=True)
 predictions=pep.video_inference(it,runner)
 if len(predictions)!=len(records):raise ValueError('DLC output frame count does not match proposals.')
 # Full-image boxes => full-image DLC coordinates. Preserve every raw prediction.
 np.savez_compressed(out/'raw_poses.npz',poses=np.asarray([r['bodyparts'] for r in predictions]),bodyparts=np.asarray(cfg.metadata.bodyparts))
 parts=list(cfg.metadata.bodyparts);rows=[]
 for i,(r,pred) in enumerate(zip(records,predictions)):
  poses=np.asarray(pred['bodyparts']);row={'frame':i,'source':r['source'],'source_frame':r['source_frame'],'source_time_s':r['source_time_s'],'localization_status':r['status'],'body_proposed':r.get('bbox_xyxy') is not None,'candidate_index':0 if r.get('bbox_xyxy') else np.nan}
  b=r.get('bbox_xyxy');validpose=b is not None and len(poses)>0
  for canonical,part in [('nose','nose'),('center','mouse_center'),('left_ear','left_ear'),('right_ear','right_ear'),('tail_base','tail_base')]:
   point=poses[0,parts.index(part)] if validpose else np.full(3,np.nan)
   x,y,q=map(float,point)
   valid=validpose and np.isfinite(point).all() and 0<=q<=1 and 0<=x<1024 and 0<=y<768
   row.update({canonical+'_x':x if valid else np.nan,canonical+'_y':y if valid else np.nan,canonical+'_likelihood':q if valid else np.nan})
  rows.append(row)
 df=pd.DataFrame(rows);df.to_csv(out/'selected_tracks.csv',index=False)
 for part in ['nose','center']:print(part,'>=0.6',int((df[part+'_likelihood']>=.6).sum()),'/',len(df),flush=True)
 manifest={'status':'experimental_unvalidated','video':str(a.video.resolve()),'source_sha256':hashlib.file_digest(a.video.open('rb'),'sha256').hexdigest(),'proposal_file':str(a.proposals.resolve()),'proposal_sha256':hashlib.file_digest(a.proposals.open('rb'),'sha256').hexdigest(),'model':'SuperAnimal TopViewMouse '+a.model,'model_sha256':hashlib.file_digest(snapshot.open('rb'),'sha256').hexdigest(),'deeplabcut_version':dlc.__version__,'method':'Foreground-conditioned DLC pose; anatomical landmarks are neural predictions, never foreground centroid or contour tips.','coordinate_space':'full_source_pixels','bbox_scale':a.bbox_scale,'device':device,'frames':len(df),'seconds':time.time()-started,'weights_finetuned':False,'accuracy_validated':False,'nose_above_cutoff':int((df.nose_likelihood>=.6).sum()),'center_above_cutoff':int((df.center_likelihood>=.6).sum())}
 (out/'selection_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',manifest['seconds'],flush=True)
if __name__=='__main__':main()


def build_comparison():
 """Create inspectable comparison sheets; counts are confidence coverage, not accuracy."""
 import cv2,numpy as np,pandas as pd
 names=['resnet-1','hrnet-1','resnet-075','resnet-125']
 paths=[ROOT/'outputs/roi-pilot']+[ROOT/'outputs/roi-variants'/n for n in names[1:]]
 tables=[pd.read_csv(p/'selected_tracks.csv') for p in paths]
 proposals=[r for r in json.loads((ROOT/'outputs/localization-benchmark/proposals.json').read_text()) if r['dataset']=='pilot']
 out=ROOT/'outputs/roi-variants';cap=cv2.VideoCapture(str(ROOT/'outputs/pretrained-pilot/pilot.mp4'));frames=[]
 while True:
  ok,im=cap.read()
  if not ok:break
  frames.append(im)
 cap.release();counts=[]
 for name,table in zip(names,tables):
  counts.append({'variant':name,'frames':len(table),'nose_ge_06':int((table.nose_likelihood>=.6).sum()),'center_ge_06':int((table.center_likelihood>=.6).sum()),'both_ge_06':int(((table.nose_likelihood>=.6)&(table.center_likelihood>=.6)).sum())})
 disagreements=[]
 for i,(a,b) in enumerate(zip(tables[0].to_dict('records'),tables[1].to_dict('records'))):
  d=float(np.hypot(a['nose_x']-b['nose_x'],a['nose_y']-b['nose_y']))
  disagreements.append({'pilot_frame':i,'source':a['source'],'source_frame':a['source_frame'],'nose_distance_px':d if np.isfinite(d) else None,'resnet_p':a['nose_likelihood'] if np.isfinite(a['nose_likelihood']) else None,'hrnet_p':b['nose_likelihood'] if np.isfinite(b['nose_likelihood']) else None})
 for start in range(0,len(frames),3):
  sheet=np.full((3*245,5*235,3),245,np.uint8)
  for n,i in enumerate(range(start,min(start+3,len(frames)))):
   b=proposals[i].get('bbox_xyxy');b=b or [190,175,750,575];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
   side=max(b[2]-b[0],b[3]-b[1])+50;x1=max(0,int(cx-side/2));y1=max(0,int(cy-side/2));x2=min(1024,int(cx+side/2));y2=min(768,int(cy+side/2))
   for j in range(5):
    im=frames[i].copy();label=f'#{i} Original' if j==0 else names[j-1]
    if j:
     r=tables[j-1].iloc[i];label+=f' N:{r.nose_likelihood:.2f} C:{r.center_likelihood:.2f}'
     for part,color in [('nose',(30,40,240)),('center',(240,180,0)),('tail_base',(200,0,180)),('left_ear',(30,220,40)),('right_ear',(30,220,40))]:
      x,y,p=[r[part+'_'+k] for k in ['x','y','likelihood']]
      if not np.isfinite([x,y,p]).all():continue
      cv2.drawMarker(im,(round(x),round(y)),color,cv2.MARKER_CROSS,6,1)
    tile=cv2.resize(im[y1:y2,x1:x2],(230,210),interpolation=cv2.INTER_NEAREST)
    sheet[n*245+28:n*245+238,j*235:j*235+230]=tile
    cv2.putText(sheet,label,(j*235+3,n*245+18),cv2.FONT_HERSHEY_SIMPLEX,.38,(30,30,30),1)
  cv2.imwrite(str(out/f'comparison-{start:02d}.jpg'),sheet)
 result={'counts':counts,'resnet_hrnet_disagreement':disagreements,'cutoff':.6,'accuracy_validated':False,'weights_finetuned':False,'note':'Identical36pilotstills and foreground boxes; each model uses raw inference. Qualitative sheet review is not human-landmark accuracy evaluation.'}
 result['qualitative_review']={'reviewer':'assistant visual inspection; not human ground-truth labels','sheets_inspected':['comparison-00.jpg','comparison-03.jpg','comparison-09.jpg','comparison-15.jpg','comparison-33.jpg'],'scope':'15 of36pilotframes visually compared acrossall4variants; selected contact and disagreement examples.','observations':[{'pilot_frame':33,'finding':'HRNet nose p0.907 appears on visible tail-base end and tail_base at head: apparent head-tail reversal. Baseline ResNet snout prediction is lower-confidence but nearer visually apparent head.'},{'pilot_frame':34,'finding':'Foreground proposal is on partition/floor away from main mouse body. Tight ResNet box reports nose p0.81 on this nonmouse region.'},{'pilot_frame':15,'finding':'HRNet center appears within mouse body while baseline center lies on cup edge; illustrates a potential localized improvement requiring labels.'},{'pilot_frame':10,'finding':'Contact/occlusion remains low-confidence for nose across all variants; changing model orboxscale does not recover reliable nose.'}],'conclusion':'Do not promote HRNet or changedboxscale from confidence counts. Both localization errors and pose/head-tail errors remain; reviewed domain labels are required.'}
 (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(counts,indent=2))
