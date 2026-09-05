"""Run actual DLC pose inference conditioned on experimental foreground subject boxes."""
from pathlib import Path
import sys,os,json,argparse,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k,v in {'MPLCONFIGDIR':'.cache/matplotlib','TORCH_HOME':'.cache/torch','HF_HOME':'.cache/huggingface','XDG_CACHE_HOME':'.cache'}.items():os.environ[k]=str(ROOT/v)
os.environ['DLC_LIGHT']='True'
from threechamber.live import LivePublisher,FrameCache,PredictionTap
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True,type=Path);p.add_argument('--proposals',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--dataset');p.add_argument('--bbox-scale',type=float,default=1);a=p.parse_args()
 if not .5<=a.bbox_scale<=2:p.error('Box scale must be between0.5 and2.')
 import numpy as np,pandas as pd,torch,deeplabcut as dlc
 import deeplabcut.pose_estimation_pytorch as pep
 from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
 torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
 records=json.loads(a.proposals.read_text());records=records['frames'] if isinstance(records,dict) else records
 if a.dataset:records=[r for r in records if r.get('dataset')==a.dataset]
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 live=LivePublisher.from_environment()
 if live:live.phase('tracking','Loading DeepLabCut pose model',len(records))
 cfg=PoseConfig.build_for_superanimal_inference('superanimal_topviewmouse',model_name='resnet_50',detector_name='fasterrcnn_mobilenet_v3_large_fpn',max_individuals=1,device=device)
 snapshot=Path(dlc.__file__).parent/'modelzoo/checkpoints/superanimal_topviewmouse_resnet_50.pt'
 cfg.to_yaml(out/'model_config.yaml',overwrite=True)
 cache=FrameCache()
 class ObservedVideoIterator(pep.VideoIterator):
  def __next__(self):
   index=self._index
   value=super().__next__()
   if live:
    image=value[0] if isinstance(value,tuple) else value
    x1,y1,x2,y2=live.context['crop']
    cache.remember(index,image[y1:y2,x1:x2].copy())
   return value
 it=ObservedVideoIterator(a.video) if live else pep.VideoIterator(a.video)
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
 # Include queued batches, the active batch and producer look-ahead. Retain only arena crops.
 cache.capacity=runner.batch_size*(runner.inference_cfg.multithreading.queue_length+3)
 started=time.time();print('Running DLC subject ROI on',device,'frames',len(records),flush=True)
 parts=list(cfg.metadata.bodyparts)
 def observe(index,prediction):
  image=cache.take(index)
  if image is None or not (live.due() or index==len(records)-1):return
  import cv2
  poses=np.asarray(prediction['bodyparts']);r=records[index];row={}
  for canonical,part in [('nose','nose'),('center','mouse_center'),('tail_base','tail_base')]:
   values=poses[0,parts.index(part)] if r.get('bbox_xyxy') is not None and len(poses)>0 else [np.nan]*3
   row.update({canonical+'_'+c:float(v) for c,v in zip(['x','y','likelihood'],values)})
  live.frame(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),index,r['source_time_s'],row=row,force=index==len(records)-1,image_is_crop=True)
 if live:
  tap=PredictionTap(observe)
  pep.video_inference(it,runner,shelf_writer=tap)
  predictions=tap.predictions
 else:predictions=pep.video_inference(it,runner)
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
 if live:print('Live preview frames published:',live.published_frames,flush=True)
 for part in ['nose','center']:print(part,'>=0.6',int((df[part+'_likelihood']>=.6).sum()),'/',len(df),flush=True)
 manifest={'status':'experimental_unvalidated','video':str(a.video.resolve()),'source_sha256':hashlib.file_digest(a.video.open('rb'),'sha256').hexdigest(),'proposal_file':str(a.proposals.resolve()),'proposal_sha256':hashlib.file_digest(a.proposals.open('rb'),'sha256').hexdigest(),'model':'SuperAnimal TopViewMouse ResNet50','model_sha256':hashlib.file_digest(snapshot.open('rb'),'sha256').hexdigest(),'deeplabcut_version':dlc.__version__,'method':'Foreground-conditioned DLC pose; anatomical landmarks are neural predictions, never foreground centroid or contour tips.','coordinate_space':'full_source_pixels','bbox_scale':a.bbox_scale,'device':device,'frames':len(df),'seconds':time.time()-started,'weights_finetuned':False,'accuracy_validated':False,'nose_above_cutoff':int((df.nose_likelihood>=.6).sum()),'center_above_cutoff':int((df.center_likelihood>=.6).sum())}
 (out/'selection_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',manifest['seconds'],flush=True)
if __name__=='__main__':main()
