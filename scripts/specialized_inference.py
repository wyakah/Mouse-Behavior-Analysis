"""Three-chamber-specific multi-candidate DLC inference. Run in .dlc-env."""
from pathlib import Path
import os,argparse,json,time,sys,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k,v in {'MPLCONFIGDIR':'.cache/matplotlib','TORCH_HOME':'.cache/torch','HF_HOME':'.cache/huggingface','XDG_CACHE_HOME':'.cache'}.items():os.environ[k]=str(ROOT/v)
os.environ['DLC_LIGHT']='True'
p=argparse.ArgumentParser();p.add_argument('--video',default='outputs/pretrained-pilot/pilot.mp4');p.add_argument('--output',default='outputs/specialized-pilot');p.add_argument('--profile',default='profiles/ethovision_three_chamber.json');p.add_argument('--frame-map');p.add_argument('--detector-threshold',type=float);a=p.parse_args()
import cv2,numpy as np,pandas as pd,torch,deeplabcut as dlc
from threechamber.subject import SubjectSelector,validate_profile
from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
profile=json.loads((ROOT/a.profile).read_text());out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
if a.detector_threshold is not None:profile['detector_score_threshold']=a.detector_threshold
cap=cv2.VideoCapture(str(ROOT/a.video));crop=validate_profile(profile,int(cap.get(3)),int(cap.get(4)));cap.release()
if a.frame_map:mapping=json.loads((ROOT/a.frame_map).read_text())
elif (ROOT/a.video).resolve()==(ROOT/'outputs/pretrained-pilot/pilot.mp4').resolve():mapping=json.loads((ROOT/'outputs/pretrained-pilot/frame_map.json').read_text())
else:
 probe=json.loads(subprocess.check_output([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/source_timeline.py'),str(ROOT/a.video)],text=True,cwd=ROOT))
 mapping=[{'source':Path(a.video).name,'source_frame':i,'source_time_s':t} for i,t in enumerate(probe['timestamps'])]
(out/'frame_map.json').write_text(json.dumps(mapping,indent=2))
# Continuous clips use a source map prepared by the clip extractor; full videos retain their timestamps via postprocessing.
torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu';start=time.time()
cfg=PoseConfig.build_for_superanimal_inference('superanimal_topviewmouse',model_name='resnet_50',detector_name='fasterrcnn_mobilenet_v3_large_fpn',max_individuals=profile['max_candidates'],device=device)
cfg.detector.model['box_score_thresh']=profile['detector_score_threshold']
cfg.to_yaml(out/'model_config.yaml',overwrite=True)
print('Device',device,'profile',profile,flush=True)
results=dlc.video_inference_superanimal(videos=[str(ROOT/a.video)],superanimal_name='superanimal_topviewmouse',model_name='resnet_50',detector_name='fasterrcnn_mobilenet_v3_large_fpn',dest_folder=out,cropping=crop,video_adapt=False,max_individuals=profile['max_candidates'],device=device,create_labeled_video=False,batch_size=4,detector_batch_size=4,customized_model_config=str(out/'model_config.yaml'))
rawfile=next(out.glob('*before_adapt.json'));predictions=json.loads(rawfile.read_text())
if len(predictions)!=len(mapping):raise ValueError(f'Prediction/source frame mismatch: {len(predictions)} vs {len(mapping)}. No aligned tracks exported.')
bodyparts=list(cfg.metadata.bodyparts)
selector=SubjectSelector(profile);rows=[]
for i,pred in enumerate(predictions):
 src=mapping[i]['source'];t=mapping[i]['source_time_s'];source_frame=mapping[i]['source_frame']
 poses=np.asarray(pred['bodyparts']);choice=selector.select(poses,bodyparts,t,src)
 r={'frame':i,'source':src,'source_frame':source_frame,'source_time_s':t,**choice}
 for canonical,part in [('nose','nose'),('center','mouse_center'),('left_ear','left_ear'),('right_ear','right_ear'),('tail_base','tail_base')]:
  if choice['candidate_index'] is None:x=y=likelihood=np.nan
  else:
   x,y,likelihood=poses[choice['candidate_index'],bodyparts.index(part)]
   x+=profile['crop_xyxy'][0];y+=profile['crop_xyxy'][1]
  r.update({canonical+'_x':x,canonical+'_y':y,canonical+'_likelihood':likelihood})
 rows.append(r)
pd.DataFrame(rows).to_csv(out/'selected_tracks.csv',index=False)
(out/'selection_manifest.json').write_text(json.dumps({'status':'experimental_unvalidated','profile':profile,'bodyparts':bodyparts,'video':str(ROOT/a.video),'device':device,'seconds':time.time()-start,'frames':len(rows),'weights_finetuned':False,'source_sha256':hashlib.file_digest(open(ROOT/a.video,'rb'),'sha256').hexdigest(),'temporal_selection':'Only between adjacent source timestamps separated by <= 0.25s; resets across recordings and pilot stills.'},indent=2))
print('DONE',len(rows),'frames',time.time()-start,flush=True)
