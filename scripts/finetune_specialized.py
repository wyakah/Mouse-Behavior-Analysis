"""Prepare a distinct DLC project from reviewed cropped labels; --train starts fine-tuning."""
from pathlib import Path
from datetime import datetime
import argparse,json,os,shutil,uuid
ROOT=Path(__file__).resolve().parents[1]
for k,v in {'MPLCONFIGDIR':'.cache/matplotlib','TORCH_HOME':'.cache/torch','HF_HOME':'.cache/huggingface','XDG_CACHE_HOME':'.cache'}.items():os.environ[k]=str(ROOT/v)
os.environ['DLC_LIGHT']='True'

def validate_export(manifest, split, annotations):
 """Validate the known human-review export contract before importing DLC or training."""
 import math
 expected_source='Explicitly reviewed annotations only; no automatic acceptance of model suggestions.'
 if manifest.get('label_source') != expected_source:
  raise ValueError('Unrecognized label source. Export explicitly reviewed labels from /labeling.')
 if not split or manifest.get('reviewed_frames') != len(split):
  raise ValueError('Export reviewed frame count does not match its split.')
 parts=manifest.get('parts', [])
 if len(parts)!=len(set(parts)) or set(parts) not in ({'nose','center','tail_base'},{'nose','left_ear','right_ear','center','tail_base'}):
  raise ValueError('Export must contain nose, center and tail base (or a complete legacy five-landmark export).')
 crop=manifest.get('profile',{}).get('crop_xyxy',[])
 if len(crop)!=4 or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in crop) or crop[0]>=crop[2] or crop[1]>=crop[3]:
  raise ValueError('Invalid exported crop.')
 seen_ids=set();seen_frames=set();sources=set();groups={'train':set(),'validation':set()}
 for e in split:
  source=e.get('source','');frame=e.get('source_frame');id=e.get('annotation_id')
  if not source or Path(source).name!=source or not isinstance(frame,int) or isinstance(frame,bool) or frame<0:
   raise ValueError('Invalid source frame in export.')
  subject=source.split('_')[0].split('.')[0]
  expected='validation' if subject=='685' else 'train' if subject in {'675','678'} else None
  if expected is None or e.get('split')!=expected:
   raise ValueError('Subject split violation: 675/678 train; 685 development validation.')
  if id in seen_ids or (source,frame) in seen_frames:
   raise ValueError('Duplicate exported annotation or source frame.')
  expected_index=['labeled-data',Path(source).stem,f'img{frame:06d}.png']
  if e.get('index')!=expected_index:raise ValueError('Label image index does not match source frame.')
  ann=annotations.get(id,{})
  if ann.get('reviewed') is not True or not str(ann.get('annotator','')).strip() or not ann.get('reviewed_at'):
   raise ValueError('Every exported frame needs explicit human review provenance.')
  if ann.get('source')!=source or ann.get('source_frame')!=frame:
   raise ValueError('Annotation provenance does not match split source frame.')
  if set(ann.get('points',{}))!=set(manifest['parts']):raise ValueError('Incomplete landmark review.')
  for point in ann['points'].values():
   if point is None:continue
   if not isinstance(point,list) or len(point)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in point):raise ValueError('Invalid reviewed coordinate.')
   if not crop[0]<=point[0]<crop[2] or not crop[1]<=point[1]<crop[3]:raise ValueError('Reviewed coordinate outside crop.')
  seen_ids.add(id);seen_frames.add((source,frame));sources.add(source);groups[expected].add(subject)
 if set(annotations)!=seen_ids or set(manifest.get('source_videos',[]))!=sources:
  raise ValueError('Export annotations or video inventory does not match split.')
 if not groups['train'] or not groups['validation'] or groups['train'] & groups['validation']:
  raise ValueError('Both disjoint training and development validation subjects are required.')
 return {'annotators':sorted({a['annotator'] for a in annotations.values()}),'reviewed_frames':len(split),'subjects':{k:sorted(v) for k,v in groups.items()},'independent_test_available':False}


def validate_label_values(merged, manifest, split, annotations):
 """Ensure authoritative DLC HDF points are precisely the exported human labels."""
 import numpy as np
 import pandas as pd
 expected_columns=pd.MultiIndex.from_product([['Researcher'],manifest['parts'],['x','y']],names=['scorer','bodyparts','coords'])
 if not merged.columns.equals(expected_columns):raise ValueError('DLC label columns do not match the reviewed export.')
 if merged.index.has_duplicates:raise ValueError('Duplicate DLC image labels.')
 expected={tuple(e['index']):e for e in split}
 if set(merged.index)!=set(expected):raise ValueError('DLC label images differ from review records.')
 x1,y1,_,_=manifest['profile']['crop_xyxy']
 for index,row in merged.iterrows():
  ann=annotations[expected[index]['annotation_id']];values=[]
  for part in manifest['parts']:
   pt=ann['points'][part];values.extend([np.nan,np.nan] if pt is None else [pt[0]-x1,pt[1]-y1])
  if not np.allclose(row.to_numpy(dtype=float),values,rtol=0,atol=1e-6,equal_nan=True):
   raise ValueError('DLC HDF coordinates differ from explicitly reviewed labels.')


def require_detector_snapshot(snapshots):
 if not snapshots or any(not s.path.is_file() for s in snapshots):
  raise RuntimeError('No trained detector snapshot. Refusing ground-truth-box fallback for end-to-end evaluation.')
 return snapshots[-1]


def evaluate_training(config, dlc, af, device):
 """Keep GT-box diagnostics distinct and require a real deployed detector snapshot."""
 from deeplabcut.pose_estimation_pytorch.data import DLCLoader
 from deeplabcut.pose_estimation_pytorch.task import Task
 from deeplabcut.pose_estimation_pytorch.apis.utils import get_model_snapshots
 loader=DLCLoader(config=config,shuffle=1,trainset_index=0)
 require_detector_snapshot(get_model_snapshots('all',loader.model_folder,Task.DETECT))
 snapshot=require_detector_snapshot(get_model_snapshots(-1,loader.model_folder,Task.DETECT))
 cfg=af.read_config(config);pose_config=Path(config).parent/'pose_only_diagnostic_config.yaml'
 cfg['detector_snapshotindex']=None;af.write_config(pose_config,cfg)
 dlc.evaluate_network(pose_config,shuffles=[1],plotting=True,engine=dlc.Engine.PYTORCH,device=device,per_keypoint_evaluation=True,detector_snapshot_index=None)
 dlc.evaluate_network(config,shuffles=[1],plotting=True,engine=dlc.Engine.PYTORCH,device=device,per_keypoint_evaluation=True,detector_snapshot_index=-1)
 return {'pose_only':'Ground-truth boxes; diagnostic only.','end_to_end':'Detector plus pose, development validation only.','detector_snapshot':str(snapshot.path),'accuracy_validated':False}


def write_status(project, status):
 from datetime import timezone
 status['updated_at']=datetime.now(timezone.utc).isoformat()
 tmp=project/'fine_tuning_status.tmp';tmp.write_text(json.dumps(status,indent=2,allow_nan=False));tmp.replace(project/'fine_tuning_status.json')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',required=True,type=Path);p.add_argument('--train',action='store_true');p.add_argument('--epochs',type=int,default=100);p.add_argument('--detector-epochs',type=int,default=30);a=p.parse_args()
 export=a.export.resolve()
 if not (export/'manifest.json').exists():p.error('Export manually reviewed labels from /labeling first.')
 manifest=json.loads((export/'manifest.json').read_text());split=json.loads((export/'split.json').read_text());annotations=json.loads((export/'annotations.json').read_text())
 try:provenance=validate_export(manifest,split,annotations)
 except ValueError as exc:p.error(str(exc))
 if a.epochs<1 or a.detector_epochs<0:p.error('Epoch counts must be positive (detector may be zero).')
 import numpy as np,pandas as pd,deeplabcut as dlc,torch
 from deeplabcut.utils import auxiliaryfunctions as af
 from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
 from deeplabcut.modelzoo import build_weight_init
 from deeplabcut.modelzoo.utils import create_conversion_table
 # Ignore wholly hidden poses for training; retain their reviewed originals in the export.
 frames=[]
 for file in sorted((export/'labeled-data').glob('*/CollectedData_Researcher.h5')):
  d=pd.read_hdf(file);frames.append(d)
 if not frames:p.error('No authoritative human-label HDF files in export.')
 merged=pd.concat(frames).sort_index()
 try:validate_label_values(merged,manifest,split,annotations)
 except ValueError as exc:p.error(str(exc))
 merged=merged.dropna(how='all')
 lookup={tuple(e['index']):e['split'] for e in split}
 train=[i for i,idx in enumerate(merged.index) if lookup[tuple(idx)]=='train'];test=[i for i,idx in enumerate(merged.index) if lookup[tuple(idx)]=='validation']
 if not train or not test:p.error('Each split needs visible landmarks; wholly hidden frames cannot train or evaluate pose.')
 # Enough to avoid a meaningless one-frame training run; validation is still required after training.
 if a.train and (len(train)<20 or len(test)<10):p.error('Review at least 20 training frames and 10 validation frames before training. More diverse labels will usually be needed.')
 videos=[ROOT/'prepared'/s for s in manifest['source_videos']]
 if any(not v.is_file() for v in videos):p.error('A prepared source video is missing.')
 project_name='ThreeChamberSpecialized'+datetime.now().strftime('%H%M%S')+uuid.uuid4().hex[:4]
 config=Path(dlc.create_new_project(project_name,'Researcher',[str(v) for v in videos],working_directory=str(ROOT/'dlc-projects'),copy_videos=False,multianimal=False))
 project=config.parent
 import hashlib
 hashes={str(f.relative_to(export)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [export/'manifest.json',export/'split.json',export/'annotations.json',*sorted((export/'labeled-data').glob('*/CollectedData_Researcher.h5'))]}
 status={'status':'preparing','export':str(export),'train_frames':len(train),'validation_frames':len(test),'memory_replay':False,'source_crop':manifest['profile'],'accuracy_validated':False,'review_provenance':provenance,'export_sha256':hashes}
 write_status(project,status)
 try:
  shutil.copytree(export/'labeled-data',project/'labeled-data',dirs_exist_ok=True)
  for f in (project/'labeled-data').glob('*/CollectedData_Researcher.h5'):
   d=pd.read_hdf(f).dropna(how='all');d.to_hdf(f,key='df_with_missing',mode='w');d.to_csv(f.with_suffix('.csv'))
  cfg=af.read_config(config);profile=manifest['profile'];x1,y1,x2,y2=profile['crop_xyxy']
  skeleton=[edge for edge in [['nose','left_ear'],['nose','right_ear'],['nose','center'],['center','tail_base']] if all(part in manifest['parts'] for part in edge)]
  cfg.update(bodyparts=manifest['parts'],skeleton=skeleton,pcutoff=.6,cropping=True,x1=x1,x2=x2,y1=y1,y2=y2,TrainingFraction=[round(len(train)/(len(train)+len(test)),2)],default_net_type='top_down_resnet_50',engine='pytorch')
  for entry in cfg['video_sets'].values():entry['crop']=f'{x1}, {x2}, {y1}, {y2}'
  af.write_config(config,cfg)
  mapping={part:('mouse_center' if part=='center' else part) for part in manifest['parts']}
  create_conversion_table(config,'superanimal_topviewmouse',mapping)
  cfg=af.read_config(config);folder=project/af.get_training_set_folder(cfg);folder.mkdir(parents=True,exist_ok=True)
  actual=merge_annotateddatasets(cfg,folder)
  if list(actual.index)!=list(merged.index):raise RuntimeError('DLC merged label order differs; refusing a potentially incorrect split.')
  weights=build_weight_init(config,'superanimal_topviewmouse','resnet_50','fasterrcnn_mobilenet_v3_large_fpn',with_decoder=True,memory_replay=False)
  dlc.create_training_dataset(config,Shuffles=[1],trainIndices=[np.asarray(train)],testIndices=[np.asarray(test)],net_type='top_down_resnet_50',detector_type='fasterrcnn_mobilenet_v3_large_fpn',weight_init=weights,engine=dlc.Engine.PYTORCH,userfeedback=False)
  shutil.copy2(export/'split.json',project/'source_split.json');shutil.copy2(export/'manifest.json',project/'label_export_manifest.json')
  status['status']='dataset_prepared';write_status(project,status)
  print('Prepared project:',config,flush=True)
  if a.train:
   device='mps' if torch.backends.mps.is_available() else 'cpu'
   status.update(status='training',device=device,pose_epochs=a.epochs,detector_epochs=a.detector_epochs);write_status(project,status)
   dlc.train_network(config,shuffle=1,engine=dlc.Engine.PYTORCH,device=device,epochs=a.epochs,detector_epochs=a.detector_epochs)
   status['status']='evaluating';write_status(project,status)
   status['evaluation']=evaluate_training(config,dlc,af,device)
   status['status']='training_and_development_evaluation_complete';write_status(project,status)
   print('Training and development evaluation finished. Inspect held-out nose error and cup-contact failures before using behavior totals.')
  else:print('No training started. Rerun this script with the reviewed export and --train for detector training and guarded end-to-end evaluation.')
 except Exception as exc:
  status.update(status='failed',failed_stage=status['status'],error=f'{type(exc).__name__}: {exc}');write_status(project,status)
  raise
if __name__=='__main__':main()
