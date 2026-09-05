"""Compare pretrained side-view pose with foreground-conditioned mouse boxes."""
from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k,v in {'MPLCONFIGDIR':'.cache/matplotlib','TORCH_HOME':'.cache/torch','HF_HOME':'.cache/huggingface','XDG_CACHE_HOME':'.cache'}.items():os.environ[k]=str(ROOT/v)
os.environ['DLC_LIGHT']='True'
import cv2,numpy as np,torch,deeplabcut as dlc
import deeplabcut.pose_estimation_pytorch as pep
from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
from stereotypy.cage import CageProposer
torch.set_num_threads(4)
out=ROOT/'outputs/stereotypy-pilot/pose';records=json.loads((out/'sample-manifest.json').read_text())
cap=cv2.VideoCapture(str(out/'sampled-cages.mp4'));contexts=[];images=[];proposals=[]
for r in records:
    ok,image=cap.read()
    if not ok:raise ValueError('Pilot frame missing')
    bg=cv2.imread(str(ROOT/'outputs/stereotypy-pilot'/r['video'][:3]/'background.jpg'),0)
    if bg is None:raise ValueError('Build both cage backgrounds first')
    # The trial video uses a top-left-aligned, padded cage crop.
    ch=round((r['crop_xyxy'][3]-r['crop_xyxy'][1])*r['scale_y']);cw=round((r['crop_xyxy'][2]-r['crop_xyxy'][0])*r['scale_x'])
    background=np.zeros(image.shape[:2],np.uint8);background[:ch,:cw]=cv2.resize(bg,(cw,ch))
    mapping=dict(crop_xyxy=[0,0,1280,560],floor_y=420)
    proposal=CageProposer(background,mapping).frame(image,0);proposals.append(proposal)
    if proposal['status']=='proposed':
        x1,y1,x2,y2=proposal['bbox_source'];pad=.18*max(x2-x1,y2-y1)
        x1=max(0,x1-pad);y1=max(0,y1-pad);x2=min(1280,x2+pad);y2=min(560,y2+pad)
        boxes=np.array([[x1,y1,x2-x1,y2-y1]],float)
    else:boxes=np.zeros((0,4),float)
    contexts.append(dict(bboxes=boxes,bbox_scores=np.ones(len(boxes))));images.append(image)
cap.release()
device='mps' if torch.backends.mps.is_available() else 'cpu'
cfg=PoseConfig.build_for_superanimal_inference('superanimal_quadruped',model_name='hrnet_w32',detector_name='fasterrcnn_resnet50_fpn_v2',max_individuals=1,device=device)
snapshot=Path(dlc.__file__).parent/'modelzoo/checkpoints/superanimal_quadruped_hrnet_w32.pt'
it=pep.VideoIterator(out/'sampled-cages.mp4');it.set_context(contexts)
runner=pep.get_pose_inference_runner(cfg,snapshot,batch_size=4,device=device,max_individuals=1)
predictions=pep.video_inference(it,runner)
parts=list(cfg.metadata.bodyparts);poses=np.asarray([p['bodyparts'] for p in predictions])
np.savez_compressed(out/'conditioned-poses.npz',poses=poses,bodyparts=parts)
stats=[]
writer=cv2.VideoWriter(str(out/'conditioned-review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2,(1280,560))
for i,(image,p,c) in enumerate(zip(images,poses,contexts)):
    valid=len(c['bboxes'])>0
    if valid:
        x,y,w,h=c['bboxes'][0];cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+h)),(90,220,170),2)
        for part,(x,y,q) in zip(parts,p[0]):
            if q>=.6 and np.isfinite([x,y,q]).all():cv2.circle(image,(round(x),round(y)),4,(0,190,255),-1)
    stats.append(dict(**records[i],body_proposed=valid,proposer_status=proposals[i]['status'],
        nose_above_cutoff=bool(valid and p[0,parts.index('nose'),2]>=.6),
        keypoints_above_cutoff=int((p[0,:,2]>=.6).sum()) if valid else 0))
    cv2.putText(image,f'Sampled frame {i} | EXPERIMENTAL - not behavior labels',(15,27),0,.6,(255,255,255),2);writer.write(image)
writer.release()
(out/'conditioned-assessment.json').write_text(json.dumps(dict(status='experimental_unvalidated',cutoff=.6,frames=stats,bodyparts=parts,model='SuperAnimal Quadruped HRNet-W32',weights_finetuned=False),indent=2))
print('Proposed',sum(r['body_proposed'] for r in stats),'nose >= .6',sum(r['nose_above_cutoff'] for r in stats),flush=True)
