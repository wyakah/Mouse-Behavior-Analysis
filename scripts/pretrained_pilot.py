from pathlib import Path
import os,time,json
import argparse
p=argparse.ArgumentParser();p.add_argument('--strong',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
for k,v in {'MPLCONFIGDIR':'.cache/matplotlib','TORCH_HOME':'.cache/torch','HF_HOME':'.cache/huggingface','XDG_CACHE_HOME':'.cache'}.items():os.environ[k]=str(root/v)
os.environ['DLC_LIGHT']='True'
import torch
import deeplabcut as dlc
torch.set_num_threads(4)
out=root/('outputs/pretrained-pilot-strong' if a.strong else 'outputs/pretrained-pilot');out.mkdir(exist_ok=True);device='mps' if torch.backends.mps.is_available() else 'cpu'
print('DLC',dlc.__version__,'device',device,flush=True);start=time.time()
dlc.video_inference_superanimal(videos=[str(root/'outputs/pretrained-pilot/pilot.mp4')],superanimal_name='superanimal_topviewmouse',model_name='hrnet_w32' if a.strong else 'resnet_50',detector_name='fasterrcnn_resnet50_fpn_v2' if a.strong else 'fasterrcnn_mobilenet_v3_large_fpn',dest_folder=out,cropping=[190,750,175,575],video_adapt=False,max_individuals=1,device=device,pcutoff=.6,create_labeled_video=True,batch_size=4,detector_batch_size=4)
(out/'runtime.json').write_text(json.dumps({'device':device,'seconds':time.time()-start,'model':'superanimal_topviewmouse/hrnet_w32' if a.strong else 'superanimal_topviewmouse/resnet_50','detector':'fasterrcnn_resnet50_fpn_v2' if a.strong else 'fasterrcnn_mobilenet_v3_large_fpn','purpose':'Preliminary visual quality assessment; no behavior scores'},indent=2))
print('Pilot complete',time.time()-start,flush=True)
