"""Diagnostic control with the published CBAS DINOv2-base + LSTM classifier.

Compatible head adapted from CBAS v2-stable, Copyright (c) 2024 logjperry,
MIT license: third_party/cbas/LICENSE. This never promotes a model for scoring.
The shortest published test file was chosen by duration, not model performance.
"""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
import cv2,numpy as np,pandas as pd,torch
from transformers import AutoModel
from sklearn.metrics import average_precision_score
from stereotypy.priority_training import CLASSES,binary_metrics
from stereotypy.training import sha256,save_json
torch.set_num_threads(4)
BEHAVIORS=['eating','drinking','rearing','climbing','digging','nesting','resting','grooming','exploration']


class ReferenceHead(torch.nn.Module):
    def __init__(self):
        super().__init__();self.lin0=torch.nn.Linear(768,256);self.lin1=torch.nn.Linear(768,9);self.lin2=torch.nn.Linear(128,9)
        self.batch_norm=torch.nn.BatchNorm1d(768);self.lstm=torch.nn.LSTM(256,64,1,batch_first=True,bidirectional=True)
    def forward(self,x):
        x=self.batch_norm(x.transpose(1,2)).transpose(1,2)
        appearance=self.lin1(x)[:,10:21].mean(1)
        motion=self.lin0(x);motion=motion-motion.mean(1,keepdim=True)
        temporal=self.lin2(self.lstm(motion)[0][:,10:21].mean(1))
        return appearance+temporal


def main():
    started=time.monotonic();out=ROOT/'outputs/stereotypy-training/reference-control';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'research/stereotypy-round2/cbas-data/test/clip_3.mp4';annotations=source.with_name('clip_3_labels.csv')
    weights=ROOT/'research/stereotypy-round2/cbas-code/models_JonesLabModel_model.pth'
    recipe=dict(source_sha256=sha256(source),backbone='facebook/dinov2-base',revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415',
                input='Native 256x256 green channel divided by 255, replicated to RGB; no ImageNet normalization',
                precision='float32 Mac MPS; CLS stored float16 as in author pipeline')
    cache=out/'features.npz';device='mps' if torch.backends.mps.is_available() else 'cpu'
    if cache.exists() and json.loads(str(np.load(cache)['recipe']))!=recipe:raise ValueError('Reference cache recipe changed')
    if not cache.exists():
        model=AutoModel.from_pretrained(recipe['backbone'],revision=recipe['revision'],local_files_only=True).eval().to(device)
        cap=cv2.VideoCapture(str(source));pending=[];features=[];decoded=0
        def flush():
            if not pending:return
            x=torch.from_numpy(np.stack(pending)).to(device).float()[:,None].repeat(1,3,1,1)/255
            with torch.inference_mode():z=model(x).last_hidden_state[:,0].float().cpu().numpy()
            features.extend(z);pending.clear()
        while True:
            ok,im=cap.read()
            if not ok:break
            if im.shape[:2]!=(256,256):raise ValueError('Unexpected reference resolution')
            pending.append(im[:,:,1]);decoded+=1
            if len(pending)==16:flush()
            if decoded%512==0:print('Reference encoded',decoded,'frames',device,flush=True)
        flush();cap.release();del model
        if decoded!=len(pd.read_csv(annotations)):raise ValueError('Reference frame/annotation mismatch')
        np.savez_compressed(cache,x=np.array(features,np.float16),recipe=json.dumps(recipe))
    features=np.load(cache)['x'];head=ReferenceHead();head.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));head.eval()
    truth=pd.read_csv(annotations);results={};stored={}
    # Author inference uses a vector mean; its training converter uses a scalar.
    # Report both as specified code-path diagnostics, never choose using test scores.
    for centering in ['author_inference_vector','author_training_scalar']:
        axis=0 if centering=='author_inference_vector' else None
        x=(features-np.mean(features,axis=axis)).astype(np.float16).astype(np.float32);probabilities=[]
        for start in range(0,len(x),256):
            centers=np.clip(np.arange(start,min(start+256,len(x))),15,len(x)-16)
            batch=torch.from_numpy(x[centers[:,None]+np.arange(-15,16)])
            with torch.inference_mode():probabilities.extend(head(batch).softmax(1).numpy())
        p=np.array(probabilities);predicted=p.argmax(1);metrics={}
        for c in CLASSES:
            k=BEHAVIORS.index(c);y=truth[c].to_numpy();m=binary_metrics(y,(predicted==k).astype(float),.5)
            m['average_precision']=float(average_precision_score(y,p[:,k]));m['decision_rule']='Nine-class argmax; no tuned per-class threshold';metrics[c]=m
        results[centering]=metrics;stored[centering]=p;print(centering,json.dumps(metrics),flush=True)
    np.savez_compressed(out/'predictions.npz',**stored)
    save_json(out/'report.json',dict(status='single_file_diagnostic_not_local_validation',production_ready=False,
        selection='Shortest official CBAS test file, clip_3, selected by duration before this control. All native frames included.',
        source_recipe=recipe,annotation_sha256=sha256(annotations),head_sha256=sha256(weights),frames=len(features),
        implementation_sha256=sha256(Path(__file__)),wall_seconds=time.monotonic()-started,results=results,
        limitation='One file, not a new independent-mouse validation. Two existing author code paths are audited; neither is promoted based on these results.'))

if __name__=='__main__':main()
