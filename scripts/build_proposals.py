"""Publish real model suggestions and review reasons; never modify human annotations."""
from pathlib import Path
import json,hashlib,argparse
import cv2,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PARTS=['nose','left_ear','right_ear','center','tail_base']
def main():
 p=argparse.ArgumentParser();p.add_argument('--tracks',type=Path,action='append',required=True);p.add_argument('--add-hard-frames',type=int,default=0);a=p.parse_args()
 qpath=ROOT/'labeling/queue.json';queue=json.loads(qpath.read_text());bykey={(r['source'],r['source_frame']):r for r in queue};byid={r['id']:r for r in queue}
 proposals={};existing=ROOT/'labeling/proposals.json'
 if existing.exists():proposals=json.loads(existing.read_text())
 candidates=[]
 for path in a.tracks:
  d=pd.read_csv(path);manifest=path.parent/'selection_manifest.json';model=json.loads(manifest.read_text()).get('model','DLC candidate selector') if manifest.exists() else 'DLC'
  for _,r in d.iterrows():
   reasons=[]
   for part in ['nose','center']:
    if pd.isna(r[part+'_likelihood']):reasons.append(part+'_missing')
    elif r[part+'_likelihood']<.6:reasons.append(part+'_low_confidence')
   key=(r.source,int(r.source_frame));candidates.append((len(reasons),key,r,model,reasons))
 # Add diverse errors with at least2s source separation, never hundreds of adjacent duplicates.
 used={};added=0
 for priority,key,r,model,reasons in sorted(candidates,key=lambda x:-x[0]):
  if added>=a.add_hard_frames:break
  if not priority or key in bykey or any(abs(key[1]-i)<45 for i in used.get(key[0],[])):continue
  src=ROOT/'prepared'/key[0]
  if not src.exists():continue
  cap=cv2.VideoCapture(str(src));cap.set(cv2.CAP_PROP_POS_FRAMES,key[1]);ok,im=cap.read();cap.release()
  if not ok:continue
  id=hashlib.sha256(f'{key[0]}:{key[1]}'.encode()).hexdigest()[:16];image=ROOT/'labeling/images'/f'{id}.png';cv2.imwrite(str(image),im)
  e={'id':id,'source':key[0],'source_frame':key[1],'image':str(image.relative_to(ROOT)),'kind':'model_failure_review','width':im.shape[1],'height':im.shape[0]};queue.append(e);bykey[key]=e;used.setdefault(key[0],[]).append(key[1]);added+=1
 for priority,key,r,model,reasons in candidates:
  if key not in bykey:continue
  e=bykey[key];points={};likelihoods={}
  for part in PARTS:
   x,y,lik=r[part+'_x'],r[part+'_y'],r[part+'_likelihood'];points[part]=[float(x),float(y)] if np.isfinite([x,y,lik]).all() else None;likelihoods[part]=float(lik) if np.isfinite(lik) else None
  proposals[e['id']]={'points':points,'likelihoods':likelihoods,'source_model':model,'reviewed':False};e['review_priority']=priority;e['review_reasons']=reasons
 tmp=qpath.with_suffix('.tmp');tmp.write_text(json.dumps(queue,indent=2));tmp.replace(qpath)
 tmp=existing.with_suffix('.tmp');tmp.write_text(json.dumps(proposals,indent=2,allow_nan=False));tmp.replace(existing)
 print(len(proposals),'model suggestions;',added,'new difficult frames;',len(queue),'total frames. Human annotations untouched.')
if __name__=='__main__':main()
