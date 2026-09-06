"""Recompute temporal evidence from verified existing raw pose, without re-inference."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from stereotypy.pose_signals import PoseSignals
from stereotypy.training import save_json,sha256
for mid in ['710','711','712','743','745']:
    old=ROOT/'outputs/stereotypy-robustness'/mid;meta=json.loads((old/'manifest.json').read_text());data=np.load(old/'pose.npz');records=json.loads(str(data['records']));a,b,c,d=meta['signature']['mapping']['crop_xyxy'];path=next((ROOT/'stereotypy_videos').glob(mid+'_*.mov'))
    if sha256(path)!=meta['signature']['source_sha256']:raise ValueError('Cached pose source changed')
    signal=PoseSignals();rows=[signal.add(r['start_s'],p,list(data['parts']),(d-b,c-a,3)) for r,p in zip(records,data['poses'])]
    out=ROOT/'outputs/stereotypy-distinction'/mid;out.mkdir(parents=True,exist_ok=True)
    save_json(out/'pose.json',dict(signature=dict(source_sha256=sha256(path),duration_s=1200,roi=[a,b,c,d],model=meta['signature']['pose_model'],pose_hz=5,implementation_sha256=sha256(ROOT/'stereotypy/pose_signals.py'),raw_pose_sha256=sha256(old/'pose.npz')),mouse_id=mid,samples=rows,decoded_frames=meta['decoded_frames'],wall_seconds=None,accuracy=None,training_performed=False,reused_raw_pose=True,method='New temporal evidence from unchanged 5 Hz pose cache; no new anatomical inference'))
    print(mid,len(rows))
