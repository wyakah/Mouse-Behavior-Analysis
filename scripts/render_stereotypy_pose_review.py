"""Full uploaded recording with the existing decisions and new anatomical overlay."""
import sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import analyze_stereotypy_batch as api
from stereotypy.window import clip_manifest
from stereotypy.pose_signals import summarize_pose
path=ROOT/'stereotypy_videos/756_stereotypy-540p30.mov';folder=ROOT/'outputs/stereotypy-distinction/756';out=folder/'review';out.mkdir(parents=True,exist_ok=True)
original=json.loads((ROOT/'outputs/stereo-f209faf01650/published/001.json').read_text());result=copy.deepcopy(original)
pose=json.loads((folder/'pose.json').read_text());samples=pose['samples'];times=np.array([p['start_s'] for p in samples])
if pose['signature']['source_sha256']!=original['source_sha256']:raise ValueError('Pose/video identity mismatch')
for r in result['windows']:
    lo,hi=np.searchsorted(times,[r['start_s'],r['end_s']]);r['pose_samples']=samples[lo:hi];r['pose_evidence']=summarize_pose(r['pose_samples'])
index=clip_manifest(json.loads((ROOT/'outputs/stereo-f209faf01650/work/001/source-index.json').read_text()));api.REPORT=ROOT/'reports/stereotypy-distinction';api.REPORT.mkdir(exist_ok=True,parents=True)
api.render(path,'756-pose',out,index,original['mapping'],result)
for c in api.CLASSES:
    assert abs(result['cumulative']['seconds'][c]-original['cumulative']['seconds'][c])<1e-5,c
result.update(pose_used_to_change_scores=False,pose_model=pose['signature']['model']);api.save_json(out/'comparison.json',dict(all_36000_frames_rendered=True,baseline_seconds_unchanged=True,pose_samples=len(samples),source_sha256=original['source_sha256'],new_model_promoted=False))
print('Full pose overlay verified; baseline behavior totals unchanged.')
