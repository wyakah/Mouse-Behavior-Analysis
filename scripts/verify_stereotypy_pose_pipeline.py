"""Real 12-second integration check with live segments and the deployed scorers."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import analyze_stereotypy_batch as api
from stereotypy.window import clip_manifest
path=ROOT/'stereotypy_videos/756_stereotypy-540p30.mov';out=ROOT/'outputs/stereotypy-distinction/integration-smoke';out.mkdir(parents=True,exist_ok=True)
index=clip_manifest(json.loads((ROOT/'outputs/stereo-f209faf01650/work/001/source-index.json').read_text()))
index['frames']=[r for r in index['frames'] if r['start_s']<12];index['frame_count']=len(index['frames']);index['duration_s']=12;index['source_gaps']=[g for g in index['source_gaps'] if g[0]<12]
mapping=api.validate_mapping(dict(crop_xyxy=[0,0,index['width'],index['height']],floor_y=index['height']*.8),index['width'],index['height'])
api.REPORT=out;result,stream=api.incremental_result(path,out,index,mapping,'mps' if api.torch.backends.mps.is_available() else 'cpu')
api.render(path,'756',out,index,mapping,result,stream_folder=out/'live',prediction_stream=stream)
assert len(result['windows'])==24
assert sum(len(r.get('pose_samples',[])) for r in result['windows'])==120
assert result['cumulative']['through_s']==12
print('integration verified: 360 frames, 120 pose samples, 24 score windows, 12 seconds')
