"""External benchmark and explicitly sampled local comparison, without promotion."""
import sys,json,csv,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,joblib
from stereotypy.distinction import feature_vector
from stereotypy.pose_signals import summarize_pose
from stereotypy.exclusive import ExclusiveScorer
from stereotypy.training import save_json,CLASSES
OUT=ROOT/'outputs/stereotypy-distinction';REPORT=ROOT/'reports/stereotypy-distinction';REPORT.mkdir(parents=True,exist_ok=True)
model=joblib.load(OUT/'external/joint-action.joblib');evaluation=json.loads((OUT/'external/evaluation.json').read_text());subjects=[];csvrows=[]
for mid in ['710','711','712','743','745','756']:
    folder=OUT/mid;pose=json.loads((folder/'pose.json').read_text());clips=json.loads((folder/'local-clips.json').read_text())
    basepath=ROOT/('outputs/stereo-f209faf01650/published/001.json' if mid=='756' else f'outputs/stereotypy-five-video/{mid}/predictions.json');base=json.loads(basepath.read_text())
    if len({pose['signature']['source_sha256'],clips['source_sha256'],base['source_sha256']})!=1:raise ValueError('Input identity mismatch')
    if 'exclusive_intervals' in base:intervals=base['exclusive_intervals'];cumulative=base['cumulative']
    else:
        scorer=ExclusiveScorer(base['thresholds']);times=np.array([r['start_s'] for r in base['windows']]);tracks=list(csv.DictReader((ROOT/f'outputs/stereotypy-five-video/{mid}/tracking.csv').open()))
        for t in tracks:
            start=float(t['start_s']);end=float(t['end_s']);i=max(0,int(np.searchsorted(times,start,side='right')-1));scorer.add(start,end,base['windows'][i]['scores'],t['status']=='proposed')
        intervals=scorer.intervals;cumulative=scorer.snapshot()
    samples=pose['samples'];pt=np.array([r['start_s'] for r in samples]);results=[]
    for clip in clips['windows']:
        start,end=clip['start_s'],clip['end_s'];bw=[r for r in base['windows'] if r['start_s']>=start-1e-6 and r['start_s']<end-1e-6]
        if not bw:raise ValueError('No baseline window')
        b=np.mean([[r['scores']['grooming'],r['scores']['rearing']] for r in bw],axis=0)
        lo,hi=np.searchsorted(pt,[start+.5-1e-6,start+1.3+1e-6]);evidence=summarize_pose(samples[lo:hi]);labels={};probabilities={}
        for name,use_pose in [('image_motion',False),('pose_fusion',True)]:
            x=feature_vector(clip['scores'],evidence,b,use_pose)[None];labels[name]=str(model['models'][name].predict(x)[0]);probabilities[name]={str(c):float(v) for c,v in zip(model['models'][name].classes_,model['models'][name].predict_proba(x)[0])}
        durations={c:sum(max(0,min(end,r['end_s'])-max(start,r['start_s'])) for r in intervals if r['behavior']==c) for c in list(CLASSES)+['other','unknown']}
        dominant=max(durations,key=durations.get)
        result=dict(start_s=start,end_s=end,selection=clip['selection'],baseline=dominant,native=clip['native_label'],**labels,probabilities=probabilities,pose_evidence=evidence,baseline_seconds=durations)
        results.append(result);csvrows.append(dict(mouse_id=mid,start_s=start,end_s=end,selection=clip['selection'],baseline=dominant,native=clip['native_label'],**labels,human_reference=''))
    review=[dict(t=round(r['start_s'],4),points={k:[round(v[0],2),round(v[1],2)] for k,v in r['points'].items() if v is not None}) for i,r in enumerate(samples) if i%max(1,pose['signature']['pose_hz']//5)==0]
    summary=dict(mouse_id=mid,pose_hz=pose['signature']['pose_hz'],pose_samples=len(samples),nose_usable_percent=100*sum(r['points'].get('nose') is not None for r in samples)/len(samples),forepaw_motion_percent=100*sum(r['forepaw_speed'] is not None for r in samples)/len(samples),test_clips=len(results),sampled_seconds=sum(r['end_s']-r['start_s'] for r in results),changed_clips=sum(r['pose_fusion']!=r['baseline'] for r in results if r['baseline'] in ('grooming','rearing','other')),gnawing_to_grooming_clips=sum(r['baseline']=='gnawing_nonfood' and r['pose_fusion']=='grooming' for r in results),gnawing_to_rearing_clips=sum(r['baseline']=='gnawing_nonfood' and r['pose_fusion']=='rearing' for r in results))
    name=mid+'.json';save_json(REPORT/name,dict(summary=summary,source='stereotypy_videos/'+mid+'_stereotypy-540p30.mov',roi=pose['signature']['roi'],pose=review,clips=results,baseline_intervals=intervals,baseline_cumulative=cumulative));subjects.append(dict(id=mid,file=name,**summary))
with (REPORT/'comparison-clips.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
save_json(REPORT/'index.json',dict(evaluation=evaluation,subjects=subjects,model_promoted=False,local_accuracy=None,local_scope='18 two-second clips per mouse; only the first 20 minutes sampled. Full recording pose evidence, 5 Hz cache for five mice and new 10 Hz inference for756.',scoring_change='Existing six-behavior model retained. Anatomical evidence and overlays added.'))
shutil.copyfile(ROOT/'static/stereotypy-distinction-report.html',REPORT/'index.html')
print(json.dumps(subjects,indent=2))
