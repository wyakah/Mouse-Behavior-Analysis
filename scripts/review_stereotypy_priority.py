"""Join experimental priority scores to source PTS for development review only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from stereotypy.training import save_json,sha256
from stereotypy.priority_training import CLASSES


def main():
    run=ROOT/'outputs/stereotypy-training/v2';catalog=json.loads((run/'feature-catalog.json').read_text())
    prediction=np.load(run/'predictions.npz',allow_pickle=False);selection=json.loads((run/'selection.json').read_text())
    sessions={r['animal_id']:r for r in json.loads((ROOT/'reports/stereotypy-pilot/sessions.json').read_text())};videos={}
    for row in catalog:
        if row['dataset']!='local':continue
        mouse=Path(row['path']).stem.split('_')[0];session=sessions[mouse]
        index=json.loads(Path(row['source_index']).read_text())
        if index['source_sha256']!=row['sha256']:raise ValueError('Source timestamps and model source differ')
        features=np.load(row['feature_path'],allow_pickle=False)
        scores=prediction['scores'][prediction['source_ids']==row['id']]
        if len(scores)!=len(features['times']):raise ValueError('Prediction/source sample mismatch')
        times=np.array([index['frames'][i]['start_s'] for i in features['frame_ids']])
        if not np.allclose(times,features['times'],rtol=0,atol=1e-7):raise ValueError('Nominal time cannot replace local source timestamps')
        ends=np.r_[times[1:],index['duration_s']]
        windows=[dict(start_s=float(t),end_s=float(end),scores={c:float(p[k]) for k,c in enumerate(CLASSES)}) for t,end,p in zip(times,ends,scores)]
        candidates={}
        for k,c in enumerate(CLASSES):
            chosen=[]
            rankings=[('high model score',np.argsort(-scores[:,k]),3),
                      ('near decision threshold',np.argsort(np.abs(scores[:,k]-selection['selected'][c]['threshold'])),2),
                      ('systematic check',np.argsort(np.abs(times-index['duration_s']*.65)),1)]
            for reason,ranking,quota in rankings:
                added=0
                for i in ranking:
                    t=float(times[i])
                    if any(a<=t<b for a,b in index['source_gaps']) or any(abs(v['start_s']-t)<8 for v in chosen):continue
                    chosen.append(dict(start_s=t,reason=reason,score=float(scores[i,k])));added+=1
                    if added==quota:break
            candidates[c]=chosen
        videos[mouse]=dict(video=session['base']+'/review.mp4',poster=session['base']+'/poster.jpg',
            reviewer='/stereotypy?session='+session['session_id'],duration_s=index['duration_s'],
            source_sha256=row['sha256'],source_gaps=index['source_gaps'],windows=windows,candidates=candidates,
            behavior_totals=None,scores_calibrated=False,status='unreviewed_model_suggestions',
            prior_unlabeled_development=True,local_accuracy=None)
    if set(videos)!={'745','729'}:raise ValueError('Both supplied mice need completed source-indexed features')
    save_json(run/'local-review.json',dict(videos=videos,selection_sha256=sha256(run/'selection.json'),production_ready=False))

if __name__=='__main__':main()
