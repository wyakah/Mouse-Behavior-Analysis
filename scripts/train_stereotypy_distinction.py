"""Fit two supported-action models, select on validation, then report legacy test.

No local pseudo-labels and no claim to have trained nonfood gnawing.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from stereotypy.distinction import *
from stereotypy.training import save_json,sha256
from train_stereotypy_priority import make_model
from stereotypy.priority_models import make_binary_model
OUT=ROOT/'outputs/stereotypy-distinction/external';V2=ROOT/'outputs/stereotypy-training/v2'
torch.set_num_threads(4)

def baseline_scores(rows):
    data=np.load(V2/'prepared-features.npz');selection=json.loads((V2/'selection.json').read_text());ids=np.flatnonzero(data['datasets']=='mit');seq=data['sequence'][ids];p=[]
    for k,c in [(0,'grooming'),(2,'rearing')]:
        if selection['selected'][c]['model']=='temporal':
            state=torch.load(V2/'temporal.pt',weights_only=True,map_location='cpu');model=make_model(state['input_dim']);model.load_state_dict(state['state_dict']);column=k
        else:
            state=torch.load(V2/'binary-temporal.pt',weights_only=True,map_location='cpu');model=make_binary_model(state['input_dim']);model.load_state_dict(state['models'][c]['state_dict']);column=None
        model.eval();result=[]
        for batch in np.array_split(seq,max(1,len(seq)//256)):
            with torch.inference_mode():v=model(torch.from_numpy(batch.astype(np.float32))).sigmoid().numpy()
            result.extend(v[:,column] if column is not None else v)
        p.append(result)
    probabilities=np.stack(p,1);source_ids=data['source_ids'][ids];catalog=json.loads((V2/'feature-catalog.json').read_text());by={}
    for r in catalog:
        if r['dataset']=='mit':
            features=np.load(r['feature_path']);by[r['id']]=(features['times'],probabilities[source_ids==r['id']])
    references=json.loads((ROOT/'outputs/stereotypy-training/v1/dataset.json').read_text());lookup={r['id']:r for r in references['windows']};result=[]
    for r in rows:
        ref=lookup[r['id']];times,ps=by[ref['source'].replace(':','-')];selected=ps[(times>=ref['start_s']-1e-6)&(times<ref['end_s']-1e-6)]
        if not len(selected):raise ValueError('No aligned baseline frames')
        result.append(selected.mean(0))
    return np.array(result)

def main():
    if (OUT/'evaluation.json').exists():raise ValueError('Test already opened; do not tune against it')
    variants={name:json.loads((OUT/file).read_text()) for name,file in [('dark_inverted','labgym.json'),('original_image','labgym-original.json')]}
    polarity_validation={}
    for name,doc in variants.items():
        vr=[r for r in doc['rows'] if r['split']=='validation']
        polarity_validation[name]=metrics([r['label'] for r in vr],[native_label(r['classes'][int(np.argmax(r['scores']))]) if r['scores'] is not None else 'unknown' for r in vr])
    selected_polarity=max(polarity_validation,key=lambda n:polarity_validation[n]['priority_macro_f1'])
    native=variants[selected_polarity];pose=json.loads((OUT/'pose.json').read_text());pmap={r['id']:r for r in pose['rows']};rows=native['rows'];base=baseline_scores(rows);y=np.array([r['label'] for r in rows]);split=np.array([r['split'] for r in rows]);train=split=='train';val=split=='validation';test=split=='test'
    groups=[{r['group'] for r in rows if r['split']==s} for s in ['train','validation','test']]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Source date groups overlap')
    models={};features={};selection={}
    for use_pose,name in [(False,'image_motion'),(True,'pose_fusion')]:
        x=np.stack([feature_vector(r['scores'],pmap[r['id']]['pose'],b,use_pose) for r,b in zip(rows,base)]);features[name]=x;options=[]
        for C in [.01,.1,1.]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=C,class_weight='balanced',max_iter=2000,random_state=42));model.fit(x[train],y[train]);m=metrics(y[val],model.predict(x[val]));options.append((m['priority_macro_f1'],C,model,m))
        best=max(options,key=lambda r:(r[0],-r[1]));models[name]=best[2];selection[name]=dict(C=best[1],validation=best[3])
    # This selection file is written BEFORE any legacy-test metric is computed.
    save_json(OUT/'selection.json',dict(polarity_validation=polarity_validation,selected_polarity=selected_polarity,models=selection,protocol_sha256=sha256(OUT/'protocol.json'),selection_metric='validation priority macro F1',local_reference_used=False))
    joblib.dump(dict(models=models,classes=LABELS,selected_polarity=selected_polarity,labgym_classes=rows[0]['classes'],production_ready=False),OUT/'joint-action.joblib')
    base_labels=[]
    for g,r in base:
        eligible=[(g,'grooming')] if g>=.5 else []
        if r>=.77:eligible.append((r,'rearing'))
        base_labels.append(max(eligible)[1] if eligible else 'other')
    native_labels=[native_label(r['classes'][int(np.argmax(r['scores']))]) if r['scores'] is not None else 'unknown' for r in rows]
    reports={name:metrics(y[test],model.predict(features[name][test])) for name,model in models.items()}
    reports['baseline']=metrics(y[test],np.array(base_labels)[test]);reports['native_labgym']=metrics(y[test],np.array(native_labels)[test])
    save_json(OUT/'evaluation.json',dict(models=reports,selection=selection,selected_polarity=selected_polarity,polarity_validation=polarity_validation,training_windows=int(train.sum()),validation_windows=int(val.sum()),test_windows=int(test.sum()),test_scope='Previously inspected MIT test windows, original date groups preserved; no local accuracy claim',trained_behaviors=['grooming','rearing','other'],unsupported=['digging','gnawing_nonfood','jumping','circling'],production_ready=False,local_accuracy=None))
    save_json(OUT/'window-comparison.json',[dict(id=r['id'],label=r['label'],split=r['split'],baseline=list(map(float,b)),native=n,**{name:str(model.predict(features[name][i:i+1])[0]) for name,model in models.items()}) for i,(r,b,n) in enumerate(zip(rows,base,native_labels))])
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
