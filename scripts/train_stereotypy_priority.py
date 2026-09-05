"""Compare cached DINO classifiers on validation, then explicitly unlock test once.

The test report is deliberately a separate command. All hyperparameters and
thresholds are fixed in selection.json before any test metrics are calculated.
"""
import argparse, copy, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import joblib
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
from stereotypy.priority_training import CLASSES,validate_catalog,context_indices,binary_metrics,select_threshold
from stereotypy.training import sha256,save_json
torch.set_num_threads(4)
SEED=20260905


def load_data(run):
    rows=validate_catalog(json.loads((run/'feature-catalog.json').read_text()))
    xs=[];ys=[];splits=[];datasets=[];groups=[];source_ids=[];contexts=[];offset=0
    for row in rows:
        data=np.load(row['feature_path'],allow_pickle=False)
        motion_path=Path(row['feature_path']).with_name(Path(row['feature_path']).stem+'-motion.npz')
        motion=np.load(motion_path,allow_pickle=False)
        if str(motion['feature_sha256'])!=sha256(row['feature_path']) or not np.array_equal(motion['frame_ids'],data['frame_ids']):raise ValueError('Motion cache does not match visual frames')
        x=np.c_[data['x'].astype(np.float32),data['geometry'],motion['x']]
        y=data['labels'];n=len(x)
        if y.shape!=(n,3) or not np.isin(y,[-1,0,1]).all():raise ValueError('Invalid labels')
        if row['split']=='development' and (y!=-1).any():raise ValueError('Local labels must stay unknown')
        xs.append(x);ys.append(y);splits.extend([row['split']]*n);datasets.extend([row['dataset']]*n)
        groups.extend([row['group']]*n);source_ids.extend([row['id']]*n)
        contexts.append(context_indices(data['cuts'])+offset);offset+=n
    return rows,np.concatenate(xs),np.concatenate(ys),np.array(splits),np.array(datasets),np.array(groups),np.array(source_ids),np.concatenate(contexts)


def make_model(dim):
    class Temporal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project=torch.nn.Sequential(torch.nn.Linear(dim,96),torch.nn.GELU())
            self.conv=torch.nn.Sequential(torch.nn.Conv1d(96,96,3,padding=1),torch.nn.GELU(),torch.nn.Dropout(.25),
                                          torch.nn.Conv1d(96,96,3,padding=1),torch.nn.GELU())
            self.head=torch.nn.Linear(192,3)
        def forward(self,x):
            z=self.project(x);t=self.conv(z.transpose(1,2))
            return self.head(torch.cat([z[:,3],t.mean(2)],1))
    return Temporal()


def validation_choices(y,scores,datasets):
    return {c:select_threshold(y[:,k],scores[:,k],datasets) for k,c in enumerate(CLASSES)}


def train(run,epochs):
    started=time.monotonic()
    if (run/'test-report.json').exists():raise ValueError('Test has been opened. Use a new protocol/run for further tuning; disclose reuse.')
    completion=json.loads((run/'extraction-complete.json').read_text())
    if completion['catalog_sha256']!=sha256(run/'feature-catalog.json'):raise ValueError('Extraction incomplete or changed')
    motion_completion=json.loads((run/'motion-complete.json').read_text())
    if motion_completion['catalog_sha256']!=completion['catalog_sha256']:raise ValueError('Motion extraction incomplete or changed')
    rows,x,y,splits,datasets,groups,source_ids,contexts=load_data(run)
    required={'cbas-train-'+str(i) for i in range(16)}|{'cbas-test-'+str(i) for i in range(4)}
    if not required.issubset({r['id'] for r in rows}):raise ValueError('CBAS feature extraction is incomplete')
    train_ids=np.flatnonzero(splits=='train');val_ids=np.flatnonzero(splits=='validation')
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    fit_ids=np.concatenate([rng.choice(train_ids[datasets[train_ids]==d],min(4000,int((datasets[train_ids]==d).sum())),replace=False) for d in sorted(set(datasets[train_ids]))])
    scaler=StandardScaler().fit(x[fit_ids]);scaled=scaler.transform(x)
    pca=PCA(n_components=128,whiten=True,svd_solver='randomized',random_state=SEED).fit(scaled[fit_ids,:775])
    motion_pca=PCA(n_components=32,whiten=True,svd_solver='randomized',random_state=SEED).fit(scaled[fit_ids,775:])
    za=np.clip(pca.transform(scaled[:,:775]),-12,12).astype(np.float32)
    zm=np.clip(motion_pca.transform(scaled[:,775:]),-12,12).astype(np.float32)
    z=np.c_[za,zm];sequence=z[contexts].astype(np.float16).astype(np.float32)
    def descriptors(current,seq):return np.c_[current,seq.mean(1),seq.std(1),seq[:,-1]-seq[:,0]].astype(np.float16).astype(np.float32)
    static_arrays={'appearance':descriptors(za,sequence[:,:,:128]),'motion':descriptors(z,sequence)}
    np.savez_compressed(run/'prepared-features.npz',sequence=sequence.astype(np.float16),
                        static_appearance=static_arrays['appearance'].astype(np.float16),static_motion=static_arrays['motion'].astype(np.float16),
                        labels=y,splits=splits,datasets=datasets,groups=groups,source_ids=source_ids)
    joblib.dump(dict(scaler=scaler,pca=pca,motion_pca=motion_pca),run/'preprocessing.joblib')
    candidates={};validation_predictions={}
    def record(name,scores):
        choices=validation_choices(y[val_ids],scores,datasets[val_ids]);candidates[name]=choices;validation_predictions[name]=scores
        save_json(run/'validation-experiments.json',candidates)
        print(name,{c:round(v['selection_score'],3) for c,v in choices.items()},flush=True)
    for name,kwargs in [('logistic-appearance',dict(C=.1)),('logistic-motion',dict(C=.1)),
                        ('trees-appearance',dict(min_samples_leaf=2)),('trees-motion',dict(min_samples_leaf=2))]:
        static=static_arrays[name.split('-')[-1]]
        models=[];scores=np.zeros((len(val_ids),3),np.float32)
        for k,c in enumerate(CLASSES):
            ids=train_ids[y[train_ids,k]>=0];target=y[ids,k]
            # Equal dataset contribution, then balance positive/negative within each dataset.
            weights=np.ones(len(ids))
            for d in set(datasets[ids]):
                for label in [0,1]:
                    mask=(datasets[ids]==d)&(target==label)
                    if mask.any():weights[mask]=len(ids)/(len(set(datasets[ids]))*2*mask.sum())
            if name.startswith('logistic'):model=LogisticRegression(max_iter=1000,random_state=SEED,**kwargs)
            else:model=ExtraTreesClassifier(n_estimators=320,max_features='sqrt',n_jobs=4,random_state=SEED,**kwargs)
            model.fit(static[ids],target,sample_weight=weights);scores[:,k]=model.predict_proba(static[val_ids])[:,1];models.append(model)
        joblib.dump(models,run/(name+'.joblib'),compress=3);record(name,scores)
    model=make_model(sequence.shape[-1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.03)
    tx=torch.from_numpy(sequence);ty=torch.from_numpy(y.astype(np.float32));weight=np.zeros_like(y,dtype=np.float32)
    for k in range(3):
        known=train_ids[y[train_ids,k]>=0];ds=set(datasets[known])
        for d in ds:
            for label in [0,1]:
                ids=known[(datasets[known]==d)&(y[known,k]==label)]
                if len(ids):weight[ids,k]=len(train_ids)/(len(ds)*2*len(ids))
    tw=torch.from_numpy(weight);best_score=-1;history=[]
    for epoch in range(epochs):
        model.train();losses=[]
        order=rng.permutation(train_ids)
        for start in range(0,len(order),128):
            ids=order[start:start+128]
            logits=model(tx[ids]+torch.randn_like(tx[ids])*.025)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,ty[ids].clamp(min=0),reduction='none')*tw[ids]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():scores=np.concatenate([model(tx[i]).sigmoid().numpy() for i in np.array_split(val_ids,max(1,len(val_ids)//256))])
        choices=validation_choices(y[val_ids],scores,datasets[val_ids]);score=np.mean([v['selection_score'] for v in choices.values()])
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_macro_f1=float(score)))
        if score>best_score:best_score=score;best=copy.deepcopy(model.state_dict());best_epoch=epoch+1;best_predictions=scores.copy()
        if epoch%5==0:print('Temporal epoch',epoch+1,'validation macro F1',round(score,3),flush=True)
        if epoch+1>=20 and epoch+1-best_epoch>=10:
            print('Early stop: no validation improvement for ten epochs',flush=True);break
    torch.save(dict(state_dict=best,input_dim=sequence.shape[-1],epoch=best_epoch,production_ready=False),run/'temporal.pt')
    record('temporal',best_predictions);save_json(run/'temporal-history.json',history)
    selected={c:dict(model=max(candidates,key=lambda name:candidates[name][c]['selection_score']),**max((v[c] for v in candidates.values()),key=lambda v:v['selection_score'])) for c in CLASSES}
    selection=dict(classes=list(CLASSES),selected=selected,production_ready=False,local_accuracy=None,
        method='Model and threshold maximize per-class mean validation F1 across supported camera datasets; no test selection.',
        normalization_fit_policy='Seeded up-to-4000 training frames per dataset; all validation/test/local frames excluded',
        candidates=list(candidates),seed=SEED,training_wall_seconds=time.monotonic()-started,
        backbone=dict(model='facebook/dinov2-small',revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',frozen=True,
            weights_sha256=sha256(ROOT/'.cache/huggingface/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors')),
        legacy_manifest_sha256=sha256(ROOT/'outputs/stereotypy-training/v1/dataset.json'),feature_catalog_sha256=sha256(run/'feature-catalog.json'),
        implementation_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'stereotypy/priority_training.py',ROOT/'scripts/stereotypy_priority_features.py',ROOT/'scripts/stereotypy_motion_features.py']},
        source_counts={s:sum(r['split']==s for r in rows) for s in set(splits)},
        support={c:{s:dict(positive=int(((y[:,k]==1)&(splits==s)).sum()),negative=int(((y[:,k]==0)&(splits==s)).sum())) for s in set(splits)} for k,c in enumerate(CLASSES)})
    selection['artifact_sha256']={p.name:sha256(p) for p in [run/'preprocessing.joblib',run/'prepared-features.npz',run/'temporal.pt',*[run/(name+'.joblib') for name in candidates if name!='temporal']]}
    save_json(run/'selection.json',selection)
    print('Selection locked',json.dumps(selected),flush=True)


def evaluate(run):
    selection=json.loads((run/'selection.json').read_text())
    if sha256(run/'feature-catalog.json')!=selection['feature_catalog_sha256']:raise ValueError('Feature catalog changed after selection')
    for name,fingerprint in selection['artifact_sha256'].items():
        if sha256(run/name)!=fingerprint:raise ValueError('Selected model or features changed after selection')
    data=np.load(run/'prepared-features.npz',allow_pickle=False);scores=np.zeros((len(data['labels']),3),np.float32)
    model_scores={}
    for name in {v['model'] for v in selection['selected'].values()}:
        if name=='binary-temporal':
            from stereotypy.priority_models import make_binary_model
            checkpoint=torch.load(run/'binary-temporal.pt',map_location='cpu',weights_only=True);columns=[]
            for c in CLASSES:
                model=make_binary_model(checkpoint['input_dim']);model.load_state_dict(checkpoint['models'][c]['state_dict']);model.eval()
                with torch.inference_mode():column=np.concatenate([model(torch.from_numpy(x.astype(np.float32))).sigmoid().numpy() for x in np.array_split(data['sequence'],max(1,len(scores)//256))])
                columns.append(column)
            p=np.stack(columns,axis=1)
        elif name=='temporal':
            checkpoint=torch.load(run/'temporal.pt',map_location='cpu',weights_only=True);model=make_model(checkpoint['input_dim']);model.load_state_dict(checkpoint['state_dict']);model.eval()
            with torch.inference_mode():p=np.concatenate([model(torch.from_numpy(x.astype(np.float32))).sigmoid().numpy() for x in np.array_split(data['sequence'],max(1,len(scores)//256))])
        else:
            models=joblib.load(run/(name+'.joblib'));static=data['static_'+name.split('-')[-1]].astype(np.float32);p=np.stack([m.predict_proba(static)[:,1] for m in models],1)
        model_scores[name]=p
    for k,c in enumerate(CLASSES):scores[:,k]=model_scores[selection['selected'][c]['model']][:,k]
    report=dict(status='experimental_not_for_scored_totals',production_ready=False,local_accuracy=None,
        selection_sha256=sha256(run/'selection.json'),metrics_unit='Sampled frames; 2 Hz CBAS, four frames per selected MIT window. Frames within a video are correlated.',
        independence_limit=json.loads((run/'protocol.json').read_text())['independence_limit'],
        tests={},by_source={})
    for split in ['test','legacy_test','validation']:
        for d in sorted(set(data['datasets'][data['splits']==split])):
            mask=(data['splits']==split)&(data['datasets']==d)
            report['tests'][d+'-'+split]={c:binary_metrics(data['labels'][mask,k],scores[mask,k],selection['selected'][c]['threshold']) for k,c in enumerate(CLASSES)}
    for source in sorted(set(data['source_ids'][data['splits']=='test'])):
        mask=data['source_ids']==source
        report['by_source'][source]={c:binary_metrics(data['labels'][mask,k],scores[mask,k],selection['selected'][c]['threshold']) for k,c in enumerate(CLASSES)}
    # Compare the same MIT two-second windows used by v1, without retuning.
    window_labels=[];window_scores=[]
    for row in json.loads((run/'feature-catalog.json').read_text()):
        if row['dataset']!='mit' or row['split']!='legacy_test':continue
        features=np.load(row['feature_path'],allow_pickle=False);source_scores=scores[data['source_ids']==row['id']]
        for window in row['windows']:
            requested=np.array([int(t*float(features['fps'])) for t in np.linspace(window['start_s'],window['end_s'],4,endpoint=False)])
            positions=np.searchsorted(features['frame_ids'],requested)
            if not np.array_equal(features['frame_ids'][positions],requested):raise ValueError('MIT comparison window alignment changed')
            window_labels.append([window['labels'][0],-1,window['labels'][3]])
            window_scores.append(source_scores[positions].mean(0))
    if window_labels:
        wy,wp=np.array(window_labels),np.array(window_scores)
        report['tests']['mit-legacy_windows']={c:binary_metrics(wy[:,k],wp[:,k],selection['selected'][c]['threshold']) for k,c in enumerate(CLASSES)}
        report['legacy_window_note']='Mean of four sampled-frame scores per original v1 two-second window, using already selected frame-validation thresholds. Previously observed test; no threshold retuning.'
    np.savez_compressed(run/'predictions.npz',scores=scores,labels=data['labels'],splits=data['splits'],datasets=data['datasets'],source_ids=data['source_ids'])
    save_json(run/'test-report.json',report);print(json.dumps(report['tests'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=ROOT/'outputs/stereotypy-training/v2');parser.add_argument('--epochs',type=int,default=40);parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    (evaluate if args.evaluate else lambda run:train(run,args.epochs))(args.run)

if __name__=='__main__':main()
