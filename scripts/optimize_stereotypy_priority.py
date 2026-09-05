"""Compare regularized per-behavior heads before opening the external test.

Each class selects its own epoch. Validation-only revisions are recorded;
optimization refuses to run once a test report exists.
"""
import copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from stereotypy.priority_training import CLASSES,select_threshold
from stereotypy.priority_models import make_binary_model
from stereotypy.training import sha256,save_json
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4);torch.set_num_threads(4)


def main():
    run=ROOT/'outputs/stereotypy-training/v2'
    if (run/'test-report.json').exists():raise ValueError('Test already opened; no further tuning in this run')
    if (run/'selection-baseline.json').exists():raise ValueError('This optimization pass is already recorded')
    selection=json.loads((run/'selection.json').read_text())
    for name,value in selection['artifact_sha256'].items():
        if sha256(run/name)!=value:raise ValueError('Baseline artifacts changed')
    started=time.monotonic();data=np.load(run/'prepared-features.npz',allow_pickle=False)
    x=torch.from_numpy(data['sequence'].astype(np.float32));y=data['labels'];splits=data['splits'];datasets=data['datasets']
    val=np.flatnonzero(splits=='validation');rng=np.random.default_rng(20260906)
    states={};choices={};histories={};predictions=np.zeros((len(val),3),np.float32)
    for k,c in enumerate(CLASSES):
        torch.manual_seed(20260906+k);train=np.flatnonzero((splits=='train')&(y[:,k]>=0));labels=torch.tensor(y[:,k],dtype=torch.float32)
        weights=np.zeros(len(y),np.float32);ds=set(datasets[train])
        for d in ds:
            for label in [0,1]:
                ids=train[(datasets[train]==d)&(y[train,k]==label)]
                if len(ids):weights[ids]=len(train)/(len(ds)*2*len(ids))
        weights=torch.from_numpy(weights);model=make_binary_model(x.shape[-1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
        best_score=-1;history=[]
        for epoch in range(30):
            model.train();order=rng.permutation(train);losses=[]
            for offset in range(0,len(order),128):
                ids=order[offset:offset+128];batch=x[ids]+torch.randn_like(x[ids])*.1
                batch=batch*(torch.rand((len(ids),1,batch.shape[-1]))>.05)
                logits=model(batch);loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels[ids],reduction='none')*weights[ids]).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():p=np.concatenate([model(x[ids]).sigmoid().numpy() for ids in np.array_split(val,max(1,len(val)//256))])
            choice=select_threshold(y[val,k],p,datasets[val]);value=choice['selection_score'];history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_f1=value))
            if value>best_score:best_score=value;best_epoch=epoch+1;best=copy.deepcopy(model.state_dict());best_choice=choice;best_p=p.copy()
            if epoch%5==0:print(c,'epoch',epoch+1,'validation F1',round(value,3),flush=True)
            if epoch+1>=12 and epoch+1-best_epoch>=7:break
        states[c]=dict(state_dict=best,epoch=best_epoch);choices[c]=best_choice;histories[c]=history;predictions[:,k]=best_p
        print('Best',c,best_epoch,round(best_score,3),flush=True)
    if (run/'test-report.json').exists():raise ValueError('Test was opened during optimization; refusing to revise selection')
    checkpoint=run/'binary-temporal.pt';torch.save(dict(classes=list(CLASSES),models=states,input_dim=x.shape[-1],production_ready=False),checkpoint)
    experiments=json.loads((run/'validation-experiments.json').read_text());experiments['binary-temporal']=choices
    save_json(run/'selection-baseline.json',selection);parent_hash=sha256(run/'selection.json')
    selection['selected']={c:dict(model=max(experiments,key=lambda n:experiments[n][c]['selection_score']),**max((v[c] for v in experiments.values()),key=lambda v:v['selection_score'])) for c in CLASSES}
    selection['candidates']=list(experiments);selection['parent_selection_sha256']=parent_hash
    selection['artifact_sha256'][checkpoint.name]=sha256(checkpoint)
    selection['optimization']=dict(wall_seconds=time.monotonic()-started,rule='Per-class regularized temporal heads; validation-only model, epoch and threshold selection',
        implementation_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'stereotypy/priority_models.py']})
    save_json(run/'binary-temporal-history.json',histories);save_json(run/'validation-experiments.json',experiments);save_json(run/'selection.json',selection)
    print('Revised selection locked',json.dumps(selection['selected']),flush=True)

if __name__=='__main__':main()
