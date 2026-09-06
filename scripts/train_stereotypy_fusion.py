"""Fit only with explicit human interval labels and mouse-disjoint train/val/test.

Example: --reference reference.json --train A B --validation C --test D E
All six behaviors are attempted; unsupported classes remain unavailable.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from stereotypy.fusion import feature_matrix,reference_samples,binary_metrics,FEATURES
from stereotypy.training import CLASSES,sha256,save_json
from stereotypy.core import validate_annotations


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--train',nargs='+',required=True);p.add_argument('--validation',nargs='+',required=True);p.add_argument('--test',nargs='+',required=True);p.add_argument('--out',type=Path,default=ROOT/'outputs/stereotypy-fusion');args=p.parse_args()
    groups=[set(args.train),set(args.validation),set(args.test)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)) or len(groups[0])<2:raise ValueError('Use disjoint mice, with at least two training mice')
    doc=json.loads(args.reference.read_text())
    if not doc.get('annotator') or not doc.get('behaviors'):raise ValueError('Explicit human annotator and behavior reference intervals required; predictions cannot be training labels')
    data={};labels={}
    for mouse in set.union(*groups):
        report=ROOT/'reports/stereotypy-robustness'/f'{mouse}.json'
        data[mouse]=json.loads(report.read_text())['rows']
        expected=json.loads((ROOT/'outputs/stereotypy-robustness'/mouse/'manifest.json').read_text())['signature']['source_sha256']
        chosen=[r for r in doc['behaviors'] if r['mouse_id']==mouse]
        if any(r.get('source_sha256')!=expected for r in chosen):raise ValueError('Behavior reference source hash mismatch')
        labels[mouse]=validate_annotations(chosen,0,1200)
    out=args.out
    if out.exists():raise ValueError('Choose a fresh output directory to preserve prior training')
    reports={};models={}
    selection=json.loads((ROOT/'outputs/stereotypy-training/v2/selection.json').read_text())
    baseline_thresholds={c:selection['selected'][c]['threshold'] if c in selection['selected'] else .5 for c in CLASSES}
    for behavior in CLASSES:
        split=[]
        for mice in groups:
            xx=[];yy=[];ww=[]
            for mouse in sorted(mice):
                ids,y,w=reference_samples(data[mouse],labels[mouse],behavior);xx.extend(feature_matrix(data[mouse])[ids]);yy.extend(y);ww.extend(w)
            split.append((np.array(xx),np.array(yy),np.array(ww)))
        if any(len(set(y))<2 for x,y,w in split):
            reports[behavior]=dict(status='unavailable',reason='Explicit positive and negative intervals required in every split');continue
        model=make_pipeline(SimpleImputer(strategy='constant',fill_value=0,add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,C=.1,random_state=42))
        x,y,w=split[0];model.fit(x,y,logisticregression__sample_weight=w)
        vx,vy,vw=split[1];vp=model.predict_proba(vx)[:,1]
        threshold=max(np.arange(.1,.91,.02),key=lambda t:binary_metrics(vy,vp,vw,t)['f1'] or 0)
        tx,ty,tw=split[2];pred=model.predict_proba(tx)[:,1]
        reports[behavior]=dict(status='experimental',threshold=float(threshold),test=binary_metrics(ty,pred,tw,threshold),
                              baseline_test=binary_metrics(ty,tx[:,CLASSES.index(behavior)],tw,baseline_thresholds[behavior]),
                              baseline_threshold=baseline_thresholds[behavior], note='Paired comparison on identical explicitly labeled and body-eligible intervals; excluded time is not evaluated.')
        models[behavior]=dict(model=model,threshold=float(threshold))
    if not models:raise ValueError('No behavior has a usable labeled train/validation/test split; no model fitted')
    out.mkdir(parents=True)
    joblib.dump(dict(models=models,features=FEATURES),out/'fusion.joblib')
    save_json(out/'evaluation.json',dict(reference_sha256=sha256(args.reference),splits=dict(train=args.train,validation=args.validation,test=args.test),behaviors=reports,production_ready=False))
if __name__=='__main__':main()
