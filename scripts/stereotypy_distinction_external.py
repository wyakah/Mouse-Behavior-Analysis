"""External, human-labeled MIT windows for a grooming/rearing distinction test.

Retains original date groups/splits. Legacy test has been used in earlier work;
this is a paired diagnostic, not a newly sealed test or local validation.
"""
import os,sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('TF_USE_LEGACY_KERAS','1');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib');os.environ['HF_HOME']=str(ROOT/'.cache/huggingface');os.environ['TORCH_HOME']=str(ROOT/'.cache/torch')
import cv2,numpy as np
from stereotypy.training import save_json,sha256
OUT=ROOT/'outputs/stereotypy-distinction/external'

def plan():
    data=json.loads((ROOT/'outputs/stereotypy-training/v1/dataset.json').read_text());rng=np.random.default_rng(42);rows=[]
    for split,cap in [('train',240),('validation',120),('test',100000)]:
        source=[r for r in data['windows'] if r['source'].startswith('mit:') and r['split']==split and r['labels'][0]>=0 and r['labels'][3]>=0]
        # Deterministic per-class budget, independent of model predictions.
        for label in ['grooming','rearing','other']:
            chosen=[r for r in source if ('grooming' if r['labels'][0] else 'rearing' if r['labels'][3] else 'other')==label]
            if len(chosen)>cap//3:chosen=[chosen[i] for i in sorted(rng.choice(len(chosen),cap//3,replace=False))]
            for r in chosen:rows.append(dict(r,label=label))
    return data['sources'],sorted(rows,key=lambda r:(r['source'],r['start_s']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['labgym','pose'],required=True);p.add_argument('--animal-vs-bg',type=int,choices=[1,2],default=1);args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);sources,rows=plan();save_json(OUT/'protocol.json',dict(seed=42,window_ids=[r['id'] for r in rows],train_cap=240,validation_cap=120,test='all original MIT test windows; previously inspected',local_labels_used=False,labels=['grooming','rearing','other'],gnawing_labels_available=False))
    if args.backend=='labgym':
        from benchmark_stereotypy_labgym import NativeLabGym,NativeInputs,background
        model=NativeLabGym()
    else:
        import torch
        torch.set_num_threads(4)
        from stereotypy.pose_signals import PoseRuntime,PoseSignals,summarize_pose
        model=PoseRuntime()
    cv2.setNumThreads(2);current=None;output=[];started=time.monotonic()
    target=OUT/(args.backend+('-original' if args.backend=='labgym' and args.animal_vs_bg==2 else '')+'.json')
    if target.exists():raise ValueError('Refusing overwrite of external feature run')
    for number,r in enumerate(rows):
        s=sources[r['source']]
        if current!=r['source']:
            if current is not None:cap.release()
            current=r['source'];path=Path(s['path']);cap=cv2.VideoCapture(str(path));a,b,c,d=s['crop']
            if sha256(path)!=s['sha256']:raise ValueError('External source changed')
            if args.backend=='labgym':bg=background(path,s['crop'],min(1200,s['frames_header']/s['fps']))
        fps=s['fps'];endpoints=[r['start_s']+v for v in (.5,1,1.5)]
        if args.backend=='labgym':
            animations=[];patterns=[]
            for t in endpoints:
                endpoint=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,endpoint-13));adapter=NativeInputs(bg,args.animal_vs_bg)
                for n in range(14):
                    ok,im=cap.read()
                    if not ok:break
                    im=im[b:d,a:c];adapter.add(im)
                sample=adapter.sample(im) if ok else None
                if sample is not None:animations.append(sample[0]);patterns.append(sample[1])
            scores=model.predict(animations,patterns).mean(0).tolist() if animations else None
            output.append(dict(id=r['id'],label=r['label'],group=r['group'],split=r['split'],scores=scores,usable_samples=len(animations),classes=model.classes))
        else:
            images=[];endpoints=[r['start_s']+.5+j*.1 for j in range(9)]
            for t in endpoints:
                cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*fps));ok,im=cap.read()
                if not ok:raise ValueError('Missing external frame')
                images.append(cv2.cvtColor(im[b:d,a:c],cv2.COLOR_BGR2RGB))
            model.signals=PoseSignals();samples=model.infer(images,endpoints);output.append(dict(id=r['id'],label=r['label'],group=r['group'],split=r['split'],pose=summarize_pose(samples)))
        if number%50==0:print(args.backend,number,'/',len(rows),'wall',round(time.monotonic()-started),flush=True)
    cap.release();save_json(target,dict(rows=output,wall_seconds=time.monotonic()-started,protocol_sha256=sha256(OUT/'protocol.json'),reference='MIT author human annotations, annotation group1; original date splits',accuracy=None))
if __name__=='__main__':main()
