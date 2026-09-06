"""Targeted local checks after the candidate failed the external benchmark.

Twelve systematic and up to six gnawing-candidate two-second clips per upload;
all within the first 20 minutes. These do not constitute full candidate totals.
"""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_stereotypy_labgym import NativeLabGym,NativeInputs,background
from stereotypy.training import sha256,save_json
import cv2,numpy as np
OUT=ROOT/'outputs/stereotypy-distinction'
model=NativeLabGym();cv2.setNumThreads(2)
for mid in ['710','711','712','743','745','756']:
    path=next((ROOT/'stereotypy_videos').glob(mid+'_*.mov'));basepath=ROOT/('outputs/stereo-f209faf01650/published/001.json' if mid=='756' else f'outputs/stereotypy-five-video/{mid}/predictions.json');base=json.loads(basepath.read_text());roi=base['mapping']['crop_xyxy'];a,b,c,d=roi
    windows=[dict(start_s=float(t),end_s=float(t+2),selection='systematic') for t in np.linspace(0,1198,12)]
    for r in sorted(base['windows'],key=lambda r:r['scores']['gnawing_nonfood'],reverse=True):
        t=min(r['start_s'],1198.)
        if all(abs(t-w['start_s'])>=10 for w in windows):windows.append(dict(start_s=t,end_s=t+2,selection='baseline gnawing candidate'))
        if len(windows)==18:break
    bg=background(path,roi,1200);cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);rows=[];started=time.monotonic()
    for w in sorted(windows,key=lambda w:w['start_s']):
        animations=[];patterns=[]
        for t in [w['start_s']+v for v in [.5,1,1.5]]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,round(t*fps)-13));adapter=NativeInputs(bg,2)
            for _ in range(14):
                ok,im=cap.read()
                if not ok:raise ValueError('Truncated test clip')
                im=im[b:d,a:c];adapter.add(im)
            sample=adapter.sample(im)
            if sample is not None:animations.append(sample[0]);patterns.append(sample[1])
        scores=model.predict(animations,patterns).mean(0) if animations else None
        rows.append(dict(w,scores=scores.tolist() if scores is not None else None,native_label=model.classes[int(np.argmax(scores))] if scores is not None else 'unobservable'))
    cap.release();save_json(OUT/mid/'local-clips.json',dict(mouse_id=mid,source_sha256=sha256(path),roi=roi,classes=model.classes,windows=rows,selection='12 systematic +6 gnawing candidates; not a random accuracy sample',animal_vs_bg=2,accuracy=None,total_candidate_test_seconds=sum(r['end_s']-r['start_s'] for r in rows),wall_seconds=time.monotonic()-started))
    print(mid,'local clips complete',len(rows),round(time.monotonic()-started,1),flush=True)
