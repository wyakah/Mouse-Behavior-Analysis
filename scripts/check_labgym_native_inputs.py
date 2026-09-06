"""Sanity-check released model on its supplied examples (not held-out accuracy)."""
import sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
from benchmark_stereotypy_labgym import NativeLabGym,NativeInputs,background
import cv2,numpy as np
from stereotypy.training import save_json
model=NativeLabGym();examples=ROOT/'research/stereotypy-round2/labgym-examples/mouse behavior examples';rows=[]
for name in model.classes:
    files=sorted((examples/name).glob('*.avi'))[:3]
    for path in files:
        jpg=path.with_suffix('.jpg')
        if not jpg.exists():continue
        cap=cv2.VideoCapture(str(path));images=[]
        while True:
            ok,im=cap.read()
            if not ok:break
            images.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,64),interpolation=cv2.INTER_AREA)[...,None])
        cap.release()
        if len(images)!=14:continue
        # Upstream cv2 input order is BGR for its pattern images.
        pattern=cv2.resize(cv2.imread(str(jpg)),(64,64),interpolation=cv2.INTER_AREA)
        p=model.predict([np.array(images)],[pattern])[0]
        rows.append(dict(expected=name,predicted=model.classes[int(np.argmax(p))],score=float(p.max())))
save_json(ROOT/'outputs/stereotypy-distinction/native-sanity.json',dict(rows=rows,matches=sum(r['expected']==r['predicted'] for r in rows),examples=len(rows),scope='Released training examples; loader/preprocessing sanity, not test accuracy'))
print('supplied example matches',sum(r['expected']==r['predicted'] for r in rows),'/',len(rows))
path=ROOT/'stereotypy_videos/710_stereotypy-540p30.mov';roi=[30,20,546,278];bg=background(path,roi,1200);cap=cv2.VideoCapture(str(path));tiles=[]
for t in [60,300,600]:
    cap.set(cv2.CAP_PROP_POS_MSEC,(t-.5)*1000);adapter=NativeInputs(bg)
    for _ in range(15):
        ok,im=cap.read();im=im[20:278,30:546];adapter.add(im)
    sample=adapter.sample(im);canvas=np.full((280,780,3),245,np.uint8);canvas[:258,:516]=im
    if sample:
        canvas[:256,520:776]=cv2.cvtColor(cv2.resize(sample[0][-1],(256,256)),cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas,str(t)+'s native input',(12,276),0,.5,(20,20,20),1);tiles.append(canvas)
cap.release();cv2.imwrite(str(ROOT/'outputs/stereotypy-distinction/native-inputs.jpg'),np.vstack(tiles))
