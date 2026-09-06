"""Bounded native LabGym comparator. Preserves the author's 20-class vocabulary.

Run in .labgym-env (TF 2.16 / legacy Keras / LabGym 2.2.2). Uses upstream
contour_frame, extract_blob and generate_patternimage without copying their code.
Background estimation is ours (32-frame bright percentile); report this adapter
rather than claiming an exact reproduction of the full LabGym application.
"""
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLCONFIGDIR','.cache/matplotlib')
import argparse,json,sys,time,csv
from pathlib import Path
from collections import deque
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import av,cv2,numpy as np,tensorflow as tf
from LabGym.tools import contour_frame,generate_patternimage
from stereotypy.training import sha256,save_json
MODEL=ROOT/'research/stereotypy-round2/labgym-model/Mouse_NonSocial_SideView_30fps'

class NativeLabGym:
    def __init__(self):
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        self.model=tf.keras.models.load_model(str(MODEL),compile=False)
        self.parameters=list(csv.DictReader((MODEL/'model_parameters.txt').open()))
        self.classes=[r['classnames'] for r in self.parameters]
        p=self.parameters[0]
        if (int(p['time_step']),int(p['inner_code']),int(p['background_free']))!=(14,0,0):raise ValueError('Unexpected native preprocessing settings')
        if self.model.output_shape[-1]!=len(self.classes):raise ValueError('Model class order mismatch')
    def predict(self,animations,patterns):
        return self.model([np.asarray(animations,np.float32)/255.,np.asarray(patterns,np.float32)/255.],training=False).numpy()

class NativeInputs:
    def __init__(self,background,animal_vs_bg=1):
        self.animal_vs_bg=animal_vs_bg;self.background=255-background if animal_vs_bg==1 else background;self.kernel=3 if min(background.shape[:2])<250 else 5
        self.contours=deque(maxlen=14);self.inners=deque(maxlen=14);self.blobs=deque(maxlen=14)
    def add(self,frame):
        # Upstream single-animal routine raises on empty/zero-area contours.
        try:
            contours,_,_,inners,blobs=contour_frame(frame,1,self.background,self.background,self.background,1.2,0,animal_vs_bg=self.animal_vs_bg,include_bodyparts=True,animation_analyzer=True,channel=1,kernel=self.kernel)
            contour=contours[0];inner=inners[0];blob=cv2.resize(blobs[0],(64,64),interpolation=cv2.INTER_AREA)[...,None]
            valid=.0015*frame.shape[0]*frame.shape[1]<cv2.contourArea(contour)<.20*frame.shape[0]*frame.shape[1]
        except (IndexError,ZeroDivisionError,ValueError,cv2.error):contour=None;inner=None;blob=np.zeros((64,64,1),np.uint8);valid=False
        if not valid:contour=None;inner=None;blob=np.zeros((64,64,1),np.uint8)
        self.contours.append(contour);self.inners.append(inner);self.blobs.append(blob)
        return valid
    def sample(self,frame):
        if len(self.blobs)!=14 or any(x is None for x in self.contours):return None
        pattern=generate_patternimage(frame,list(self.contours),inners=list(self.inners),std=50)
        return np.array(self.blobs),cv2.resize(pattern,(64,64),interpolation=cv2.INTER_AREA)

def background(path,roi,duration):
    images=[];a,b,c,d=roi
    with av.open(str(path)) as con:
        st=con.streams.video[0];origin=st.start_time or 0
        for t in np.linspace(0,max(0,duration-.1),32):
            con.seek(origin+int(t/st.time_base),stream=st)
            for f in con.decode(st):
                if f.pts>=origin+int(t/st.time_base):break
            images.append(f.to_ndarray(format='bgr24')[b:d,a:c])
    return np.percentile(images,90,axis=0).astype(np.uint8)

def run(path,dest,model,limit=1200,animal_vs_bg=1):
    mid=path.stem.split('_')[0];out=dest/mid;out.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'outputs/stereotypy-five-video'/mid/'input.json'
    with av.open(str(path)) as con:
        s=con.streams.video[0];f=next(con.decode(s));duration=min(limit,float(s.duration*s.time_base) if s.duration else limit)
        roi=json.loads(baseline.read_text())['mapping']['crop_xyxy'] if baseline.exists() else [0,0,f.width,f.height]
    sig=dict(source_sha256=sha256(path),model_sha256=sha256(MODEL/'saved_model.pb'),variables_sha256=sha256(MODEL/'variables/variables.data-00000-of-00001'),parameters_sha256=sha256(MODEL/'model_parameters.txt'),implementation_sha256=sha256(__file__),roi=roi,duration_s=duration,recipe='labgym-native14-2hz-percentile-background-v1',animal_vs_bg=animal_vs_bg)
    target=out/'labgym.json'
    if target.exists():
        if json.loads(target.read_text())['signature']!=sig:raise ValueError('Choose a fresh output directory')
        print(mid,'verified cached LabGym',flush=True);return
    started=time.monotonic();bg=background(path,roi,duration);adapter=NativeInputs(bg,animal_vs_bg);rows=[];pending=[];patterns=[];pending_ids=[];decoded=0;next_sample=0.;last=None;a,b,c,d=roi
    def flush():
        if not pending:return
        scores=model.predict(pending,patterns)
        for i,p in zip(pending_ids,scores):
            rows[i].update(scores={c:float(v) for c,v in zip(model.classes,p)},label=model.classes[int(np.argmax(p))])
        pending.clear();patterns.clear();pending_ids.clear()
    with av.open(str(path)) as con:
        st=con.streams.video[0];st.thread_type='AUTO';origin=None
        for f in con.decode(st):
            if origin is None:origin=float(f.pts*f.time_base)
            t=float(f.pts*f.time_base)-origin
            if t>=duration:break
            if last is not None and t-last>.2:adapter=NativeInputs(bg,animal_vs_bg)
            last=t;decoded+=1;im=f.to_ndarray(format='bgr24')[b:d,a:c];adapter.add(im)
            if t+1e-6>=next_sample:
                rows.append(dict(start_s=t,end_s=min(duration,t+.5),scores=None,label='unobservable'))
                sample=adapter.sample(im)
                if sample is not None:pending.append(sample[0]);patterns.append(sample[1]);pending_ids.append(len(rows)-1)
                next_sample+=.5
                if len(pending)>=32:flush()
            if decoded%9000==0:print(mid,'LabGym decoded',decoded,'scored windows',len(rows),flush=True)
    flush()
    for i,r in enumerate(rows):r['end_s']=rows[i+1]['start_s'] if i+1<len(rows) else duration
    totals={c:sum(r['end_s']-r['start_s'] for r in rows if r['label']==c) for c in model.classes+['unobservable']}
    save_json(target,dict(signature=sig,source=str(path.relative_to(ROOT)),mouse_id=mid,classes=model.classes,windows=rows,summary_seconds=totals,decoded_frames=decoded,wall_seconds=time.monotonic()-started,accuracy=None,training_performed=False,score_semantics='Native 20-class comparator; chewing is not nonfood gnawing',preprocessing='Upstream LabGym 2.2.2 native14-frame animation/internal-contour pattern; our percentile background and invalid-contour gate'))
    print(mid,'LabGym complete',round(time.monotonic()-started,1),'s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',nargs='+',default=['710','711','712','743','745','756']);p.add_argument('--out',type=Path,default=ROOT/'outputs/stereotypy-distinction');p.add_argument('--limit',type=float,default=1200);p.add_argument('--animal-vs-bg',type=int,choices=[1,2],default=1);args=p.parse_args();cv2.setNumThreads(2);model=NativeLabGym()
    for mid in args.only:
        paths=list((ROOT/'stereotypy_videos').glob(mid+'_*.mov'))
        if len(paths)!=1:raise ValueError('Expected one source per mouse')
        run(paths[0],args.out,model,min(1200,args.limit),args.animal_vs_bg)
