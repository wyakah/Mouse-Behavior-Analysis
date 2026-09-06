"""Run the same 10 Hz anatomical evidence used by automatic processing."""
import os,sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface');os.environ['TORCH_HOME']=str(ROOT/'.cache/torch');os.environ['DLC_LIGHT']='True'
import av,cv2,numpy as np,torch
from stereotypy.pose_signals import PoseRuntime
from stereotypy.training import save_json,sha256

def run(path,out,runtime,limit):
    mid=path.stem.split('_')[0];dest=out/mid;dest.mkdir(parents=True,exist_ok=True)
    old=ROOT/'outputs/stereotypy-five-video'/mid/'input.json'
    with av.open(str(path)) as c:
        st=c.streams.video[0];f=next(c.decode(st));duration=min(limit,float(st.duration*st.time_base));roi=json.loads(old.read_text())['mapping']['crop_xyxy'] if old.exists() else [0,0,f.width,f.height]
    signature=dict(source_sha256=sha256(path),duration_s=duration,roi=roi,model=runtime.metadata,pose_hz=10,implementation_sha256=sha256(ROOT/'stereotypy/pose_signals.py'))
    target=dest/'pose.json'
    if target.exists():
        if json.loads(target.read_text())['signature']!=signature:raise ValueError('Choose fresh output folder')
        return
    from stereotypy.pose_signals import PoseSignals
    runtime.signals=PoseSignals();images=[];times=[];rows=[];next_sample=0.;started=time.monotonic();n=0;a,b,c,d=roi
    def flush():
        if images:rows.extend(runtime.infer(images,times));images.clear();times.clear()
    with av.open(str(path)) as con:
        st=con.streams.video[0];st.thread_type='AUTO';origin=None
        for f in con.decode(st):
            if origin is None:origin=float(f.pts*f.time_base)
            t=float(f.pts*f.time_base)-origin
            if t>=duration:break
            n+=1
            if t+1e-6>=next_sample:
                images.append(f.to_ndarray(format='rgb24')[b:d,a:c]);times.append(t);next_sample+=.1
                if len(images)>=32:flush()
            if n%9000==0:print(mid,'pose frames decoded',n,'samples',len(rows),'wall',round(time.monotonic()-started),flush=True)
    flush();save_json(target,dict(signature=signature,mouse_id=mid,samples=rows,decoded_frames=n,wall_seconds=time.monotonic()-started,accuracy=None,training_performed=False))
    print(mid,'pose complete',len(rows),round(time.monotonic()-started,1),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',nargs='+',default=['710','711','712','743','745','756']);p.add_argument('--limit',type=float,default=1200);p.add_argument('--out',type=Path,default=ROOT/'outputs/stereotypy-distinction');args=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(2);runtime=PoseRuntime()
    for mid in args.only:run(next((ROOT/'stereotypy_videos').glob(mid+'_*.mov')),args.out,runtime,min(1200,args.limit))
