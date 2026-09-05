"""Cache label-free local optical-flow descriptors alongside visual features.

Uses native neighboring frames, not differences between widely sampled images.
This is a lightweight motion ablation inspired by two-stream classifiers, not a
reproduction of DeepEthogram's learned flow network.
"""
import argparse,concurrent.futures,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from stereotypy.training import sha256,save_json
RECIPE='farneback-native-neighbors-hof4x4x8-v1'


def descriptor(first,second,box):
    flow=cv2.calcOpticalFlowFarneback(first,second,None,.5,2,11,2,5,1.1,0)
    a,b,c,d=box;flow=flow[b:d,a:c]
    out=[]
    for row in np.array_split(flow,4,axis=0):
        for cell in np.array_split(row,4,axis=1):
            if not cell.size:out.extend([0.]*12);continue
            fx,fy=cell[...,0],cell[...,1];magnitude=np.minimum(np.hypot(fx,fy),15)
            angle=(np.arctan2(fy,fx)+2*np.pi)%(2*np.pi)
            hist=np.bincount(np.minimum((angle/(2*np.pi)*8).astype(int).ravel(),7),weights=magnitude.ravel(),minlength=8)/magnitude.size
            out.extend([*hist,float(magnitude.mean()),float(magnitude.std()),float(np.clip(fx,-15,15).mean()),float(np.clip(fy,-15,15).mean())])
    return np.array(out,np.float32)


def extract(row):
    cv2.setNumThreads(1)
    path=Path(row['feature_path']);dest=path.with_name(path.stem+'-motion.npz')
    fingerprint=sha256(path)
    data=np.load(path,allow_pickle=False)
    if row.get('source_index'):
        index=json.loads(Path(row['source_index']).read_text())
        if index['source_sha256']!=row['sha256']:raise ValueError('Motion/source timestamp fingerprint mismatch')
        actual_times=np.array([index['frames'][i]['start_s'] for i in data['frame_ids']])
        if not np.allclose(actual_times,data['times'],rtol=0,atol=1e-7):raise ValueError('Motion source clock differs from visual features')
    if dest.exists():
        old=np.load(dest,allow_pickle=False)
        if str(old['feature_sha256'])==fingerprint and str(old['recipe'])==RECIPE:return row['id']
    ids=data['frame_ids'];geometry=data['geometry'];fps=float(data['fps'])
    cap=cv2.VideoCapture(row['path']);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if row.get('source_index'):total=json.loads(Path(row['source_index']).read_text())['frame_count']
    step=max(1,round(fps*.1)) if row['dataset']!='labgym' else 1
    requests=np.clip(np.stack([ids-step,ids,ids+step],1),0,total-1);wanted=set(requests.ravel().tolist());frames={}
    a,b,c,d=row.get('crop',[0,0,int(cap.get(3)),int(cap.get(4))]);i=0
    while cap.grab():
        if i in wanted:
            ok,im=cap.retrieve()
            if not ok:raise ValueError('Motion frame decode failed')
            frames[i]=cv2.resize(cv2.cvtColor(im[b:d,a:c],cv2.COLOR_BGR2GRAY),(160,120))
        i+=1
        if i>int(requests.max()):break
    cap.release()
    if wanted-set(frames):raise ValueError('Missing source frames for motion')
    gaps=[];timestamps=None
    if row.get('source_index'):
        index=json.loads(Path(row['source_index']).read_text());gaps=index['source_gaps'];timestamps=[f['start_s'] for f in index['frames']]
    features=[]
    for sample,(previous,current,following) in enumerate(requests):
        g=geometry[sample]
        if g[-1]>0:
            cx,cy,bw,bh=g[:4];bw=max(bw,.08)*1.3;bh=max(bh,.08)*1.3
            box=(max(0,int((cx-bw/2)*160)),max(0,int((cy-bh/2)*120)),min(160,int((cx+bw/2)*160)+1),min(120,int((cy+bh/2)*120)+1))
        else:box=(0,0,160,120)
        valid=previous!=current and following!=current
        if timestamps and any(a<timestamps[following] and b>timestamps[previous] for a,b in gaps):valid=False
        if not valid:features.append(np.zeros(385,np.float32));continue
        features.append(np.r_[descriptor(frames[previous],frames[current],box),descriptor(frames[current],frames[following],box),1.])
    np.savez_compressed(dest,x=np.array(features,np.float32),frame_ids=ids,feature_sha256=fingerprint,recipe=RECIPE)
    return row['id']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    run=ROOT/'outputs/stereotypy-training/v2';done=set();pending={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            try:rows=json.loads((run/'feature-catalog.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):rows=[]
            for future in list(pending):
                if future.done():
                    identity=future.result();done.add(identity);pending.pop(future)
                    if len(done)%100==0 or not identity.startswith('labgym'):print(identity,'motion complete;',len(done),'sources',flush=True)
            eligible=[r for r in rows if r['id'] not in done and r['id'] not in pending.values()]
            for row in eligible[:max(0,args.workers*2-len(pending))]:pending[pool.submit(extract,row)]=row['id']
            complete=(run/'extraction-complete.json').exists()
            if not pending and len(done)==len(rows) and (not args.watch or complete):break
            time.sleep(.05)
    save_json(run/'motion-complete.json',dict(recipe=RECIPE,sources=len(done),catalog_sha256=sha256(run/'feature-catalog.json')))
    print('Motion extraction complete',len(done),flush=True)

if __name__=='__main__':main()
