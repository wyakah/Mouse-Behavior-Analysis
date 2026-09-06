"""Frozen-model end-to-end stereotypy pilot. Suggestions never become annotations.

The full first 1200 seconds are decoded. Priority heads sample at 2 Hz with
native optical flow; weak heads use 16 frames per 2-second window. Timings are
integrated from source PTS, not rounded video headers. No fitting or statistics.
"""
import argparse, csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface');os.environ['TORCH_HOME']=str(ROOT/'.cache/torch')
import av,cv2,numpy as np,torch,joblib
from stereotypy.training import CLASSES,save_json,sha256,make_model as weak_model
from stereotypy.window import clip_manifest
from stereotypy.video import index_video
from stereotypy.cage import validate_mapping,crop_frame,CageProposer
from stereotypy.priority_training import context_indices
from stereotypy.priority_models import make_binary_model
from stereotypy.batch import summarize_predictions, candidate_windows, select_sources
from stereotypy_priority_features import letterbox,mouse_view
from stereotypy_motion_features import extract as extract_motion
from train_stereotypy_priority import make_model
from train_stereotypy_video import input_image

torch.set_num_threads(4);cv2.setNumThreads(2)
RUN=ROOT/'outputs/stereotypy-five-video';REPORT=ROOT/'reports/stereotypy-five-video'
V2=ROOT/'outputs/stereotypy-training/v2';V1=ROOT/'outputs/stereotypy-training/v1'
# Visually inspected cage bounds in original, unrotated source pixels.
MAPPINGS={'710':([30,20,546,278],215),'711':([40,20,542,280],222),
          '712':([50,20,514,265],217),'743':([40,20,534,286],215),
          '745':([25,20,520,278],221)}


def inputs(path):
    mouse=path.stem.split('_')[0];out=RUN/mouse;out.mkdir(parents=True,exist_ok=True)
    index_path=out/'source-index.json'
    if not index_path.exists():save_json(index_path,index_video(path))
    full=json.loads(index_path.read_text());stat=path.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(full['source_size'],full['source_mtime_ns']) or sha256(path)!=full['source_sha256']:raise ValueError('Source changed since indexing')
    if full['rotation_degrees']!=0:raise ValueError('Normalize orientation explicitly before analysis')
    index=clip_manifest(full);roi,floor=MAPPINGS[mouse];mapping=validate_mapping(dict(crop_xyxy=roi,floor_y=floor),full['width'],full['height'])
    signature=dict(source_sha256=full['source_sha256'],source_index_sha256=sha256(index_path),mapping=mapping,
                   end_s=index['duration_s'],recipe='frozen-v2-plus-weak-v1-batch-1')
    old=out/'input.json'
    if old.exists() and json.loads(old.read_text())!=signature:raise ValueError('Inputs changed: choose a fresh run directory')
    save_json(old,signature);return mouse,out,index,mapping


def background(path,index,mapping,size=None):
    images=[];a,b,c,d=mapping['crop_xyxy']
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        for t in np.linspace(0,index['duration_s']-.1,32):
            target=t+index['source_origin_s'];container.seek(int(target/stream.time_base),stream=stream)
            for frame in container.decode(stream):
                if float(frame.pts*frame.time_base)>=target:break
            gray=cv2.cvtColor(frame.to_ndarray(format='bgr24')[b:d,a:c],cv2.COLOR_BGR2GRAY)
            images.append(cv2.resize(gray,size) if size else gray)
    return np.percentile(images,90,axis=0).astype(np.uint8)


def features(path,mouse,out,index,mapping,device):
    from transformers import AutoModel
    dest=out/'visual.npz';times=np.array([f['start_s'] for f in index['frames']]);a,b,c,d=mapping['crop_xyxy']
    ids=np.unique(np.searchsorted(times,np.arange(0,index['duration_s'],.5),side='right')-1)
    if not dest.exists():
        print(mouse,'extracting priority visual features',flush=True)
        bg=background(path,index,mapping,(320,240));model=AutoModel.from_pretrained('facebook/dinov2-small',revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',local_files_only=True).eval().to(device)
        mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        pending=[];encoded=[];geometry=[];cuts=[];previous=None;cursor=0
        def flush():
            for i in range(0,len(pending),16):
                x=torch.from_numpy(np.stack(pending[i:i+16])).to(device).float()[:,None].repeat(1,3,1,1)/255
                with torch.inference_mode():encoded.extend(model((x-mean)/std).last_hidden_state[:,0,:].cpu().numpy())
            pending.clear()
        with av.open(str(path)) as container:
            stream=container.streams.video[0];stream.thread_type='AUTO'
            for i,frame in enumerate(container.decode(stream)):
                if i>=index['frame_count']:break
                if frame.pts!=index['frames'][i]['pts']:raise ValueError('Frame alignment changed')
                if cursor<len(ids) and i==ids[cursor]:
                    gray=cv2.cvtColor(frame.to_ndarray(format='bgr24')[b:d,a:c],cv2.COLOR_BGR2GRAY)
                    body,g=mouse_view(cv2.resize(gray,(320,240)),bg);full=letterbox(gray)
                    cuts.append(cursor==0 or times[i]-times[ids[cursor-1]]>1 or previous is not None and np.abs(full.astype(float)-previous).mean()>40 or any(s<times[i] and e>times[ids[max(0,cursor-1)]] for s,e in index['source_gaps']))
                    previous=full.astype(float);geometry.append(g);pending.extend([full,letterbox(body)]);cursor+=1
                    if len(pending)>=128:flush()
                    if cursor%600==0:print(mouse,'priority samples',cursor,'/',len(ids),flush=True)
        flush()
        if cursor!=len(ids):raise ValueError('Incomplete visual decoding')
        np.savez_compressed(dest,x=np.array(encoded,np.float16).reshape(-1,768),geometry=np.array(geometry,np.float32),frame_ids=ids,times=times[ids],cuts=cuts,fps=index['nominal_fps'])
        del model
        if device=='mps':torch.mps.empty_cache()
    # A clipped index prevents motion from borrowing frames beyond minute 20.
    save_json(out/'analysis-index.json',index)
    row=dict(id=mouse,path=str(path),dataset='local',feature_path=str(dest),crop=mapping['crop_xyxy'],source_index=str(out/'analysis-index.json'),sha256=index['source_sha256'])
    extract_motion(row);print(mouse,'native optical flow complete',flush=True)


def priority_predict(out):
    selection=json.loads((V2/'selection.json').read_text())
    for name,digest in selection['artifact_sha256'].items():
        if name=='prepared-features.npz':continue
        if sha256(V2/name)!=digest:raise ValueError('Frozen model artifact changed')
    data=np.load(out/'visual.npz');motion=np.load(out/'visual-motion.npz');x=np.c_[data['x'].astype(np.float32),data['geometry'],motion['x']]
    prep=joblib.load(V2/'preprocessing.joblib');scaled=prep['scaler'].transform(x)
    z=np.c_[np.clip(prep['pca'].transform(scaled[:,:775]),-12,12),np.clip(prep['motion_pca'].transform(scaled[:,775:]),-12,12)].astype(np.float32)
    seq=z[context_indices(data['cuts'])].astype(np.float16).astype(np.float32);scores=[]
    for k,c in enumerate(['grooming','digging','rearing']):
        chosen=selection['selected'][c]['model']
        if chosen=='temporal':
            state=torch.load(V2/'temporal.pt',weights_only=True,map_location='cpu');model=make_model(state['input_dim']);model.load_state_dict(state['state_dict'])
        elif chosen=='binary-temporal':
            state=torch.load(V2/'binary-temporal.pt',weights_only=True,map_location='cpu');model=make_binary_model(state['input_dim']);model.load_state_dict(state['models'][c]['state_dict'])
        else:raise ValueError('Selected model requires an explicit inference implementation')
        model.eval()
        with torch.inference_mode():p=np.concatenate([model(torch.from_numpy(b)).sigmoid().numpy() for b in np.array_split(seq,max(1,len(seq)//256))])
        scores.append(p[:,k] if chosen=='temporal' else p)
    return data['times'],np.stack(scores,1),selection


def weak_predict(path,mouse,out,index,mapping,device):
    from torchvision.models import mobilenet_v3_large,MobileNet_V3_Large_Weights
    cache=out/'weak-features.npz';times=np.array([f['start_s'] for f in index['frames']]);starts=np.arange(0,index['duration_s'],2.);ends=np.minimum(starts+2,index['duration_s'])
    requests=np.array([np.searchsorted(times,np.linspace(s,e,16,endpoint=False),side='right')-1 for s,e in zip(starts,ends)])
    if not cache.exists():
        print(mouse,'extracting six-head review features',flush=True)
        model=mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).eval().to(device);model.classifier=torch.nn.Identity()
        mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        lookup={}
        for w,row in enumerate(requests):
            for j,i in enumerate(row):lookup.setdefault(int(i),[]).append((w,j))
        feats=np.zeros((len(starts),16,960),np.float16);pending=[];targets=[];seen=0
        def flush():
            x=torch.from_numpy(np.stack(pending)).to(device).float()/255
            with torch.inference_mode():p=model((x-mean)/std).cpu().numpy()
            for f,items in zip(p,targets):
                for w,j in items:feats[w,j]=f
            pending.clear();targets.clear()
        with av.open(str(path)) as container:
            stream=container.streams.video[0];stream.thread_type='AUTO'
            for i,frame in enumerate(container.decode(stream)):
                if i>=index['frame_count']:break
                if frame.pts!=index['frames'][i]['pts']:raise ValueError('Weak head source alignment changed')
                if i in lookup:
                    pending.append(input_image(frame.to_ndarray(format='bgr24'),mapping['crop_xyxy']));targets.append(lookup[i]);seen+=1
                    if len(pending)==64:flush()
                    if seen%2400==0:print(mouse,'weak-head sampled frames',seen,flush=True)
        if pending:flush()
        if seen!=len(lookup):raise ValueError('Incomplete weak feature decoding')
        np.savez_compressed(cache,features=feats);del model
        if device=='mps':torch.mps.empty_cache()
    checkpoint=V1/'temporal-model.pt';report=json.loads((V1/'training-report.json').read_text())
    if sha256(checkpoint)!=report['checkpoint_sha256']:raise ValueError('Weak checkpoint changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);model=weak_model();model.load_state_dict(state['state_dict']);model.eval()
    x=np.clip((np.load(cache)['features'].astype(np.float32)-state['mean'].numpy())/state['scale'].numpy(),-10,10)
    with torch.inference_mode():scores=np.concatenate([model(torch.from_numpy(b))[0].sigmoid().numpy() for b in np.array_split(x,max(1,len(x)//128))])
    return starts,ends,scores,report['checkpoint_sha256']


def predict(path,mouse,out,index,mapping,device):
    times,p,selection=priority_predict(out);ws,we,weak,digest=weak_predict(path,mouse,out,index,mapping,device)
    rows=[];thresholds={c:float(selection['selected'][c]['threshold']) if c in selection['selected'] else .5 for c in CLASSES}
    for i,t in enumerate(times):
        end=float(times[i+1]) if i+1<len(times) else index['duration_s'];weak_i=min(int(np.searchsorted(ws,t,side='right')-1),len(ws)-1)
        scores={c:float(p[i,['grooming','digging','rearing'].index(c)]) if c in selection['selected'] else float(weak[weak_i,k]) for k,c in enumerate(CLASSES)}
        rows.append(dict(start_s=float(t),end_s=end,scores=scores))
    summary,bouts=summarize_predictions(rows,thresholds,index['duration_s'],index['source_gaps'])
    result=dict(mouse_id=mouse,source=path.name,source_sha256=index['source_sha256'],mapping=mapping,duration_s=index['duration_s'],original_duration_s=index['original_duration_s'],frame_count=index['frame_count'],
        source_gaps=index['source_gaps'],production_ready=False,accuracy=None,accepted_behavior_seconds=None,
        status='unreviewed_model_candidates',thresholds=thresholds,summary=summary,bouts=bouts,windows=rows,candidates=candidate_windows(rows,thresholds,index['duration_s']),
        models=dict(priority_selection_sha256=sha256(V2/'selection.json'),weak_checkpoint_sha256=digest),
        sampling=dict(priority_hz=2,weak_frames_per_2s=16,review='Every source frame within the analysis window'),
        prior_mouse_development=mouse=='745',training_performed=False,statistics_performed=False)
    save_json(out/'predictions.json',result);return result


def render(path,mouse,out,index,mapping,result):
    dest=REPORT/(mouse+'.mp4');partial=REPORT/(mouse+'.partial.mp4');REPORT.mkdir(parents=True,exist_ok=True)
    bg=background(path,index,mapping);track=[];a,b,c,d=mapping['crop_xyxy'];width=c-a;width-=width%2;height=round((d-b)*width/(c-a)/2)*2
    proposer=CageProposer(cv2.resize(bg,(width,height)),mapping)
    # Separate title panel preserves all cage pixels in the review video.
    times=np.array([r['start_s'] for r in result['windows']]);last=None
    with av.open(str(path)) as source,av.open(str(partial),'w',options={'movflags':'+faststart'}) as output:
        stream=source.streams.video[0];stream.thread_type='AUTO';enc=output.add_stream('libx264',rate=30);enc.width=width;enc.height=height+96;enc.pix_fmt='yuv420p';enc.options={'crf':'28','preset':'veryfast'}
        from fractions import Fraction
        enc.time_base=Fraction(1,60000);enc.codec_context.time_base=Fraction(1,60000);enc.codec_context.max_b_frames=0
        for i,frame in enumerate(source.decode(stream)):
            if i>=index['frame_count']:break
            timing=index['frames'][i]
            if frame.pts!=timing['pts']:raise ValueError('Review source alignment changed')
            raw=frame.to_ndarray(format='bgr24');crop=crop_frame(raw,mapping);t=timing['start_s'];f=proposer.frame(crop,t-last if last is not None else 0);last=t;track.append(dict(start_s=t,end_s=timing['end_s'],**f))
            canvas=np.full((height+96,width,3),25,np.uint8);canvas[96:]=crop
            if f['bbox_source']:
                x1,y1,x2,y2=f['bbox_source'];cv2.rectangle(canvas,(round((x1-a)/(c-a)*width),96+round((y1-b)/(d-b)*height)),(round((x2-a)/(c-a)*width),96+round((y2-b)/(d-b)*height)),(90,220,170) if f['status']=='proposed' else (40,190,245),1)
            r=result['windows'][min(len(times)-1,max(0,int(np.searchsorted(times,t,side='right')-1)))];scores=r['scores']
            cv2.putText(canvas,f'Mouse {mouse}   {int(t)//60:02}:{t%60:04.1f} / 20:00   Body: {f["status"]}',(8,18),0,.42,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(canvas,'EXPERIMENTAL MODEL CANDIDATES - UNREVIEWED',(8,36),0,.39,(110,195,245),1,cv2.LINE_AA)
            for k,cname in enumerate(CLASSES):
                label={'gnawing_nonfood':'Gnawing'}.get(cname,cname.title());val=scores[cname];color=(90,220,170) if val>=result['thresholds'][cname] else (170,170,170)
                cv2.putText(canvas,f'{label} {val:.2f}',(8+(k%3)*(width//3),57+(k//3)*19),0,.38,color,1,cv2.LINE_AA)
            if i==0:cv2.imwrite(str(REPORT/(mouse+'.jpg')),canvas)
            encoded=av.VideoFrame.from_ndarray(canvas,format='bgr24');encoded.pts=round(t*60000);encoded.time_base=Fraction(1,60000)
            for packet in enc.encode(encoded):output.mux(packet)
            if i%9000==0:print(mouse,'review frames',i,'/',index['frame_count'],flush=True)
        for packet in enc.encode():output.mux(packet)
    if len(track)!=index['frame_count']:raise ValueError('Incomplete annotated video')
    partial.replace(dest)
    with (out/'tracking.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(track[0]));writer.writeheader();writer.writerows(track)
    seconds={s:sum(r['end_s']-r['start_s'] for r in track if r['status']==s) for s in ['proposed','ambiguous','edge','missing','camera_motion']}
    result['tracking']=dict(status_seconds=seconds,proposal_percent=100*seconds['proposed']/index['duration_s'],accuracy=None)
    result['review_sha256']=sha256(dest);save_json(out/'predictions.json',result)
    # Model-blind systematic body checks, plus behavior candidates for qualitative review.
    audit=[]
    for t in np.linspace(10,index['duration_s']-10,12):
        i=int(np.searchsorted([r['start_s'] for r in track],t));audit.append(dict(frame_id=i,**track[i]))
    save_json(out/'tracking-audit.json',audit)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='+');parser.add_argument('--phase',choices=['features','predict','render','all'],default='all');args=parser.parse_args()
    device='mps' if torch.backends.mps.is_available() else 'cpu';print('Inference device:',device,flush=True)
    RUN.mkdir(parents=True,exist_ok=True)
    # One encoder works in queue order while the GPU prepares the next mouse.
    # No simultaneous writes to a mouse's artifacts; failures propagate before exit.
    futures=[]
    with ThreadPoolExecutor(max_workers=1) as encoder:
        selected=set(args.only or MAPPINGS)
        if selected-set(MAPPINGS):raise ValueError('A reviewed cage mapping is required for every selected mouse.')
        for path in select_sources(ROOT/'stereotypy_videos',selected):
            mouse,out,index,mapping=inputs(path)
            if args.phase in ('features','all'):features(path,mouse,out,index,mapping,device)
            if args.phase in ('predict','all'):result=predict(path,mouse,out,index,mapping,device)
            else:result=json.loads((out/'predictions.json').read_text()) if (out/'predictions.json').exists() else None
            if args.phase in ('render','all'):futures.append((mouse,encoder.submit(render,path,mouse,out,index,mapping,result)))
            print(mouse,args.phase,'inference complete',flush=True)
        for mouse,future in futures:
            future.result();print(mouse,'annotated review complete',flush=True)

if __name__=='__main__':main()
