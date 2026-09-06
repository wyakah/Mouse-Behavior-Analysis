"""Bounded incremental inference with the frozen models and original context.

Publish eight seconds at a time, retaining three future priority samples for
the temporal heads. Scores are final before a section is emitted.
"""
import json,os
import av,cv2,numpy as np,torch,joblib
from stereotypy.incremental import sections

def prepare(api,path,out,index,mapping,device):
    selection=json.loads((api.V2/'selection.json').read_text())
    thresholds={c:float(selection['selected'][c]['threshold']) if c in selection['selected'] else .5 for c in api.CLASSES}
    result=dict(windows=[],summary=[],thresholds=thresholds,duration_s=index['duration_s'],original_duration_s=index['original_duration_s'],
        source_gaps=index['source_gaps'],source_sha256=index['source_sha256'],mapping=mapping,frame_count=index['frame_count'],
        production_ready=False,accuracy=None,training_performed=False,statistics_performed=False,
        sampling=dict(priority_hz=2,weak_frames_per_2s=16,publication_seconds=8,priority_future_samples=3,motion_decoder='PyAV',recipe='incremental-native-av-v1'),
        pose_enabled=os.environ.get('STEREOTYPY_POSE','1')!='0',
        models=dict(priority_selection_sha256=api.sha256(api.V2/'selection.json'),weak_checkpoint_sha256=api.sha256(api.V1/'temporal-model.pt')))
    return result,generate(api,path,out,index,mapping,device,selection,result)

def generate(api,path,out,index,mapping,device,selection,result):
    from transformers import AutoModel
    from torchvision.models import mobilenet_v3_large,MobileNet_V3_Large_Weights
    from stereotypy_motion_features import descriptor
    for name,digest in selection['artifact_sha256'].items():
        if name!='prepared-features.npz' and api.sha256(api.V2/name)!=digest:raise ValueError('Frozen model changed')
    report=json.loads((api.V1/'training-report.json').read_text())
    if api.sha256(api.V1/'temporal-model.pt')!=report['checkpoint_sha256']:raise ValueError('Frozen weak model changed')
    prep=joblib.load(api.V2/'preprocessing.joblib');heads=[]
    for k,c in enumerate(['grooming','digging','rearing']):
        chosen=selection['selected'][c]['model']
        if chosen=='temporal':
            state=torch.load(api.V2/'temporal.pt',weights_only=True,map_location='cpu');model=api.make_model(state['input_dim']);model.load_state_dict(state['state_dict'])
        elif chosen=='binary-temporal':
            state=torch.load(api.V2/'binary-temporal.pt',weights_only=True,map_location='cpu');model=api.make_binary_model(state['input_dim']);model.load_state_dict(state['models'][c]['state_dict'])
        else:raise ValueError('Unsupported priority model')
        heads.append((model.eval(),k if chosen=='temporal' else None))
    state=torch.load(api.V1/'temporal-model.pt',weights_only=True,map_location='cpu')
    weak=api.weak_model();weak.load_state_dict(state['state_dict']);weak.eval()
    weak_mean=state['mean'].numpy();weak_scale=state['scale'].numpy()
    dino=AutoModel.from_pretrained('facebook/dinov2-small',revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',local_files_only=True).eval().to(device)
    mobile=mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).eval().to(device);mobile.classifier=torch.nn.Identity()
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    bg=api.background(path,index,mapping,(320,240));a,b,c,d=mapping['crop_xyxy']
    times=np.array([f['start_s'] for f in index['frames']]);ids=np.unique(np.searchsorted(times,np.arange(0,index['duration_s'],.5),side='right')-1)
    step=max(1,round(index['nominal_fps']*.1));motion_ids=np.clip(np.stack([ids-step,ids,ids+step],1),0,len(times)-1)
    starts=np.arange(0,index['duration_s'],2.);ends=np.minimum(starts+2,index['duration_s'])
    weak_ids=np.array([np.searchsorted(times,np.linspace(s,e,16,endpoint=False),side='right')-1 for s,e in zip(starts,ends)])
    pose_runtime=None;pose_error=None;pose_rows=[];pose_done=0
    if os.environ.get('STEREOTYPY_POSE','1')!='0':
        try:
            from stereotypy.pose_signals import PoseRuntime,summarize_pose,infer_or_error
            pose_runtime=PoseRuntime();result['models']['pose']=pose_runtime.metadata;result['sampling']['pose_hz']=10;result['pose_status']='available';result['pose_used_to_change_scores']=False
        except Exception as exc:
            pose_error=str(exc);result['pose_status']='unavailable';result['pose_error']=pose_error;print('Pose unavailable; retaining video classifier:',exc,flush=True)
    pose_ids=np.unique(np.searchsorted(times,np.arange(0,index['duration_s'],.1),side='right')-1) if pose_runtime else np.array([],int)
    wanted_raw=set(ids)|set(weak_ids.ravel())|set(pose_ids);wanted_gray=set(motion_ids.ravel())
    raw={};gray={};visual=[];geometry=[];motion=[];cuts=[];weak_features=[];weak_scores=[];previous=None;decoded=-1
    def encode(images,network,grayscale=False,batch=16):
        values=[]
        for offset in range(0,len(images),batch):
            x=torch.from_numpy(np.stack(images[offset:offset+batch])).to(device).float()/255
            if grayscale:x=x[:,None].repeat(1,3,1,1)
            with torch.inference_mode():
                y=network((x-mean)/std)
                if grayscale:y=y.last_hidden_state[:,0,:]
                values.extend(y.cpu().numpy())
        return np.array(values,np.float16)
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO';frames=iter(container.decode(stream))
        def ensure(last):
            nonlocal decoded
            while decoded<last:
                frame=next(frames);decoded+=1
                if frame.pts!=index['frames'][decoded]['pts']:raise ValueError('Incremental source alignment changed')
                if decoded in wanted_raw or decoded in wanted_gray:
                    im=frame.to_ndarray(format='bgr24')
                    if decoded in wanted_raw:raw[decoded]=im
                    if decoded in wanted_gray:gray[decoded]=cv2.resize(cv2.cvtColor(im[b:d,a:c],cv2.COLOR_BGR2GRAY),(160,120))
        for first,stop,visual_stop,end_s in sections(times[ids],index['duration_s']):
            weak_stop=int(np.searchsorted(starts,end_s))
            pose_stop=int(np.searchsorted(times[pose_ids],end_s)) if pose_runtime else 0
            ensure(max(int(motion_ids[visual_stop-1].max()),int(weak_ids[weak_stop-1].max()),int(pose_ids[pose_stop-1]) if pose_stop else 0))
            if pose_runtime:
                for offset in range(pose_done,pose_stop,32):
                    selected=pose_ids[offset:min(pose_stop,offset+32)]
                    inferred,error=infer_or_error(pose_runtime,[cv2.cvtColor(raw[int(i)][b:d,a:c],cv2.COLOR_BGR2RGB) for i in selected],times[selected])
                    if error:
                        pose_error=error;pose_runtime=None;result['pose_status']='unavailable';result['pose_error']=error
                        print('Pose failed; retaining video classifier:',error,flush=True);break
                    pose_rows.extend(inferred)
                pose_done=pose_stop
            images=[]
            for n in range(len(visual),visual_stop):
                im=cv2.cvtColor(raw[int(ids[n])][b:d,a:c],cv2.COLOR_BGR2GRAY)
                body,g=api.mouse_view(cv2.resize(im,(320,240)),bg);full=api.letterbox(im)
                cuts.append(n==0 or times[ids[n]]-times[ids[n-1]]>1 or previous is not None and np.abs(full.astype(float)-previous).mean()>40 or any(s<times[ids[n]] and e>times[ids[max(0,n-1)]] for s,e in index['source_gaps']))
                previous=full.astype(float);geometry.append(g);images.extend([full,api.letterbox(body)])
                before,now,after=map(int,motion_ids[n]);box=(0,0,160,120)
                if g[-1]>0:
                    cx,cy,bw,bh=g[:4];bw=max(bw,.08)*1.3;bh=max(bh,.08)*1.3
                    box=(max(0,int((cx-bw/2)*160)),max(0,int((cy-bh/2)*120)),min(160,int((cx+bw/2)*160)+1),min(120,int((cy+bh/2)*120)+1))
                valid=before!=now and after!=now and not any(s<times[after] and e>times[before] for s,e in index['source_gaps'])
                motion.append(np.r_[descriptor(gray[before],gray[now],box),descriptor(gray[now],gray[after],box),1.] if valid else np.zeros(385,np.float32))
            if images:visual.extend(encode(images,dino,True).reshape(-1,768))
            requests=weak_ids[len(weak_scores):weak_stop]
            if len(requests):
                unique=np.unique(requests);encoded=encode([api.input_image(raw[int(i)],mapping['crop_xyxy']) for i in unique],mobile,batch=64)
                features=encoded[np.searchsorted(unique,requests)];weak_features.extend(features)
                x=np.clip((features.astype(np.float32)-weak_mean)/weak_scale,-10,10)
                with torch.inference_mode():weak_scores.extend(weak(torch.from_numpy(x))[0].sigmoid().numpy())
            x=np.c_[np.array(visual,np.float32),np.array(geometry,np.float32),np.array(motion,np.float32)]
            scaled=prep['scaler'].transform(x)
            z=np.c_[np.clip(prep['pca'].transform(scaled[:,:775]),-12,12),np.clip(prep['motion_pca'].transform(scaled[:,775:]),-12,12)].astype(np.float32)
            seq=z[api.context_indices(cuts)[first:stop]].astype(np.float16).astype(np.float32);pred=[]
            for model,column in heads:
                with torch.inference_mode():p=model(torch.from_numpy(seq)).sigmoid().numpy()
                pred.append(p[:,column] if column is not None else p)
            pred=np.stack(pred,1);rows=[]
            for offset,n in enumerate(range(first,stop)):
                t=float(times[ids[n]]);wi=min(int(np.searchsorted(starts,t,side='right')-1),len(weak_scores)-1)
                scores={name:float(pred[offset,['grooming','digging','rearing'].index(name)]) if name in selection['selected'] else float(weak_scores[wi][k]) for k,name in enumerate(api.CLASSES)}
                end=float(times[ids[n+1]]) if n+1<len(ids) else index['duration_s']
                row=dict(start_s=t,end_s=end,scores=scores)
                if pose_runtime:
                    pt=times[pose_ids[:pose_done]];lo,hi=np.searchsorted(pt,[t,end])
                    samples=pose_rows[lo:hi];row.update(pose_samples=samples,pose_evidence=summarize_pose(samples))
                elif pose_error:row['pose_error']=pose_error
                rows.append(row)
            # Preserve only the overlap needed by the next section.
            keep=max(0,int(np.searchsorted(times,end_s))-step-2)
            for store in (raw,gray):
                for key in list(store):
                    if key<keep:del store[key]
            if end_s==index['duration_s']:
                api.save_json(out/'pose-evidence.json',dict(samples=pose_rows,model=pose_runtime.metadata if pose_runtime else None,error=pose_error,pose_hz=10,used_to_change_scores=False))
                np.savez_compressed(out/'incremental-features.npz',x=np.array(visual,np.float16),geometry=np.array(geometry,np.float32),motion=np.array(motion,np.float32),cuts=cuts,frame_ids=ids,times=times[ids],weak_features=np.array(weak_features,np.float16))
            print('Scored through',end_s,'seconds; decoded through',round(times[decoded],2),flush=True)
            yield rows
