"""Reproducible five-mouse pose/motion ablation; preserves the published baseline.

No ground truth is inferred from model predictions. Frozen video heads are rerun
from verified cached features; pose is sampled at 5 Hz, motion at source rate.
"""
import os,sys,json,csv,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
for k,v in {'MPLCONFIGDIR':'.cache/matplotlib','TORCH_HOME':'.cache/torch','HF_HOME':'.cache/huggingface','XDG_CACHE_HOME':'.cache'}.items():os.environ[k]=str(ROOT/v)
os.environ['DLC_LIGHT']='True'
import av,cv2,numpy as np,torch
from stereotypy.robustness import *
from stereotypy.training import save_json,sha256
from stereotypy.batch import select_sources
from analyze_stereotypy_batch import MAPPINGS,inputs,priority_predict,weak_predict,predict

OUT=ROOT/'outputs/stereotypy-robustness';REPORT=ROOT/'reports/stereotypy-robustness'


def setup():
    import deeplabcut as dlc
    import deeplabcut.pose_estimation_pytorch as pep
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    cfg=PoseConfig.build_for_superanimal_inference('superanimal_quadruped',model_name='hrnet_w32',detector_name='fasterrcnn_resnet50_fpn_v2',max_individuals=1,device=device)
    checkpoint=Path(dlc.__file__).parent/'modelzoo/checkpoints/superanimal_quadruped_hrnet_w32.pt'
    runner=pep.get_pose_inference_runner(cfg,checkpoint,batch_size=16,device=device,max_individuals=1)
    return runner,list(cfg.metadata.bodyparts),dict(device=device,checkpoint_sha256=sha256(checkpoint),finetuned=False)


def infer(runner,parts,images,boxes):
    contexts=[]
    for im,box in zip(images,boxes):
        x1,y1,x2,y2=pad_box(box,im.shape)
        contexts.append(dict(bboxes=np.array([[x1,y1,x2-x1,y2-y1]],float),bbox_scores=np.ones(1)))
    predictions=runner.inference(list(zip(images,contexts)))
    if len(predictions)!=len(images):raise ValueError('Pose frame alignment failed')
    return [np.asarray(p['bodyparts'])[0] for p in predictions]


def run(path,runner,parts,model,limit=None):
    mouse,baseline,index,mapping=inputs(path);out=OUT/mouse;out.mkdir(parents=True,exist_ok=True)
    dest=out/'pose.npz';duration=min(index['duration_s'],limit or 1200)
    a,b,c,d=mapping['crop_xyxy'];records=[];poses=[];motion=[];variants=[];variant_poses=[]
    # Rerun frozen heads without modifying the saved baseline prediction artifact.
    vt,vp,selection=priority_predict(baseline)
    ws,we,wp,weak_hash=weak_predict(path,mouse,baseline,index,mapping,model['device'])
    old=json.loads((baseline/'predictions.json').read_text());rows=old['windows']
    delta=max(abs(float(vp[i,k])-rows[i]['scores'][name]) for i in range(len(vt)) for k,name in enumerate(['grooming','digging','rearing']))
    if delta>1e-5:raise ValueError('Frozen priority baseline failed reproduction')
    # Check CLASSES ordering rather than trusting the displayed order.
    from stereotypy.training import CLASSES
    weak_delta=max(abs(float(wp[min(int(np.searchsorted(ws,r['start_s'],side='right')-1),len(ws)-1),CLASSES.index(name)])-r['scores'][name]) for r in rows for name in CLASSES if name not in selection['selected'])
    if weak_delta>1e-5:raise ValueError('Frozen weak baseline failed reproduction')
    signature=dict(source_sha256=sha256(path),duration_s=duration,mapping=mapping,version=VERSION,pose_model=model,pose_hz=5,
                   implementation_sha256={p:sha256(ROOT/p) for p in ['stereotypy/robustness.py','scripts/stereotypy_robustness_run.py']})
    if dest.exists():
        saved=json.loads((out/'manifest.json').read_text())
        if saved['signature']!=signature:raise ValueError('Existing run has different inputs or implementation; choose a fresh output directory')
        print(mouse,'verified cached robustness pass',flush=True);return
    pending=[];pending_boxes=[];pending_ids=[];reference=None;previous=None;last_t=None
    systematic=set(np.round(np.linspace(0,max(0,duration-1),12)*5).astype(int));sample=0;next_sample=0;decoded=0
    def flush():
        if not pending:return
        pred=infer(runner,parts,pending,pending_boxes)
        for ix,p in zip(pending_ids,pred):poses[ix]=p
        pending.clear();pending_boxes.clear();pending_ids.clear()
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        for i,frame in enumerate(container.decode(stream)):
            if i>=index['frame_count'] or index['frames'][i]['start_s']>=duration:break
            timing=index['frames'][i];t=timing['start_s'];decoded+=1
            if frame.pts!=timing['pts']:raise ValueError('Source PTS mismatch')
            im=cv2.cvtColor(frame.to_ndarray(format='bgr24')[b:d,a:c],cv2.COLOR_BGR2RGB)
            gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            if reference is None:reference=im.copy()
            # Full-rate optical flow; retain magnitude and vertical movement of dark pixels.
            small=cv2.resize(gray,(160,80));dt=t-last_t if last_t is not None else 0
            if previous is not None and 0<dt<.2:
                flow=cv2.calcOpticalFlowFarneback(previous,small,None,.5,2,13,2,5,1.1,0)
                mask=small<min(110,np.percentile(small,25));mask[:15]=False
                mag=np.linalg.norm(flow,axis=2)
                motion.append([t,float(np.percentile(mag[mask],90)) if mask.any() else 0,float(np.median(flow[:,:,1][mask])) if mask.any() else 0])
            previous=small;last_t=t
            if t+1e-6<next_sample:continue
            next_sample+=.2
            box,ambiguous=appearance_box(im);similarity=scene_similarity(reference,im)
            record=dict(start_s=t,frame_id=i,box=box,ambiguous=bool(ambiguous),scene_similarity=similarity)
            records.append(record);poses.append(np.full((len(parts),3),np.nan,np.float32))
            if box is not None:
                pending.append(im);pending_boxes.append(box);pending_ids.append(len(poses)-1)
            if sample in systematic:
                cv2.imwrite(str(out/f'label-{sample:05d}.jpg'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
                if box is not None:
                    enhanced=[enhance(im,mode) for mode in ['raw','gamma','clahe']]
                    pp=infer(runner,parts,enhanced,[box]*3)
                    for mode,p in zip(['raw','gamma','clahe'],pp):
                        variants.append(dict(sample=sample,start_s=t,mode=mode,**pose_evidence(p,parts,im.shape)));variant_poses.append(p)
            sample+=1
            if len(pending)>=32:flush()
            if sample%500==0:print(mouse,'pose samples',sample,'/',round(duration*5),flush=True)
    flush()
    np.savez_compressed(dest,poses=np.array(poses),parts=parts,records=np.array(json.dumps(records)),motion=np.array(motion),variant_poses=np.array(variant_poses))
    save_json(out/'variants.json',variants)
    save_json(out/'manifest.json',dict(signature=signature,decoded_frames=decoded,pose_samples=sample,priority_max_delta=delta,weak_max_delta=weak_delta,reference_labels=0,training_performed=False))
    print(mouse,'complete',decoded,'source frames',sample,'pose samples',flush=True)


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='+',default=list(MAPPINGS));parser.add_argument('--limit',type=float);args=parser.parse_args()
    if args.limit:OUT=ROOT/'outputs/stereotypy-robustness-smoke'
    torch.set_num_threads(4);cv2.setNumThreads(2)
    runner,parts,model=setup()
    for path in select_sources(ROOT/'stereotypy_videos',args.only):run(path,runner,parts,model,args.limit)
if __name__=='__main__':main()
