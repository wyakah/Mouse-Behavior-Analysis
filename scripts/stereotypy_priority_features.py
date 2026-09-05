"""Cache DINOv2 whole-cage and isolated-mouse features for priority behaviors.

Source-video splits and label authorities are retained. No predicted labels enter
training and local user recordings remain unlabeled development data.
"""
import os, sys, json, argparse, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
import cv2, numpy as np, pandas as pd, torch
from transformers import AutoModel
from stereotypy.training import sha256,save_json
from stereotypy.priority_training import labgym_labels
torch.set_num_threads(4)
CLASSES=['grooming','digging','rearing']
RECIPE='dino-dual-view-small-224-fp32-2hz-stratified-v2'


def letterbox(image):
    h,w=image.shape;scale=224/max(h,w)
    small=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    out=np.full((224,224),127,np.uint8);y=(224-small.shape[0])//2;x=(224-small.shape[1])//2
    out[y:y+small.shape[0],x:x+small.shape[1]]=small
    return out


def mouse_view(gray,background):
    # Independent per-video background; no class labels involved.
    mask=((gray.astype(float)<background.astype(float)-15)&(gray<145)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    h,w=gray.shape
    ids=[i for i in range(1,n) if .003*h*w<stats[i,4]<.28*h*w]
    if not ids:return gray,np.zeros(7,np.float32)
    i=max(ids,key=lambda i:stats[i,4]);x,y,bw,bh,area=stats[i];cx,cy=centers[i]
    pad=round(.15*max(bw,bh));a,b,c,d=max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)
    isolated=np.where(labels==i,gray,0).astype(np.uint8)[b:d,a:c]
    return isolated,np.array([cx/w,cy/h,bw/w,bh/h,area/(w*h),bh/max(bw,1),1],np.float32)


def entries():
    data=ROOT/'research/stereotypy-round2';rows=[]
    for original_split in ['train','test']:
        for path in sorted((data/'cbas-data'/original_split).glob('*.mp4')):
            number=int(path.stem.split('_')[-1])
            split='test' if original_split=='test' else 'validation' if number in (3,9,14) else 'train'
            rows.append(dict(id=f'cbas-{original_split}-{number}',path=str(path),dataset='cbas',split=split,
                group=f'cbas-{original_split}-{number}',annotation=str(path.with_name(path.stem+'_labels.csv')),
                authority='CBAS published per-frame human annotations; original animal identities unavailable'))
    legacy=json.loads((ROOT/'outputs/stereotypy-training/v1/dataset.json').read_text())
    for key,s in legacy['sources'].items():
        if s['dataset'] not in ('mit','local'):continue
        win=[w for w in legacy['windows'] if w['source']==key]
        split=win[0]['split']
        if s['dataset']=='local':split='development'
        elif split=='test':split='legacy_test'
        rows.append(dict(id=key.replace(':','-'),path=s['path'],dataset=s['dataset'],split=split,
            group=win[0]['group'],crop=s['crop'],windows=win,
            authority='MIT source labels' if s['dataset']=='mit' else 'Unlabeled local development recording'))
        if s.get('source_index'):rows[-1]['source_index']=s['source_index']
    lab=data/'labgym-examples/mouse behavior examples'
    for folder in sorted(lab.iterdir()):
        if not folder.is_dir():continue
        name=folder.name
        labels=labgym_labels(name)
        if max(labels)<0:continue
        # Foraging is not equivalent to digging; it remains unknown for that head.
        for path in sorted(folder.glob('*.avi')):
            rows.append(dict(id='labgym-'+hashlib.sha256(str(path).encode()).hexdigest()[:16],path=str(path),
                dataset='labgym',split='train',group='labgym-parent-unknown',labels=labels,
                authority='LabGym curated example label; training only because original mouse/source IDs unavailable'))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='+');args=parser.parse_args()
    out=ROOT/'outputs/stereotypy-training/v2';out.mkdir(parents=True,exist_ok=True)
    all_entries=entries();save_json(out/'protocol.json',dict(recipe=RECIPE,classes=CLASSES,
        validation='CBAS train clips 3,9,14 and existing MIT validation dates',
        test='Official CBAS four test files. Previously observed MIT test retained as legacy diagnostic.',
        independence_limit='CBAS files concatenate behavior examples; original individual/segment provenance unavailable. File-disjoint, not proven animal-disjoint.',
        target=dict(accuracy=.90,balanced_accuracy=.90,precision=.90,recall=.90),
        test_used_for_selection=False,production_ready=False))
    selected=[r for r in all_entries if not args.only or r['dataset'] in args.only]
    device='mps' if torch.backends.mps.is_available() else 'cpu';dtype=torch.float32
    model=AutoModel.from_pretrained('facebook/dinov2-small',revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',local_files_only=True).eval().to(device=device,dtype=dtype)
    mean=torch.tensor([.485,.456,.406],device=device,dtype=dtype)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device,dtype=dtype)[None,:,None,None]
    def encode(images):
        outputs=[]
        for start in range(0,len(images),16):
            x=torch.from_numpy(np.stack(images[start:start+16])).to(device=device,dtype=dtype)[:,None].repeat(1,3,1,1)/255
            with torch.inference_mode():z=model((x-mean)/std).last_hidden_state[:,0,:].float().cpu().numpy()
            outputs.append(z)
        return np.concatenate(outputs)
    records=[];lab_images=[];lab_items=[]
    def checkpoint():
        catalog=out/'feature-catalog.json'
        old=json.loads(catalog.read_text()) if catalog.exists() else []
        combined={r['id']:r for r in old if r.get('recipe')==RECIPE};combined.update({r['id']:r for r in records})
        save_json(catalog,list(combined.values()))
    def flush_lab():
        if not lab_images:return
        encoded=encode(lab_images);offset=0
        for info,dest,ids,ys,geometries,cuts,fps in lab_items:
            z=encoded[offset:offset+len(ids)];offset+=len(ids)
            np.savez_compressed(dest,x=np.c_[z,z].astype(np.float16),geometry=np.asarray(geometries,np.float32),
                labels=np.asarray(ys,np.int8),frame_ids=ids,times=ids/fps,cuts=np.asarray(cuts),fps=fps)
            records.append(info)
        checkpoint();print('Batched LabGym:',len(records),'completed in this pass',flush=True)
        lab_images.clear();lab_items.clear()
    for number,row in enumerate(selected):
        path=Path(row['path']);fingerprint=sha256(path)
        annotation_hash=sha256(row['annotation']) if row.get('annotation') else ''
        index_hash=sha256(row['source_index']) if row.get('source_index') else ''
        signature=hashlib.sha256((RECIPE+fingerprint+annotation_hash+index_hash+json.dumps(row,sort_keys=True)).encode()).hexdigest()
        dest=out/'dino-features'/(signature+'.npz');dest.parent.mkdir(exist_ok=True)
        info=dict(**row,sha256=fingerprint,feature_path=str(dest),recipe=RECIPE)
        if row.get('annotation'):info['annotation_sha256']=sha256(row['annotation'])
        if dest.exists():records.append(info);continue
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if row.get('source_index'):
            index=json.loads(Path(row['source_index']).read_text())
            if index['source_sha256']!=fingerprint:raise ValueError('Source PTS index mismatch')
            total=index['frame_count']
        if row['dataset']=='labgym':
            ids=np.unique(np.linspace(0,total-1,4).astype(int));ys=np.tile(row['labels'],(len(ids),1))
        elif row['dataset']=='mit':
            ids=[];ys=[]
            for win in row['windows']:
                for t in np.linspace(win['start_s'],win['end_s'],4,endpoint=False):
                    ids.append(int(t*fps));ys.append([win['labels'][0],-1,win['labels'][3]])
            ids=np.array(ids);ys=np.array(ys)
            order=np.argsort(ids);ids,ys=ids[order],ys[order]
        else:
            ids=np.arange(0,total,max(1,round(fps/2)))
            ys=np.full((len(ids),3),-1)
            if row['dataset']=='cbas':
                table=pd.read_csv(row['annotation'])
                if len(table)!=total:raise ValueError('Annotation/frame count mismatch')
                ys=table[CLASSES].to_numpy()[ids]
                # All-zero label rows without an explicit source category are unknown.
                known=table.drop(columns=[c for c in table if c.startswith('Unnamed')]).sum(axis=1).to_numpy()[ids]>0
                ys[~known]=-1
                if row['split']=='train':
                    # Stratify training only; preserve full evaluation sampling.
                    rng=np.random.default_rng(20260905)
                    columns=[c for c in table if not c.startswith('Unnamed')]
                    selected_indices=set()
                    for col in columns:
                        candidates=np.flatnonzero(table[col].to_numpy()[ids]>0)
                        centers=rng.choice(candidates,min(12,len(candidates)),replace=False)
                        for center in centers:
                            selected_indices.update(range(max(0,center-2),min(len(ids),center+3)))
                    selected_indices=np.array(sorted(selected_indices),int)
                    ids,ys=ids[selected_indices],ys[selected_indices]
        crop=row.get('crop',[0,0,int(cap.get(3)),int(cap.get(4))]);a,b,c,d=crop
        backgrounds=[]
        if row['dataset']!='labgym':
            for frame_id in np.linspace(0,total-1,32).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_id));ok,im=cap.read()
                if ok:backgrounds.append(cv2.resize(cv2.cvtColor(im[b:d,a:c],cv2.COLOR_BGR2GRAY),(320,240)))
            background=np.percentile(backgrounds,90,axis=0).astype(np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        pending=[];features=[];geometries=[];cuts=[];previous=None;cursor=0;decoded=0
        while cap.grab():
            if cursor<len(ids) and decoded==ids[cursor]:
                ok,im=cap.retrieve()
                if not ok:raise ValueError('Frame decode failure')
                gray=cv2.cvtColor(im[b:d,a:c],cv2.COLOR_BGR2GRAY)
                if row['dataset']=='labgym':body=gray;geom=np.zeros(7,np.float32)
                else:body,geom=mouse_view(cv2.resize(gray,(320,240)),background)
                full=letterbox(gray);focus=letterbox(body)
                # Cuts / sparse-source gaps prevent temporal context mixing.
                cut=(cursor==0 or ids[cursor]-ids[cursor-1]>fps or
                     previous is not None and np.abs(full.astype(float)-previous).mean()>40)
                cuts.append(cut);previous=full.astype(float)
                geometries.append(geom);pending.extend([full] if row['dataset']=='labgym' else [full,focus]);cursor+=1
                if len(pending)>=128:z=encode(pending);features.extend(np.concatenate([z,z],axis=1) if row['dataset']=='labgym' else z.reshape(-1,768));pending=[]
            decoded+=1
        if pending and row['dataset']!='labgym':
            z=encode(pending);features.extend(np.concatenate([z,z],axis=1) if row['dataset']=='labgym' else z.reshape(-1,768))
        cap.release()
        if cursor!=len(ids) or row['dataset']!='mit' and decoded!=total:raise ValueError(f'Frame indexing failed {row["id"]}: {cursor}/{len(ids)} {decoded}/{total}')
        if row['dataset']=='labgym':
            lab_images.extend(pending);lab_items.append((info,dest,ids,ys,geometries,cuts,fps))
            if len(lab_images)>=128:flush_lab()
            continue
        times=ids/fps
        if row.get('source_index'):
            index=json.loads(Path(row['source_index']).read_text())
            if index['source_sha256']!=fingerprint:raise ValueError('Source PTS index mismatch')
            times=np.array([index['frames'][i]['start_s'] for i in ids])
            for j in range(1,len(times)):
                if any(a<times[j] and b>times[j-1] for a,b in index['source_gaps']):cuts[j]=True
        np.savez_compressed(dest,x=np.asarray(features,np.float16),geometry=np.asarray(geometries,np.float32),
            labels=np.asarray(ys,np.int8),frame_ids=ids,times=times,cuts=np.asarray(cuts),fps=fps)
        records.append(info)
        catalog=out/'feature-catalog.json'
        old=json.loads(catalog.read_text()) if catalog.exists() else []
        combined={r['id']:r for r in old if r.get('recipe')==RECIPE};combined.update({r['id']:r for r in records})
        save_json(catalog,list(combined.values()))
        if row['dataset']!='labgym' or number%100==0:print(row['id'],len(ids),'samples',device,flush=True)
    flush_lab()
    # Include completed caches from all parts; successive --only runs are composable.
    catalog=out/'feature-catalog.json'
    old=json.loads(catalog.read_text()) if catalog.exists() else []
    combined={r['id']:r for r in old if r.get('recipe')==RECIPE};combined.update({r['id']:r for r in records})
    save_json(catalog,list(combined.values()))
    expected={r['id'] for r in all_entries}
    if expected==set(combined):
        save_json(out/'extraction-complete.json',dict(recipe=RECIPE,catalog_sha256=sha256(catalog),sources=len(combined),expected_sources=len(expected)))
    print('Completed',len(records),'sources',flush=True)


if __name__=='__main__':main()
