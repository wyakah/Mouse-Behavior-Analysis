"""Export explicitly corrected cage-relative landmarks to DeepLabCut tables.

No automatic points, guessed hidden anatomy, or implicit train/test assignment.
"""
import argparse,json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from stereotypy.robustness import LANDMARKS
from stereotypy.training import save_json,sha256


def export(reference,output,train,validation):
    if set(train)&set(validation) or not train or not validation:raise ValueError('Specify disjoint training and validation mouse IDs')
    doc=json.loads(reference.read_text());template=json.loads((ROOT/'reports/stereotypy-robustness/reference-template.json').read_text())
    trusted={(f['mouse_id'],f['frame_id']):f for f in template['frames']}
    if not str(doc.get('annotator') or '').strip():raise ValueError('Human annotator is required')
    groups={};split=[];seen=set()
    for frame in doc['frames']:
        key=(frame['mouse_id'],frame['frame_id']);truth=trusted.get(key)
        if key in seen or truth is None or frame['source_sha256']!=truth['source_sha256']:raise ValueError('Duplicate or unrecognized reference frame')
        seen.add(key)
        if frame['mouse_id'] not in set(train)|set(validation):continue
        points=frame['landmarks']
        if set(points)!=set(LANDMARKS):raise ValueError('Landmark schema differs')
        if any(v is None for v in points.values()):continue # incomplete review remains unreviewed
        coords=[]
        for name in LANDMARKS:
            p=points[name]
            if p.get('visibility')=='occluded':coords.extend([np.nan,np.nan])
            elif p.get('visibility')=='visible' and np.isfinite([p.get('x',np.nan),p.get('y',np.nan)]).all() and 0<=p['x']<truth['width'] and 0<=p['y']<truth['height']:
                coords.extend([p['x'],p['y']])
            else:raise ValueError('Invalid reference coordinate')
        if np.isnan(coords).all():continue
        mouse=frame['mouse_id'];name=f'img{frame["frame_id"]:06d}.jpg';index=('labeled-data',mouse,name)
        groups.setdefault(mouse,[]).append((index,coords,truth['image']))
        split.append(dict(index=index,split='train' if mouse in train else 'validation',source_sha256=truth['source_sha256'],frame_id=frame['frame_id']))
    if not any(s['split']=='train' for s in split) or not any(s['split']=='validation' for s in split):raise ValueError('Both splits need fully reviewed visible poses; no automatic labels will be exported')
    if output.exists():raise ValueError('Choose a new export directory')
    for mouse,rows in groups.items():
        folder=output/'labeled-data'/mouse;folder.mkdir(parents=True)
        for idx,coords,image in rows:shutil.copyfile(ROOT/'reports/stereotypy-robustness'/image,folder/idx[2])
        columns=pd.MultiIndex.from_product([['Researcher'],LANDMARKS,['x','y']],names=['scorer','bodyparts','coords'])
        df=pd.DataFrame([r[1] for r in rows],index=pd.MultiIndex.from_tuples([r[0] for r in rows]),columns=columns)
        df.to_hdf(folder/'CollectedData_Researcher.h5',key='df_with_missing');df.to_csv(folder/'CollectedData_Researcher.csv')
    save_json(output/'split.json',split)
    save_json(output/'manifest.json',dict(reference_sha256=sha256(reference),annotator=doc['annotator'],parts=LANDMARKS,coordinate_space='native cage-crop pixels',
              train_mice=train,validation_mice=validation,training_performed=False,reviewed_frames=len(split),pretrained_model='superanimal_quadruped',architecture='hrnet_w32'))
    print(output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--train',nargs='+',required=True);p.add_argument('--validation',nargs='+',required=True);a=p.parse_args();export(a.reference,a.out,a.train,a.validation)
