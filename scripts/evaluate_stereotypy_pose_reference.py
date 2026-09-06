"""Evaluate visible human landmarks for raw/gamma/CLAHE on identical frames.

Reports availability separately from error conditional on a returned landmark.
Unlabeled/occluded reference points never count as prediction errors or successes.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from stereotypy.robustness import LANDMARKS
from stereotypy.training import save_json,sha256


def evaluate(reference):
    doc=json.loads(reference.read_text())
    if not str(doc.get('annotator') or '').strip():raise ValueError('Human annotator required')
    template=json.loads((ROOT/'reports/stereotypy-robustness/reference-template.json').read_text())
    truth={(f['mouse_id'],f['frame_id']):f for f in template['frames']};cache={};results={mode:{p:dict(visible_reference_points=0,available_predictions=0,errors=[]) for p in LANDMARKS} for mode in ['raw','gamma','clahe']};seen=set()
    for frame in doc['frames']:
        key=(frame['mouse_id'],frame['frame_id'])
        if key in seen or key not in truth or frame['source_sha256']!=truth[key]['source_sha256']:raise ValueError('Duplicate or mismatched source frame')
        seen.add(key);mouse=key[0]
        if mouse not in cache:
            folder=ROOT/'outputs/stereotypy-robustness'/mouse;npz=np.load(folder/'pose.npz');variants=json.loads((folder/'variants.json').read_text())
            rec=json.loads(str(npz['records']));lookup={(rec[v['sample']]['frame_id'],v['mode']):p for v,p in zip(variants,npz['variant_poses'])}
            cache[mouse]=(list(npz['parts']),lookup)
        parts,lookup=cache[mouse]
        for name,p in frame['landmarks'].items():
            if name not in LANDMARKS:raise ValueError('Unexpected landmark')
            if p is None or p.get('visibility')=='occluded':continue
            if p.get('visibility')!='visible' or not np.isfinite([p.get('x',np.nan),p.get('y',np.nan)]).all() or not (0<=p['x']<truth[key]['width'] and 0<=p['y']<truth[key]['height']):raise ValueError('Invalid visible coordinate')
            for mode in results:
                item=results[mode][name];item['visible_reference_points']+=1
                pose=lookup.get((key[1],mode));point=pose[parts.index(name)] if pose is not None else None
                if point is not None and np.isfinite(point).all() and point[2]>=.6 and 0<=point[0]<truth[key]['width'] and 0<=point[1]<truth[key]['height']:
                    item['available_predictions']+=1;item['errors'].append(float(np.linalg.norm(point[:2]-[p['x'],p['y']])))
    if not sum(r['visible_reference_points'] for r in results['raw'].values()):raise ValueError('No visible human reference points; accuracy is unavailable')
    for bypart in results.values():
        for item in bypart.values():
            errors=item.pop('errors');item['median_error_native_pixels']=float(np.median(errors)) if errors else None
            item['p90_error_native_pixels']=float(np.percentile(errors,90)) if errors else None
            item['available_fraction']=item['available_predictions']/item['visible_reference_points'] if item['visible_reference_points'] else None
    return dict(reference_sha256=sha256(reference),annotator=doc['annotator'],cutoff=.6,results=results,
                caveat='Development frames already inspected; errors conditional on availability. This is not a blinded independent behavioral accuracy benchmark.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new output path')
    save_json(a.out,evaluate(a.reference))
