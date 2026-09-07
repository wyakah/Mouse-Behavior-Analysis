"""Human review queue and reproducible DLC label export."""
from pathlib import Path
from datetime import datetime,timezone
import json,threading,uuid
import cv2,numpy as np,pandas as pd
from flask import Blueprint,request,jsonify,send_file
PARTS=['nose','center','tail_base']
LEGACY_PARTS=['nose','left_ear','right_ear','center','tail_base']
LOCK=threading.Lock()

def validate_annotation(data,entry,profile):
    # Accept already-open five-point editors and existing annotations without discarding their work.
    if set(data.get('points',{})) not in (set(PARTS),set(LEGACY_PARTS)):raise ValueError('Review nose, body center and tail base; use Not visible for hidden points.')
    x1,y1,x2,y2=profile['crop_xyxy']
    for part,p in data['points'].items():
        if p is None:continue
        if not isinstance(p,list) or len(p)!=2 or not np.isfinite(p).all():raise ValueError(f'{part}: provide a finite x,y point or null.')
        if not x1<=p[0]<x2 or not y1<=p[1]<y2:raise ValueError(f'{part}: landmark lies outside the saved arena crop.')
    if not str(data.get('annotator','')).strip():raise ValueError('Enter an annotator name.')
    if data.get('reviewed') is not True:raise ValueError('Confirm that the landmarks were manually reviewed.')
    return {'points':data['points'],'annotator':str(data['annotator']).strip(),'reviewed':True,'reviewed_at':datetime.now(timezone.utc).isoformat(),'source':entry['source'],'source_frame':entry['source_frame']}

def export_labels(root):
    queuepath=root/'labeling/queue.json';queue=json.loads(queuepath.read_text()) if queuepath.exists() else [];annpath=root/'labeling/annotations.json';annotations=json.loads(annpath.read_text()) if annpath.exists() else {}
    reviewed=[e for e in queue if annotations.get(e['id'],{}).get('reviewed')]
    if not reviewed:raise ValueError('No manually reviewed labels yet. Open Review & label first.')
    profile=json.loads((root/'profiles/ethovision_three_chamber.json').read_text());x1,y1,x2,y2=profile['crop_xyxy']
    out=root/'labeling/exports'/('labels-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    groups={};split=[];export_annotations={}
    for e in reviewed:
        ann=validate_annotation(annotations[e['id']],e,profile)
        export_annotations[e['id']]={**annotations[e['id']],'points':{part:ann['points'][part] for part in PARTS}}
        stem=Path(e['source']).stem;name=f'img{e["source_frame"]:06d}.png';folder=out/'labeled-data'/stem;folder.mkdir(parents=True,exist_ok=True)
        im=cv2.imread(str(root/e['image']))
        if im is None:raise ValueError('Missing source label image.')
        cv2.imwrite(str(folder/name),im[y1:y2,x1:x2])
        row=[]
        for part in PARTS:
            pt=ann['points'][part];row.extend([np.nan,np.nan] if pt is None else [pt[0]-x1,pt[1]-y1])
        groups.setdefault(stem,[]).append((('labeled-data',stem,name),row))
        split.append({'index':['labeled-data',stem,name],'split':'validation' if stem.startswith('685') else 'train','annotation_id':e['id'],'source':e['source'],'source_frame':e['source_frame']})
    cols=pd.MultiIndex.from_product([['Researcher'],PARTS,['x','y']],names=['scorer','bodyparts','coords'])
    for stem,rows in groups.items():
        df=pd.DataFrame([r[1] for r in rows],index=pd.MultiIndex.from_tuples([r[0] for r in rows]),columns=cols)
        df.to_csv(out/'labeled-data'/stem/'CollectedData_Researcher.csv');df.to_hdf(out/'labeled-data'/stem/'CollectedData_Researcher.h5',key='df_with_missing')
    (out/'split.json').write_text(json.dumps(split,indent=2));(out/'annotations.json').write_text(json.dumps(export_annotations,indent=2));(out/'profile.json').write_text(json.dumps(profile,indent=2))
    (out/'original_annotations.json').write_text(json.dumps({e['id']:annotations[e['id']] for e in reviewed},indent=2))
    manifest={'reviewed_frames':len(reviewed),'coordinate_space':'Cropped image; native pixels; x offset 190 and y offset 175 for original coordinates.','profile':profile,'parts':PARTS,'source_videos':sorted({e['source'] for e in reviewed}),'split_policy':'Subjects 675 and 678 train; 685 development validation. All three have been inspected already; final independent accuracy needs additional footage.','label_source':'Explicitly reviewed annotations only; no automatic acceptance of model suggestions.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return out,manifest

def register_labeling(app,root_getter):
    def root():return root_getter() if callable(root_getter) else root_getter
    bp=Blueprint('labeling',__name__)
    def queue():
        p=root()/'labeling/queue.json';return json.loads(p.read_text()) if p.exists() else []
    def annotations():
        p=root()/'labeling/annotations.json';return json.loads(p.read_text()) if p.exists() else {}
    @bp.get('/labeling')
    def page():return send_file(root()/'static/labeling.html')
    @bp.get('/api/labeling')
    def get_queue():return jsonify({'proposals':json.loads((root()/'labeling/proposals.json').read_text()) if (root()/'labeling/proposals.json').exists() else {},'entries':queue(),'annotations':annotations(),'parts':PARTS,'profile':json.loads((root()/'profiles/ethovision_three_chamber.json').read_text())})
    @bp.get('/api/labeling/image/<id>')
    def image(id):
        entry=next((e for e in queue() if e['id']==id),None)
        if entry is None:raise ValueError('Unknown label frame.')
        return send_file(root()/entry['image'],mimetype='image/png')
    @bp.post('/api/labeling/<id>')
    def save(id):
        entry=next((e for e in queue() if e['id']==id),None)
        if entry is None:raise ValueError('Unknown label frame.')
        profile=json.loads((root()/'profiles/ethovision_three_chamber.json').read_text());record=validate_annotation(request.get_json(),entry,profile)
        with LOCK:
            all_annotations=annotations();old=all_annotations.get(id)
            if old:
                hist=root()/'labeling/history';hist.mkdir(exist_ok=True);(hist/f'{id}-{uuid.uuid4().hex}.json').write_text(json.dumps(old,indent=2))
            all_annotations[id]=record;tmp=root()/'labeling/annotations.tmp';tmp.write_text(json.dumps(all_annotations,indent=2));tmp.replace(root()/'labeling/annotations.json')
        return jsonify(saved=True)
    @bp.post('/api/labeling/export')
    def export():
        with LOCK:out,manifest=export_labels(root())
        return jsonify(path=str(out.relative_to(root())),**manifest)
    app.register_blueprint(bp)
