"""Collect explicit human revisions from the five existing interval-review sessions."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stereotypy.training import save_json,sha256
from stereotypy.core import validate_annotations


def collect():
    published=json.loads((ROOT/'reports/stereotypy-five-video/results.json').read_text())
    reference=json.loads((ROOT/'reports/stereotypy-robustness/reference-template.json').read_text())
    labels=[];annotators=set();provenance=[]
    for video in published['videos']:
        path=ROOT/'labeling/stereotypy'/(video['reviewer_session_id']+'.json');session=json.loads(path.read_text());revision=session['revisions'][-1]
        if not revision['annotations']:continue
        if not revision.get('annotator_id'):raise ValueError('An explicit human annotator is required')
        if session['video_manifest']['source_sha256']!=video['source_sha256'] or sha256(ROOT/session['video'])!=video['source_sha256']:raise ValueError('Source identity changed')
        if session['start_s']!=0 or session['end_s']!=1200:raise ValueError('Reference window must be the first 20 minutes')
        labels.extend(dict(mouse_id=video['mouse_id'],source_sha256=video['source_sha256'],**r) for r in validate_annotations(revision['annotations'],0,1200))
        annotators.add(revision['annotator_id']);provenance.append(dict(mouse_id=video['mouse_id'],session_id=session['id'],revision=revision['revision'],session_sha256=sha256(path)))
    if not labels:raise ValueError('No explicit human behavior annotations exist; no reference exported')
    reference.update(annotator=', '.join(sorted(annotators)),status='human_reference_draft',behaviors=labels,provenance=provenance)
    return reference
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a fresh output path')
    save_json(a.out,collect())
