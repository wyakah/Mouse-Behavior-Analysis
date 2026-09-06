"""Publish a local five-mouse candidate report and machine-readable exports."""
import csv,json,sys,shutil,uuid
import av,cv2
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stereotypy.training import CLASSES,save_json,sha256
from stereotypy.core import BEHAVIORS,ETHOGRAM_VERSION
from stereotypy.batch import apply_observability
RUN=ROOT/'outputs/stereotypy-five-video';REPORT=ROOT/'reports/stereotypy-five-video'


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    records=[];summaries=[];segments=[];windows=[]
    audit=json.loads((RUN/'qualitative-audit.json').read_text())
    decisions={r['mouse_id']:r['localization_approved'] for r in audit['body_spot_checks']}
    if set(decisions)!=set(['710','711','712','743','745']):raise ValueError('Complete the five-mouse localization audit first')
    for mouse in ['710','711','712','743','745']:
        r=json.loads((RUN/mouse/'predictions.json').read_text())
        if sha256(ROOT/'stereotypy_videos'/r['source'])!=r['source_sha256']:raise ValueError('Source changed before publication')
        if r['duration_s']!=1200 or r['frame_count']!=36000 or not r.get('tracking') or sha256(REPORT/(mouse+'.mp4'))!=r['review_sha256']:raise ValueError('Incomplete first-20-minute review')
        if cv2.imread(str(REPORT/(mouse+'.jpg'))) is None:raise ValueError('Missing or unreadable review poster')
        source=json.loads((RUN/mouse/'analysis-index.json').read_text())
        with av.open(str(REPORT/(mouse+'.mp4'))) as container:
            stream=container.streams.video[0]
            packets=[p for p in container.demux(stream) if p.pts is not None]
            if len(packets)!=36000 or stream.frames!=36000:raise ValueError('Annotated video frame count differs')
            if any(abs(float(p.pts*stream.time_base)-f['start_s'])>1e-6 for p,f in zip(packets,source['frames'])):raise ValueError('Annotated video timestamps differ from source')
            if abs(float((packets[-1].pts+packets[-1].duration)*stream.time_base)-1200)>1e-6:raise ValueError('Annotated video endpoint differs')
        r['review_verified']=dict(frames=36000,source_timestamps_match=True,end_s=1200)
        with (RUN/mouse/'tracking.csv').open() as handle:
            r=apply_observability(r,list(csv.DictReader(handle)),decisions[mouse])
        sid=uuid.uuid5(uuid.NAMESPACE_URL,r['source_sha256']+'first-1200-seconds-v1').hex
        session_path=ROOT/'labeling/stereotypy'/(sid+'.json')
        if not session_path.exists():
            now=datetime.now(timezone.utc).isoformat()
            ids=dict(animal_id=mouse,sex='unknown',genotype='',session_id='twenty-minute-'+mouse,apparatus_id='Not recorded')
            save_json(session_path,dict(id=sid,video='stereotypy_videos/'+r['source'],**ids,view='side',created_at=now,
                video_manifest=json.loads((RUN/mouse/'source-index.json').read_text()),
                metadata_history=[dict(changed_at=now,**ids)],start_s=0,end_s=1200,merge_gap_s=0,
                ethogram_version=ETHOGRAM_VERSION,ethogram_status='draft',definitions=dict(BEHAVIORS),
                cage_mapping=r['mapping'],cage_revision=1,cage_history=[dict(revision=1,changed_at=now,mapping=r['mapping'])],
                revisions=[dict(revision=0,created_at=now,annotator_id=None,annotations=[],start_s=0,end_s=1200,merge_gap_s=0)]))
        r['reviewer_session_id']=sid
        records.append(r)
        summaries.extend(dict(mouse_id=mouse,analysis_seconds=r['duration_s'],**s) for s in r['summary'])
        segments.extend(dict(mouse_id=mouse,**s,status='unreviewed_model_candidate') for s in r['bouts'])
        windows.extend(dict(mouse_id=mouse,start_s=w['start_s'],end_s=w['end_s'],
            eligible_seconds=max(0,w['end_s']-w['start_s']-sum(max(0,min(b,w['end_s'])-max(a,w['start_s'])) for a,b in r['excluded_intervals'])),
            localization_audit_passed=r['localization_audit_passed'],**w['scores']) for w in r['windows'])
    save_json(REPORT/'results.json',dict(videos=records,statistics_performed=False,training_performed=False,production_ready=False,
        implementation_sha256={name:sha256(ROOT/name) for name in ['scripts/analyze_stereotypy_batch.py','scripts/report_stereotypy_batch.py','stereotypy/batch.py','stereotypy/window.py']}))
    write_csv(REPORT/'candidate-summary.csv',summaries);write_csv(REPORT/'candidate-segments.csv',segments);write_csv(REPORT/'model-scores.csv',windows)
    audit_path=RUN/'qualitative-audit.json'
    if audit_path.exists():shutil.copyfile(audit_path,REPORT/'qualitative-audit.json')
    else:save_json(REPORT/'qualitative-audit.json',dict(status='pending',accuracy=None))
    shutil.copyfile(ROOT/'static/stereotypy-batch-report.html',REPORT/'index.html')
    print(json.dumps([{k:r[k] for k in ['mouse_id','duration_s','frame_count','tracking']} for r in records],indent=2))

if __name__=='__main__':main()
