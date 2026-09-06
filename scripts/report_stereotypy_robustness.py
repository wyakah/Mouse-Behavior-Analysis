"""Publish paired diagnostics with explicit unknowns, never inferred accuracy."""
import sys,json,csv,shutil
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import av,cv2,numpy as np
from stereotypy.robustness import LANDMARKS,pose_evidence,disagreements,temporal_features
from stereotypy.training import CLASSES,save_json,sha256
RUN=ROOT/'outputs/stereotypy-robustness';REPORT=ROOT/'reports/stereotypy-robustness'


def build(mouse):
    folder=RUN/mouse;meta=json.loads((folder/'manifest.json').read_text());data=np.load(folder/'pose.npz')
    records=json.loads(str(data['records']));poses=data['poses'];parts=list(data['parts']);
    if meta['decoded_frames']!=36000 or len(records)!=6000 or poses.shape!=(6000,len(parts),3):raise ValueError('Incomplete first-20-minute pose pass')
    if any(b['start_s']<=a['start_s'] for a,b in zip(records,records[1:])):raise ValueError('Pose times not increasing')
    mapping=meta['signature']['mapping'];a,b,c,d=mapping['crop_xyxy']
    baseline=json.loads((ROOT/'outputs/stereotypy-five-video'/mouse/'predictions.json').read_text())
    source=ROOT/'stereotypy_videos'/baseline['source']
    if sha256(source)!=meta['signature']['source_sha256']:raise ValueError('Source changed')
    flow=data['motion'];flow_times=flow[:,0] if len(flow) else np.array([])
    oldrows=baseline['windows'];times=np.array([r['start_s'] for r in oldrows]);previous=None;previous_t=None
    result=[];summary=dict(mouse_id=mouse,duration_s=1200,pose_hz=5,pose_samples=len(records),
                         baseline_body_proposed_seconds=baseline['tracking']['status_seconds']['proposed'],
                         scene_flag_seconds=0.,nose_available_seconds=0.,axis_available_seconds=0.,
                         appearance_available_seconds=0.,pose_rearing_disagreement_seconds=0.,
                         upright_video_disagreement_seconds=0.,pose_unknown_seconds=0.)
    behavior=[dict(mouse_id=mouse,behavior=name,baseline_raw_candidate_seconds=next(s['candidate_seconds'] for s in baseline['summary'] if s['behavior']==name),
                   scene_eligible_candidate_seconds=0.,pose_conflict_candidate_seconds=0. if name=='rearing' else None,accepted_seconds=None,accuracy=None) for name in CLASSES]
    review=[]
    for i,(r,p) in enumerate(zip(records,poses)):
        t=r['start_s'];end=records[i+1]['start_s'] if i+1<len(records) else 1200;dt=end-t
        evidence=pose_evidence(p,parts,(d-b,c-a,3))
        # Require both a persistent apparatus match and a nonambiguous body proposal.
        scene_valid=r['scene_similarity'] is not None and r['scene_similarity']>=.45
        body_valid=scene_valid and r['box'] is not None and not r['ambiguous']
        if not body_valid:
            evidence={**evidence,'axial_available':False,'nose_available':False,'upright_support':None,'horizontal_support':None,'center':None,'body_length':None,'angle':None}
        motion=temporal_features(previous,evidence,t-previous_t if previous_t is not None else 0)
        # Keep a box-center proxy separate from anatomical mid-back; no confidence is invented.
        evidence['body_box_center']=[(r['box'][0]+r['box'][2])/2,(r['box'][1]+r['box'][3])/2] if body_valid else None
        motion['box_speed_cage_width_s']=None
        if previous is not None and previous.get('body_box_center') is not None and evidence['body_box_center'] is not None and 0<t-previous_t<=.25:
            motion['box_speed_cage_width_s']=float(np.linalg.norm(np.array(evidence['body_box_center'])-previous['body_box_center'])/(c-a)/(t-previous_t))
        fi=np.searchsorted(flow_times,[t,end]);segment=flow[fi[0]:fi[1]]
        motion['flow_peak_cage_width_s']=float(np.max(segment[:,1])*30/160) if len(segment) and body_valid else None
        motion['flow_upward_cage_width_s']=float(-np.min(segment[:,2])*30/160) if len(segment) and body_valid else None
        # Fine-contact features remain unknown if the relevant landmarks are hidden.
        pts=dict(zip(parts,p));length=evidence['body_length'];floor=mapping['floor_y']-b
        for feature,names in [('forepaw_bedding_distance',['front_left_paw','front_right_paw']),('nose_bedding_distance',['nose'])]:
            valid=[pts[n] for n in names if n in pts and np.isfinite(pts[n]).all() and pts[n][2]>=.6 and 0<=pts[n][0]<c-a and 0<=pts[n][1]<d-b]
            evidence[feature]=float(min(abs(v[1]-floor) for v in valid)/length) if valid and length and body_valid else None
        score=oldrows[max(0,int(np.searchsorted(times,t,side='right')-1))]['scores']
        flags=disagreements(score,baseline['thresholds'],evidence,scene_valid)
        if r['ambiguous']:flags.append('body_ambiguous')
        if not body_valid:flags.append('body_unreliable')
        # Fast motion is a review cue, never an automatic jumping label.
        if motion['ascent_body_lengths_s'] is not None and motion['ascent_body_lengths_s']>2:
            flags.append('rapid_ascent_review')
        if body_valid and motion['flow_peak_cage_width_s'] is not None and motion['flow_peak_cage_width_s']>1:
            flags.append('rapid_motion_review')
        summary['scene_flag_seconds']+=dt*(not scene_valid)
        summary['appearance_available_seconds']+=dt*body_valid
        summary['nose_available_seconds']+=dt*evidence['nose_available']
        summary['axis_available_seconds']+=dt*evidence['axial_available']
        summary['pose_unknown_seconds']+=dt*(not evidence['axial_available'])
        summary['pose_rearing_disagreement_seconds']+=dt*('rearing_pose_disagreement' in flags)
        summary['upright_video_disagreement_seconds']+=dt*('upright_video_disagreement' in flags)
        # Integrate video-window intersections exactly; pose sampling does not shift baseline boundaries.
        j=max(0,int(np.searchsorted(times,t,side='right')-1))
        while j<len(oldrows) and oldrows[j]['start_s']<end:
            row=oldrows[j];overlap=max(0,min(end,row['end_s'])-max(t,row['start_s']))
            for item in behavior:
                name=item['behavior']
                if row['scores'][name]>=baseline['thresholds'][name]:
                    item['scene_eligible_candidate_seconds']+=overlap*body_valid
                    if name=='rearing' and body_valid and evidence['horizontal_support']:
                        item['pose_conflict_candidate_seconds']+=overlap
            j+=1
        result.append(dict(**r,end_s=end,scene_valid=scene_valid,body_valid=body_valid,evidence=evidence,motion=motion,scores=score,flags=flags))
        if any(f in flags for f in ['rearing_pose_disagreement','upright_video_disagreement','rapid_ascent_review','rapid_motion_review','scene_unreliable']) and (not review or t-review[-1]['start_s']>15):
            review.append(dict(start_s=max(0,t-2),end_s=min(1200,t+4),reasons=flags))
        previous=evidence;previous_t=t
    variants=json.loads((folder/'variants.json').read_text());ablation=[]
    for mode in ['raw','gamma','clahe']:
        vr=[r for r in variants if r['mode']==mode]
        ablation.append(dict(mode=mode,sampled_frames=len(vr),nose_available=sum(r['nose_available'] for r in vr),axis_available=sum(r['axial_available'] for r in vr),
                            landmark_error_pixels=None,accuracy=None,selected_for_production=False))
    published=json.loads((ROOT/'reports/stereotypy-five-video/results.json').read_text())
    reviewer=next(r['reviewer_session_id'] for r in published['videos'] if r['mouse_id']==mouse)
    save_json(REPORT/f'{mouse}.json',dict(reviewer_session_id=reviewer,mouse_id=mouse,summary=summary,behavior=behavior,rows=result,variants=ablation,review=review[:60],model=meta['signature']['pose_model'],
                                         baseline_priority_max_delta=meta['priority_max_delta'],baseline_weak_max_delta=meta['weak_max_delta']))
    # Raw unannotated reference frames, deliberately not initialized from model points.
    references=[]
    for image in sorted(folder.glob('label-*.jpg')):
        sample=int(image.stem.split('-')[1]);dst=f'{mouse}-{image.name}';shutil.copyfile(image,REPORT/dst)
        references.append(dict(mouse_id=mouse,frame_id=records[sample]['frame_id'],start_s=records[sample]['start_s'],image=dst,
                               width=c-a,height=d-b,crop_xyxy=[a,b,c,d],source_sha256=meta['signature']['source_sha256'],landmarks={name:None for name in LANDMARKS}))
    return summary,behavior,references,result,poses,parts,source,mapping


def render(mouse,rows,poses,parts,source,mapping):
    dest=REPORT/f'{mouse}.mp4'
    digest=sha256(Path(__file__))
    pose_digest=sha256(RUN/mouse/'pose.npz')
    receipt=REPORT/f'{mouse}-review.json'
    if dest.exists() and receipt.exists() and json.loads(receipt.read_text()).get('implementation_sha256')==digest and json.loads(receipt.read_text()).get('pose_sha256')==pose_digest and json.loads(receipt.read_text()).get('video_sha256')==sha256(dest):return
    partial=REPORT/f'{mouse}.partial.mp4';a,b,c,d=mapping['crop_xyxy'];w=(c-a)//2*2;h=(d-b)//2*2
    index=json.loads((ROOT/'outputs/stereotypy-five-video'/mouse/'analysis-index.json').read_text());times=np.array([r['start_s'] for r in rows]);n=0
    with av.open(str(source)) as container,av.open(str(partial),'w',options={'movflags':'+faststart'}) as output:
        src=container.streams.video[0];src.thread_type='AUTO';enc=output.add_stream('libx264',rate=30);enc.width=w;enc.height=h+80;enc.pix_fmt='yuv420p';enc.options={'crf':'26','preset':'veryfast'}
        enc.time_base=Fraction(1,60000);enc.codec_context.time_base=Fraction(1,60000);enc.codec_context.max_b_frames=0
        for i,frame in enumerate(container.decode(src)):
            if i>=index['frame_count']:break
            f=index['frames'][i];t=f['start_s']
            if frame.pts!=f['pts']:raise ValueError('Source clock mismatch')
            j=max(0,int(np.searchsorted(times,t,side='right')-1));r=rows[j]
            canvas=np.full((h+80,w,3),24,np.uint8);canvas[80:]=cv2.resize(frame.to_ndarray(format='bgr24')[b:d,a:c],(w,h))
            color=(95,215,165) if r['body_valid'] else (80,180,245)
            if r['box']:
                x1,y1,x2,y2=r['box'];cv2.rectangle(canvas,(round(x1*w/(c-a)),80+round(y1*h/(d-b))),(round(x2*w/(c-a)),80+round(y2*h/(d-b))),color,1)
            if r['body_valid']:
                cx,cy=r['evidence']['body_box_center'];cv2.drawMarker(canvas,(round(cx*w/(c-a)),80+round(cy*h/(d-b))),(95,215,165),cv2.MARKER_CROSS,7,1)
                for name,(x,y,q) in zip(parts,poses[j]):
                    if name in LANDMARKS and np.isfinite([x,y,q]).all() and q>=.6 and 0<=x<c-a and 0<=y<d-b:
                        cv2.circle(canvas,(round(x*w/(c-a)),80+round(y*h/(d-b))),2,(0,210,255),-1)
            cv2.putText(canvas,f'Mouse {mouse}  {int(t)//60:02}:{t%60:04.1f} / 20:00',(8,18),0,.45,(240,240,240),1,cv2.LINE_AA)
            cv2.putText(canvas,'EXPERIMENTAL POSE 5 Hz | source video 30 fps',(8,37),0,.38,(100,190,245),1,cv2.LINE_AA)
            text=' | '.join(r['flags']) or 'No rule disagreement (not validated)'
            cv2.putText(canvas,text[:85],(8,57),0,.33,color,1,cv2.LINE_AA)
            cv2.putText(canvas,'Yellow: pose | green cross: box center (not anatomical)',(8,73),0,.32,(180,180,180),1,cv2.LINE_AA)
            vf=av.VideoFrame.from_ndarray(canvas,format='bgr24');vf.pts=round(t*60000);vf.time_base=enc.time_base
            for packet in enc.encode(vf):output.mux(packet)
            n+=1
        for packet in enc.encode():output.mux(packet)
    if n!=36000:raise ValueError('Incomplete review')
    with av.open(str(partial)) as v:
        s=v.streams.video[0];packets=[p for p in v.demux(s) if p.pts is not None]
        if len(packets)!=36000 or any(abs(float(p.pts*s.time_base)-f['start_s'])>1e-6 for p,f in zip(packets,index['frames'])):raise ValueError('Encoded clock mismatch')
        if abs(float((packets[-1].pts+packets[-1].duration)*s.time_base)-1200)>1e-6:raise ValueError('Wrong endpoint')
    partial.replace(dest);save_json(receipt,dict(implementation_sha256=digest,pose_sha256=pose_digest,video_sha256=sha256(dest),frames=n,source_timestamps_match=True,end_s=1200));print(mouse,'review verified',n,flush=True)


def main():
    REPORT.mkdir(parents=True,exist_ok=True);summaries=[];behaviors=[];refs=[]
    for mouse in ['710','711','712','743','745']:
        summary,behavior,references,rows,poses,parts,source,mapping=build(mouse)
        summaries.append(summary);behaviors.extend(behavior);refs.extend(references)
        render(mouse,rows,poses,parts,source,mapping)
    for name,data in [('comparison',summaries),('behavior-diagnostics',behaviors)]:
        with (REPORT/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save_json(REPORT/'reference-template.json',dict(schema=1,annotator=None,status='unlabeled',frames=refs,behaviors=[]))
    save_json(REPORT/'comparison.json',dict(videos=summaries,reference_labels=0,accuracy_improvement=None,production_ready=False,training_performed=False,
                                          caveat='Availability and disagreements are diagnostics, not accuracy. 743 remains withheld under the previous visual audit.'))
    external=json.loads((ROOT/'outputs/stereotypy-training/v2/test-report.json').read_text())
    save_json(REPORT/'external-baseline.json',dict(source='Previously measured CBAS file-disjoint test; not proven mouse-disjoint. No new external evaluation or retraining in this pass.',metrics=external['tests']['cbas-test'],accuracy_improvement=None))
    shutil.copyfile(ROOT/'static/stereotypy-robustness-report.html',REPORT/'index.html')
    shutil.copyfile(ROOT/'static/stereotypy-pose-reference.html',REPORT/'labeling.html')
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
