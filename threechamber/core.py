"""Calibrated, timestamp-weighted three-chamber scoring. No inferred/interpolated tracks."""
from pathlib import Path
import hashlib, json
import cv2
import numpy as np
import pandas as pd
import av

VERSION = '0.4.0'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def metadata(path):
    with av.open(str(path)) as c:
        s=c.streams.video[0]
        return dict(width=s.width,height=s.height,fps=float(s.average_rate),frames=s.frames,
                    duration_seconds=float(s.duration*s.time_base) if s.duration else c.duration/1e6)

def strict_frames(container,stream):
    try:
        yield from container.decode(stream)
    except av.error.InvalidDataError as e:
        raise ValueError('Damaged video packet. Use a fully verified prepared copy and generate DLC tracks from that exact copy; no frames were silently skipped.') from e

def timeline(path):
    with av.open(str(path)) as c:
        s=c.streams.video[0]; ts=[]; durations=[]
        for f in strict_frames(c,s):
            if f.pts is None: raise ValueError('Missing video timestamps; remux and verify alignment first.')
            ts.append(float(f.pts*f.time_base)); durations.append(float(f.duration*f.time_base) if f.duration else 0)
        t=np.asarray(ts); t-=t[0]
        if len(t)<2 or np.any(np.diff(t)<=0): raise ValueError('Video timestamps must increase strictly.')
        dt=np.r_[np.diff(t),durations[-1] or np.median(np.diff(t))]
        return t,dt

def polygon(points,name):
    a=np.asarray(points,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<3 or not np.isfinite(a).all(): raise ValueError(f'{name}: draw at least 3 finite points.')
    if not cv2.isContourConvex(a) or abs(cv2.contourArea(a))<1e-4: raise ValueError(f'{name}: use a non-crossing convex polygon.')
    return a

def calibration(cfg):
    if cfg.get('confirmed') is not True: raise ValueError('Confirm the measured calibration and cup floor footprints first.')
    w,h=float(cfg['width_cm']),float(cfg['depth_cm'])
    if not np.isfinite([w,h]).all() or min(w,h)<=0: raise ValueError('Enter measured arena dimensions in cm.')
    corners=polygon(cfg['arena'],'arena')
    if len(corners)!=4: raise ValueError('Arena needs four floor corners: top-left, top-right, bottom-right, bottom-left.')
    H=cv2.getPerspectiveTransform(corners,np.float32([[0,0],[w,0],[w,h],[0,h]]))
    if not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-12: raise ValueError('Degenerate calibration.')
    bounds=cfg.get('dividers_cm',[w/3,2*w/3])
    if not 0<float(bounds[0])<float(bounds[1])<w: raise ValueError('Chamber dividers must be ordered inside arena width.')
    cups={k:transform(polygon(cfg['cups'][k],k),H) for k in ['left','right']}
    for key,p in cups.items():
        if np.any(p<0) or np.any(p[:,0]>w) or np.any(p[:,1]>h): raise ValueError(f'{key} cup must lie within calibrated floor.')
    if cv2.intersectConvexConvex(cups['left'],cups['right'])[0]>0: raise ValueError('Cup footprints overlap.')
    return H,cups,w,h,list(map(float,bounds))

def analysis_geometry(cfg):
    """Unit-square rectification for occupancy needs no physical length."""
    mode=cfg.get('analysis_mode','calibrated')
    if mode=='calibrated':return calibration(cfg)
    if mode not in ('chambers_only','drawn_zones','circle_zones'):raise ValueError('Unknown analysis mode.')
    if cfg.get('confirmed') is not True:raise ValueError('Confirm floor corners and chamber dividers first.')
    corners=polygon(cfg.get('arena',[]),'arena')
    if len(corners)!=4:raise ValueError('Arena needs four floor corners in perimeter order.')
    H=cv2.getPerspectiveTransform(corners,np.float32([[0,0],[1,0],[1,1],[0,1]]))
    if not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-12:raise ValueError('Degenerate calibration.')
    bounds=cfg.get('dividers_fraction',[1/3,2/3])
    if len(bounds)!=2 or not np.isfinite(bounds).all() or not 0<float(bounds[0])<float(bounds[1])<1:raise ValueError('Chamber divider fractions must be ordered between 0 and 1.')
    regions={}
    if mode=='circle_zones':
        circles=validate_circles(cfg)
        regions={side:transform(points,H) for side,points in circle_polygons(circles).items()}
    if mode=='drawn_zones':
        for side in ('left','right'):
            regions[side]=transform(polygon(cfg.get('interaction_zones',{}).get(side,[]),f'{side} interaction zone'),H)
            if np.any(regions[side]<-1e-6) or np.any(regions[side]>1+1e-6):raise ValueError(f'{side} interaction zone must lie inside the arena.')
        if cv2.intersectConvexConvex(regions['left'],regions['right'])[0]>0:raise ValueError('Interaction zones overlap. Draw separate left and right regions.')
    return H,regions,1.,1.,list(map(float,bounds))

def transform(points,H):
    return cv2.perspectiveTransform(np.asarray(points,dtype=np.float32).reshape(-1,1,2),H).reshape(-1,2)

def validate_circles(cfg):
    circles=cfg.get('cup_circles',{})
    try:
        diameter=float(circles['diameter_px'])
        centers=np.asarray([circles['left'],circles['right']],dtype=float)
    except (KeyError,TypeError,ValueError) as e:raise ValueError('Set one shared circle diameter and both cup centers.') from e
    if not np.isfinite(diameter) or diameter<=0 or centers.shape!=(2,2) or not np.isfinite(centers).all():raise ValueError('Circle diameter and centers must be finite; diameter must be positive.')
    arena=polygon(cfg.get('arena',[]),'arena')
    for side,center in zip(('left','right'),centers):
        if cv2.pointPolygonTest(arena,tuple(center),True)<diameter/2-1e-5:raise ValueError(f'{side} circle must fit inside the arena. Move its center or reduce the shared diameter.')
    if np.linalg.norm(centers[0]-centers[1])<diameter-1e-5:raise ValueError('The cup circles overlap. Move their centers or reduce the shared diameter.')
    return dict(diameter_px=diameter,left=centers[0].tolist(),right=centers[1].tolist())

def circle_polygons(circles):
    angle=np.linspace(0,2*np.pi,128,endpoint=False)
    offsets=np.column_stack([np.cos(angle),np.sin(angle)])*float(circles['diameter_px'])/2
    return {side:(np.asarray(circles[side])+offsets).astype(np.float32) for side in ('left','right')}

def propose_circles(cfg):
    """One shared image-space diameter, independently movable centers."""
    centers={};diameters=[]
    for side in ('left','right'):
        points=cfg.get('interaction_zones',{}).get(side) or cfg.get('cups',{}).get(side)
        if points:
            center,radius=cv2.minEnclosingCircle(np.asarray(points,dtype=np.float32))
            centers[side]=list(center);diameters.append(radius*2)
        else:
            arena=polygon(cfg.get('arena',[]),'arena')
            H=cv2.getPerspectiveTransform(np.float32([[0,0],[1,0],[1,1],[0,1]]),arena)
            centers[side]=transform([[1/6 if side=='left' else 5/6,.5]],H)[0].tolist()
    if not diameters:
        arena=polygon(cfg.get('arena',[]),'arena');diameters=[np.linalg.norm(arena[1]-arena[0])*.2]
    return dict(cup_circles=dict(diameter_px=round(float(max(diameters)),1),**centers),zone_geometry_source='Linked circles estimated from existing outlines or arena position; one shared diameter in video pixels. Review both centers and the common size.')

def propose_interaction_zones(cfg, expansion=1.25):
    """Explicit image-space suggestion; never infer centimeters or alter the cup outlines."""
    arena=polygon(cfg.get('arena',[]),'arena')
    zones={}
    for side in ('left','right'):
        points=np.asarray(cfg.get('cups',{}).get(side,[]),dtype=np.float32)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
            raise ValueError('Draw both cup outlines first, or draw interaction zones directly.')
        hull=polygon(cv2.convexHull(points).reshape(-1,2),side+' cup outline')
        center=hull.mean(axis=0); expanded=(hull-center)*expansion+center
        area,clipped=cv2.intersectConvexConvex(expanded,arena)
        if area<=0 or clipped is None:raise ValueError(f'{side} cup outline is outside the arena.')
        zones[side]=polygon(clipped.reshape(-1,2),side+' suggested zone').tolist()
    return {'interaction_zones':zones,'zone_geometry_source':f'Estimated from saved cup outlines: convex envelope expanded by {(expansion-1)*100:.0f}% about its center and clipped to the arena. No physical size inferred; review the selected regions.'}

def load_tracks(path,nose='nose',center='center',scorer=None,individual=None):
    """DLC 3/4-row CSV, HDF5, or explicit flat frame/nose_x/... CSV. Fail on ambiguous identities."""
    path=Path(path)
    if path.suffix.lower() in ('.h5','.hdf5'):
        df=pd.read_hdf(path)
    else:
        first=pd.read_csv(path,nrows=0)
        if {'frame',f'{nose}_x',f'{center}_x'}<=set(first.columns): df=pd.read_csv(path).set_index('frame')
        else:
            with path.open() as f: lines=[next(f,'') for _ in range(4)]
            depth=4 if lines[1].split(',')[0].strip()=='individuals' else 3
            df=pd.read_csv(path,header=list(range(depth)),index_col=0)
    if isinstance(df.columns,pd.MultiIndex):
        scorers=list(df.columns.get_level_values(0).unique())
        if scorer is None and len(scorers)!=1: raise ValueError(f'Select one scorer: {scorers}')
        df=df.xs(scorer or scorers[0],axis=1,level=0)
        if df.columns.nlevels==3:
            individuals=list(df.columns.get_level_values(0).unique())
            if individual is None: raise ValueError(f'Select subject individual explicitly: {individuals}')
            df=df.xs(individual,axis=1,level=0)
        df.columns=['_'.join(map(str,c)) for c in df.columns]
    try: index=pd.to_numeric(df.index).to_numpy(dtype=float)
    except Exception as e: raise ValueError('Track frame indices must be zero-based integers.') from e
    if not np.array_equal(index,np.arange(len(df))): raise ValueError('Tracks must contain every video frame once, in zero-based order.')
    out=pd.DataFrame(index=np.arange(len(df)))
    for canonical,bodypart in [('nose',nose),('center',center)]:
        for coord in ['x','y','likelihood']:
            col=f'{bodypart}_{coord}'
            if col not in df: raise ValueError(f'Missing {col}. Available: {list(df.columns)}')
            out[f'{canonical}_{coord}']=pd.to_numeric(df[col],errors='raise').to_numpy()
    for k in ['nose','center']:
        p=out[f'{k}_likelihood'].to_numpy()
        if ((p[np.isfinite(p)]<0)|(p[np.isfinite(p)]>1)).any(): raise ValueError('Likelihood must be between 0 and 1.')
    return out

def score(tracks,t,dt,cfg):
    H,cups,w,h,bounds=analysis_geometry(cfg)
    circle_zones=cfg.get('analysis_mode')=='circle_zones'
    drawn_zones=cfg.get('analysis_mode') in ('drawn_zones','circle_zones')
    if len(tracks)!=len(t): raise ValueError(f'Track/video mismatch: {len(tracks)} rows versus {len(t)} decoded frames.')
    cutoff=float(cfg.get('pcutoff',.6))
    if not 0<=cutoff<=1: raise ValueError('Confidence cutoff must be 0–1.')
    start=float(cfg.get('start_s',0)); end=cfg.get('end_s')
    end=float(end) if end is not None else float(t[-1]+dt[-1])
    if not np.isfinite([start,end]).all() or start<0 or end<=start or end>t[-1]+dt[-1]+.001: raise ValueError('Invalid scoring time window.')
    weights=np.maximum(0,np.minimum(t+dt,end)-np.maximum(t,start))
    out=tracks.copy(); out.insert(0,'frame',np.arange(len(t))); out['time_s']=t; out['duration_s']=weights
    for part in ['nose','center']:
        xy=tracks[[f'{part}_x',f'{part}_y']].to_numpy()
        valid=np.isfinite(xy).all(axis=1)&np.isfinite(tracks[f'{part}_likelihood'])&(tracks[f'{part}_likelihood']>=cutoff)
        cm=transform(np.nan_to_num(xy,nan=0,posinf=0,neginf=0),H)
        valid &= np.isfinite(cm).all(axis=1)
        out[f'{part}_x_cm']=cm[:,0]; out[f'{part}_y_cm']=cm[:,1]
        out[f'{part}_valid']=valid
    # For unit-square scoring, temporary coordinates below are normalized, never exported as centimeters.
    nx,ny=out.nose_x_cm.to_numpy(),out.nose_y_cm.to_numpy()
    for side in ['left','right']:
        out[f'{side}_interaction']=False
        out[f'{side}_cup_distance_cm']=np.nan
    nose_floor=out.nose_valid.to_numpy()&(nx>=0)&(nx<=w)&(ny>=0)&(ny<=h)
    for side,p in cups.items():
        d=np.array([-cv2.pointPolygonTest(p,(float(x),float(y)),True) for x,y in zip(nx,ny)])
        if circle_zones:
            center=np.asarray(cfg['cup_circles'][side],dtype=float)
            xy=tracks[['nose_x','nose_y']].to_numpy()
            out[f'{side}_interaction']=nose_floor&(np.linalg.norm(xy-center,axis=1)<=float(cfg['cup_circles']['diameter_px'])/2+1e-7)
        elif drawn_zones:
            out[f'{side}_interaction']=nose_floor&(d<=1e-7)
        else:
            out[f'{side}_cup_distance_cm']=np.where(out.nose_valid,d,np.nan)
            out[f'{side}_interaction']=nose_floor&(d>=-1e-6)&(d<=1+1e-6)
    inside_cup=np.zeros(len(t),dtype=bool)
    for side in cups: inside_cup|=out[f'{side}_cup_distance_cm'].to_numpy() < -1e-6
    out['nose_scoreable']=nose_floor&~inside_cup
    overlap=out.left_interaction&out.right_interaction
    out.loc[overlap,['left_interaction','right_interaction']]=False
    out.loc[overlap,'nose_scoreable']=False
    out.loc[~out.nose_scoreable,['left_interaction','right_interaction']]=False
    cx,cy=out.center_x_cm.to_numpy(),out.center_y_cm.to_numpy()
    labels=np.full(len(t),'unknown',dtype=object)
    floor=out.center_valid.to_numpy()&(cx>=0)&(cx<=w)&(cy>=0)&(cy<=h)
    labels[out.center_valid.to_numpy()&~floor]='outside'
    # Shared divider belongs to chamber on its right; all chambers form a disjoint partition.
    labels[floor]=np.array(['left','center','right'])[np.searchsorted(bounds,cx[floor],side='right')]
    out['chamber']=labels
    total=float(weights.sum()); sec=lambda mask:float(weights[np.asarray(mask)].sum())
    summary={'version':VERSION,'window_start_s':start,'window_end_s':end,'analyzed_seconds':total,
             'decoded_frames':len(t),'scored_frames':int((weights>0).sum()),'pcutoff':cutoff,
             'max_source_frame_interval_s':float(dt.max()),'median_source_frame_interval_s':float(np.median(dt)),'intervals_over_twice_median':int((dt>2*np.median(dt)).sum()),
             'interaction_threshold_cm':1,'occupancy_landmark':cfg.get('center_bodypart','center'),
             'nose_valid_seconds':sec(out.nose_valid),'nose_unscoreable_seconds':sec(~out.nose_scoreable),
             'nose_scoreable_fraction':sec(out.nose_scoreable)/total,
             'center_valid_fraction':sec(out.center_valid)/total,
             'left_nose_seconds':sec(out.left_interaction),'right_nose_seconds':sec(out.right_interaction)}
    for key in ['left','center','right','unknown','outside']: summary[f'{key}_chamber_seconds']=sec(labels==key)
    denom=summary['left_nose_seconds']+summary['right_nose_seconds']
    target=cfg.get('target_side','unspecified')
    summary['target_side']=target
    summary['preference_index']=(summary[f'{target}_nose_seconds']-summary[f'{"right" if target=="left" else "left"}_nose_seconds'])/denom if target in ('left','right') and denom else None
    summary['analysis_mode']=cfg.get('analysis_mode','calibrated');summary['accuracy_validated']=False
    summary['cup_metrics_status']=('drawn_zones_calculated' if drawn_zones else 'calculated') if cups else 'not_requested'
    summary['interaction_measure']='nose_in_linked_circle' if circle_zones else ('nose_in_selected_zone' if drawn_zones else ('nose_in_external_1cm_ring' if cups else None))
    summary['zone_geometry_source']=cfg.get('zone_geometry_source','User-selected interaction regions') if drawn_zones else None
    if drawn_zones:summary['interaction_threshold_cm']=None
    if circle_zones:
        summary['cup_diameter_px']=float(cfg['cup_circles']['diameter_px'])
        for side in ('left','right'):
            for axis,value in zip(('x','y'),cfg['cup_circles'][side]):summary[f'{side}_cup_center_{axis}_px']=float(value)
    summary['geometry_source']=cfg.get('geometry_source','User-confirmed floor geometry')
    summary['occupancy_bounds_note']='Bounds allocate unknown tracking time only; they do not account for incorrect identities, landmarks, or geometry.'
    # Unknown/outside times are kept; valid tracking coverage is not measured accuracy.
    summary['center_occupancy_coverage']=sec(np.isin(labels,['left','center','right']))/total
    for key in ['left','center','right']:
        summary[f'{key}_chamber_lower_seconds']=summary[f'{key}_chamber_seconds']
        summary[f'{key}_chamber_upper_seconds']=summary[f'{key}_chamber_seconds']+summary['unknown_chamber_seconds']
    if cfg.get('analysis_mode') in ('chambers_only','drawn_zones','circle_zones'):
        for part in ['nose','center']:
            out.rename(columns={f'{part}_x_cm':f'{part}_x_fraction',f'{part}_y_cm':f'{part}_y_fraction'},inplace=True)
    if not cups:
        for key in ['interaction_threshold_cm','left_nose_seconds','right_nose_seconds','nose_scoreable_fraction','nose_unscoreable_seconds','preference_index']:summary[key]=None
        out['nose_scoreable']=False
        out['cup_metrics_available']=False
    else:out['cup_metrics_available']=True
    for part in ['nose','center']:
        for suffix in ['cm','fraction']:
            cols=[f'{part}_x_{suffix}',f'{part}_y_{suffix}']
            if cols[0] in out:out.loc[~out[f'{part}_valid'],cols]=np.nan
    return out,summary

def bouts(rows):
    records=[]
    for kind in ['left_interaction','right_interaction']:
        mask=rows[kind].to_numpy()&(rows.duration_s.to_numpy()>0)
        for a,b in zip(np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))==1),np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))==-1)):
            records.append(dict(behavior=kind,start_frame=int(a),end_frame_exclusive=int(b),start_s=float(max(rows.iloc[a].time_s,rows.attrs.get('start_s',0))),duration_s=float(rows.iloc[a:b].duration_s.sum())))
    return pd.DataFrame(records,columns=['behavior','start_frame','end_frame_exclusive','start_s','duration_s'])

def review_video(video,rows,cfg,destination,progress=lambda x:None):
    """Preserve every source frame and its timestamps in an H.264 review video (no audio)."""
    from fractions import Fraction
    H,cups,w,h,bounds=analysis_geometry(cfg); inv=np.linalg.inv(H)
    drawn_zones=cfg.get('analysis_mode') in ('drawn_zones','circle_zones')
    region_key='interaction_zones' if drawn_zones else 'cups'
    source_regions=circle_polygons(cfg['cup_circles']) if cfg.get('analysis_mode')=='circle_zones' else cfg.get(region_key,{})
    contours={k:np.rint(np.asarray(source_regions[k])).astype(np.int32) for k in cups}
    # Render 1 cm external rings using a metric raster, then project contour back to camera.
    scale=20; rings={}
    for key,p in cups.items():
        if drawn_zones:continue
        mask=np.zeros((int(h*scale)+4,int(w*scale)+4),np.uint8)
        cv2.fillPoly(mask,[np.rint(p*scale).astype(np.int32)],255)
        dist=cv2.distanceTransform(255-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        outer=np.uint8(dist<=scale)*255
        cs,_=cv2.findContours(outer,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rings[key]=[np.rint(transform(c.reshape(-1,2)/scale,inv)).astype(np.int32) for c in cs]
    with av.open(str(video)) as source, av.open(str(destination),'w') as target:
        ins=source.streams.video[0]; stream=target.add_stream('libx264',rate=ins.average_rate)
        view=cfg.get('review_crop_xyxy')
        if view is not None:
            vx1,vy1,vx2,vy2=map(int,view)
            if not 0<=vx1<vx2<=ins.width or not 0<=vy1<vy2<=ins.height:raise ValueError('Review crop outside source frame.')
            stream.width=vx2-vx1;stream.height=vy2-vy1+82
        else:stream.width=ins.width;stream.height=ins.height
        stream.pix_fmt='yuv420p'
        stream.time_base=Fraction(1,1000000); stream.codec_context.time_base=stream.time_base
        stream.options={'crf':'20','preset':'fast'}
        count=0
        for i,frame in enumerate(strict_frames(source,ins)):
            if i>=len(rows): raise ValueError('Review frame count exceeds tracking rows.')
            im=frame.to_ndarray(format='bgr24'); r=rows.iloc[i]
            for side,color in [('left',(200,180,70)),('right',(100,180,245))]:
                if side in contours:
                    cv2.polylines(im,[contours[side]],True,color,2)
                    if side in rings:cv2.polylines(im,rings[side],True,color,1)
            for x in bounds:
                line=np.rint(transform([[x,0],[x,h]],inv)).astype(int)
                cv2.line(im,tuple(line[0]),tuple(line[1]),(190,190,190),1)
            for part,color in [('nose',(30,245,240)),('center',(100,230,120))]:
                if r[f'{part}_valid']:
                    x,y=r[f'{part}_x'],r[f'{part}_y']
                    if 0<=x<ins.width and 0<=y<ins.height: cv2.circle(im,(round(x),round(y)),4,color,-1)
            if view is not None:im=cv2.copyMakeBorder(im[vy1:vy2,vx1:vx2],82,0,0,0,cv2.BORDER_CONSTANT,value=(20,24,30))
            cv2.rectangle(im,(0,0),(stream.width,82),(20,24,30),-1)
            status='OUTSIDE SCORING WINDOW' if r.duration_s<=0 else ('NOSE UNRESOLVED' if not r.nose_scoreable else f'L cup: {bool(r.left_interaction)}   R cup: {bool(r.right_interaction)}')
            if not cups and r.duration_s>0:status='CHAMBER OCCUPANCY ONLY | 1 cm cup metrics unavailable'
            caption='Selected zones | yellow nose; green center' if drawn_zones else ('1 cm outside cup floor edge' if cups else 'No physical scale | yellow nose; green center')
            for y,text in [(23,f'Frame {i} | {r.time_s:.3f}s | Chamber: {r.chamber}'),(48,status),(71,f'Nose p={r.nose_likelihood:.3f} | cutoff={cfg.get("pcutoff",.6):.2f} | {caption}')]:
                cv2.putText(im,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.42 if view is not None else .52,(235,240,245),1,cv2.LINE_AA)
            out=av.VideoFrame.from_ndarray(im,format='bgr24'); out.pts=round(r.time_s*1e6); out.time_base=stream.time_base
            for packet in stream.encode(out): target.mux(packet)
            count+=1
            if i%200==0: progress(i/len(rows))
        for packet in stream.encode(): target.mux(packet)
        if count!=len(rows): raise ValueError('Review decode ended early.')

def analyze(video,tracks_path,cfg,outdir,progress=lambda s:None):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=False)
    try:
        progress('Reading source timestamps'); t,dt=timeline(video)
        tracks=load_tracks(tracks_path,cfg.get('nose_bodypart','nose'),cfg.get('center_bodypart','center'),cfg.get('scorer') or None,cfg.get('individual') or None)
        rows,summary=score(tracks,t,dt,cfg); rows.attrs['start_s']=summary['window_start_s']
        summary['video']=Path(video).name; summary['tracking_file']=Path(tracks_path).name
        progress('Exporting tables')
        rows.to_csv(outdir/'frames.csv',index=False); bouts(rows).to_csv(outdir/'bouts.csv',index=False)
        pd.DataFrame([summary]).to_csv(outdir/'summary.csv',index=False)
        (outdir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        (outdir/'calibration.json').write_text(json.dumps(cfg,indent=2,allow_nan=False))
        progress('Rendering annotated review video')
        review_video(video,rows,cfg,outdir/'review.mp4',lambda p:progress(f'Rendering review: {p:.0%}'))
        progress('Hashing source files')
        method={'chambers_only':'Body-center chamber occupancy in rectified unit-square; cup metrics not requested.', 'circle_zones':'Free-subject nose inside two image-space circles with one shared diameter, exact radial boundary included; body-center chamber occupancy. No physical distance inferred.', 'drawn_zones':'Free-subject nose inside selected image-space interaction zones, boundary included; body-center chamber occupancy. Zone proximity is a behavioral proxy; no physical distance is inferred.'}.get(cfg.get('analysis_mode'),'Nose in external 0–1 cm cup ring; body-center chamber occupancy.')
        manifest={'version':VERSION,'video':str(Path(video).resolve()),'video_sha256':sha256(video),'tracks':str(Path(tracks_path).resolve()),'tracks_sha256':sha256(tracks_path),'method':method+' No interpolation; actual PTS durations.','reference':'https://doi.org/10.1016/j.heliyon.2024.e36352','status':'complete','review':'All source frames retained; silent H.264, original presentation timestamps.','validation':'Outputs require manual tracking and geometry review before scientific use.'}
        (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2))
        return summary
    except Exception as e:
        (outdir/'FAILED.txt').write_text(str(e)); raise
