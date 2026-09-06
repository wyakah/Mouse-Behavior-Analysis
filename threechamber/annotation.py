"""Shared source-coordinate annotations for streamed and exported review video."""
import cv2
import numpy as np
from threechamber.live import landmark_states


class AnnotationRenderer:
    HEADER = 126
    COLORS = {'nose': (94,216,248), 'center': (182,228,133), 'tail_base': (255,164,201)}

    def __init__(self, cfg, source_size):
        from threechamber.core import analysis_geometry, circle_polygons, transform
        self.cfg, self.source_size = cfg, source_size
        self.totals=dict.fromkeys(['left','center','right','left_nose','right_nose'],0.)
        self.nose_observed=0.
        self.last_index=-1
        self.crop = list(map(int, cfg.get('review_crop_xyxy', [0,0,*source_size])))
        x1,y1,x2,y2 = self.crop
        if not 0<=x1<x2<=source_size[0] or not 0<=y1<y2<=source_size[1]:
            raise ValueError('Review crop outside source frame.')
        self.width = (x2-x1+1)//2*2
        self.height = (y2-y1+self.HEADER+1)//2*2
        H,cups,w,h,bounds = analysis_geometry(cfg)
        self.geometry=(H,cups,w,h,bounds)
        from threechamber.social import stranger_side
        self.side=stranger_side(cfg)
        inverse = np.linalg.inv(H)
        self.role_positions=transform([[bounds[0]/2,h*.06],[(bounds[1]+w)/2,h*.06]],inverse)-[x1,y1]
        self.floor = np.rint(np.asarray(cfg['arena'])-[x1,y1]).astype(np.int32)
        self.dividers = [np.rint(transform([[x,0],[x,h]],inverse)-[x1,y1]).astype(np.int32) for x in bounds]
        mode = cfg.get('analysis_mode')
        regions = circle_polygons(cfg['cup_circles']) if mode=='circle_zones' else cfg.get('interaction_zones' if mode=='drawn_zones' else 'cups',{})
        self.contours = {side:np.rint(np.asarray(points)-[x1,y1]).astype(np.int32) for side,points in regions.items() if side in cups}
        self.rings = {}
        if mode not in ('circle_zones','drawn_zones'):
            scale=20
            for side,p in cups.items():
                mask=np.zeros((int(h*scale)+4,int(w*scale)+4),np.uint8)
                cv2.fillPoly(mask,[np.rint(p*scale).astype(np.int32)],255)
                dist=cv2.distanceTransform(255-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                cs,_=cv2.findContours(np.uint8(dist<=scale)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                self.rings[side]=[np.rint(transform(c.reshape(-1,2)/scale,inverse)-[x1,y1]).astype(np.int32) for c in cs]

    def draw(self, image, index, time_s, row, image_is_crop=False):
        if 'chamber' not in row and row.get('duration_s',0)>0:
            import pandas as pd
            from threechamber.core import classify_tracks
            values={f'{part}_{axis}':row.get(f'{part}_{axis}',np.nan)
                    for part in ('nose','center') for axis in ('x','y','likelihood')}
            classified=classify_tracks(pd.DataFrame([values]),self.cfg,self.geometry).iloc[0].to_dict()
            dt=float(row['duration_s'])
            end=self.cfg.get('end_s')
            duration=max(0,min(time_s+dt,float(end) if end is not None else time_s+dt)-max(time_s,float(self.cfg.get('start_s',0))))
            row=dict(row,**classified,duration_s=duration)
        if index>self.last_index and 'chamber' in row:
            dt=max(0,float(row.get('duration_s',0)))
            if row['chamber'] in ('left','center','right'):self.totals[row['chamber']]+=dt
            if row.get('nose_scoreable'):self.nose_observed+=dt
            for side in ('left','right'):
                if row.get('nose_scoreable') and row.get(side+'_interaction'):self.totals[side+'_nose']+=dt
            self.last_index=index
        x1,y1,x2,y2=self.crop
        if image_is_crop:
            if image.shape[:2]!=(y2-y1,x2-x1):raise ValueError('Preview crop does not match its source geometry.')
            im=image.copy()
        else:im=image[y1:y2,x1:x2].copy()
        cv2.polylines(im,[self.floor],True,(171,190,179),1,cv2.LINE_AA)
        for line in self.dividers:cv2.line(im,tuple(line[0]),tuple(line[1]),(171,190,179),1,cv2.LINE_AA)
        for side,color in [('left',(242,215,139)),('right',(128,189,255))]:
            if side in self.contours:
                cv2.polylines(im,[self.contours[side]],True,color,2,cv2.LINE_AA)
                if side in self.rings:cv2.polylines(im,self.rings[side],True,color,1,cv2.LINE_AA)
        points=landmark_states(row,self.cfg.get('pcutoff',.6),*self.source_size)
        for a,b in [('nose','center'),('center','tail_base')]:
            p,q=points[a],points[b]
            if p['state']==q['state']=='accepted':
                cv2.line(im,(round(p['x']-x1),round(p['y']-y1)),(round(q['x']-x1),round(q['y']-y1)),(222,232,226),1,cv2.LINE_AA)
        for name,p in points.items():
            if p['state']=='missing':continue
            center=(round(p['x']-x1),round(p['y']-y1));color=self.COLORS[name]
            if p['state']=='accepted':
                cv2.circle(im,center,5,(20,30,25),-1,cv2.LINE_AA)
                cv2.circle(im,center,3,color,-1,cv2.LINE_AA)
            else:
                for start in range(0,360,60):cv2.ellipse(im,center,(6,6),0,start,start+30,color,1,cv2.LINE_AA)
                cv2.drawMarker(im,center,color,cv2.MARKER_TILTED_CROSS,4,1,cv2.LINE_AA)
        im=cv2.copyMakeBorder(im,self.HEADER,self.height-im.shape[0]-self.HEADER,0,self.width-im.shape[1],cv2.BORDER_CONSTANT,value=(26,38,32))
        def text(value,x,y,scale=.43,color=(220,231,224),max_width=None):
            value=str(value).encode('ascii','replace').decode()
            limit=max_width or self.width-x-10
            while cv2.getTextSize(value,0,scale,1)[0][0]>limit and len(value)>3:value=value[:-4]+'...'
            cv2.putText(im,value,(x,y),0,scale,color,1,cv2.LINE_AA)
        scale=min(1.,self.width/560)
        meta=self.cfg.get('recording_metadata',{})
        text('Mouse '+str(meta.get('id') or '--'),12,23,.55*scale,max_width=self.width*.55-12)
        ticks=int(time_s*100+1e-7)
        text(f'{ticks//6000:02}:{ticks%6000/100:05.2f}',int(self.width*.78),23,.48*scale)
        text(f"Sex: {meta.get('sex') or '--'}   |   Genotype: {meta.get('genotype') or '--'}",12,44,.4*scale)
        total=sum(self.totals[k] for k in ('left','center','right'))
        self.si_percent=100*self.totals[self.side+'_nose']/total if self.side in ('left','right') and total>0 and self.nose_observed>0 else None
        from threechamber.social import chamber_label
        metrics=[(chamber_label(side,self.cfg),f'{self.totals[side]:.1f} s') for side in ('left','center','right')]
        metrics.append(('Stranger Interaction %',f'{self.si_percent:.1f}%' if self.si_percent is not None else '--'))
        for i,(label,value) in enumerate(metrics):
            x=8+i*(self.width-16)//4;cw=(self.width-16)//4-5
            cv2.rectangle(im,(x,54),(x+cw,118),(39,55,46),-1)
            label_scale=min(.33*scale,(cw-14)/max(1,cv2.getTextSize(label,0,1,1)[0][0]))
            text(label,x+7,75,label_scale,max_width=cw-12)
            text(value if 'chamber' in row else '--',x+7,104,.58*scale,(182,228,133) if i==3 else (242,247,243),cw-12)
        # Rectified chamber centers keep role labels at the top inside each chamber.
        for i,(side,color) in enumerate([('left',(242,215,139)),('right',(128,189,255))]):
            if side not in self.contours:continue
            label=('Stranger' if side==self.side else 'Object') if self.side in ('left','right') else side.title()
            tw,th=cv2.getTextSize(label,0,.43*scale,1)[0]
            px,py=self.role_positions[i]
            x=int(np.clip(px-tw/2,3,max(3,self.width-tw-3)))
            y=int(py)+self.HEADER+th//2
            cv2.rectangle(im,(x-3,y-th-3),(x+tw+3,y+4),(26,38,32),-1)
            text(label,x,y,.43*scale,color)
        return im
