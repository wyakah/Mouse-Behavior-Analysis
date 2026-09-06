"""Shared source-coordinate annotations for streamed and exported review video."""
import cv2
import numpy as np
from threechamber.live import landmark_states


class AnnotationRenderer:
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
        self.height = (y2-y1+106+1)//2*2
        H,cups,w,h,bounds = analysis_geometry(cfg)
        inverse = np.linalg.inv(H)
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
        im=cv2.copyMakeBorder(im,106,self.height-im.shape[0]-106,0,self.width-im.shape[1],cv2.BORDER_CONSTANT,value=(26,38,32))
        small=.42 if self.width>=400 else .24
        ticks=int(time_s*100+1e-7)
        cv2.putText(im,f'Frame {index+1:,}   |   {ticks//6000:02}:{ticks%6000/100:05.2f}',(12,22),cv2.FONT_HERSHEY_SIMPLEX,small,(235,243,236),1,cv2.LINE_AA)
        for i,(name,label) in enumerate([('nose','Nose'),('center','Center'),('tail_base','Tail')]):
            p=points[name];q='missing' if p['likelihood'] is None else f'{p["likelihood"]:.2f}'+(' ?' if p['state']=='uncertain' else '')
            cv2.putText(im,f'{label} {q}',(12+i*(self.width//3),46),cv2.FONT_HERSHEY_SIMPLEX,small,self.COLORS[name],1,cv2.LINE_AA)
        if 'chamber' in row:
            status='Outside scoring window' if row.get('duration_s',1)<=0 else f'Chamber: {row["chamber"]} | '+(f'Left: {bool(row.get("left_interaction"))}  Right: {bool(row.get("right_interaction"))}' if row.get('nose_scoreable') else 'Nose unscored')
            from threechamber.social import stranger_side
            side=stranger_side(self.cfg);total=sum(self.totals[k] for k in ('left','center','right'))
            if side in ('left','right'):
                pct=f'{100*self.totals[side+"_nose"]/total:.1f}%' if total>0 and self.nose_observed>0 else '--'
                status=f'Chamber: {row["chamber"]} | Stranger {side}: {pct} cumulative'
        else:status='Tracking preview | measurements finalize after processing'
        cv2.putText(im,status,(12,70),cv2.FONT_HERSHEY_SIMPLEX,small*.93,(205,219,208),1,cv2.LINE_AA)
        if 'chamber' in row:
            t=self.totals
            text=f'Total s | Chambers L {t["left"]:.1f} C {t["center"]:.1f} R {t["right"]:.1f} | Nose L {t["left_nose"]:.1f} R {t["right_nose"]:.1f}'
            scale=min(small*.9,(self.width-24)/max(1,cv2.getTextSize(text,0,1,1)[0][0]))
            cv2.putText(im,text,(12,94),0,scale,(205,219,208),1,cv2.LINE_AA)
        return im
