"""Measure proposal availability, not anatomical accuracy, on original sample frames."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,pandas as pd
from threechamber.localization import build_background,ForegroundLocalizer

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/localization-benchmark');args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    profile=json.loads((ROOT/'profiles/ethovision_three_chamber.json').read_text())
    mappings={'pilot':json.loads((ROOT/'outputs/pretrained-pilot/frame_map.json').read_text()),'contact':json.loads((ROOT/'outputs/specialized-continuous/frame_map.json').read_text())}
    sources=sorted({r['source'] for rows in mappings.values() for r in rows}); backgrounds={}
    for source in sources:
        path=out/(Path(source).stem+'_background.npz')
        if path.exists():
            stored=np.load(path);bg,noise=stored['background'],stored['noise']
        else:
            bg,noise=build_background(ROOT/'prepared'/source,profile['crop_xyxy']);np.savez_compressed(path,background=bg,noise=noise)
        backgrounds[source]=(bg,noise)
    results=[];tiles=[];all_detail=[]
    for dataset,mapping in mappings.items():
        source=None;cap=None;selector=None
        for i,item in enumerate(mapping):
            if source!=item['source']:
                if cap:cap.release()
                source=item['source'];cap=cv2.VideoCapture(str(ROOT/'prepared'/source));selector=ForegroundLocalizer(profile,*backgrounds[source])
            cap.set(cv2.CAP_PROP_POS_FRAMES,item['source_frame']);ok,frame=cap.read()
            if not ok:raise ValueError(f'Cannot read original source frame {item}')
            result,mask=selector.propose(frame,item['source_time_s'],source,return_mask=True)
            row={'dataset':dataset,'index':i,**item,**{k:v for k,v in result.items() if k not in ('candidates','rejected_components')}};results.append(row)
            all_detail.append({'dataset':dataset,'index':i,**item,**result})
            if dataset=='pilot' or i%15==0 or result['status']!='proposal' and i%5==0:
                x1,y1,x2,y2=profile['crop_xyxy'];tile=frame[y1:y2,x1:x2].copy()
                for rank,c in enumerate(result['candidates'][:3]):
                    bx1,by1,bx2,by2=c['bbox_xyxy'];color=(80,220,100) if rank==0 and result['status']=='proposal' else (0,170,255)
                    cv2.rectangle(tile,(bx1-x1,by1-y1),(bx2-x1,by2-y1),color,2)
                    cv2.putText(tile,str(rank),(bx1-x1,by1-y1+15),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
                header=np.zeros((43,tile.shape[1],3),np.uint8)
                cv2.putText(header,f'{dataset} {i} | {source[:3]} {item["source_time_s"]:.2f}s',(8,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.putText(header,f'{result["status"]} | {len(result["candidates"])} candidates',(8,34),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                tile=np.vstack((header,tile));cv2.imwrite(str(out/f'{dataset}_{i:03d}.jpg'),tile)
                tiles.append((dataset,tile))
        cap.release()
    pd.DataFrame(results).to_csv(out/'localization.csv',index=False)
    (out/'proposals.json').write_text(json.dumps(all_detail,indent=2))
    summary={'status':'experimental_unvalidated','purpose':'Bounding-box proposals for DLC pose conditioning; no nose or anatomical center estimated.','limitations':['Coverage is not accuracy. Human box/landmark labels are needed.','Temporal median can absorb a subject immobile for much of recording.','Cup occlusion, reflections and stimulus-mouse movement can create ambiguous or incomplete support.'],'datasets':{}}
    for dataset in mappings:
        subset=[r for r in results if r['dataset']==dataset];summary['datasets'][dataset]={'frames':len(subset),'proposals':sum(r['status']=='proposal' for r in subset),'ambiguous':sum(r['status']=='ambiguous' for r in subset),'unresolved':sum(r['status']=='unresolved' for r in subset)}
        images=[im for d,im in tiles if d==dataset];chunks=[]
        for start in range(0,len(images),18):
            chunk=images[start:start+18];h,w=chunk[0].shape[:2];sheet=np.zeros((int(np.ceil(len(chunk)/3))*h,3*w,3),np.uint8)
            for j,im in enumerate(chunk):sheet[j//3*h:(j//3+1)*h,j%3*w:(j%3+1)*w]=im
            path=out/f'{dataset}_sheet_{start//18+1}.jpg';cv2.imwrite(str(path),sheet)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
