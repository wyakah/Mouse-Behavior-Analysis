"""Sample source videos without modifying them; write technical metadata and contact sheets."""
from pathlib import Path
import json
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / 'reports'; out.mkdir(exist_ok=True)
reports = []
for p in sorted(root.glob('*.mp4')):
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w,h = int(cap.get(3)),int(cap.get(4))
    cells=[]; samples=[]
    for idx in np.linspace(0,n-1,12).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok, frame = cap.read()
        if not ok: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        samples.append({'frame':int(idx),'seconds':round(idx/fps,3),'laplacian_variance':round(float(cv2.Laplacian(gray,cv2.CV_64F).var()),2),'mean_luma':round(float(gray.mean()),2),'clipped_dark_fraction':round(float((gray<6).mean()),4),'clipped_bright_fraction':round(float((gray>249).mean()),4)})
        cv2.imwrite(str(out/f'{p.stem}_frame_{idx}.jpg'),frame)
        thumb=cv2.resize(frame,(480,round(h*480/w)))
        thumb=cv2.copyMakeBorder(thumb,28,0,0,0,cv2.BORDER_CONSTANT,value=(22,25,29))
        cv2.putText(thumb,f'{idx/fps:.1f}s | frame {idx}',(10,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1)
        cells.append(thumb)
    cap.release()
    sheet=np.vstack([np.hstack(cells[i:i+3]) for i in range(0,len(cells),3)])
    sheet_name=f'{p.stem}_contact.jpg'; cv2.imwrite(str(out/sheet_name),sheet)
    reports.append({'name':p.name,'width':w,'height':h,'fps':fps,'frames':n,'duration_seconds':n/fps,'size_bytes':p.stat().st_size,'contact_sheet':sheet_name,'samples':samples})
(out/'video_inspection.json').write_text(json.dumps(reports,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in reports],indent=2))
