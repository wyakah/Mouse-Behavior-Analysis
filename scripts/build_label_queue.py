from pathlib import Path
import json,hashlib,cv2,re
ROOT=Path(__file__).resolve().parents[1];dest=ROOT/'labeling';(dest/'images').mkdir(parents=True,exist_ok=True)
manifest=dest/'queue.json'
existing=json.loads(manifest.read_text()) if manifest.exists() else []
entries={e['id']:e for e in existing}

def add(src,frame,image,kind):
 id=hashlib.sha256(f'{src}:{frame}'.encode()).hexdigest()[:16]
 if id in entries:return
 path=dest/'images'/f'{id}.png';cv2.imwrite(str(path),image)
 entries[id]={'id':id,'source':src,'source_frame':int(frame),'image':str(path.relative_to(ROOT)),'kind':kind,'width':image.shape[1],'height':image.shape[0]}
# Seed failure-review frames first. These remain unreviewed; predictions are not truth.
mp=json.loads((ROOT/'outputs/pretrained-pilot/frame_map.json').read_text())
for m in mp:
 cap=cv2.VideoCapture(str(ROOT/'prepared'/m['source']));cap.set(cv2.CAP_PROP_POS_FRAMES,m['source_frame']);ok,im=cap.read();cap.release()
 if not ok:raise RuntimeError('Source extraction failed')
 # Preserve identifiers and reviewed annotations while replacing the image with native source pixels.
 id=hashlib.sha256(f"{m['source']}:{m['source_frame']}".encode()).hexdigest()[:16]
 cv2.imwrite(str(dest/'images'/f'{id}.png'),im)
 add(m['source'],m['source_frame'],im,'pilot_review')
for folder in sorted((ROOT/'dlc-projects').glob('*/labeled-data/*')):
 for p in sorted(folder.glob('*.png')):
  match=re.search(r'img(\d+)',p.stem)
  if match:add(folder.name+'.mp4',int(match[1]),cv2.imread(str(p)),'diverse_training_frame')
manifest.write_text(json.dumps(list(entries.values()),indent=2));print(len(entries),'unreviewed frame entries available')
