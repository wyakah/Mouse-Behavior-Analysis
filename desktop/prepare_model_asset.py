"""Create a platform-neutral release asset containing only shipped inference weights."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];bundle=root/'desktop/bundle';out=root/'desktop/dist/Behavior-Studio-models-v1.zip'
files={p.relative_to(bundle).as_posix():p for p in (bundle/'models').rglob('*') if p.is_file()}
for p in (bundle/'runtime/lib/python3.12/site-packages/deeplabcut/modelzoo/checkpoints').glob('*.pt'):files['checkpoints/'+p.name]=p
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    hashes={}
    for name,p in files.items():
        z.write(p,name)
        with p.open('rb') as f:hashes[name]=hashlib.file_digest(f,'sha256').hexdigest()
    z.writestr('model-hashes.json',json.dumps(hashes,indent=2))
with out.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
(root/'desktop/models-asset.json').write_text(json.dumps(dict(tag='v0.1.3',file=out.name,sha256=digest),indent=2)+'\n')
print(out,digest)
