"""Assemble a relocatable offline macOS payload from a tested Python environment.

Use a python-build-standalone distribution as --python-base (not a framework
installation or a venv). No recordings, predictions, or training examples ship.
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.is_dir():shutil.copytree(source,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store'))
    else:shutil.copy2(source,destination)

def build(args):
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise ValueError('This build currently supports Apple Silicon Macs only')
    bundle=args.output.resolve();bundle.mkdir(parents=True,exist_ok=True)
    if (bundle/'manifest.json').exists() and not args.refresh:raise ValueError('Use --refresh to update an existing build payload')
    runtime=bundle/'runtime';site=runtime/'lib/python3.12/site-packages'
    if not (runtime/'.complete').exists():
        runtime.mkdir(exist_ok=True)
        copy(args.python_base/'bin/python3.12',runtime/'bin/python3.12')

        for name in ('python3','python'):
            link=runtime/'bin'/name
            if not link.exists():link.symlink_to('python3.12')
        # Standard library and bundled extension dependencies, without host packages.
        shutil.copytree(args.python_base/'lib',runtime/'lib',dirs_exist_ok=True,ignore=shutil.ignore_patterns('site-packages','__pycache__','*.pyc','pkgconfig'))
        copy(args.environment/'lib/python3.12/site-packages',site)
        checkpoints=site/'deeplabcut/modelzoo/checkpoints'
        keep={'superanimal_topviewmouse_resnet_50.pt','superanimal_quadruped_hrnet_w32.pt'}
        for path in checkpoints.glob('*.pt'):
            if path.name not in keep:path.unlink()
        # Executable entry-point scripts with host shebangs are not bundled.
        subprocess.run([str(runtime/'bin/python3'),'-m','pip','install','--disable-pip-version-check','--upgrade','--target',str(site),'flask==3.1.2','openpyxl==3.1.5'],check=True)
        (runtime/'.complete').touch()
    payload=bundle/'payload';payload.mkdir(exist_ok=True)
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    owned={}
    for name in sorted(set(files)):
        p=Path(name)
        if p.parts[0] not in ('app.py','threechamber','stereotypy','scripts','static','profiles','desktop'):continue
        if p.parts[0]=='desktop' and name!='desktop/service.py':continue
        if p.suffix not in ('.py','.json','.html','.css','.js','.mjs'):continue
        copy(ROOT/p,payload/p);owned[name]=sha(payload/p)
    models=bundle/'models'
    for name in ('selection.json','preprocessing.joblib','temporal.pt','binary-temporal.pt','logistic-appearance.joblib','logistic-motion.joblib','trees-appearance.joblib','trees-motion.joblib'):
        copy(ROOT/'outputs/stereotypy-training/v2'/name,models/'stereotypy-training/v2'/name)
    for name in ('temporal-model.pt','training-report.json'):
        copy(ROOT/'outputs/stereotypy-training/v1'/name,models/'stereotypy-training/v1'/name)
    copy(ROOT/'.cache/huggingface/hub/models--facebook--dinov2-small',models/'huggingface/hub/models--facebook--dinov2-small')
    copy(ROOT/'.cache/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth',models/'torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth')
    hashes={str(p.relative_to(bundle)):sha(p) for p in models.rglob('*') if p.is_file()}
    for p in (site/'deeplabcut/modelzoo/checkpoints').glob('*.pt'):hashes[str(p.relative_to(bundle))]=sha(p)
    copy(ROOT/'desktop/THIRD_PARTY_NOTICES.md',bundle/'THIRD_PARTY_NOTICES.md')
    copy(ROOT/'desktop/notices',bundle/'notices')
    copy(ROOT/'desktop/runtime-lock.txt',bundle/'runtime-lock.txt')
    manifest=dict(version='0.1.2',platform='macos-arm64',python='3.12',program_files=owned,model_files=hashes)
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run([str(runtime/'bin/python3'),'-I','-c','import flask,openpyxl,cv2,av,torch,transformers,statsmodels,tables; print("Portable runtime imports OK")'],check=True)
    print('Payload ready:',bundle)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--python-base',type=Path,required=True);parser.add_argument('--environment',type=Path,default=ROOT/'.dlc-env');parser.add_argument('--output',type=Path,default=ROOT/'desktop/bundle');parser.add_argument('--refresh',action='store_true')
    build(parser.parse_args())
