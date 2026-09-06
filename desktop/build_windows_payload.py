"""Build a self-contained Windows x64 CPU runtime and verified offline model payload."""
import hashlib,json,os,shutil,subprocess,sys,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'desktop/bundle'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    if os.name!='nt':raise RuntimeError('Build this payload on Windows x64')
    runtime=BUNDLE/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'.cache/python-embed.zip';archive.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve('https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip',archive)
    with zipfile.ZipFile(archive) as z:z.extractall(runtime)
    (runtime/'python312._pth').write_text('python312.zip\n.\nLib/site-packages\nimport site\n')
    # Ship application-local C++ runtime DLLs; do not rely on Visual Studio being installed.
    redists=[]
    for variable in ('ProgramFiles','ProgramFiles(x86)'):
        base=Path(os.environ.get(variable,'C:/Program Files'))/'Microsoft Visual Studio/2022'
        redists.extend(base.glob('*/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT'))
    if not redists:raise RuntimeError('Microsoft redistributable CRT directory not found')
    redist=max(redists,key=lambda p:tuple(int(v) for v in p.parent.parent.name.split('.')))
    for dll in redist.glob('*.dll'):shutil.copy2(dll,runtime/dll.name)
    if not (runtime/'msvcp140.dll').is_file():raise RuntimeError('C++ standard library was not bundled')
    site=runtime/'Lib/site-packages';site.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','pip','install','--target',str(site),'pip'],check=True)
    python=runtime/'python.exe'
    subprocess.run([str(python),'-m','pip','install','--index-url','https://download.pytorch.org/whl/cpu','torch==2.14.0','torchvision==0.29.0'],check=True)
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'desktop/runtime-lock.txt')],check=True)
    model_meta=json.loads((ROOT/'desktop/models-asset.json').read_text());archive=ROOT/'.cache'/model_meta['file']
    # gh handles the release URL and redirects; no credentials are embedded in the installer.
    subprocess.run(['gh','release','download',model_meta['tag'],'--repo',os.environ['GITHUB_REPOSITORY'],'--pattern',model_meta['file'],'--dir',str(archive.parent),'--clobber'],check=True)
    if sha(archive)!=model_meta['sha256']:raise ValueError('Model archive checksum mismatch')
    with zipfile.ZipFile(archive) as z:z.extractall(BUNDLE)
    hashes=json.loads((BUNDLE/'model-hashes.json').read_text())
    for name,expected in hashes.items():
        if sha(BUNDLE/name)!=expected:raise ValueError('Model integrity check failed: '+name)
    checkpoints=site/'deeplabcut/modelzoo/checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
    for p in (BUNDLE/'checkpoints').glob('*.pt'):shutil.copy2(p,checkpoints/p.name)
    shutil.rmtree(BUNDLE/'checkpoints');(BUNDLE/'model-hashes.json').unlink()
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines();owned={}
    for name in sorted(set(files)):
        p=Path(name)
        if p.parts[0] not in ('app.py','threechamber','stereotypy','scripts','static','profiles','desktop'):continue
        if p.parts[0]=='desktop' and name!='desktop/service.py':continue
        if p.suffix not in ('.py','.json','.html','.css','.js','.mjs'):continue
        target=BUNDLE/'payload'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,target);owned[name]=sha(target)
    models={p.relative_to(BUNDLE).as_posix():sha(p) for folder in [BUNDLE/'models',checkpoints] for p in folder.rglob('*') if p.is_file()}
    for name in ['THIRD_PARTY_NOTICES.md','runtime-lock.txt']:shutil.copy2(ROOT/'desktop'/name,BUNDLE/name)
    shutil.copytree(ROOT/'desktop/notices',BUNDLE/'notices',dirs_exist_ok=True)
    version=json.loads((ROOT/'desktop/package.json').read_text())['version']
    (BUNDLE/'manifest.json').write_text(json.dumps(dict(version=version,platform='windows-x64',python='3.12',program_files=owned,model_files=models),indent=2))
    subprocess.run([str(python),'-I','-B','-c','import flask,openpyxl,cv2,av,torch,transformers,statsmodels,tables; print("Windows runtime imports OK")'],check=True)
if __name__=='__main__':main()
