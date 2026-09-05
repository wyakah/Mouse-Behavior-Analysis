"""Fetch the public CBAS and LabGym priority-behavior training material.

Public file IDs and known CBAS hashes are versioned; recordings remain local.
Google's large-file scan notice is handled through its normal download form.
Authentication, quota errors, and access requests are never bypassed.
"""
import argparse, hashlib, json, shutil, sys, urllib.parse, urllib.request, zipfile
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stereotypy.training import sha256,save_json


class DownloadForm(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.values={};self.active=False
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='form' and attrs.get('id')=='download-form':self.action=attrs.get('action');self.active=True
        if self.active and tag=='input' and attrs.get('name'):self.values[attrs['name']]=attrs.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.active=False


def download(public_id,path,expected=None):
    if path.exists() and (expected is None or sha256(path)==expected):return
    path.parent.mkdir(parents=True,exist_ok=True)
    url='https://drive.google.com/uc?export=download&id='+public_id
    response=urllib.request.urlopen(url,timeout=180)
    if 'text/html' in response.headers.get('Content-Type',''):
        parser=DownloadForm();parser.feed(response.read().decode());response.close()
        if not parser.action:raise ValueError('Public download unavailable; authentication/quota/access notices are not bypassed')
        parsed=urllib.parse.urlparse(parser.action)
        if parsed.scheme!='https' or parsed.hostname!='drive.usercontent.google.com' or parser.values.get('id')!=public_id:
            raise ValueError('Unexpected download form')
        response=urllib.request.urlopen(parser.action+'?'+urllib.parse.urlencode(parser.values),timeout=180)
    if 'text/html' in response.headers.get('Content-Type',''):raise ValueError('Expected data, received a web page')
    partial=path.with_suffix(path.suffix+'.part')
    with response,partial.open('wb') as stream:shutil.copyfileobj(response,stream,1024*1024)
    if expected and sha256(partial)!=expected:raise ValueError('Source fingerprint differs from the audited dataset')
    partial.replace(path)


def unpack(archive,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Corrupt archive')
        for item in z.infolist():
            if item.filename.startswith('__MACOSX/') or Path(item.filename).name in ('.DS_Store',''):continue
            target=(destination/item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):raise ValueError('Unsafe archive member')
            if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',choices=['cbas','labgym'],default=['cbas','labgym']);args=parser.parse_args()
    catalog=json.loads((ROOT/'docs/data/stereotypy-priority-sources.json').read_text());out=ROOT/'research/stereotypy-round2';records=[]
    if 'cbas' in args.datasets:
        for row in catalog['cbas']:
            path=out/'cbas-data'/row['split']/row['name'];download(row['public_id'],path,row['sha256']);records.append(dict(**row,path=str(path)));print(row['split'],row['name'],flush=True)
    if 'labgym' in args.datasets:
        dest=out/'labgym-examples';marker=dest/'acquisition.json'
        if not marker.exists():
            archive=out/'labgym-examples.zip';download(catalog['labgym']['examples_public_id'],archive)
            archive_hash=sha256(archive);unpack(archive,dest)
            files=[dict(path=str(p.relative_to(dest)),sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(dest.rglob('*.avi')) if '__MACOSX' not in p.parts]
            save_json(marker,dict(public_id=catalog['labgym']['examples_public_id'],archive_sha256=archive_hash,files=files))
        records.append(dict(dataset='labgym',manifest=str(marker),manifest_sha256=sha256(marker)))
    save_json(out/'priority-acquisition.json',records)

if __name__=='__main__':main()
