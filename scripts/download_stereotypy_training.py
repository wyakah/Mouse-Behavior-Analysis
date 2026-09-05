"""Fetch the public pilot training sources, retaining source and access records.

Downloads stay local under research/ (gitignored). No credentials, author emails,
or dataset access requests are submitted. Use prepare_stereotypy_training.py next.
"""
import argparse
import io
import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stereotypy.training import save_json, sha256

MIT_URL = 'https://www.dropbox.com/s/1vo4jcdbslgsfzb/full_database.zip?dl=1'
OSF_TARGETS = 'https://api.osf.io/v2/nodes/rwhtd/files/osfstorage/61f77ad72082b8053c4dfad6/?page[size]=100'
OSF_VIDEOS = 'https://api.osf.io/v2/nodes/rwhtd/files/osfstorage/61f7791d026ee60654b50683/?page[size]=100'
ENTRIES = {'circling':'1_ggs8x7g2', 'jumping':'1_dgatnabm', 'gnawing_nonfood':'1_5mzyurjp'}


def fetch(url, **kwargs):
    return urllib.request.urlopen(urllib.request.Request(url, **kwargs), timeout=180)


class RemoteZip(io.RawIOBase):
    """Read archive members without a second full archive on disk."""
    def __init__(self, url):
        self.url, self.position = url, 0
        with fetch(url, method='HEAD') as response:
            self.size = int(response.headers['Content-Length'])

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b''
        with fetch(self.url, headers={'Range':f'bytes={self.position}-{self.position+size-1}'}) as response:
            if response.status != 206:
                raise ValueError('Server does not honor byte ranges')
            data = response.read()
        if len(data) != size:
            raise ValueError('Truncated range response')
        self.position += len(data)
        return data


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix+'.part')
        with fetch(url) as response, partial.open('wb') as stream:
            shutil.copyfileobj(response, stream, 1024*1024)
        partial.replace(destination)
    print(destination.name, destination.stat().st_size, flush=True)


def osf_files(url):
    result = []
    while url:
        with fetch(url) as response:
            data = json.load(response)
        result.extend(data['data']); url = data['links']['next']
    return result


def kaltura(**kwargs):
    url = 'https://cdnapisec.kaltura.com/api_v3/index.php?' + urllib.parse.urlencode({'format':1, **kwargs})
    with fetch(url) as response:
        result = json.load(response)
    if isinstance(result, dict) and result.get('objectType') == 'KalturaAPIException':
        raise ValueError(result['message'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'research/stereotypy-external')
    parser.add_argument('--datasets', nargs='+', choices=['mit','osf','stanford'], default=['mit','osf','stanford'])
    args = parser.parse_args(); out = args.output
    out.mkdir(parents=True, exist_ok=True)
    records = []
    if 'mit' in args.datasets:
        with zipfile.ZipFile(RemoteZip(MIT_URL)) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.endswith(('.mpg','.txt')):
                    continue
                dest = (out/'mit'/member.filename).resolve()
                if not dest.is_relative_to((out/'mit').resolve()):
                    raise ValueError('Unsafe archive member')
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    data = archive.read(member)  # ZIP CRC is verified by zipfile.
                    temp = dest.with_suffix(dest.suffix+'.part'); temp.write_bytes(data); temp.replace(dest)
                records.append(dict(dataset='mit', page='https://cbmm.mit.edu/mouse-dataset',
                    download=MIT_URL, member=member.filename, path=str(dest), sha256=sha256(dest)))
                print(member.filename, flush=True)
    if 'osf' in args.datasets:
        videos = {f['attributes']['name'].removesuffix('.mp4'):f for f in osf_files(OSF_VIDEOS)}
        for annotation in osf_files(OSF_TARGETS):
            stem = annotation['attributes']['name'].removesuffix('.csv')
            if stem not in videos:
                continue
            for item in (annotation, videos[stem]):
                name = item['attributes']['name']
                if Path(name).name != name:
                    raise ValueError('Unsafe remote filename')
                dest = out/'osf'/name; url = item['links']['download']
                download(url, dest)
                records.append(dict(dataset='osf', page='https://osf.io/rwhtd/', download=url,
                    path=str(dest.resolve()), sha256=sha256(dest)))
    if 'stanford' in args.datasets:
        # This is the same public anonymous session used by the embedded player.
        session = kaltura(service='session', action='startWidgetSession', widgetId='_1392761')
        for behavior, entry in ENTRIES.items():
            context = kaltura(service='baseentry', action='getContextData', entryId=entry, ks=session['ks'],
                **{'contextDataParams:objectType':'KalturaEntryContextDataParams',
                   'contextDataParams:referrer':'https://med.stanford.edu/'})
            if any(context.get(k) for k in ('isSiteRestricted','isCountryRestricted','isSessionRestricted','isIpAddressRestricted')):
                raise ValueError('Public media access is restricted')
            asset = max(context['flavorAssets'], key=lambda row:row['width'])
            url = kaltura(service='flavorasset', action='getUrl', id=asset['id'], ks=session['ks'])
            dest = out/'stanford'/(behavior+'.mp4')
            download(url.replace('http://','https://'), dest)
            # Do not retain ephemeral player sessions or signed stream URLs.
            records.append(dict(dataset='stanford', behavior=behavior, entry_id=entry,
                page='https://med.stanford.edu/mousebehavior/ethogram/ethogram-index.html',
                path=str(dest.resolve()), sha256=sha256(dest), annotation='positive video bag only'))
    save_json(out/'acquisition-manifest.json', dict(records=records,
        redistribution='Not established; local research copies only. Do not publish data or weights.'))


if __name__ == '__main__':
    main()
