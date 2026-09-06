"""Exercise a packaged service using its real HTTP upload and queue APIs."""
import argparse,json,time,subprocess
from pathlib import Path
import httpx
import imageio_ffmpeg

def run(args):
    ready=json.loads(args.ready.read_text());url=ready['url'];base=url.split('/desktop/')[0]
    client=httpx.Client(base_url=base,timeout=60,follow_redirects=True)
    assert httpx.get(base+'/api/videos').status_code==403
    assert client.get(url).status_code==200
    assert client.get('/api/desktop').json()['offline']
    assert client.get('/api/stereotypy/automatic/readiness').json()['ready']
    clip=args.ready.parent/'smoke-input.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(args.video),'-t',str(args.seconds),'-an','-c:v','libx264','-preset','ultrafast',str(clip)],check=True)
    with clip.open('rb') as f:r=client.post('/api/videos/upload',files={'file':('desktop-smoke.mp4',f,'video/mp4')})
    r.raise_for_status();video=r.json()['name']
    if args.assay=='stereotypy':
        body={'entries':[{'video':video,'id':'desktop-smoke','sex':'unknown','genotype':''}]};endpoint='/api/stereotypy/batches'
    else:
        seeded=client.post('/api/batch/seed',json={'video':video}).json()
        seeded['config'].update(pcutoff=.6,confirmed=True);seeded['reviewed']=True
        body=dict(entries=[seeded],diameter_px=seeded['config']['cup_circles']['diameter_px'],pcutoff=.6)
        endpoint='/api/batches'
    r=client.post(endpoint,json=body);r.raise_for_status();identifier=r.json()['id']
    start=time.monotonic();last=None;live=False
    while time.monotonic()-start<900:
        state=client.get(endpoint+'/'+identifier).json();entry=state['entries'][0]
        value=(state['status'],entry['status'],entry.get('stage'))
        if value!=last:print(value,flush=True);last=value
        preview=client.get(endpoint+'/'+identifier+'/video/0').json().get('manifest')
        if preview and preview.get('segments'):live=True
        if state['status'] not in ('queued','running'):break
        time.sleep(2)
    assert state['status']=='complete',state
    if args.assay=='stereotypy':
        assert 'results.xlsx' in state['downloads'],state
        assert abs(entry['duration_s']-args.seconds)<.2
        assert sum(s['candidate_seconds'] or 0 for s in entry['summary'])<=args.seconds+.1
        download=client.get('/stereotypy-runs/'+identifier+'/results.xlsx')
        assert entry['video_file']
    else:download=client.get('/batches/'+identifier+'/results.xlsx')
    assert download.status_code==200 and download.content[:2]==b'PK'
    (args.ready.parent/(args.assay+'-results.xlsx')).write_bytes(download.content)
    report=dict(assay=args.assay,status=state['status'],batch_id=identifier,live_segments_observed=live,elapsed_seconds=round(time.monotonic()-start,1),entry=entry)
    (args.ready.parent/(args.assay+'-smoke.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='entry'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ready',type=Path,required=True);p.add_argument('--video',type=Path,required=True);p.add_argument('--seconds',type=float,default=12);p.add_argument('--assay',choices=['stereotypy','three_chamber'],default='stereotypy');run(p.parse_args())
