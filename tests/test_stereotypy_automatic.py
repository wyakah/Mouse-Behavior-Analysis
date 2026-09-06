import json
from pathlib import Path
from flask import Flask
from stereotypy.automatic import register_automatic,reconcile

class Pool:
    def __init__(self):self.calls=[]
    def submit(self,*args):self.calls.append(args)

def client(tmp_path,monkeypatch):
    monkeypatch.setattr('stereotypy.automatic.model_readiness',lambda root:dict(ready=True))
    app=Flask(__name__);pool=Pool();register_automatic(app,lambda:tmp_path,pool)
    @app.errorhandler(ValueError)
    def error(e):return dict(error=str(e)),400
    (tmp_path/'test.mov').write_bytes(b'test')
    return app.test_client(),pool

def test_starts_without_annotations_or_mapping_and_deduplicates(tmp_path,monkeypatch):
    c,p=client(tmp_path,monkeypatch);body=dict(entries=[dict(video='test.mov',id='Mouse 1',sex='female',genotype='WT')])
    r=c.post('/api/stereotypy/batches',json=body);assert r.status_code==202
    b=r.get_json();assert b['window_seconds']==1200 and b['entries'][0]['sex']=='female'
    assert c.post('/api/stereotypy/batches',json=body).get_json()['id']==b['id']
    assert len(p.calls)==1
    assert not (tmp_path/'sessions').exists()
    assert c.get('/api/stereotypy/batches/'+b['id']).get_json()['status']=='queued'
    assert c.get('/stereotypy-runs/'+b['id']+'/../batch.json').status_code==400

def test_rejects_invalid_queue(tmp_path,monkeypatch):
    c,p=client(tmp_path,monkeypatch)
    for entries in [[],[dict(video='../outside.mov',id='a')],[dict(video='test.mov',id='')],[dict(video='test.mov',id='a')]*21]:
        assert c.post('/api/stereotypy/batches',json=dict(entries=entries)).status_code==400
    assert not p.calls

def test_interrupted_worker_is_retryable(tmp_path):
    path=tmp_path/'batch.json';path.write_text(json.dumps(dict(status='running',entries=[dict(status='running')])))
    assert reconcile(path)['status']=='failed'

def test_legacy_product_urls_never_open_manual_scoring():
    from app import app
    c=app.test_client()
    for path in ['/stereotypy','/stereotypy?session=old-session','/stereotypy/review']:
        response=c.get(path)
        assert response.status_code==200
        assert b'stereotypy-auto.js' in response.data
        assert b'Mark an interval' not in response.data

def test_worker_continues_after_failure_and_excludes_unknown_time(tmp_path,monkeypatch):
    import importlib.util,types,csv
    from stereotypy.training import CLASSES
    spec=importlib.util.spec_from_file_location('queue_worker',Path(__file__).parents[1]/'scripts/run_stereotypy_queue.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    monkeypatch.setattr(worker,'automatic_mapping',lambda _:([0,0,100,100],80))
    monkeypatch.setattr(worker,'export_results',lambda dest,state:None)
    dest=tmp_path/'run';dest.mkdir()
    entries=[dict(id=str(i),video=f'{i}.mov',sex='female',genotype='WT',status='queued') for i in range(2)]
    for e in entries:(tmp_path/e['video']).touch()
    worker.save(dest/'batch.json',dict(entries=entries,status='queued'))
    calls=[]
    def inputs(path):
        key=path.stem.split('_')[0];calls.append(key)
        if key=='001':raise ValueError('Corrupt recording')
        out=dest/'work'/key;out.mkdir(parents=True)
        return key,out,dict(duration_s=1),{}
    def predict(*args):return dict(windows=[dict(start_s=0,end_s=1,scores=dict.fromkeys(CLASSES,.9))],thresholds=dict.fromkeys(CLASSES,.5),duration_s=1,source_gaps=[],summary=[])
    def render(path,key,out,index,mapping,result,**kwargs):
        from stereotypy.exclusive import ExclusiveScorer
        scorer=ExclusiveScorer(result['thresholds'])
        scores=dict.fromkeys(CLASSES,.1);scores['grooming']=.9
        scorer.add(0,.4,scores);scorer.add(.4,1,scores,False)
        result.update(summary=scorer.summary(),cumulative=scorer.snapshot(),bouts=[],scoring_policy='exclusive-highest-qualified-v1')
        kwargs['progress'](scorer.snapshot())
    fake=types.SimpleNamespace(torch=types.SimpleNamespace(backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda:False))),MAPPINGS={},inputs=inputs,features=lambda *args:None,predict=predict,render=render)
    worker.run(dest,fake)
    state=json.loads((dest/'batch.json').read_text())
    assert calls==['001','002'] and state['status']=='complete_with_errors'
    good=state['entries'][1];assert good['sex']=='female' and good['genotype']=='WT'
    assert len(good['summary'])==6
    assert sum(b['candidate_seconds'] for b in good['summary'])==.4
    assert all(b['unknown_seconds']==.6 for b in good['summary'])
