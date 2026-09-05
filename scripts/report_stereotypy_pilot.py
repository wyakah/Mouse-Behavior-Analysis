"""Register immutable local pilot copies and build a two-recording review report."""
from pathlib import Path
import sys,json,uuid,shutil,hashlib,argparse
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stereotypy.core import BEHAVIORS,ETHOGRAM_VERSION
parser=argparse.ArgumentParser();parser.add_argument('--register-only',choices=['745','729']);args=parser.parse_args()

now=datetime.now(timezone.utc).isoformat()
report=ROOT/'reports/stereotypy-pilot';report.mkdir(parents=True,exist_ok=True)
sessions=[];cards=[]
for animal in ['745','729']:
    if args.register_only and animal!=args.register_only:continue
    pilot=ROOT/'outputs/stereotypy-pilot'/animal
    result=json.loads((pilot/'manifest.json').read_text());source=json.loads((pilot/'source-index.json').read_text())
    sid=uuid.uuid5(uuid.NAMESPACE_URL,'stereotypy-pilot:'+source['source_sha256']).hex
    runid=hashlib.sha256((pilot/'manifest.json').read_bytes()).hexdigest()[:32]
    destination=ROOT/'outputs/stereotypy'/sid/runid
    if not destination.exists():shutil.copytree(pilot,destination)
    file=ROOT/'labeling/stereotypy'/f'{sid}.json';file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists():record=json.loads(file.read_text())
    else:
        record=dict(id=sid,video=result['source'],animal_id=animal,sex='unknown',genotype='',
            session_id='feasibility-'+animal,apparatus_id='Not recorded',view='side',created_at=now,
            video_manifest=source,start_s=0,end_s=source['duration_s'],merge_gap_s=0,
            ethogram_version=ETHOGRAM_VERSION,ethogram_status='draft',definitions=dict(BEHAVIORS),
            revisions=[dict(revision=0,created_at=now,annotator_id=None,annotations=[],start_s=0,end_s=source['duration_s'],merge_gap_s=0)],
            cage_mapping=result['mapping'],cage_revision=1,
            cage_history=[dict(revision=1,changed_at=now,mapping=result['mapping'])])
    if not any(r['id']==runid for r in record.get('cage_runs',[])):
        record.setdefault('cage_runs',[]).append(dict(id=runid,created_at=now,cage_revision=1,
            directory=str(destination.relative_to(ROOT)),result=result))
    temp=file.with_suffix('.tmp');temp.write_text(json.dumps(record,allow_nan=False));temp.replace(file)
    base=f'/api/stereotypy/sessions/{sid}/cage/{runid}'
    sessions.append(dict(animal_id=animal,session_id=sid,run_id=runid,base=base))
    cards.append(f'''<article><div class="card-title"><h2>Mouse {animal}</h2><span>{source['duration_s']:.2f} seconds · {source['frame_count']:,} frames</span></div>
    <video controls preload="metadata" poster="{base}/poster.jpg" src="{base}/review.mp4"></video>
    <div class="card-body"><div class="metric"><strong>{result['proposal_coverage_percent']:.2f}%</strong><span>body proposal availability<br><b>Accuracy is not validated</b></span></div>
    <p>All source frames processed. No automatic behavior totals have been produced.</p>
    <div class="actions"><a class="primary" href="/stereotypy?session={sid}">Review & label behaviors →</a><a href="{base}/frame-features.csv?download=1">Frame data CSV</a><a href="{base}/manifest.json">Run details</a></div></div></article>''')
if args.register_only:
    print(json.dumps(sessions));sys.exit(0)
(report/'sessions.json').write_text(json.dumps(sessions,indent=2))
# Diagnostic frame grids are review images, not measurement videos.
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stereotypy · Feasibility results</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f1;color:#213e34;font:15px/1.6 system-ui,sans-serif}main{max-width:1220px;margin:auto;padding:36px 28px}header{display:flex;justify-content:space-between;font-size:13px;margin-bottom:42px}a{color:#27684f;text-decoration:none}h1{font-size:38px;line-height:1.2;letter-spacing:-1px;margin:8px 0 14px}h2{font-size:23px;margin:0}h3{margin:0 0 10px}.eyebrow{font-size:11px;letter-spacing:2px;font-weight:700}.intro{max-width:790px;color:#65756b;margin-bottom:28px}.notice{background:#f0e9d7;border:1px solid #e1d2a8;padding:17px 22px;border-radius:12px;margin:24px 0;font-size:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}article,.panel{background:white;border:1px solid #dce2d8;border-radius:16px;overflow:hidden}.card-title{padding:20px 22px;display:flex;justify-content:space-between;align-items:center}.card-title span{font-size:12px;color:#69796f}video{width:100%;display:block;background:#1b2622}.card-body{padding:22px}.metric{display:flex;gap:18px;align-items:center}.metric strong{font-size:35px;letter-spacing:-1px}.metric span{font-size:12px;color:#67776e}.metric b{font-weight:500;color:#926621}.card-body p{font-size:13px;color:#67776e}.actions{display:flex;align-items:center;flex-wrap:wrap;gap:16px;font-size:12px}.primary{background:#255d46;color:white;padding:11px 15px;border-radius:8px}.panel{padding:26px;margin-top:24px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e4e9e0}th{font-size:12px;color:#69796f}.panel p{color:#63736a;max-width:980px}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:25px;margin-top:20px}.steps b{font-size:14px}.steps p{font-size:13px}details{margin-top:20px}summary{cursor:pointer;font-size:14px;font-weight:600}details img{width:100%;margin-top:18px;border-radius:8px}footer{font-size:12px;color:#79877d;margin-top:28px}@media(max-width:850px){.grid,.steps{grid-template-columns:1fr}.card-title{display:block}main{padding:22px 16px}h1{font-size:30px}}
</style></head><body><main><header><a href="/#tests">← Behavior Analysis</a><span>LOCAL FEASIBILITY STUDY · 05 SEP 2026</span></header><div class="eyebrow">STEREOTYPY / TWO REAL RECORDINGS</div><h1>A clearer view of the mouse.</h1><p class="intro">The cage is mapped, the full recordings are processed, and the first side-view pose models have been tested. Watch the focused reviews below, then inspect suggested clips in the behavior reviewer.</p>
<div class="notice"><b>Ready for assisted review. Automatic behavior scoring still needs training.</b><br>Body boxes and upright flags are experimental proposals. Grooming, digging, gnawing, rearing, jumping and circling remain unscored until reviewed.</div><div class="grid">'''+''.join(cards)+'''</div>
<section class="panel"><h2>What the pretrained trial showed</h2><p>24 time-spaced source frames per mouse. A tighter mouse box improves nose availability, but it does not establish anatomical accuracy. The full review videos above use foreground body proposals, not these unvalidated pose predictions.</p><table><thead><tr><th>Nose likelihood ≥ 0.6</th><th>745</th><th>729</th></tr></thead><tbody><tr><td>Default side-view DeepLabCut detector</td><td>2 / 24</td><td>3 / 24</td></tr><tr><td>DeepLabCut given a tighter mouse box</td><td>8 / 24</td><td>13 / 24</td></tr></tbody></table><p>All 48 diagnostic frames received a foreground body proposal. Paws remained inconsistent. Likelihood and proposal coverage are not accuracy measures.</p>
<details><summary>Inspect all 48 pose samples</summary><p>Each numbered image is a separate time-spaced sample. Yellow points are generic Quadruped model predictions above the cutoff, including points that may be incorrect or irrelevant to mice.</p><img src="conditioned-0.jpg" alt="24 experimental body-conditioned pose samples from mouse 745"><img src="conditioned-1.jpg" alt="24 experimental body-conditioned pose samples from mouse 729"></details></section>
<section class="panel"><h2>The next accuracy step</h2><div class="steps"><div><b>01 · Correct visible anatomy</b><p>Nose, mid-back, neck, tail base and paws. Include turns, upright poses and occlusions. Hidden points stay unknown.</p></div><div><b>02 · Label the six behaviors</b><p>Use complete short clips with explicit positives, negatives and uncertainty. Keep ordinary turns separate from completed circles, and rearing separate from jumping.</p></div><div><b>03 · Test on new mice</b><p>Validate durations and event counts separately. Two mice establish feasibility; they cannot establish 95–99% behavioral accuracy.</p></div></div><p>Methods follow the side-view <a href="https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2024.1340357/full">DLC/SimBA grooming study</a> and <a href="https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html">DeepLabCut’s side-view model guidance</a>. Fine movements hidden by bedding, scratches or the hopper need review; cropping cannot recover them.</p></section><footer>Original recordings preserved · original frame timestamps retained · audited timing corrections · local files only · no genotype inferred from cage labels</footer></main></body></html>'''
if (ROOT/'reports/stereotypy-training/index.html').exists():
    page=page.replace('<div class="notice">','<p><a class="primary" href="/reports/stereotypy-training/index.html">Open the six-behavior training results →</a></p><div class="notice">',1)
(report/'index.html').write_text(page)
print('http://127.0.0.1:8765/reports/stereotypy-pilot/index.html')
