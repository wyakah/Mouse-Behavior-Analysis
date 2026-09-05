"""Generate the local priority-model audit report without accepting annotations."""
import argparse,html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stereotypy.training import save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=ROOT/'outputs/stereotypy-training/v2');args=parser.parse_args()
    out=ROOT/'reports/stereotypy-priority';out.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.run/'test-report.json').read_text());selection=json.loads((args.run/'selection.json').read_text())
    experiments=json.loads((args.run/'validation-experiments.json').read_text())
    payload=dict(report=report,selection=selection,experiments=experiments)
    for name in ['test-report.json','selection.json','protocol.json','validation-experiments.json']:
        shutil.copy2(args.run/name,out/name)
    reviews=args.run/'local-review.json'
    if reviews.exists():payload['local']=json.loads(reviews.read_text())
    template=(ROOT/'static/stereotypy-priority-report.html').read_text()
    (out/'index.html').write_text(template.replace('__DATA__',json.dumps(payload,allow_nan=False).replace('<','\\u003c')))
    print('http://127.0.0.1:8765/reports/stereotypy-priority/index.html')

if __name__=='__main__':main()
