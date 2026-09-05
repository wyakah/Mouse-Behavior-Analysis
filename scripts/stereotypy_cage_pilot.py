"""Run the two supplied feasibility recordings; mappings are recording-specific."""
from pathlib import Path
import sys
import json
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stereotypy.pilot import run_pilot
from stereotypy.video import index_video
parser=argparse.ArgumentParser();parser.add_argument('--only',choices=['745','729']);args=parser.parse_args()

for name,roi,floor in [('745_stereotypy.MOV',[1200,245,3120,1090],865),
                       ('729_stereotypy.mov',[300,340,3210,1590],1270)]:
    if args.only and not name.startswith(args.only):continue
    out=ROOT/'outputs/stereotypy-pilot'/name[:3]
    out.mkdir(parents=True,exist_ok=True)
    print('Indexing',name,flush=True)
    manifest=json.loads((out/'source-index.json').read_text()) if (out/'source-index.json').exists() else index_video(ROOT/name)
    (out/'source-index.json').write_text(json.dumps(manifest))
    result=run_pilot(ROOT/name,dict(crop_xyxy=roi,floor_y=floor),out,manifest,
        lambda n,total:print(name,n,'/',total,flush=True))
    print(name,result['proposal_seconds'],flush=True)
