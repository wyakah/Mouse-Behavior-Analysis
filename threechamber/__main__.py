"""Headless scoring using the same engine as the local UI."""
import argparse,json
from .core import analyze
p=argparse.ArgumentParser(description='Score DLC nose proximity and body-center chamber occupancy.')
p.add_argument('--video',required=True);p.add_argument('--tracks',required=True)
p.add_argument('--calibration',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
with open(a.calibration) as f:cfg=json.load(f)
print(json.dumps(analyze(a.video,a.tracks,cfg,a.output,print),indent=2))
