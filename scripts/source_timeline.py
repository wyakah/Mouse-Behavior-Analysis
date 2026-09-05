"""Strict source timestamp probe, run with the lightweight .venv interpreter."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from threechamber.core import timeline
if __name__=='__main__':
 t,dt=timeline(sys.argv[1]);print(json.dumps({'timestamps':t.tolist(),'durations':dt.tolist()}))
