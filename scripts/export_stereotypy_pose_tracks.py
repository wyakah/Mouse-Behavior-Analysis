"""Export raw landmark coordinates with explicit scene/body eligibility per sample."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from stereotypy.robustness import LANDMARKS


def main():
    report=ROOT/'reports/stereotypy-robustness'
    for mouse in ['710','711','712','743','745']:
        folder=ROOT/'outputs/stereotypy-robustness'/mouse;npz=np.load(folder/'pose.npz');parts=list(npz['parts']);meta=json.loads((folder/'manifest.json').read_text())
        a,b,c,d=meta['signature']['mapping']['crop_xyxy'];rows=json.loads((report/f'{mouse}.json').read_text())['rows']
        with (report/f'{mouse}-body-center.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['mouse_id','start_s','end_s','source_x','source_y','speed_cage_width_s','method','anatomy_validated'])
            for row in rows:
                center=row['evidence']['body_box_center']
                writer.writerow([mouse,row['start_s'],row['end_s'],center[0]+a if center else '',center[1]+b if center else '',row['motion']['box_speed_cage_width_s'],'untrained_body_box_center',False])
        with (report/f'{mouse}-landmarks.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['mouse_id','source_frame','start_s','end_s','landmark','crop_x','crop_y','source_x','source_y','model_likelihood','above_cutoff_in_crop','scene_valid','body_eligible','anatomy_validated'])
            for row,pose in zip(rows,npz['poses']):
                for name in LANDMARKS:
                    x,y,q=pose[parts.index(name)];finite=bool(np.isfinite([x,y,q]).all());available=finite and q>=.6 and 0<=x<c-a and 0<=y<d-b
                    writer.writerow([mouse,row['frame_id'],row['start_s'],row['end_s'],name,*([float(x),float(y),float(x+a),float(y+b),float(q)] if finite else ['']*5),bool(available),row['scene_valid'],row['body_valid'],False])
if __name__=='__main__':main()
