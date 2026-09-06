"""Refresh saved Three Chamber presentation from existing scored frames, without rerunning inference."""
import json, shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
from threechamber.core import review_video, metadata
from threechamber.live import LivePublisher
from threechamber.batch import export_workbook
from threechamber.social import stranger_metrics


def refresh(root):
    root=Path(root).resolve()
    backup=root/'.cache'/('review-backup-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    # Derive new fields from saved measured durations, without changing classifications.
    for summary_file in (root/'outputs').glob('*/summary.json'):
        summary=json.loads(summary_file.read_text())
        if 'left_chamber_seconds' not in summary:continue
        summary.update(stranger_metrics(summary))
        summary_file.write_text(json.dumps(summary,indent=2,allow_nan=False))
        pd.DataFrame([summary]).to_csv(summary_file.with_suffix('.csv'),index=False)
    for file in sorted((root/'batches').glob('*/batch.json')):
        batch=json.loads(file.read_text())
        if batch.get('status') not in ('complete','complete_with_errors'):continue
        for i,e in enumerate(batch['entries']):
            if e.get('status')!='complete':continue
            out=root/'outputs'/e['run_id'];rows=pd.read_csv(out/'frames.csv')
            e['summary']=json.loads((out/'summary.json').read_text())
            manifest=json.loads((out/'manifest.json').read_text());video=Path(manifest['video'])
            cfg=json.loads((out/'calibration.json').read_text())
            cfg['recording_metadata']={k:e.get(k,'') for k in ('id','sex','genotype')}
            info=metadata(video)
            temp=file.parent/'presentation-refresh'
            live=LivePublisher(temp,dict(recording_id=e['id'],recording_index=i,source_duration_s=info['duration_seconds'],source_size=[info['width'],info['height']]))
            target=out/'review-refreshed.mp4'
            print('Refreshing',e['run_id'],flush=True)
            review_video(video,rows,cfg,target,live=live)
            old=backup/e['run_id'];old.mkdir(parents=True,exist_ok=True)
            shutil.copy2(out/'review.mp4',old/'review.mp4');target.replace(out/'review.mp4')
            shutil.copy2(out/'calibration.json',old/'calibration.json')
            (out/'calibration.json').write_text(json.dumps(cfg,indent=2))
            generated=temp/'video'/str(i);existing=file.parent/'live/video'/str(i)
            if generated.exists():
                if existing.exists():existing.rename(old/'stream')
                existing.parent.mkdir(parents=True,exist_ok=True);generated.rename(existing)
            shutil.rmtree(temp,ignore_errors=True)
        export_workbook(root,batch,file.parent)
        file.write_text(json.dumps(batch,indent=2,allow_nan=False))
    print('Previous review artifacts preserved at',backup,flush=True)

if __name__=='__main__':
    refresh(Path(__file__).resolve().parents[1])
