"""Run in a separate DeepLabCut environment. Real DLC inference, never substitute blob tracking."""
import os
os.environ.setdefault("MPLCONFIGDIR",str(__import__("pathlib").Path(__file__).resolve().parents[1]/".cache/matplotlib"))
os.environ.setdefault("TORCH_HOME",str(__import__("pathlib").Path(__file__).resolve().parents[1]/".cache/torch"))
import argparse
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('action',choices=['create','extract','label','train','evaluate','infer'])
p.add_argument('--config'); p.add_argument('--video',action='append'); p.add_argument('--output',default='dlc-projects'); p.add_argument('--experimenter',default='Researcher'); p.add_argument('--shuffle',type=int,default=1)
a=p.parse_args()
import deeplabcut as dlc
if a.action=='create':
    config=dlc.create_new_project('ThreeChamber',a.experimenter,[str(Path(v).resolve()) for v in a.video],working_directory=str(Path(a.output).resolve()),copy_videos=False,multianimal=False)
    from deeplabcut.utils import auxiliaryfunctions
    cfg=auxiliaryfunctions.read_config(config)
    cfg['bodyparts']=['left_ear','nose','right_ear','center','left_lateral','right_lateral','tail_base','tail_end']
    cfg['numframes2pick']=30; cfg['pcutoff']=0.6
    auxiliaryfunctions.write_config(config,cfg); print(config)
elif a.action=='extract': dlc.extract_frames(a.config,mode='automatic',algo='kmeans',userfeedback=False)
elif a.action=='label': dlc.label_frames(a.config)
elif a.action=='train':
    dlc.check_labels(a.config)
    dlc.create_training_dataset(a.config,Shuffles=[a.shuffle])
    dlc.train_network(a.config,shuffle=a.shuffle)
elif a.action=='evaluate': dlc.evaluate_network(a.config,Shuffles=[a.shuffle],plotting=True)
elif a.action=='infer':
    Path(a.output).mkdir(parents=True,exist_ok=True)
    import shutil,json,hashlib
    shutil.copy2(a.config,Path(a.output)/'model_config.yaml')
    (Path(a.output)/'inference_manifest.json').write_text(json.dumps({'deeplabcut_version':dlc.__version__,'model_config':str(Path(a.config).resolve()),'config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),'shuffle':a.shuffle,'videos':a.video},indent=2))
    dlc.analyze_videos(a.config,[str(Path(v).resolve()) for v in a.video],shuffle=a.shuffle,save_as_csv=True,destfolder=str(Path(a.output).resolve()))
