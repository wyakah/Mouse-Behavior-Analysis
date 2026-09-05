"""Preserve source videos and prepare the user-confirmed first600second interval."""
from pathlib import Path
import json,subprocess,uuid
import imageio_ffmpeg
from threechamber.core import metadata,timeline,sha256

def prepare_trial(root,source,seconds=600):
    root=Path(root);source=Path(source)
    if source.parent==root/'prepared' or metadata(source)['duration_seconds']<=seconds+.1:return source
    dest=root/'prepared'/f'{source.stem}__first_{seconds:g}s.mp4';dest.parent.mkdir(exist_ok=True)
    # Existing supplied prepared samples have already been strictly decoded and verified.
    side=dest.with_suffix('.json')
    if dest.exists() and side.exists():
        record=json.loads(side.read_text())
        if record.get('requested_interval_s')==[0,seconds] and record.get('decoded_frames',0)>0:return dest
        raise ValueError('A prepared filename exists without a matching verified interval; review the source.')
    partial=dest.with_name(dest.stem+'-'+uuid.uuid4().hex[:6]+'.partial.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-i',str(source),'-t',str(seconds),'-map','0:v:0','-an','-c:v','copy','-movflags','+faststart',str(partial)],check=True)
    t,dt=timeline(partial)
    if t[-1]+dt[-1]<seconds-.2:raise ValueError('Prepared recording ends before the requested scoring window.')
    partial.replace(dest)
    side.write_text(json.dumps({'original':str(source.relative_to(root)),'source_sha256':sha256(source),'prepared':str(dest.relative_to(root)),'prepared_sha256':sha256(dest),'requested_interval_s':[0,seconds],'decoded_frames':len(t),'validation':'Strict complete decoding passed; source untouched.'},indent=2))
    return dest
