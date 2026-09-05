"""Bounded all-frame video preview. Immutable fMP4 segments, atomic manifests."""
from fractions import Fraction
from pathlib import Path
from queue import Queue, Full
from threading import Thread
import json
import time
import uuid
import warnings
import av
from threechamber.live import atomic_json
from threechamber.annotation import AnnotationRenderer


class SegmentWriter:
    """Encode consecutive source frames; each independently playable segment starts at zero."""
    def __init__(self, folder, cfg, source_size, fps, context, segment_seconds=2.):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.renderer=AnnotationRenderer(cfg,source_size)
        self.fps=Fraction(str(fps)).limit_denominator(100000)
        self.segment_seconds=segment_seconds;self.target=None;self.count=0;self.last_time=-1
        self.expected_frames=context.get('expected_frames')
        self.generation=uuid.uuid4().hex
        self.manifest=dict(schema=1,generation=self.generation,status='streaming',segments=[],frame_count=0,
                           available_until_s=0.,mime='video/mp4; codecs="avc1.42c01f"',
                           recording_id=context['recording_id'],recording_index=context['recording_index'],
                           source_duration_s=context['source_duration_s'],stage=context.get('stage','tracking'))
        self.persist()

    def persist(self):
        self.manifest['updated_at']=time.time()
        atomic_json(self.folder/'manifest.json',self.manifest)

    def _open(self,index,time_s):
        self.start=time_s;self.first=index
        name=f'{self.generation}-{len(self.manifest["segments"]):05d}.mp4'
        self.path=self.folder/name;self.temporary=self.path.with_suffix('.part')
        self.target=av.open(str(self.temporary),'w',format='mp4',options={'movflags':'empty_moov+default_base_moof+frag_keyframe+skip_trailer'})
        self.stream=self.target.add_stream('libx264',rate=self.fps)
        self.stream.width=self.renderer.width;self.stream.height=self.renderer.height
        self.stream.pix_fmt='yuv420p';self.stream.time_base=Fraction(1,1000000)
        self.stream.codec_context.time_base=self.stream.time_base
        self.stream.options={'preset':'veryfast','tune':'zerolatency','crf':'22','profile':'baseline','level':'3.1','bf':'0','g':'100000'}

    def _publish(self):
        if self.target is None:return
        for packet in self.stream.encode():self.target.mux(packet)
        self.target.close();self.target=None
        self.temporary.replace(self.path)
        self.manifest['segments'].append(dict(file=self.path.name,start_s=self.start,end_s=self.end,
                                              first_frame=self.first,last_frame=self.count-1,frames=self.count-self.first))
        self.manifest.update(frame_count=self.count,available_until_s=self.end)
        self.persist()

    def write(self,image,index,time_s,duration_s,row,image_is_crop=False,annotated=False):
        if index!=self.count or time_s<=self.last_time or duration_s<=0:
            raise ValueError('Preview frames must be consecutive with increasing source timestamps.')
        if self.target is not None and time_s-self.start>=self.segment_seconds:self._publish()
        if self.target is None:self._open(index,time_s)
        im=image if annotated else self.renderer.draw(image,index,time_s,row,image_is_crop)
        frame=av.VideoFrame.from_ndarray(im,format='bgr24');frame.pts=round((time_s-self.start)*1e6)
        frame.time_base=Fraction(1,1000000);frame.duration=round(duration_s*1e6)
        for packet in self.stream.encode(frame):
            packet.duration=round(duration_s/float(packet.time_base));self.target.mux(packet)
        self.count+=1;self.last_time=time_s;self.end=time_s+duration_s

    def close(self,error=None):
        if not error and self.expected_frames is not None and self.count!=self.expected_frames:
            error=f'Preview ended after {self.count} of {self.expected_frames} frames. Check the final review output.'
        if error:
            if self.target:self.target.close();self.target=None
            if hasattr(self,'temporary'):self.temporary.unlink(missing_ok=True)
            self.manifest.update(status='failed',error=str(error))
        else:
            self._publish();self.manifest['status']='complete'
        self.persist()


class VideoPublisher:
    """A bounded encoding worker; overload or failures stop preview, never scientific work."""
    def __init__(self, live, cfg, fps, capacity=64):
        self.folder=live.folder/'video'/str(live.context['recording_index'])
        self.queue=Queue(maxsize=capacity);self.failure=None;self.closed=False
        self.writer=None;self.live=live;self.cfg=cfg;self.fps=fps
        self.worker=Thread(target=self._run,name='annotated-preview',daemon=True);self.worker.start()

    def _run(self):
        try:
            self.writer=SegmentWriter(self.folder,self.cfg,self.live.context['source_size'],self.fps,
                                      dict(self.live.context,stage=self.live.stage,expected_frames=self.live.total_frames))
            while True:
                item=self.queue.get()
                if item is None:break
                if self.failure:break
                self.writer.write(*item)
            self.writer.close(self.failure)
        except Exception as exc:
            self.failure=f'Video preview unavailable: {type(exc).__name__}'
            warnings.warn(self.failure+'; analysis continues.',RuntimeWarning)
            if self.writer:
                try:self.writer.close(self.failure)
                except Exception:pass
        finally:
            while not self.queue.empty():self.queue.get_nowait()

    def submit(self,image,index,time_s,duration_s,row,image_is_crop=False,annotated=False):
        if self.failure or self.closed:return False
        try:
            self.queue.put_nowait((image.copy(),index,float(time_s),float(duration_s),dict(row),image_is_crop,annotated))
            return True
        except Full:
            self.failure='Video preview encoder could not keep up; final review remains available after analysis.'
            warnings.warn(self.failure,RuntimeWarning)
            return False

    def close(self):
        if self.closed:return
        self.closed=True
        # At most one bounded queue of frames remains; no playback-speed pacing.
        if self.worker.is_alive():
            while self.worker.is_alive():
                try:self.queue.put(None,timeout=.1);break
                except Full:continue
            self.worker.join(timeout=15)
            if self.worker.is_alive():self.failure='Preview encoder did not finish in time.'


def start_video(live,cfg,fps):
    if live is None or live.disabled or 'source_size' not in live.context:return None
    folder=live.folder/'video'/str(live.context['recording_index'])
    try:
        existing=json.loads((folder/'manifest.json').read_text())
        if existing.get('status')=='complete':return None
    except (OSError,ValueError):pass
    try:return VideoPublisher(live,cfg,fps)
    except Exception as exc:
        warnings.warn(f'Video preview unavailable; analysis continues: {type(exc).__name__}',RuntimeWarning)
        return None
