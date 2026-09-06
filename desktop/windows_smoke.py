"""Exercise both bundled pipelines using synthetic video; never an accuracy benchmark."""
import argparse,json,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'desktop'))
import cv2,numpy as np
from smoke_test import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args();bundle=a.bundle.resolve()
    temp=ROOT/'.cache'/('windows-smoke-'+uuid.uuid4().hex[:8]);temp.mkdir(parents=True)
    video=temp/'synthetic.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(1024,768))
    for i in range(120):
        im=np.full((768,1024,3),245,np.uint8);cv2.rectangle(im,(190,175),(750,575),(160,160,160),-1)
        for x in [377,563]:cv2.line(im,(x,175),(x,575),(210,210,210),2)
        for x in [290,650]:cv2.circle(im,(x,360),40,(65,65,65),-1)
        x=220+i*2;cv2.ellipse(im,(x,470),(17,30),20,0,360,(25,25,25),-1)
        writer.write(im)
    writer.release();ready=temp/'ready.json';log=(temp/'service.log').open('w')
    process=subprocess.Popen([str(bundle/'runtime/python.exe'),'-I','-B',str(bundle/'payload/desktop/service.py'),'--bundle',str(bundle),'--workspace',str(temp/'workspace'),'--ready',str(ready)],stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            if ready.exists():break
            if process.poll() is not None:raise RuntimeError((temp/'service.log').read_text())
            time.sleep(1)
        if not ready.exists():raise RuntimeError('Packaged service did not start')
        for assay in ['three_chamber','stereotypy']:
            run(argparse.Namespace(ready=ready,video=video,seconds=6,assay=assay))
    finally:
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        process.wait();log.close()
if __name__=='__main__':main()
