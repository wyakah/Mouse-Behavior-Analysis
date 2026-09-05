"""Start the local server if needed, then open it once it is ready."""
from pathlib import Path
import subprocess,sys,time,urllib.request,webbrowser
root=Path(__file__).resolve().parents[1];url='http://127.0.0.1:8765'
def ready():
    try:
        with urllib.request.urlopen(url,timeout=1) as r:return b'Three Chamber' in r.read()
    except Exception:return False
if ready():webbrowser.open(url)
else:
    server=subprocess.Popen([sys.executable,str(root/'app.py')],cwd=root)
    try:
        for _ in range(60):
            if server.poll() is not None:raise SystemExit('Server failed to start. Check messages above.')
            if ready():break
            time.sleep(.5)
        else:raise SystemExit('Server did not become ready.')
        webbrowser.open(url);server.wait()
    except KeyboardInterrupt:server.terminate();server.wait()
