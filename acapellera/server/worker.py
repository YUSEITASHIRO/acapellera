"""analyze.py kept loaded while the launcher runs, so a song pays no imports, model load or ssh connection
(3.1 s of imports and loads, measured). Reached on 127.0.0.1 only: directly, or through the launcher's ssh tunnel.

  python worker.py <port> [idle minutes = 60]

POST /run  body = the song file, headers X-Show (id), X-Ext (.flac ...), X-Title (url-encoded)
           -> 200 show.tgz bytes, header X-Seconds (json of the step times) | 500 the error text
GET  /progress -> {"stage", "frac"} of the song being made
GET  /wav/<show id> -> that show's mix as wav
GET  /ping -> "ok"
Started by acapellera/launch.py (over `ssh -tt` for a server), so it ends with the launcher (Ctrl+C). If the PC
side vanishes without closing the ssh, it ends itself after the idle minutes (keeps a forgotten one off a shared GPU).
Its songs and results live under ACAPELLERA_HOME (default ~/.acapellera): in/, out/, and songformer/ if installed.
"""
import json, os, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import analyze

ROOT = Path(os.environ.get("ACAPELLERA_HOME", Path.home() / ".acapellera")).expanduser()
(ROOT / "in").mkdir(parents=True, exist_ok=True)
(ROOT / "out").mkdir(parents=True, exist_ok=True)
PORT, IDLE = int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
# SongFormer for the sections, in its own venv (it pins torch 2.4.0; setup_songformer.sh); it ends with this process
SF = ROOT / "songformer" / "src" / "SongFormer"
SF_PY = next((p for p in (ROOT / "songformer" / "venv" / "bin" / "python", ROOT / "songformer" / "venv" / "Scripts" / "python.exe")
              if p.exists()), None)  # without SongFormer the songs are made without sections
if SF_PY:
    import subprocess
    subprocess.Popen([str(SF_PY), str(Path(__file__).parent / "sf_server.py"),
                      str(analyze.SF_PORT), str(os.getpid())], cwd=SF,
                     env=dict(os.environ, PYTHONPATH="../third_party", HF_HOME=str(ROOT / "songformer" / "hf"), OMP_NUM_THREADS="1"))
sep = analyze.separator()  # htdemucs_6s, the only separation
last, busy = time.time(), False
prog = {"stage": "待機", "frac": 0.0}  # the step the song is in, and how far into it (GET /progress)


class H(BaseHTTPRequestHandler):
    def reply(self, code, body: bytes, headers=()):
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/progress":
            return self.reply(200, json.dumps(prog).encode(), [("Content-Type", "application/json")])
        if self.path.startswith("/wav/"):  # /wav/<show id>: the mix as wav, for the page's download button
            p = ROOT / "out" / Path(self.path[5:]).name / "show" / "mix.wav"
            return self.reply(200, p.read_bytes(), [("Content-Type", "audio/wav")]) if p.exists() else self.reply(404, b"no such show")
        self.reply(200, b"ok")

    def do_POST(self):
        global last, busy
        last, busy = time.time(), True
        # a new song starts from nothing: the last song's "送信待ち 100%" left here made the bar jump to 88% the moment
        # the next song had been sent
        prog.update(stage="受信", frac=0.0)
        try:
            show = self.headers["X-Show"]
            song = ROOT / "in" / (show + self.headers.get("X-Ext", ".flac"))
            song.write_bytes(self.rfile.read(int(self.headers["Content-Length"])))
            tgz, secs = analyze.process(sep, song, ROOT / "out" / show, show, unquote(self.headers.get("X-Title", "")),
                                        lambda stage, f: prog.update(stage=stage, frac=round(f, 3)))
            prog.update(stage="送信待ち", frac=1.0)
            self.reply(200, tgz.read_bytes(), [("Content-Type", "application/gzip"), ("X-Seconds", json.dumps(secs))])
        except Exception:
            self.reply(500, traceback.format_exc().encode())
        last, busy = time.time(), False

    def log_message(self, *a):
        pass


def idle_stop():
    while busy or time.time() - last < IDLE * 60:  # never during a song, however long it takes
        time.sleep(10)
    print(f"no request for {IDLE:g} min: stopping", flush=True)
    os._exit(0)


threading.Thread(target=idle_stop, daemon=True).start()
print(f"worker ready on 127.0.0.1:{PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()  # /progress answers while /run works
