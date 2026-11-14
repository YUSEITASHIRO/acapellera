"""SongFormer (ASLP-lab/SongFormer, CC BY 4.0) kept loaded, answering one song's sections.

Run by worker.py inside SongFormer's own venv, from songformer/src/SongFormer with PYTHONPATH=.:../third_party:

  python sf_server.py <port> <parent pid>

POST /sections  body = absolute path of an audio file on this machine -> [{"label", "start", "end"}, ...] (SongFormer's
                 own infer.py `inference` loop, unchanged: 8 classes, rule post-processing on); header X-Window =
                 the window used (s)
It ends when its parent (worker.py) is gone.

GPU memory (measured on an 11 GB card, a 247 s song): SongFormer's own 420 s window peaks at 10.4 GB,
so it is never run beside the separation (worker.py asks after demucs); the memory PyTorch keeps after a song is
given back each time. When the 420 s window runs out of memory, the song is done again with a 120 s window (5.7 GB;
on that song the boundaries stayed and 4-5 of 12 labels changed). fp16 (autocast) did not fit either (8.5 GB in use
and 2.3 GB more asked) and is not offered.
"""
import json, os, sys, tempfile, threading, time
import multiprocessing as mp
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, os.getcwd())  # src/SongFormer, as infer.sh runs it
from infer.infer import inference  # noqa: E402

OUT = Path(tempfile.mkdtemp(prefix="songformer-"))
lock = threading.Lock()


class GiveBack:
    """The output queue as `inference` sees it: each answer first returns the cached GPU memory."""

    def __init__(self, q):
        self.q = q

    def put(self, x):
        import torch
        torch.cuda.empty_cache()
        self.q.put(x)


class TakeIn:
    """The input queue as `inference` sees it: an item (path, window) sets the window for that song."""

    def __init__(self, q, args):
        self.q, self.args = q, args

    def get(self):
        item = self.q.get()
        if isinstance(item, tuple):
            item, win = item
            self.args.win_size = self.args.hop_size = win
        return item


def run(rank, q_in, q_out, args, fp16=False):
    inference(rank, TakeIn(q_in, args), GiveBack(q_out), args)


WINDOWS = (420, 120)  # SongFormer's own window first; the short one only when that runs out of memory


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        path = self.rfile.read(int(self.headers["Content-Length"])).decode()
        p = OUT / f"{Path(path).stem}.json"
        p.unlink(missing_ok=True)
        win = 0
        with lock:  # one song at a time through the one model
            for win in WINDOWS:
                q_in.put((path, win))
                q_out.get()
                if p.exists():
                    break
        body = p.read_bytes() if p.exists() else b"[]"
        self.send_response(200 if p.exists() else 500)
        self.send_header("X-Window", str(win))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def watch_parent(parent):
    while True:
        try:
            os.kill(parent, 0)
        except OSError:
            os._exit(0)
        time.sleep(2)


if __name__ == "__main__":
    PORT, PARENT = int(sys.argv[1]), int(sys.argv[2])
    mp.set_start_method("spawn", force=True)
    q_in, q_out = mp.Queue(), mp.Queue()
    args = Namespace(output_dir=str(OUT), win_size=420, hop_size=420, num_classes=128, model="SongFormer",
                     checkpoint="SongFormer.safetensors", config_path="SongFormer.yaml", no_rule_post_processing=False)
    proc = mp.Process(target=run, args=(0, q_in, q_out, args), daemon=True)
    proc.start()
    threading.Thread(target=watch_parent, args=(PARENT,), daemon=True).start()
    print(f"songformer ready on 127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
