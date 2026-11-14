"""What the stage page (`?stage` in the web app) talks to: it takes the dropped song, has the worker make
the show, unpacks it, and reports how far along it is.

  python drop_server.py <port> <web app url> [worker url]

POST /upload  body = the song, header X-Name (url-encoded file name)  -> "ok" | 409 while busy
GET  /status  -> {"lines", "done", "show", "pct", "stage", "wav", "error", "wav_failed", "notices"}: pct 0-100
              over the whole way, stage = the step now; wav = the show's mix.wav has arrived (fetched after the
              show, for the download button); error = what went wrong, in plain words; notices = made, but
              without something (e.g. no sections), shown after the song only
"""
import http.client, io, json, re, sys, tarfile, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

HERE = Path(__file__).parent
SHOWS = HERE.parent / "public" / "shows"
PORT, WEB = int(sys.argv[1]), sys.argv[2]
WORKER = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8770"  # server/worker.py (here, or through the ssh tunnel)
# Where each step sits on the 0-100 bar: from the times measured on 2026-11-12 (49.7 s song: send ~1 s,
# separation 1.4-3.1, transcription 2.0-2.9, render 10.5, pack 0.4, fetch ~4 s).
SPAN = {"送信": (0, 5), "分離": (5, 20), "採譜": (20, 30), "書き出し": (30, 85), "まとめ": (85, 88), "送信待ち": (88, 88), "回収": (88, 100)}
job = {"lines": [], "done": False, "show": None, "busy": False, "pct": 0, "stage": "待機", "wav": False,
       "error": None, "wav_failed": False, "notices": []}
remote = {"on": False}  # while the worker works, /status asks the worker how far it is


# What the page's lines say: only what is being done now, one line per step. Details go to the console.
DOING = {"受信": "曲を受け取りました", "送信": "曲を送っています", "分離": "歌声と楽器を分けています",
         "採譜": "音符を読み取っています", "書き出し": "歌を作っています", "まとめ": "仕上げています",
         "回収": "舞台に運んでいます", "完了": "準備ができました"}


# Made, but not quite as asked: one plain line each (analyze.py's secs["notes"])
NOTES = {"no_sections": "曲の区切り（サビなど）が取れなかったので、サビでの強調なしで作りました",
         "short_window": "曲が長いので、曲の区切りは簡易な方法で取りました"}


def plain(err: str) -> str:
    """An error, as what happened and what to do."""
    e = err.lower()
    if "out of memory" in e or "alloc_failed" in e:
        return "計算に使うメモリが足りず、止まりました。少し待ってから、もう一度ドロップしてください。"
    if "connection refused" in e or "timed out" in e or "remote end closed" in e or "10061" in e:
        return "処理する側につながりませんでした。起動した窓を閉じて、起動し直してください。"
    if "format not recognised" in e or "could not open" in e or "audioread" in e or "nobackenderror" in e:
        return "この形式のファイルは読めませんでした。mp3 や wav の曲を落としてください。"
    return "途中で止まりました。もう一度ドロップしてください。続くときは黒い窓に詳しい内容が出ています。"


def unshuffled(b: bytes) -> bytes:
    """server/analyze.shuffled undone: the .tele's float32 bytes back in their order (the same bytes as rendered)."""
    import numpy as np
    k = (len(b) - 48) // 4
    body = np.frombuffer(b[48:48 + 4 * k], dtype=np.uint8).reshape(4, k)
    return b[:48] + body.T.tobytes() + b[48 + 4 * k:]


def say(line: str):
    """console only"""
    print(time.strftime("%H:%M:%S ") + line, flush=True)


def at(stage: str, frac: float):
    a, b = SPAN.get(stage, (job["pct"], job["pct"]))
    if stage != job["stage"] and stage in DOING:
        job["lines"].append(DOING[stage])
    job["stage"], job["pct"] = stage, max(job["pct"], round(a + (b - a) * min(1.0, frac), 1))


def worker_progress():
    try:
        with urllib.request.urlopen(WORKER + "/progress", timeout=2) as r:
            p = json.loads(r.read())
        if remote["on"] and p["stage"] in SPAN:
            at(p["stage"], p["frac"])
    except OSError:
        pass


def run(song: Path, show: str, name: str):
    t = time.time()
    try:
        if song.suffix.lower() in (".wav", ".aif", ".aiff"):  # lossless, about half the bytes to send
            import soundfile
            flac = song.with_suffix(".flac")
            data, rate = soundfile.read(song)
            soundfile.write(flac, data, rate, format="FLAC")
            say(f"FLAC に圧縮: {song.stat().st_size / 1e6:.1f} -> {flac.stat().st_size / 1e6:.1f} MB")
            song = flac
        body = song.read_bytes()
        u = urlparse(WORKER)
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
        c.putrequest("POST", "/run")
        for k, v in {"X-Show": show, "X-Ext": song.suffix, "X-Title": quote(Path(name).stem),
                     "Content-Length": str(len(body))}.items():
            c.putheader(k, v)
        c.endheaders()
        say(f"worker へ送信 ({len(body) / 1e6:.1f} MB)")
        for k in range(0, len(body), 1 << 18):
            c.send(body[k:k + (1 << 18)])
            at("送信", (k + (1 << 18)) / len(body))
        say("worker で分離・採譜・歌の書き出し")
        remote["on"] = True
        r = c.getresponse()
        remote["on"] = False
        if r.status != 200:
            raise RuntimeError("worker: " + r.read().decode(errors="replace")[-1500:])
        say(f"worker の内訳 (s): {r.getheader('X-Seconds')}")
        for n in json.loads(r.getheader("X-Seconds") or "{}").get("notes", []):
            job["notices"].append(NOTES.get(n, n))  # shown after the song only, never under Play (StagePage)
        size, buf = int(r.getheader("Content-Length")), bytearray()
        while chunk := r.read(1 << 16):
            buf += chunk
            at("回収", len(buf) / size)
        say(f"回収 {len(buf) / 1e6:.1f} MB")
        with tarfile.open(fileobj=io.BytesIO(bytes(buf))) as tar:
            tar.extractall(SHOWS / show, filter="data")
        for p in (SHOWS / show).glob("*.tele.shuf"):  # analyze.shuffled, put back in order
            p.with_suffix("").write_bytes(unshuffled(p.read_bytes()))
            p.unlink()
        at("回収", 1.0)
        at("完了", 1.0)
        say(f"完了: ドロップから {time.time() - t:.1f} s")
        job["show"] = show
    except Exception as e:  # the page says what happened and what to do; the details stay in the console
        say(f"エラー: {e}")
        job["error"] = plain(str(e))
        job["stage"] = "エラー"
    finally:
        remote["on"] = False
        job["done"], job["busy"] = True, False
    if job["show"]:  # the wav for the download button, after the show is already playable
        try:
            with urllib.request.urlopen(f"{WORKER}/wav/{show}", timeout=600) as r:
                (SHOWS / show / "mix.wav").write_bytes(r.read())
            job["wav"] = True
            say("wav をダウンロードできます")
        except OSError as e:
            say(f"wav の取得に失敗: {e}")
            job["wav_failed"] = True


class H(BaseHTTPRequestHandler):
    def send(self, code, body: bytes, ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")  # the stage page is served by the web app's port
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # the browser asks first because of the X-Name header
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "X-Name, Content-Type")  # a dropped File carries its type
        self.end_headers()

    def do_GET(self):
        if remote["on"]:
            worker_progress()
        self.send(200, json.dumps(job).encode(), "application/json")

    def do_POST(self):
        if self.path != "/upload" or job["busy"]:
            return self.send(409, "処理中です。終わってからドロップしてください。".encode())
        name = unquote(self.headers.get("X-Name", "song"))
        ext = re.sub(r"[^.\w]", "", Path(name).suffix)[:8] or ".wav"
        show = time.strftime("sing-%Y%m%d-%H%M%S")
        song = HERE / "in" / f"{show}{ext}"  # an ascii name for the worker and the URL; the title keeps the original
        song.parent.mkdir(exist_ok=True)
        song.write_bytes(self.rfile.read(int(self.headers["Content-Length"])))
        (song.parent / f"{show}.name.txt").write_text(name, encoding="utf-8")
        job.update(lines=[DOING["受信"]], done=False, show=None, busy=True, pct=0, stage="受信", wav=False,
                   error=None, wav_failed=False, notices=[])
        say(f"受信: {name} ({song.stat().st_size / 1e6:.1f} MB)")
        threading.Thread(target=run, args=(song, show, name), daemon=True).start()
        self.send(200, b"ok")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    print(f"stage API: http://127.0.0.1:{PORT}/status", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
