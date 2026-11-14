"""Acapellera's launcher: the web app, the worker (here or on a GPU server over ssh), the drop server, and the
stage page in the browser. Ctrl+C stops everything it started.

  python acapellera/launch.py                      the worker runs on this machine
  python acapellera/launch.py --remote             the worker runs on the server named in acapellera/remote.json

acapellera/remote.json (not in git; your host stays on your machine):
  {"host": "<ssh host or alias>", "repo": "<this repository's checkout on the server>", "home": "<ACAPELLERA_HOME there, optional>"}
The server needs `bash acapellera/server/setup.sh` and `bash acapellera/server/setup_engine.sh` once (README).
Locally the worker uses $ACAPELLERA_HOME/venv if it exists (default ~/.acapellera), else this Python.
"""
import json, os, shutil, subprocess, sys, threading, time, urllib.request, webbrowser
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WEB_PORT, DROP_PORT, WORKER_PORT = 5317, 8768, 8770
WEB = f"http://127.0.0.1:{WEB_PORT}/"
WORKER = f"http://127.0.0.1:{WORKER_PORT}"
started: list[subprocess.Popen] = []


def run(cmd, cwd=None, env=None, hidden=False) -> subprocess.Popen:
    if hidden and os.name == "nt":
        # ssh -tt in this console would switch it to raw input, and Ctrl+C would go to the server instead of
        # stopping this launcher: ssh gets a hidden console of its own, its output is copied here
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        threading.Thread(target=lambda: [print(l.decode("utf-8", "replace"), end="", flush=True) for l in p.stdout],
                         daemon=True).start()
    else:
        p = subprocess.Popen(cmd, cwd=cwd, env=env)
    started.append(p)
    return p


def answers(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2).close()
        return True
    except OSError:
        return False


def wait_for(url: str, proc: subprocess.Popen, what: str):
    while not answers(url):
        if proc.poll() is not None:
            raise SystemExit(f"{what} did not start (exit {proc.returncode})")
        time.sleep(0.5)


def worker_local() -> subprocess.Popen:
    home = Path(os.environ.get("ACAPELLERA_HOME", Path.home() / ".acapellera")).expanduser()
    py = next((p for p in (home / "venv" / "Scripts" / "python.exe", home / "venv" / "bin" / "python") if p.exists()),
              Path(sys.executable))
    return run([str(py), str(HERE / "server" / "worker.py"), str(WORKER_PORT)], env=dict(os.environ, PYTHONUTF8="1"))


def worker_remote() -> subprocess.Popen:
    conf = HERE / "remote.json"
    if not conf.exists():
        raise SystemExit(f"--remote needs {conf} (see the top of this file)")
    c = json.loads(conf.read_text(encoding="utf-8"))
    host, repo = c["host"], c["repo"].rstrip("/")
    home = c.get("home", "~/.acapellera")
    ssh, scp = shutil.which("ssh"), shutil.which("scp")
    if not ssh or not scp:
        raise SystemExit("ssh and scp are needed for --remote")
    # this checkout's server scripts go up at every start, so what runs there is what is here
    files = [str(p) for p in sorted((HERE / "server").glob("*.py"))]
    if subprocess.call([scp, "-q", *files, f"{host}:{repo}/acapellera/server/"]) != 0:
        raise SystemExit("could not copy the server scripts to the server")
    cmd = (f"cd {repo} && export ACAPELLERA_HOME={home} && source {home}/venv/bin/activate && PYTHONUTF8=1 "
           f"python acapellera/server/worker.py {WORKER_PORT}")
    return run([ssh, "-tt", "-o", "ExitOnForwardFailure=yes", "-L", f"{WORKER_PORT}:localhost:{WORKER_PORT}", host, cmd],
               hidden=True)


def web_app() -> subprocess.Popen | None:
    if answers(WEB):
        print(f"the web app is already running on {WEB_PORT} (not stopped by this launcher)")
        return None
    bun = shutil.which("bun")
    if not bun:
        raise SystemExit("bun is needed for the web app (https://bun.sh)")
    p = run([bun, "run", "dev", "--port", str(WEB_PORT)], cwd=REPO)
    print("starting the web app (the first start builds the engine, about a minute)...", flush=True)
    wait_for(WEB, p, "the web app")
    return p


def stop_all():
    """What this launcher started, with everything those started in turn (bun's dev server)."""
    procs = []
    for p in started:
        try:
            q = psutil.Process(p.pid)
            procs += q.children(recursive=True) + [q]
        except psutil.NoSuchProcess:
            pass
    for q in procs:
        try:
            q.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=8)
    for q in alive:
        q.kill()
    print(f"stopped {len(procs)} process(es).", flush=True)


def main():
    remote = "--remote" in sys.argv[1:]
    worker = worker_remote() if remote else worker_local()
    web_app()
    print("waiting for the worker (models loading)...", flush=True)
    wait_for(WORKER + "/ping", worker, "the worker")
    drop = run([sys.executable, str(HERE / "drop_server.py"), str(DROP_PORT), WEB, WORKER])
    time.sleep(1.5)
    webbrowser.open(f"{WEB}?stage={DROP_PORT}")
    print("Acapellera: drop a song file on the stage page. Ctrl+C to stop.", flush=True)
    while drop.poll() is None and worker.poll() is None:
        time.sleep(0.5)


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopping...", flush=True)
    finally:
        stop_all()
