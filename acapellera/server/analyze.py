"""One song -> the show the web app plays, all on the machine that runs the worker (a GPU server, or this PC).

  python analyze.py <song file> <out dir> <show id> [title]     (one song; worker.py keeps it loaded instead)

1. demucs (htdemucs_6s, GPU) splits the song into vocals, drums, bass, guitar, piano, other.
2. SongFormer (sf_server.py) finds the sections, after the separation has freed the GPU, beside step 3.
3. basic-pitch (ONNX on the CPU, one model, the four stems at once) turns vocals, bass, guitar, piano into note
   events [start s, end s, midi, amplitude]; "other" is separated but not transcribed.
4. The drum stem's onsets in three bands: kick (low), snare (mid), hat (high).
5. make_sing.py (next to this file) renders the twelve voices with the CHIMERA CLI on all cores.
6. show.tgz: what the page needs (show.json, mix.ogg, v*.tele, v*.json), for the PC to fetch.
"""
import io, json, sys, tarfile, time
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict  # ONNX: TensorFlow is not installed in this venv
from demucs.api import Separator
from concurrent.futures import ThreadPoolExecutor

# One basic-pitch model for every stem and song, the stems transcribed at once (measured 2026-11-13 on a 247 s
# song's four stems: 13.7 s one after another with a path each time, 5.8 s like this; the same 3,915 notes)
BP_MODEL = Model(ICASSP_2022_MODEL_PATH)


def shuffled(b: bytes) -> bytes:
    """A .tele file with its float32 bytes grouped by byte position (all first bytes, then all second bytes, ...):
    the same bytes, packed 18% smaller by gzip (measured: 57.3 -> 46.8 MB for nine voices of a 247 s song).
    drop_server.py puts them back in order."""
    k = (len(b) - 48) // 4
    body = np.frombuffer(b[48:48 + 4 * k], dtype=np.uint8).reshape(k, 4)
    return b[:48] + body.T.tobytes() + b[48 + 4 * k:]

sys.path.insert(0, str(Path(__file__).parent))
import make_sing  # noqa: E402

BANDS = {"kick": (20, 150), "snare": (150, 2500), "hat": (6000, 11025)}
SF_PORT = 8772  # sf_server.py (SongFormer), started by worker.py


def sections_async(song: Path) -> dict:
    """Ask SongFormer for the song's sections in a thread, while the rest goes on; `res["sections"]` when done
    (left empty when SongFormer is not there or fails: the song is then made without them)."""
    import threading, urllib.request
    res = {"sections": [], "seconds": None, "window": None}

    def ask():
        t = time.time()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{SF_PORT}/sections", data=str(song.resolve()).encode(), method="POST")
            with urllib.request.urlopen(req, timeout=600) as r:
                res["sections"] = json.loads(r.read())
                res["window"] = int(r.headers.get("X-Window", 0))
        except Exception as e:
            print("sections: none,", e, flush=True)
        res["seconds"] = round(time.time() - t, 1)

    res["thread"] = threading.Thread(target=ask, daemon=True)
    res["thread"].start()
    return res


# One separation, six stems: vocals, drums, bass, guitar, piano, other. Measured on a 247 s song: 7.9 s, 619 MiB of
# GPU (an 11 GB card).
def separator() -> Separator:
    return Separator("htdemucs_6s", device="cuda" if torch.cuda.is_available() else "cpu")


def level_db(x: np.ndarray, sr: int) -> list[float]:
    """The level every 10 ms (make_sing.ENV_HOP), in dB, for telling a held note from a rest."""
    hop = int(sr * 0.01)
    n = len(x) // hop
    rms = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9
    return [round(float(v), 1) for v in 20 * np.log10(rms)]


def process(sep: Separator, song: Path, out: Path, show_id: str, title: str = "", progress=lambda stage, f: None) -> tuple[Path, dict]:
    """progress(stage name, fraction of that stage) as it goes: 分離, 採譜, 書き出し, まとめ."""
    t = {"start": time.time()}
    out.mkdir(parents=True, exist_ok=True)
    progress("分離", 0.0)
    sep.update_parameter(callback=lambda d: d["state"] == "end" and progress("分離", min(1.0, d["segment_offset"] / max(1, d["audio_length"]))))
    _, stems = sep.separate_audio_file(song)
    sr = sep.samplerate
    stems = {k: v.cpu() for k, v in stems.items()}
    # SongFormer needs nearly all of an 11 GB GPU for a 4-minute song (sf_server.py): it runs only now, after
    # the separation has given its GPU memory back, beside the transcription (which runs on the CPU)
    torch.cuda.empty_cache()
    sec = sections_async(song)
    t["demucs"] = time.time()
    res, env = {}, {}
    # "other" is not transcribed: nobody sings it
    melodic = ("vocals", "bass", "guitar", "piano")
    progress("採譜", 0.0)
    for stem in melodic:
        x = stems[stem].mean(0).cpu().numpy()
        env[stem] = level_db(x, sr)
        soundfile.write(out / f"{stem}.wav", x, sr)
    done = [0]

    def transcribe(stem):
        _, _, events = predict(str(out / f"{stem}.wav"), BP_MODEL)
        done[0] += 1
        progress("採譜", done[0] / len(melodic))
        return sorted([round(s, 3), round(e, 3), int(n), round(float(a), 3)] for s, e, n, a, *_ in events)

    def vocal_onsets():
        """The vocals' onset strength every 10 ms, for make_sing.resung (the measure RESING's values were chosen on)."""
        x, rate = soundfile.read(out / "vocals.wav")
        y = librosa.resample(x, orig_sr=rate, target_sr=22050)
        return [round(float(v), 3) for v in librosa.onset.onset_strength(y=y, sr=22050, hop_length=220)]

    with ThreadPoolExecutor(len(melodic) + 1) as ex:
        onsets = ex.submit(vocal_onsets)
        res.update(zip(melodic, ex.map(transcribe, melodic)))
        res["onset"] = {"vocals": onsets.result()}
    res["env"] = env
    t["basic_pitch"] = time.time()
    # drums: onsets of the band energy. ponytail: fixed band edges, no drum-transcription model (kept light)
    y = librosa.resample(stems["drums"].mean(0).cpu().numpy(), orig_sr=sr, target_sr=22050)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    f = librosa.fft_frequencies(sr=22050, n_fft=1024)
    for name, (lo, hi) in BANDS.items():
        env = librosa.onset.onset_strength(S=librosa.power_to_db(S[(f >= lo) & (f < hi)]), sr=22050, hop_length=256)
        on = librosa.onset.onset_detect(onset_envelope=env, sr=22050, hop_length=256, units="time", backtrack=False)
        res[name] = [round(float(x), 3) for x in on]
    t["drums"] = time.time()
    res["duration"] = round(len(y) / 22050, 3)
    sec["thread"].join()
    res["sections"] = sec["sections"]
    res["sections_window"] = sec["window"]
    t["sections_wait"] = time.time()  # how long the sections kept everything else waiting
    print(f"sections ({sec['seconds']} s, window {sec['window']} s):", [(s["label"], round(s["start"], 1)) for s in sec["sections"]], flush=True)
    (out / "notes.json").write_text(json.dumps(res), encoding="utf-8")
    print("notes:", {k: len(v) for k, v in res.items() if isinstance(v, list)}, flush=True)
    show = out / "show"
    progress("書き出し", 0.0)
    make_sing.main(out / "notes.json", show_id, title, str(show), lambda f: progress("書き出し", f))
    t["render"] = time.time()
    progress("まとめ", 0.0)
    tgz = out / "show.tgz"
    with tarfile.open(tgz, "w:gz", compresslevel=1) as tar:
        for p in sorted(show.iterdir()):
            if p.suffix == ".tele":  # sent as <name>.tele.shuf, regrouped (see shuffled)
                data = shuffled(p.read_bytes())
                info = tarfile.TarInfo(p.name + ".shuf")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif p.suffix != ".wav":  # the wavs stay here (check_sing.py; mix.wav is fetched afterwards)
                tar.add(p, arcname=p.name)
    t["pack"] = time.time()
    steps = ["start", "demucs", "basic_pitch", "drums", "sections_wait", "render", "pack"]
    secs = {b: round(t[b] - t[a], 1) for a, b in zip(steps, steps[1:])}
    # what the page should tell (drop_server.py words it): the song was made, but without or with rougher sections
    secs["notes"] = ["no_sections"] if not sec["sections"] else ["short_window"] if sec["window"] and sec["window"] < 420 else []
    print("worker seconds:", secs, f"show.tgz {tgz.stat().st_size / 1e6:.1f} MB", flush=True)
    return tgz, secs


if __name__ == "__main__":
    process(separator(), Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
