"""Acapellera, run by analyze.py: notes.json -> twelve CHIMERA voices -> a show for the stage page.

  python make_sing.py <notes.json> <show id> [title] [out dir]

A cappella: nothing of the original recording plays.
Stems from htdemucs_6s; no note dropped for its strength (MIN_AMP all 0).
  lead     vocals' strongest note at a time; the word by the note's pitch class (LEAD; a sharp sings its natural's)
  harmony  do = the vocals' next strongest note, ree = the next after do (rank_lanes)
  drums    kick "bah", snare "pa", hat "si" (staccato)
  chorus   bass zoo (lower) / di (upper), guitar no / ku, piano ru / mi: each stem's two strongest notes at a time;
           in the drums' back row (bass left of the guitar), at the harmony's level
"other" is no longer sung. Lead, harmony and chorus hold each note to the part's own next note when that comes
within REST_MIN; same-pitch neighbours are held on or sung again per RESING (lead and harmony only).
"""
import json, os, struct, subprocess, sys, tempfile, time, wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
# the CHIMERA CLI of this repository (`cargo build --release -p chimera-cli`), or CHIMERA_CLI
CLI = Path(os.environ.get("CHIMERA_CLI", REPO / "target" / "release" / ("chimera.exe" if os.name == "nt" else "chimera")))
SHOWS = REPO / "public" / "shows"
# dictionary spellings (CMUdict): zu and ri are not in it, zoo and ree are said the same
# the lead's word by pitch class, C, C#, ..., B; a sharp sings its natural's word
LEAD = ["mi", "mi", "pa", "pa", "zoo", "di", "di", "ree", "ree", "ru", "ru", "ma"]
# Knobs (tune by ear):
MIN_NOTE = 0.08   # s; a shorter note is joined to the previous one (or dropped)
# Levels of the groups: lead > harmony and chorus > drums
LEAD_DB = 8.0
HARMONY_DB = -2.0
DRUM_DB = -5.0
# The chorus squad: from the bass, guitar and piano stems, each stem's two strongest notes at a time, the lower to
# the first syllable, the upper to the second.
CHORUS = {"bass": ("zoo", "di"), "guitar": ("no", "ku"), "piano": ("ru", "mi")}
# Notes under these amplitudes are dropped, per stem, before any voice is chosen from it (0: none dropped; kept as a
# knob). The drums have no amplitude.
MIN_AMP = {"vocals": 0.0, "bass": 0.0, "other": 0.0, "guitar": 0.0, "piano": 0.0}


def loud(src, stem):
    """The stem's notes at or above MIN_AMP[stem]."""
    return [e for e in src[stem] if e[3] >= MIN_AMP[stem]]
# each voice's group, in show.json's order: lead, do, ree, bah, pa, si, then the chorus squad zoo, di, no, ku, ru, mi
# (the chorus squad stands with the drums and sings at the harmony's level)
GROUPS = ["vocal", "harmony", "harmony", "drums", "drums", "drums"] + ["chorus"] * 6
# the chorus squad at the harmony's level
CHORUS_DB = HARMONY_DB
GROUP_DB = {"vocal": LEAD_DB, "harmony": HARMONY_DB, "drums": DRUM_DB, "chorus": CHORUS_DB}
# Everyone louder where the song lifts, by SongFormer's section labels (the stage also enlarges the lead there).
LIFT_DB = {"chorus": 3.0, "pre-chorus": 1.0, "prechorus": 1.0}
LIFT_RAMP = 0.5  # s, in and out of a section


def ramped(db: np.ndarray, ramp: float = LIFT_RAMP) -> np.ndarray:
    """dB steps -> gain, each step a `ramp` s ramp (a moving average by cumulative sums: np.convolve took ~14 s)."""
    k = int(ramp * FS)
    if k > 1 and db.any():
        c = np.concatenate([[0.0], np.cumsum(np.pad(db, (k // 2, k - k // 2 - 1), mode="edge"))])
        db = (c[k:] - c[:-k]) / k
    return 10 ** (db / 20)


def lift(n: int, sections) -> np.ndarray:
    """Every voice's gain over the song (n samples at FS): LIFT_DB of each section's label, ramped; the same for all."""
    db = np.zeros(n)
    for s in sections:
        v = LIFT_DB.get(str(s["label"]).lower(), 0.0)
        if v:
            db[int(s["start"] * FS):int(s["end"] * FS)] = v
    return ramped(db)


# Each note as loud to the ear as the others of its part, whatever its pitch. The ear's sensitivity by frequency is ISO 226:2003's equal-loudness
# contour at PHON; each rendered note's spectrum is weighted by it (not the note's fundamental alone: a voice's harmonics
# are heard too, and the fundamental alone would lift an 82 Hz note by about 18 dB). The note's gain brings its weighted
# level to the part's mean; the group levels (LEAD_DB ...) then set the parts apart as before.
# ISO 226:2003 Table 1 (f Hz, alpha_f, L_U dB, T_f dB), typed in and checked only by 1 kHz giving L_p = L_N
# (test_make_sing.py); compare with the standard before relying on it elsewhere.
ISO226 = np.array([
    (20, 0.532, -31.6, 78.5), (25, 0.506, -27.2, 68.7), (31.5, 0.480, -23.0, 59.5), (40, 0.455, -19.1, 51.1),
    (50, 0.432, -15.9, 44.0), (63, 0.409, -13.0, 37.5), (80, 0.387, -10.3, 31.5), (100, 0.367, -8.1, 26.5),
    (125, 0.349, -6.2, 22.1), (160, 0.330, -4.5, 17.9), (200, 0.315, -3.1, 14.4), (250, 0.301, -2.0, 11.4),
    (315, 0.288, -1.1, 8.6), (400, 0.276, -0.4, 6.2), (500, 0.267, 0.0, 4.4), (630, 0.259, 0.3, 3.0),
    (800, 0.253, 0.5, 2.2), (1000, 0.250, 0.0, 2.4), (1250, 0.246, -2.7, 3.5), (1600, 0.244, -4.1, 1.7),
    (2000, 0.243, -1.0, -1.3), (2500, 0.243, 1.7, -4.2), (3150, 0.243, 2.5, -6.0), (4000, 0.242, 1.2, -5.4),
    (5000, 0.242, -2.1, -1.5), (6300, 0.245, -7.1, 6.0), (8000, 0.254, -11.2, 12.6), (10000, 0.271, -10.7, 13.9),
    (12500, 0.301, -3.1, 12.3)])
PHON = 60.0         # the listening level the contour is taken at (a knob)
NOTE_GAIN_MAX = 12.0  # dB, the most a note is raised or lowered
NOTE_RAMP = 0.03    # s, the gain moves between notes over this


def equal_loudness_spl() -> np.ndarray:
    """ISO 226:2003: the SPL (dB) at each Table 1 frequency that sounds as loud as PHON phon."""
    f, a, lu, tf = ISO226.T
    af = 4.47e-3 * (10 ** (0.025 * PHON) - 1.15) + (0.4 * 10 ** ((tf + lu) / 10 - 9)) ** a
    return 10 / a * np.log10(af) - lu + 94


def ear_weight(freqs: np.ndarray) -> np.ndarray:
    """Power weight of each frequency for the ear (1 at 1 kHz): 10^((PHON - SPL needed)/10), flat beyond the table."""
    db = PHON - np.interp(np.log(np.maximum(freqs, 1.0)), np.log(ISO226[:, 0]), equal_loudness_spl())
    return 10 ** (db / 10)


def note_gains(y: np.ndarray, notes) -> np.ndarray:
    """A gain over the part's wav `y`: each note's ear-weighted level brought to the part's mean (within NOTE_GAIN_MAX)."""
    spans = [(int(s * FS), min(int(e * FS), len(y))) for s, e, *_ in notes]
    spans = [(a, b) for a, b in spans if b - a > 64]
    if not spans:
        return np.ones(len(y))
    levels = []
    for a, b in spans:
        x = np.abs(np.fft.rfft(y[a:b])) ** 2
        p = (x * ear_weight(np.fft.rfftfreq(b - a, 1 / FS))).sum() / (b - a) ** 2
        levels.append(10 * np.log10(p + 1e-20))
    sounding = [v for v in levels if v > -150]
    if not sounding:
        return np.ones(len(y))
    ref = float(np.mean(sounding))
    db = np.zeros(len(y))
    for (a, b), v in zip(spans, levels):
        db[a:b] = np.clip(ref - v, -NOTE_GAIN_MAX, NOTE_GAIN_MAX) if v > -150 else 0.0
    return ramped(db, NOTE_RAMP)


DRUM = {"kick": ("bah", 43, 0.15), "snare": ("pa", 57, 0.12), "hat": ("si", 74, 0.08)}  # syllable, midi, max s
# ponytail: one fixed range per body, octave shifts into it; measure the engine's real range if notes sound wrong
RANGE = {"male": (40, 64), "female": (55, 79), "child": (60, 84)}


OVERLAP = 0.1  # s; notes overlapping by no more than this are one after the other (legato), not together
# Legato: every part but the drums keeps singing up to its next note unless a rest comes (REST_MIN, legato).
ENV_HOP = 0.01    # s; the stems' level is given every 10 ms (analyze.py)
SILENT_DB = 30.0  # below the stem's loud level (95th percentile) by this much = silent
# A gap of REST_MIN or more before the part's own next note is a rest; every shorter gap is held over (see legato).
REST_MIN = 0.5  # s


# The lead's same-pitch runs: inside a run of one pitch there is no rest (RESING decides held or sung again there),
# however long the gap; the rest is judged only before the run's first note and after its last.
RUN_NO_REST = {"lead": True}


def legato(notes, env_db=None, same_pitch_runs=False):
    """Hold each note on to the part's own next note when that comes within REST_MIN; a longer gap is a rest.
    Judged on the part's own notes, not on its stem's level (on the stem, a harmony taken from the vocals would never
    rest while the lead sang).
    `env_db` is no longer used here (kept so the callers stay as they were). With `same_pitch_runs`, a gap between
    two notes of the same pitch is always held over (RUN_NO_REST)."""
    if not notes:
        return notes, 0
    out, held = [list(n) for n in notes], 0
    for a, b in zip(out, out[1:]):
        if a[1] < b[0] and (b[0] < a[1] + REST_MIN - 1e-9 or (same_pitch_runs and a[2] == b[2])):  # under REST_MIN held
            a[1] = b[0]
            held += 1
    return out, held


def rank_lanes(events, k):
    """The strongest note at each moment to lane 0, the next strongest to lane 1, and so on to lane k-1
    (lead = the vocals' strongest note, do = the next, ree = the next after do): notes are taken
    strongest first, each into the first lane where nothing already sounds then; with every lane taken, dropped."""
    out = [[] for _ in range(k)]
    for e in sorted(events, key=lambda e: -e[3]):
        for lane in out:
            if all(min(x[1], e[1]) - max(x[0], e[0]) <= OVERLAP for x in lane):
                lane.append(e)
                break
    return [sorted(lane) for lane in out]


# Same pitch twice: held on as one note, or sung again without a rest. A limit on the stem's silence between the
# two could not tell them apart (on a test song most same-pitch neighbours had no gap at all), so the second note is sung again only where the vocals both dip between the two (DIP_DB) and start anew at the second
# (onset at least ONSET_PCT percent of the song's onsets); otherwise it is held on (a separation dropout).
# Only neighbours closer than REST_MIN (no limit inside the lead's runs, RUN_NO_REST). The chorus is never joined.
# The lead: dip >= 2 dB and an onset in the song's top 50%.
RESING = {"lead": (2.0, 50.0), "do": (6.0, 80.0), "ree": (6.0, 80.0)}  # (DIP_DB, ONSET_PCT) per voice
# Per voice, whether RESING is used; False: no same-pitch notes are joined, each is sung again.
JOIN_SAME = {"lead": True, "do": False, "ree": False}
DIP_EDGE = 0.03  # s; the dip is looked for from this much before the gap to this much after


def resung(a, b, env_db, onset, rule) -> bool:
    """Whether same-pitch note b, after a, is sung again (see RESING). Without the onset curve (a notes.json made
    before it was stored) the dip alone decides."""
    dip_min, pct_min = rule
    if not env_db:
        return False
    env = np.asarray(env_db, dtype=float)
    k = lambda t: max(0, min(len(env), int(t / ENV_HOP)))
    around = np.r_[env[k(a[0]):k(a[1])], env[k(b[0]):k(b[1])]]
    gap = env[k(a[1] - DIP_EDGE):k(b[0] + DIP_EDGE) + 1]
    if not len(around) or not len(gap) or float(np.median(around) - gap.min()) < dip_min:
        return False
    if not onset:
        return True
    on = np.asarray(onset, dtype=float)
    i = k(b[0])
    peak = on[max(0, i - 3):i + 4].max() if i < len(on) else 0.0
    return float((on < peak).mean() * 100) >= pct_min


def merge_same(notes, env_db, onset, rule, max_gap=None):
    """Join a note to the one before when both have the same pitch, they are closer than REST_MIN, and the second is
    not sung again (resung); the joined note is sung once, held to the later one's end. `max_gap` (default REST_MIN):
    the lead passes no limit, since its same-pitch runs have no rests inside (RUN_NO_REST)."""
    max_gap = REST_MIN if max_gap is None else max_gap
    out, merged = [], 0
    for n in notes:
        if out and out[-1][2] == n[2] and n[0] - out[-1][1] < max_gap and not resung(out[-1], n, env_db, onset, rule):
            out[-1][1] = max(out[-1][1], n[1])
            merged += 1
        else:
            out.append(list(n))
    return out, merged


def pick(events, k):
    """At most k notes sounding at once, the strongest kept (events: [start, end, midi, amp])."""
    kept = []
    for e in sorted(events, key=lambda e: -e[3]):
        if sum(1 for x in kept if min(x[1], e[1]) - max(x[0], e[0]) > OVERLAP) < k:
            kept.append(e)
    return sorted(kept)


def lanes(events):
    """Two strongest notes at once -> a lower and an upper lane."""
    low, high = [], []
    for e in pick(events, 2):  # in order of start
        other = next((x for x in low + high if x[0] < e[1] and e[0] < x[1]), None)
        if other is None:  # alone: the lane whose last note is nearer
            lp, hp = (low[-1][2] if low else 48), (high[-1][2] if high else 72)
            (low if abs(e[2] - lp) <= abs(e[2] - hp) else high).append(e)
        elif e[2] < other[2]:  # under the sounding note: this one low, that one high
            if other in low:
                low.remove(other); high.append(other)
            low.append(e)
        else:
            if other in high:
                high.remove(other); low.append(other)
            high.append(e)
    return sorted(low), sorted(high)


def mono(events):
    """One line: the strongest of overlapping notes kept, overlaps cut at the next start, short notes
    joined to the previous one or dropped. Dropped counts every note lost on the way."""
    kept = pick(events, 1)
    out, joined = [], 0
    for s, e, p, a in kept:
        if out and s < out[-1][1]:
            out[-1][1] = s
        if e - s < MIN_NOTE:
            if out and s - out[-1][1] < 0.05:
                out[-1][1] = e; joined += 1
            continue
        out.append([s, e, p, a])
    out = [n for n in out if n[1] - n[0] > 0.02]
    return out, joined, len(events) - len(out) - joined


def body_for(notes):
    m = float(np.median([n[2] for n in notes])) if notes else 60
    return "male" if m < 55 else "female" if m < 70 else "child"


def to_melody(notes, syllables, body):
    """[[midi, beats]] at bpm 60 (a beat is a second), rests filling the gaps; octave-shifted into the body's range."""
    lo, hi = RANGE[body]
    seq, t, shifted = [], 0.0, 0
    for (s, e, p, _), _syl in zip(notes, syllables):
        if s - t > 0.001:
            seq.append([0, round(s - t, 4)])
        q = p
        while q < lo: q += 12
        while q > hi: q -= 12
        shifted += q != p
        seq.append([q, round(e - s, 4)])
        t = e
    return {"bpm": 60, "t0": 0, "lyrics": " ".join(syllables), "notes": seq}, shifted


FS, HOP = 48000, 768  # the CLI's rate and telemetry hop (render.rs TELE_HOP)
LEAD_IN = 0.3   # s before a chunk's first note (the CLI's default t0: consonants land before the beat)
SPLIT_GAP = 1.0  # s; a rest at least this long always starts a new chunk (rendering it would cost more)
# A longer chunk is cut at its next rest of SHORT_GAP or more, so all cores have work. Measured on the 49.7 s
# test (2026-11-12): 4.0/0.2 left a 33.3 s chunk (render 23.1 s); 2.0/0.05 gives 128 chunks, longest 6.2 s.
MAX_CHUNK = 2.0  # s
SHORT_GAP = 0.05
# A held line (legato) has no rest to cut at. Measured 2026-11-12: with legato the render went from 10.5 s to
# 29-30 s (one long chunk per phrase). So a long line is cut inside a note of at least XF_MIN: both pieces
# sing that note, and one fades into the other across XF (fractions of the note), after the next piece's
# consonant, so the voice never stops.
XF_MIN = 0.15  # s
XF = (0.35, 0.85)


def chunks(melody):
    """A voice's melody (bpm 60, t0 0) -> pieces rendered on their own:
    (start s, melody of the piece, fade-in window or None, fade-out window or None), windows in song seconds."""
    notes, t, syl = [], 0.0, iter(melody["lyrics"].split())
    for p, b in melody["notes"]:
        if p > 0:
            notes.append((t, t + b, p, next(syl)))
        t += b
    groups = []
    for n in notes:
        if groups:
            g = groups[-1]
            gap, length = n[0] - g[-1][1], g[-1][1] - g[0][0]
            if gap < SPLIT_GAP and not (length >= MAX_CHUNK and gap >= SHORT_GAP):
                g.append(n)
                continue
        groups.append([n])
    pieces = []  # (notes, fade in, fade out)
    for g in groups:
        st, xin = 0, None
        for j in range(1, len(g) - 1):
            s, e = g[j][0], g[j][1]
            if s - g[st][0] >= MAX_CHUNK and e - s >= XF_MIN and j > st:
                win = (s + XF[0] * (e - s), s + XF[1] * (e - s))
                pieces.append((g[st:j + 1], xin, win))
                st, xin = j, win
        pieces.append((g[st:], xin, None))
    out = []
    for g, xin, xout in pieces:
        s0 = int(max(0.0, g[0][0] - LEAD_IN) * FS) // HOP * HOP / FS  # on the telemetry grid
        seq, t = [], g[0][0]
        for s, e, p, _ in g:
            if s - t > 0.001:
                seq.append([0, round(s - t, 4)])
            seq.append([p, round(e - s, 4)])
            t = e
        out.append((s0, {"bpm": 60, "t0": round(g[0][0] - s0, 4), "lyrics": " ".join(n[3] for n in g), "notes": seq}, xin, xout))
    return out


def render_chunk(stem: Path, melody: dict, body: str):
    stem.with_suffix(".json").write_text(json.dumps(melody), encoding="utf-8")
    r = subprocess.run([str(CLI), "render", "notes", str(stem.with_suffix(".json")), str(stem), body,
                        "--tele", str(stem.with_suffix(".tele"))], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{stem.name}: chimera exited {r.returncode}: {r.stdout} {r.stderr}")
    with wave.open(str(stem.with_suffix(".wav"))) as w:
        y = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    return y, stem.with_suffix(".tele").read_bytes()


def fade(w: np.ndarray, s0: float, win, up: bool) -> np.ndarray:
    """w (starting at s0 s) faded in (up) or out across win = (a, b) song seconds, equal power."""
    t = s0 + np.arange(len(w)) / FS
    u = np.clip((t - win[0]) / (win[1] - win[0]), 0.0, 1.0)
    return w * (np.sin(u * np.pi / 2) if up else np.cos(u * np.pi / 2))


def assemble(pieces, n: int) -> tuple[np.ndarray, bytes]:
    """[(start s, wav, tele bytes, fade-in window, fade-out window)] of one voice, in time order -> its whole
    wav (n samples) and telemetry. Frames between pieces repeat the next piece's first frame (the voice at
    rest, before it sounds); across a fade the telemetry switches pieces at the window's middle."""
    y = np.zeros(n)
    head = pieces[0][2][:48]
    frame_len = struct.unpack_from("<I", head, 20)[0]
    total = n // HOP + 1
    frames = np.empty((total, frame_len), dtype="<f4")
    fill_from = 0
    for s0, w, tele, xin, xout in pieces:
        k0 = round(s0 * FS) // HOP
        if xin:
            w = fade(w, s0, xin, True)
        if xout:
            w = fade(w, s0, xout, False)
        y[k0 * HOP:k0 * HOP + len(w)] += w[:n - k0 * HOP]
        f = np.frombuffer(tele[48:], dtype="<f4").reshape(-1, frame_len)
        a = round(sum(xin) / 2 * FS / HOP) if xin else k0
        b = min(round(sum(xout) / 2 * FS / HOP) if xout else k0 + len(f), k0 + len(f), total)
        if a > fill_from:
            frames[fill_from:a] = f[a - k0] if a - k0 < len(f) else f[-1]
        frames[a:b] = f[a - k0:b - k0]
        fill_from = max(fill_from, b)
    frames[fill_from:] = frames[fill_from - 1]
    head = head[:16] + struct.pack("<I", total) + head[20:]
    return y, head + frames.tobytes()


def render_all(voices, d: Path, progress=None) -> list[np.ndarray]:
    """[(name, melody, body)] -> each voice's wav, its .wav/.tele/.json written into the show folder.
    Every voice is cut into chunks and all chunks run at once on every core. `progress(done fraction)`."""
    jobs = [(i, s0, m, xin, xout) for i, (_, mel, _) in enumerate(voices) for s0, m, xin, xout in chunks(mel)]
    # a part with no note at all (e.g. ree when only one voice sings, or ru/mi in a song without piano) still stands
    # on the stage: one short note is rendered only for the head's telemetry, then its sound is dropped and every frame
    # is the head at rest (before that, assemble failed on the empty part and the whole song stopped)
    silent = {i for i in range(len(voices)) if not any(j[0] == i for j in jobs)}
    jobs += [(i, 0.0, {"bpm": 60, "t0": 0.3, "lyrics": "mi", "notes": [[60, 0.1]]}, None, None) for i in sorted(silent)]
    # the longest chunks first, so the last one to finish is a short one
    order = sorted(range(len(jobs)), key=lambda k: -sum(b for _, b in jobs[k][2]["notes"]))
    res = [None] * len(jobs)
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(os.cpu_count()) as ex:
        futs = {ex.submit(render_chunk, Path(tmp) / f"c{k}", jobs[k][2], voices[jobs[k][0]][2]): k for k in order}
        for done, f in enumerate(as_completed(futs), 1):
            res[futs[f]] = f.result()
            if progress:
                progress(done / len(jobs))
    n = max(round(s0 * FS) + len(w) for (vi, s0, *_), (w, _) in zip(jobs, res) if vi not in silent) if len(silent) < len(voices) else FS
    wavs = []
    for i, (name, mel, body) in enumerate(voices):
        pieces = [(s0, w, t, xin, xout) for (vi, s0, _, xin, xout), (w, t) in zip(jobs, res) if vi == i]
        if i in silent:
            s0, w, t, _, _ = pieces[0]
            frame_len = struct.unpack_from("<I", t, 20)[0]
            rest = np.frombuffer(t[48:48 + 4 * frame_len], dtype="<f4")  # frame 0: before any sound
            total = n // HOP + 1
            y, tele = np.zeros(n), t[:16] + struct.pack("<I", total) + t[20:48] + np.tile(rest, total).tobytes()
        else:
            y, tele = assemble(pieces, n)
        (d / f"v{i + 1}.tele").write_bytes(tele)
        (d / f"v{i + 1}.json").write_text(json.dumps(mel), encoding="utf-8")
        with wave.open(str(d / f"v{i + 1}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
            w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
        wavs.append(y)
    print(f"render: {len(jobs)} chunks on {os.cpu_count()} threads")
    return wavs


def main(notes_path: Path, show_id: str, title: str = "", out_dir: str = "", progress=None):
    t_start = time.time()
    src = json.loads(notes_path.read_text(encoding="utf-8"))
    env = src.get("env", {})
    report, held, same = {}, {}, {}
    parts = []  # (name, notes [s,e,midi,amp], syllables, body)
    # the vocals in strength order: the strongest note at a time is the lead's, the next do's, the next ree's
    vocal = loud(src, "vocals")
    ranked = rank_lanes(vocal, 3)
    for name, lane in zip(("lead", "do", "ree"), ranked):
        m, j, d = mono(lane)
        if JOIN_SAME[name]:
            m, same[name] = merge_same(m, env.get("vocals"), src.get("onset", {}).get("vocals"), RESING[name],
                                       float("inf") if RUN_NO_REST.get(name) else None)
        else:
            same[name] = 0
        m, held[name] = legato(m, same_pitch_runs=RUN_NO_REST.get(name, False))
        syl = [LEAD[n[2] % 12] for n in m] if name == "lead" else [name] * len(m)
        parts.append((name, m, syl, body_for(m)))
        report[name] = (len(lane), j, d)
    report["lead"] = (len(src["vocals"]), report["lead"][1], report["lead"][2] + len(vocal) - sum(len(x) for x in ranked)
                      + len(src["vocals"]) - len(vocal))  # the lead's line also counts the notes no lane took
    for drum, (syl, midi, longest) in DRUM.items():
        on = src[drum]
        notes = [[s, min(s + longest, on[i + 1] if i + 1 < len(on) else s + longest), midi, 1.0] for i, s in enumerate(on)]
        notes = [n for n in notes if n[1] - n[0] > 0.03]
        parts.append((syl, notes, [syl] * len(notes), {"kick": "male", "snare": "female", "hat": "child"}[drum]))
        report[syl] = (len(on), 0, len(on) - len(notes))
    # the chorus squad: bass zoo/di, guitar no/ku, piano ru/mi, each stem's two strongest notes at a time, the lower
    # to the first; held like the rest, never joined (RESING is for the lead and the harmony only)
    for stem, syls in CHORUS.items():
        if stem not in src:
            continue
        for syl, lane in zip(syls, lanes(loud(src, stem))):  # lower, upper
            m, j, d = mono(lane)
            same[syl] = 0
            m, held[syl] = legato(m)
            parts.append((syl, m, [syl] * len(m), body_for(m)))
            report[syl] = (len(src[stem]), j, d)

    d = Path(out_dir) if out_dir else SHOWS / show_id
    d.mkdir(parents=True, exist_ok=True)
    voices = []
    for name, notes, syl, body in parts:
        m, shifted = to_melody(notes, syl, body)
        voices.append((name, m, body))
        report[name] = report[name] + (shifted, body)
    t_render0 = time.time()
    wavs = render_all(voices, d, progress)
    t_render = time.time()

    n = max(len(w) for w in wavs)
    mix = np.zeros((n, 2))
    sections = src.get("sections", [])
    t_gain = time.time()
    with ThreadPoolExecutor(len(wavs)) as ex:  # numpy's FFT lets go of the GIL: the parts at once
        # not the lead: it keeps its notes' own levels
        wavs = [wavs[0]] + list(ex.map(lambda wp: wp[0] * note_gains(wp[0], wp[1][1]), zip(wavs[1:], parts[1:])))
    print(f"per-note loudness: {time.time() - t_gain:.1f} s")
    for i, w in enumerate(wavs):
        gain = 10 ** (GROUP_DB[GROUPS[i]] / 20)
        pan = 0.5 if i == 0 else (i - 1) / (len(wavs) - 2)  # lead in the middle, the rest left to right
        mix[:len(w), 0] += gain * w * np.cos(pan * np.pi / 2)
        mix[:len(w), 1] += gain * w * np.sin(pan * np.pi / 2)
    mix *= lift(n, sections)[:, None]  # every voice alike
    mix *= 0.89 / np.abs(mix).max()
    # Ogg Vorbis: a tenth of the wav to fetch from the server; the page decodes it with decodeAudioData (transport.ts)
    # written a second at a time: one write of the whole mix segfaulted inside libsndfile
    import soundfile
    with soundfile.SoundFile(d / "mix.ogg", "w", 48000, 2, format="OGG", subtype="VORBIS") as f:
        for k in range(0, len(mix), 48000):
            f.write(mix[k:k + 48000].astype(np.float32))
    # the same mix as wav, for the page's download button;
    # fetched after the show, so it does not delay playback
    with wave.open(str(d / "mix.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    names = ["lead"] + [f"{p[0]} ({k})" for p, k in zip(parts[1:], ["harmony", "harmony", "kick", "snare", "hat",
                                                                       "chorus, bass low", "chorus, bass high",
                                                                       "chorus, guitar low", "chorus, guitar high",
                                                                       "chorus, piano low", "chorus, piano high"])]
    (d / "show.json").write_text(json.dumps({
        "title": f"{title or show_id}, a cappella", "audio": "mix.ogg", "bpm": 60, "downbeat": 0,
        "sections": [{"label": s["label"], "start": round(s["start"], 2), "end": round(s["end"], 2)} for s in sections],
        "voices": [{"name": nm, "tele": f"v{i + 1}.tele", "melody": f"v{i + 1}.json"} for i, nm in enumerate(names)]},
        indent=1), encoding="utf-8")
    for nm, (k, (name, *_)) in zip(names, enumerate(parts)):
        src_n, j, dr, sh, body = report[name]
        print(f"{nm:20s} {body:6s} notes {len(parts[k][1]):5d} (from {src_n}, joined {j}, dropped {dr}, "
              f"octave-shifted {sh}, held over a gap {held.get(name, 0)}, same pitch joined {same.get(name, 0)})")
    print(f"show {show_id}: {n / 48000:.1f} s; render {t_render - t_render0:.1f} s, total {time.time() - t_start:.1f} s -> {d}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], *sys.argv[3:5])  # [title] [out dir]
