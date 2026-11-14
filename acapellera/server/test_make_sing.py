"""Self-check of make_sing's note choosing: python test_make_sing.py"""
import numpy as np
from make_sing import LEAD, lanes, mono, pick, to_melody

# pick: at most 2 at once, the strongest kept
ev = [[0, 1, 60, 0.9], [0, 1, 64, 0.5], [0, 1, 67, 0.7]]
assert [e[2] for e in pick(ev, 2)] == [60, 67]
# lanes: the lower of two sounding notes goes low, whatever came first
low, high = lanes([[0, 1, 67, 0.9], [0.5, 1.5, 60, 0.8], [2, 3, 70, 0.9]])
assert [e[2] for e in low] == [60] and [e[2] for e in high] == [67, 70], (low, high)
# mono: of two notes together the stronger stays
m, _, dropped = mono([[0, 1, 60, 0.4], [0, 1, 67, 0.9]])
assert [n[2] for n in m] == [67] and dropped == 1, m
# mono: overlaps cut, a short note joined to the previous one, a lone short note dropped
m, joined, dropped = mono([[0, 1, 60, 1], [0.9, 2, 62, 1], [2, 2.05, 64, 1], [5, 5.03, 65, 1]])
assert [n[:3] for n in m] == [[0, 0.9, 60], [0.9, 2.05, 62]] and (joined, dropped) == (1, 1), (m, joined, dropped)
# lead syllables by pitch class: C = mi ... B = ma (the table LEAD), a sharp sings its natural's word
assert [LEAD[p % 12] for p in (60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71)] == \
    ["mi", "mi", "pa", "pa", "zoo", "di", "di", "ree", "ree", "ru", "ru", "ma"]
# melody: rests fill the gaps, notes shifted into the body's range keep their pitch class
mel, shifted = to_melody([[1, 2, 36, 1], [2, 3, 50, 1]], ["do", "do"], "male")
assert mel["notes"] == [[0, 1], [48, 1], [50, 1]] and shifted == 1, mel
assert mel["lyrics"] == "do do"
# legato (REST_MIN 0.5 s, on the part's own notes): gaps under 0.5 s held over (C -> C# too), 0.5 s or more a rest
from make_sing import legato, rank_lanes
held, n = legato([[0, 0.5, 60, 1], [0.6, 1.0, 61, 1], [1.49, 2.5, 64, 1], [3.0, 4.0, 65, 1]])
assert [n_[1] for n_ in held] == [0.6, 1.49, 2.5, 4.0] and n == 2, held  # 0.1 and 0.49 held, 0.5 a rest
# legato with same-pitch runs (the lead): a 0.5 s gap of one pitch held; a 3 s gap of one pitch (2 s or more)
# held 1.5 s more, then a rest; a 3 s gap between two pitches a rest, nothing added
held, n = legato([[0, 1, 60, 1], [1.5, 2, 60, 1], [5, 6, 60, 1], [9, 10, 62, 1]], same_pitch_runs=True)
assert [n_[1] for n_ in held] == [1.5, 3.5, 6, 10] and n == 2, held
# rank_lanes: the strongest at each moment to the lead, the next to do, the next to ree, a fourth dropped
ev = [[0, 1, 60, 0.9], [0, 1, 64, 0.5], [0, 1, 67, 0.7], [0, 1, 72, 0.3], [1, 2, 62, 0.4]]
lead_, do_, ree_ = rank_lanes(ev, 3)
assert [e[2] for e in lead_] == [60, 62] and [e[2] for e in do_] == [67] and [e[2] for e in ree_] == [64], (lead_, do_, ree_)
# equal loudness: 1 kHz needs PHON dB (the contour's definition); 100 Hz needs more; note_gains raises a 100 Hz note
# against a 1 kHz note of the same amplitude, and lowers the 1 kHz one
from make_sing import equal_loudness_spl, ISO226, PHON, note_gains, FS as FS_
spl = equal_loudness_spl()
assert abs(spl[list(ISO226[:, 0]).index(1000)] - PHON) < 0.05 and spl[list(ISO226[:, 0]).index(100)] > PHON + 10, spl
tt = np.arange(FS_) / FS_
y = np.concatenate([0.1 * np.sin(2 * np.pi * 100 * tt), 0.1 * np.sin(2 * np.pi * 1000 * tt)])
gn = note_gains(y, [[0, 1, 43, 1], [1, 2, 83, 1]])
assert gn[FS_ // 2] > 1.5 and gn[3 * FS_ // 2] < 0.8, (gn[FS_ // 2], gn[3 * FS_ // 2])
# merge_same (RESING): same pitch, no dip -> held on as one; a 12 dB dip with a strong onset -> sung again;
# a 12 dB dip with no onset (a dropout) -> held on; a gap of REST_MIN or more -> left apart (a rest)
from make_sing import merge_same
env = [-10.0] * 1000
env[298:303] = [-22.0] * 5   # 2.98-3.03 s: a 12 dB dip
env[598:603] = [-22.0] * 5   # 5.98-6.03 s: a 12 dB dip
onset = [0.1] * 1000
onset[300] = 5.0             # a strong start at 3.0 s only
notes = [[0, 1, 60, 1], [1, 2, 60, 1], [2, 3, 62, 1], [3, 5, 62, 1], [5, 6, 64, 1], [6, 7, 64, 1], [8, 9, 64, 1]]
m, k = merge_same(notes, env, onset, (6.0, 80.0))
assert [(n[0], n[1], n[2]) for n in m] == [(0, 2, 60), (2, 3, 62), (3, 5, 62), (5, 7, 64), (8, 9, 64)] and k == 2, m
# lift: +3 dB inside a chorus, 0 dB well outside it
from make_sing import lift, FS
g = lift(10 * FS, [{"label": "verse", "start": 0, "end": 4}, {"label": "chorus", "start": 4, "end": 8}])
assert abs(20 * np.log10(g[6 * FS]) - 3.0) < 1e-6 and abs(g[2 * FS] - 1.0) < 1e-9 and abs(g[9 * FS] - 1.0) < 1e-9, (g[2 * FS], g[6 * FS])
assert (lift(FS, []) == 1).all()
# chunks: a 5 s held line (ten 0.5 s notes, no rest) is cut inside notes, both pieces sing the cut note,
# and the windows of a cut are the same on both sides
from make_sing import chunks
ps = chunks({"bpm": 60, "t0": 0, "lyrics": " ".join(["do"] * 10), "notes": [[60, 0.5]] * 10})
assert len(ps) == 3, [(p[0], p[2], p[3]) for p in ps]
assert ps[0][2] is None and ps[-1][3] is None and ps[0][3] == ps[1][2] == (2.175, 2.425), [(p[2], p[3]) for p in ps]
assert sum(len(p[1]["notes"]) for p in ps) == 10 + 2  # the two cut notes are sung twice
# render_all with a part that has no note (local CHIMERA build): no failure; that part silent, its head at rest
import shutil, struct, tempfile
from pathlib import Path
from make_sing import CLI, render_all
if CLI.exists():
    with tempfile.TemporaryDirectory() as tmp:
        sung = {"bpm": 60, "t0": 0, "lyrics": "mi pa", "notes": [[0, 0.5], [60, 0.4], [62, 0.4]]}
        empty = {"bpm": 60, "t0": 0, "lyrics": "", "notes": []}
        wavs = render_all([("lead", sung, "female"), ("ree", empty, "female")], Path(tmp))
        assert len(wavs) == 2 and len(wavs[0]) == len(wavs[1]) and not wavs[1].any() and wavs[0].any()
        t = (Path(tmp) / "v2.tele").read_bytes()
        frames, frame_len = struct.unpack_from("<I", t, 16)[0], struct.unpack_from("<I", t, 20)[0]
        assert len(t) == 48 + 4 * frames * frame_len and frames == len(wavs[1]) // 768 + 1, (len(t), frames, frame_len)
else:
    print("(CHIMERA CLI not built here: the empty-part render was not checked)")
print("ok")
