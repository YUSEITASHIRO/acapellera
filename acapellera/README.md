# Acapellera

Drop a song on a web page; twelve singers built from CHIMERA's physically modelled voices sing it back a
cappella on a small stage, and you can download the result as a wav. Nothing of the original recording plays.

## What happens to a song

1. **Separation**: demucs (htdemucs_6s) splits the song into vocals, drums, bass, guitar, piano and other.
2. **Notes**: basic-pitch turns vocals, bass, guitar and piano into notes; the drum stem gives onsets in a low,
   a middle and a high band. SongFormer (optional) finds the sections (verse, chorus, ...).
3. **Parts** (`server/make_sing.py`):

   | Part | Sings | From |
   |---|---|---|
   | lead (front, centre, largest) | a word chosen by the note's pitch | the vocals' strongest note at each moment |
   | harmony do, ree (front, left and right) | "do", "ree" | the vocals' second and third strongest notes |
   | drums bah, pa, si (back) | short "bah", "pa", "si" at fixed pitches | low, middle and high drum onsets |
   | chorus squad zoo/di, no/ku, ru/mi (back) | their syllables | bass, guitar, piano: the two strongest notes at a time, lower and upper |

   Every part but the drums holds each note up to its next one and rests only at a gap of 0.5 s or more.
   Levels: lead > harmony and chorus > drums; everyone is louder in a chorus section.
4. **Singing**: the CHIMERA CLI renders the twelve voices in parallel on all cores; the page plays the mix while
   each head moves with its own voice.

## Run it

You need [Bun](https://bun.sh), [Rust](https://rustup.rs) (the web app builds its engine on first start) and
Python 3.11 with `psutil` for the launcher.

**The worker** (separation, notes, singing) can run on this machine or on a GPU server. A GPU makes the
separation fast; on a CPU it works, slower.

### On this machine

```
bash acapellera/server/setup.sh            # Python env in ~/.acapellera/venv (TORCH_INDEX for CPU or your CUDA)
bash acapellera/server/setup_engine.sh     # the CHIMERA CLI: target/release/chimera
bash acapellera/server/setup_songformer.sh # optional: sections (needs about 11 GB of GPU memory)
python acapellera/launch.py                # or acapellera.bat on Windows
```

On Windows without bash: make a venv at `%USERPROFILE%\.acapellera\venv`, install PyTorch from
https://pytorch.org for your machine, then `pip install -r acapellera/server/requirements.txt`, and build the CLI
with `cargo build --release -p chimera-cli`.

### On a GPU server (over ssh)

On the server, clone this repository and run `setup.sh` and `setup_engine.sh` (and `setup_songformer.sh` if
wanted). On your machine, write `acapellera/remote.json` (it is not committed):

```json
{"host": "your-ssh-host", "repo": "/path/to/the/checkout/on/the/server"}
```

then `python acapellera/launch.py --remote` (or `acapellera.bat --remote`). The launcher copies the server scripts
up, starts the worker over `ssh -tt` with a tunnel on port 8770, and the songs go through that tunnel only.

The page opens at `http://127.0.0.1:5317/?stage=8768`. Ctrl+C in the launcher's window stops everything.

## Privacy and copyright

- The songs you drop and the shows made from them stay on your machine and your server
  (`acapellera/in/`, `public/shows/`, `~/.acapellera/`); all are outside git.
- Stage sounds: see `public/stage-sfx/CREDITS.md` (four are included under CC BY 4.0; two optional ones you
  download yourself).

## Credits

- [CHIMERA](https://github.com/jamesondh/chimera-vocal-synth) by Jameson Hodge (MIT): the voices, the engine and
  the web app this is built on.
- [demucs](https://github.com/adefossez/demucs) (MIT), [basic-pitch](https://github.com/spotify/basic-pitch)
  (Apache-2.0), [SongFormer](https://github.com/ASLP-lab/SongFormer) (CC BY 4.0), [librosa](https://librosa.org) (ISC).
- Sound effects: OtoLogic (CC BY 4.0).
