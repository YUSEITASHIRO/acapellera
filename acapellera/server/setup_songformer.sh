#!/usr/bin/env bash
# Optional: SongFormer (ASLP-lab, CC BY 4.0) for the song's sections (intro / verse / chorus ...), in its own venv
# because it pins torch 2.4.0, under $ACAPELLERA_HOME/songformer (default ~/.acapellera). Needs a GPU with about
# 11 GB for a 4-minute song; without it the songs are made without sections (no louder chorus).
# Log: $ACAPELLERA_HOME/setup_songformer.log.
set -uo pipefail
ROOT=${ACAPELLERA_HOME:-$HOME/.acapellera}
mkdir -p "$ROOT"
cd "$ROOT"
exec > >(tee -a setup_songformer.log) 2>&1
date
[ -d songformer ] || git clone --depth 1 --recurse-submodules https://github.com/ASLP-lab/SongFormer songformer
cd songformer
git log -1 --format='%H %cd'
[ -d venv ] || "${PYTHON:-python3}" -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch==2.4.0 torchaudio==2.4.0 --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
# the pinned list first; what fails to install is reported, then installed without its pin
grep -v -E '^\s*(#|$)|^torch==|^torchaudio==' requirements.txt > req.txt
pip install --quiet -r req.txt || { echo "PINNED INSTALL FAILED; retrying without pins"; sed -E 's/[=<>].*//' req.txt > req_nopin.txt; pip install --quiet -r req_nopin.txt; }
export HF_HOME=$ROOT/songformer/hf
cd src/SongFormer
python utils/fetch_pretrained.py
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo SONGFORMER_DONE
