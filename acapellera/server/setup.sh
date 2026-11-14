#!/usr/bin/env bash
# The worker's Python environment (separation by demucs, transcription by basic-pitch), in $ACAPELLERA_HOME/venv
# (default ~/.acapellera). Everything is logged to $ACAPELLERA_HOME/setup.log.
#   PYTHON=python3.11 TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash acapellera/server/setup.sh
# TORCH_INDEX picks the PyTorch wheels for your GPU's CUDA (cu121 here); https://download.pytorch.org/whl/cpu
# for no GPU (separation then takes several times longer).
set -euo pipefail
ROOT=${ACAPELLERA_HOME:-$HOME/.acapellera}
mkdir -p "$ROOT/in" "$ROOT/out"
cd "$ROOT"
exec > >(tee -a setup.log) 2>&1
date
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
[ -d venv ] || "${PYTHON:-python3}" -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch torchaudio --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
pip install --quiet -r "$(dirname "$0")/requirements.txt"
python -c "import torch, demucs, basic_pitch, librosa; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
du -sh venv
echo SETUP_DONE
