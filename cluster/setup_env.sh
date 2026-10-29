#!/bin/bash
# One-time environment setup on Clemson Palmetto (run on a login node from the repo root).
# Module names change between cluster updates: check `module avail python cuda` and edit.
set -euo pipefail

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11}
VENV_DIR=${VENV_DIR:-$HOME/venvs/ncc}

module load "$PYTHON_MODULE"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
# CUDA wheels of PyTorch bundle their own runtime, so no cuda module is needed for pip.
pip install -e ".[ml,anthropic,dev]"
python -c "import torch; print('torch', torch.__version__, 'CUDA build', torch.version.cuda)"
echo "Environment ready: source $VENV_DIR/bin/activate"
