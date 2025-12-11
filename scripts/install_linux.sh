#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

ENV_NAME="sovits_prep"
CONDA_ROOT="$HOME/miniconda3"
CONDA_BIN="$CONDA_ROOT/bin/conda"

echo "[1/6] Installing system packages (ffmpeg)..."
sudo apt-get update
sudo apt-get install -y ffmpeg

echo "[2/6] Installing Miniconda (if needed)..."
if [ ! -d "$CONDA_ROOT" ]; then
  TMP_INST="/tmp/miniconda.sh"
  wget -O "$TMP_INST" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$TMP_INST" -b -p "$CONDA_ROOT"
  rm "$TMP_INST"
else
  echo "Miniconda already installed at $CONDA_ROOT"
fi

if [ ! -x "$CONDA_BIN" ]; then
  echo "ERROR: conda binary not found at $CONDA_BIN"
  exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"

echo "[3a/6] Accepting Anaconda Terms of Service for default channels..."
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "[3/6] Creating / activating conda env '$ENV_NAME'..."
if conda env list | grep -qE "^$ENV_NAME\s"; then
  echo "Conda environment '$ENV_NAME' already exists"
else
  conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

echo "[4/6] Installing PyTorch 2.8.0 (CUDA 12.9, cu129)..."
python -m pip install --upgrade pip
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129

echo "[5/6] Installing project requirements..."
python -m pip install -r requirements.txt

echo "[6/6] Running tests..."
PYTHONPATH="." pytest -q

echo
echo "======================================================="
echo "Environment '$ENV_NAME' is ready."
echo "To use it in a new shell, run:"
echo "  source \"$CONDA_ROOT/etc/profile.d/conda.sh\""
echo "  conda activate \"$ENV_NAME\""
echo "Then you can run:"
echo "  python sovits_prep.py --help"
echo "======================================================="
