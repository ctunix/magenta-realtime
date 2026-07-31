#!/usr/bin/env bash
# install_kaggle.sh — one-shot installer for Magenta RealTime 2 on Kaggle/Colab.
#
#   curl -fsSL https://raw.githubusercontent.com/ctunix/magenta-realtime/main/scripts/install_kaggle.sh | bash
#
# What it does:
#   1. Detects Kaggle vs Colab vs other, picks a writable working dir.
#   2. Clones this fork (with the vendored sequence-layers submodule).
#   3. Installs a pinned, non-drifting JAX/CUDA stack (requirements-kaggle.txt)
#      then the repo with --no-deps so the four cuda packages stay in sync.
#   4. Downloads MusicCoCa + SpectroStream resources and the mrt2_base checkpoint.
#   5. Quantizes the checkpoint to bf16 (9.84 GB -> 4.92 GB) so it fits a T4.
#   6. Prints the exact Python cell to run for sharded generation.
#
# It is idempotent: re-running skips already-done steps. Non-interactive: safe
# to pipe from curl. GPU is NOT used by this script (downloads + CPU quantize).
set -euo pipefail

# Don't let git hang on a prompt when piped from curl.
export GIT_TERMINAL_PROMPT=0

# --- 1. Detect environment -------------------------------------------------
if [ -d /kaggle/working ]; then
  WORK_DIR=/kaggle/working
elif [ -d /content ]; then
  WORK_DIR=/content
else
  WORK_DIR="$PWD"
fi
MAGENTA_HOME="${MAGENTA_HOME:-$WORK_DIR/magenta-rt-v2}"
REPO_DIR="$WORK_DIR/magenta-realtime"
# Override the source repo by setting MRT_REPO (default: this fork).
MRT_REPO="${MRT_REPO:-https://github.com/ctunix/magenta-realtime.git}"

echo "============================================================"
echo " Magenta RealTime 2 — Kaggle/Colab installer"
echo "   WORK_DIR     = $WORK_DIR"
echo "   MAGENTA_HOME = $MAGENTA_HOME"
echo "   REPO         = $MRT_REPO"
echo "============================================================"

# --- 2. Clone (with submodule) --------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  echo "[1/5] Repo exists; fetching latest..."
  git -C "$REPO_DIR" pull --quiet --recurse-submodules 2>/dev/null || \
    git -C "$REPO_DIR" submodule update --init --recursive >/dev/null
else
  echo "[1/5] Cloning $MRT_REPO -> $REPO_DIR ..."
  git clone --recurse-submodules "$MRT_REPO" "$REPO_DIR"
fi

# --- 3. Install deps + repo -------------------------------------------------
cd "$REPO_DIR"
echo "[2/5] Installing pinned JAX/CUDA stack (this can take a few minutes)..."
python -m pip install -q -r requirements-kaggle.txt
echo "[2/5] Installing magenta-rt (--no-deps so cuda packages don't drift)..."
python -m pip install -q -e . --no-deps
hash -r  # refresh bash's command cache so `mrt` is found

# Locate the `mrt` CLI (fall back to a python -c wrapper if not on PATH).
if command -v mrt >/dev/null 2>&1; then
  MRT=(mrt)
else
  # `main()` reads sys.argv[1:], so args after the -c string are picked up.
  MRT=(python -c "from magenta_rt.cli import main; main()")
fi

# --- 4. Download resources + checkpoint ------------------------------------
export MAGENTA_HOME
echo "[3/5] Downloading MusicCoCa + SpectroStream resources..."
"${MRT[@]}" models init
echo "[4/5] Downloading the mrt2_base checkpoint (~9.84 GB)..."
"${MRT[@]}" checkpoints download mrt2_base

# --- 5. Quantize to bf16 ----------------------------------------------------
echo "[5/5] Quantizing mrt2_base fp32 -> bf16 (CPU; ~4.92 GB out)..."
if "${MRT[@]}" checkpoints quantize mrt2_base --dtype bf16; then
  CKPT="mrt2_base_bf16.safetensors"
else
  echo "  (quantize failed; will use the fp32 checkpoint instead)"
  CKPT="mrt2_base.safetensors"
fi

echo "============================================================"
echo " DONE. Install location: $REPO_DIR"
echo " Checkpoint:              \$MAGENTA_HOME/checkpoints/$CKPT"
echo "============================================================"
echo
echo "Now run this Python cell (in a NEW cell) to generate 8 s of audio:"
cat <<EOF_CELL
import os
# Must be set BEFORE the first `import jax` so jaxlib reads them.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.85')
os.environ.setdefault('MAGENTA_HOME', '$MAGENTA_HOME')

import logging, jax
logging.basicConfig(level=logging.INFO, force=True)
from magenta_rt.jax import _gpu_check
print(_gpu_check.diagnose_devices())
_gpu_check.assert_gpu_available()          # fail loudly, no silent CPU fallback
n_gpu = _gpu_check.num_local_cuda_devices()

from magenta_rt import MagentaRT2Jax
from magenta_rt.config import MUSICCOCA
mrt = MagentaRT2Jax(
    size='mrt2_base',
    checkpoint='$CKPT',
    shard=(n_gpu >= 2),                     # tensor-parallel across 2 GPUs if present
    require_gpu=True,
)
print('Sharded:', mrt._sharded, '| mesh:', mrt._mesh)

embedding = mrt.embed_style('disco funk', use_mapper=True)
wav, state = mrt.generate(conditioning={MUSICCOCA.key: embedding}, frames=200)  # 8 s @ 25 Hz

import IPython.display as ipd
out = '$WORK_DIR/mrt2_base_output.wav'
wav.write(out)
print('Saved', out)
ipd.display(ipd.Audio(wav.samples.T, rate=wav.sample_rate))
EOF_CELL
echo "============================================================"
