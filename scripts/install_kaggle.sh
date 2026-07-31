#!/usr/bin/env bash
# install_kaggle.sh — one-shot installer for Magenta RealTime 2 on Kaggle/Colab.
#
#   curl -fsSL https://raw.githubusercontent.com/ctunix/magenta-realtime/main/scripts/install_kaggle.sh | bash
#
# What it does:
#   1. Detects Kaggle vs Colab vs other, picks a writable working dir.
#   2. Installs into an ISOLATED VENV so Kaggle's system packages (RAPIDS,
#      pandas, jupyter-server) are NOT touched — avoids numba/numpy conflicts.
#   3. Installs a pinned, non-drifting JAX/CUDA stack + the repo (--no-deps).
#   4. Downloads MusicCoCa + SpectroStream resources and the mrt2_base
#      checkpoint, then quantizes to bf16 (9.84 GB -> 4.92 GB) and deletes the
#      fp32 original to free ~10 GB of disk.
#   5. Prints the exact cells to run for sharded generation.
#
# Idempotent and non-interactive (safe to pipe from curl). No GPU used here.
set -euo pipefail
export GIT_TERMINAL_PROMPT=0

# --- 1. Detect environment -------------------------------------------------
if [ -d /kaggle/working ]; then WORK_DIR=/kaggle/working
elif [ -d /content ]; then WORK_DIR=/content
else WORK_DIR="$PWD"; fi
# IMPORTANT: paths.py appends "magenta-rt-v2" to MAGENTA_HOME, so set the BASE
# dir (e.g. /kaggle/working), NOT .../magenta-rt-v2 (that would double-nest).
MAGENTA_HOME="${MAGENTA_HOME:-$WORK_DIR}"
REPO_DIR="$WORK_DIR/magenta-realtime"
VENV_DIR="$WORK_DIR/mrt_venv"
MRT_REPO="${MRT_REPO:-https://github.com/ctunix/magenta-realtime.git}"

echo "============================================================"
echo " Magenta RealTime 2 — Kaggle/Colab installer (venv-isolated)"
echo "   WORK_DIR     = $WORK_DIR"
echo "   MAGENTA_HOME = $MAGENTA_HOME   (assets land in $MAGENTA_HOME/magenta-rt-v2/)"
echo "   VENV         = $VENV_DIR"
echo "   REPO         = $MRT_REPO"
echo "============================================================"

# --- Migrate the old (pre-fix) double-nested layout if present -------------
# An earlier version of this script set MAGENTA_HOME=.../magenta-rt-v2, which
# made paths.py nest another magenta-rt-v2 inside. Move those assets up.
OLD_NESTED="$MAGENTA_HOME/magenta-rt-v2/magenta-rt-v2"
if [ -d "$OLD_NESTED" ]; then
  echo "  Migrating existing assets from old nested layout -> $MAGENTA_HOME/magenta-rt-v2/"
  mkdir -p "$MAGENTA_HOME/magenta-rt-v2"
  for item in "$OLD_NESTED"/* "$OLD_NESTED"/.[!.]*; do
    [ -e "$item" ] || continue
    mv -n "$item" "$MAGENTA_HOME/magenta-rt-v2/" 2>/dev/null || true
  done
  rmdir "$OLD_NESTED" 2>/dev/null || true
fi

# --- 2. Clone (with submodule) --------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  echo "[1/5] Repo exists; fetching latest..."
  git -C "$REPO_DIR" pull --quiet --recurse-submodules 2>/dev/null || \
    git -C "$REPO_DIR" submodule update --init --recursive >/dev/null
else
  echo "[1/5] Cloning $MRT_REPO -> $REPO_DIR ..."
  git clone --recurse-submodules "$MRT_REPO" "$REPO_DIR"
fi

# --- 3. Create venv + install deps + repo ----------------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[2/5] Creating isolated venv at $VENV_DIR ..."
  python -m venv "$VENV_DIR"
fi
VPIP="$VENV_DIR/bin/pip"
VPY="$VENV_DIR/bin/python"
echo "[2/5] Installing pinned JAX/CUDA stack into the venv (a few minutes)..."
"$VPIP" install -q --upgrade pip
"$VPIP" install -q -r "$REPO_DIR/requirements-kaggle.txt"
echo "[2/5] Installing magenta-rt into the venv (--no-deps, no cuda drift)..."
"$VPIP" install -q -e "$REPO_DIR" --no-deps
# The venv's mrt CLI:
MRT="$VENV_DIR/bin/mrt"

# --- 4. Download resources + checkpoint ------------------------------------
export MAGENTA_HOME
echo "[3/5] Downloading MusicCoCa + SpectroStream resources..."
"$MRT" models init
echo "[4/5] Downloading the mrt2_base checkpoint (~9.84 GB)..."
"$MRT" checkpoints download mrt2_base

# --- 5. Quantize to bf16 + free disk ---------------------------------------
echo "[5/5] Quantizing mrt2_base fp32 -> bf16 (CPU; ~4.92 GB out)..."
if "$MRT" checkpoints quantize mrt2_base --dtype bf16; then
  CKPT="mrt2_base_bf16.safetensors"
  # Free ~10 GB by removing the fp32 original (bf16 is the default path).
  if [ "${KEEP_FP32:-0}" != "1" ]; then
    rm -f "$MAGENTA_HOME/magenta-rt-v2/checkpoints/mrt2_base.safetensors"
    echo "  (removed fp32 original to free disk; set KEEP_FP32=1 to keep it)"
  fi
else
  echo "  (quantize failed; will use the fp32 checkpoint instead)"
  CKPT="mrt2_base.safetensors"
fi

echo "============================================================"
echo " DONE. Repo: $REPO_DIR | venv: $VENV_DIR"
echo " Checkpoint: \$MAGENTA_HOME/magenta-rt-v2/checkpoints/$CKPT"
echo "============================================================"
echo
echo "Run this in a NEW cell to generate 8 s of audio (uses the venv, so"
echo "Kaggle's system packages are untouched):"
cat <<EOF_GEN
# --- Generate (sharded across 2 GPUs if present) ---
!XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 MAGENTA_HOME=$MAGENTA_HOME $MRT jax generate --model mrt2_base --checkpoint $CKPT --shard --duration 8 --prompt "disco funk"

# --- Play the result (run in the same or a new cell) ---
import IPython.display as ipd
ipd.display(ipd.Audio('$MAGENTA_HOME/magenta-rt-v2/outputs/output_audio_jax_mrt2_base.wav', rate=48000))
EOF_GEN
echo "============================================================"
