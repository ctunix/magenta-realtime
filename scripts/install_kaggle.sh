#!/usr/bin/env bash
# install_kaggle.sh — one-shot installer for Magenta RealTime 2 on Kaggle/Colab.
#
#   curl -fsSL https://raw.githubusercontent.com/ctunix/magenta-realtime/main/scripts/install_kaggle.sh | bash
#
# What it does:
#   1. Detects Kaggle vs Colab vs other, picks a writable working dir.
#   2. Installs a pinned, non-drifting JAX/CUDA stack (requirements-kaggle.txt)
#      then the repo NON-EDITABLE (--no-deps), so `import magenta_rt` works in a
#      subprocess without a kernel restart. (A venv is NOT used: `python -m venv`
#      fails at ensurepip on Kaggle/Colab.)
#   3. Downloads MusicCoCa + SpectroStream resources and the mrt2_base
#      checkpoint, then quantizes to bf16 (9.84 GB -> 4.92 GB) and deletes the
#      fp32 original to free ~10 GB of disk.
#   4. Prints the exact cells to run for sharded generation.
#
# Idempotent and non-interactive (safe to pipe from curl). No GPU used here.
#
# NOTE on pip warnings: you may see "dependency conflicts" with Kaggle's
# pre-installed RAPIDS (cuml/cudf want numba<0.62) because MRT needs numba>=0.65
# for numpy 2.x. These are unavoidable and harmless for running MRT — ignore them.
set -euo pipefail
export GIT_TERMINAL_PROMPT=0

# --- 1. Detect environment -------------------------------------------------
if [ -d /kaggle/working ]; then WORK_DIR=/kaggle/working
elif [ -d /content ]; then WORK_DIR=/content
else WORK_DIR="$PWD"; fi
# Robust Python detection (some images have `python3` but not `python`).
PY="$(command -v python3 || command -v python)"
# PEP 668: some images (Debian/Ubuntu) block system-wide pip installs. Pass
# --break-system-packages when pip supports it; harmless on images that don't
# enforce it (e.g. Kaggle/Colab). A venv can't be used (python -m venv fails at
# ensurepip on Kaggle/Colab).
if "$PY" -m pip install --help 2>&1 | grep -q -- --break-system-packages; then
  PIP_FLAGS=(--break-system-packages)
else
  PIP_FLAGS=()
fi
# IMPORTANT: paths.py appends "magenta-rt-v2" to MAGENTA_HOME, so set the BASE
# dir (e.g. /kaggle/working), NOT .../magenta-rt-v2 (that would double-nest).
MAGENTA_HOME="${MAGENTA_HOME:-$WORK_DIR}"
REPO_DIR="$WORK_DIR/magenta-realtime"
MRT_REPO="${MRT_REPO:-https://github.com/ctunix/magenta-realtime.git}"

echo "============================================================"
echo " Magenta RealTime 2 — Kaggle/Colab installer"
echo "   WORK_DIR     = $WORK_DIR"
echo "   MAGENTA_HOME = $MAGENTA_HOME   (assets land in $MAGENTA_HOME/magenta-rt-v2/)"
echo "   REPO         = $MRT_REPO"
echo "============================================================"

# --- Migrate the old double-nested layout if present (from an earlier run) --
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
# Remove a broken leftover venv from an earlier run (python -m venv fails on Kaggle).
[ -d "$WORK_DIR/mrt_venv" ] && rm -rf "$WORK_DIR/mrt_venv" || true

# --- 2. Clone (with submodule) --------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  echo "[1/5] Repo exists; fetching latest..."
  git -C "$REPO_DIR" pull --quiet --recurse-submodules 2>/dev/null || \
    git -C "$REPO_DIR" submodule update --init --recursive >/dev/null
else
  echo "[1/5] Cloning $MRT_REPO -> $REPO_DIR ..."
  git clone --recurse-submodules "$MRT_REPO" "$REPO_DIR"
fi

# --- 3. Install deps + repo (NON-EDITABLE) ---------------------------------
cd "$REPO_DIR"
echo "[2/5] Installing pinned JAX/CUDA stack (a few minutes; ignore RAPIDS numba warnings)..."
"$PY" -m pip install -q "${PIP_FLAGS[@]}" -r requirements-kaggle.txt
echo "[2/5] Installing magenta-rt (non-editable, --no-deps so cuda packages don't drift)..."
"$PY" -m pip install -q "${PIP_FLAGS[@]}" "$REPO_DIR" --no-deps
hash -r  # refresh bash's command cache so `mrt` is found

# Locate the `mrt` CLI (fall back to a python -c wrapper if not on PATH).
if command -v mrt >/dev/null 2>&1; then
  MRT=(mrt)
else
  MRT=("$PY" -c "from magenta_rt.cli import main; main()")
fi

# --- 4. Download resources + checkpoint ------------------------------------
export MAGENTA_HOME
echo "[3/5] Downloading MusicCoCa + SpectroStream resources..."
"${MRT[@]}" models init
echo "[4/5] Downloading the mrt2_base checkpoint (~9.84 GB)..."
"${MRT[@]}" checkpoints download mrt2_base

# --- 5. Quantize to bf16 + free disk ---------------------------------------
echo "[5/5] Quantizing mrt2_base fp32 -> bf16 (CPU; ~4.92 GB out)..."
if "${MRT[@]}" checkpoints quantize mrt2_base --dtype bf16; then
  CKPT="mrt2_base_bf16.safetensors"
  if [ "${KEEP_FP32:-0}" != "1" ]; then
    rm -f "$MAGENTA_HOME/magenta-rt-v2/checkpoints/mrt2_base.safetensors"
    echo "  (removed fp32 original to free disk; set KEEP_FP32=1 to keep it)"
  fi
else
  echo "  (quantize failed; will use the fp32 checkpoint instead)"
  CKPT="mrt2_base.safetensors"
fi

echo "============================================================"
echo " DONE. Repo: $REPO_DIR"
echo " Checkpoint: \$MAGENTA_HOME/magenta-rt-v2/checkpoints/$CKPT"
echo "============================================================"
echo
echo "Run this in a NEW cell to generate 8 s of audio (runs in a subprocess,"
echo "so no kernel restart is needed):"
cat <<EOF_GEN
# --- Generate (sharded across 2 GPUs if present) ---
!XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 MAGENTA_HOME=$MAGENTA_HOME $PY -m magenta_rt.jax.generate --model mrt2_base --checkpoint $CKPT --shard --duration 8 --prompt "disco funk"

# --- Play the result (run in the same or a new cell) ---
import IPython.display as ipd
ipd.display(ipd.Audio('$MAGENTA_HOME/magenta-rt-v2/outputs/output_audio_jax_mrt2_base.wav', rate=48000))
EOF_GEN
echo "============================================================"
