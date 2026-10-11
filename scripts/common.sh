#!/usr/bin/env bash
# Source this file from bash. Keep all large, recoverable files on scratch.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export VLA_ROOT="$ROOT"
export VLA_RUNTIME="$ROOT/.runtime"
export OPENPI_REPO="$VLA_RUNTIME/repos/openpi"
export PLUS_REPO="$VLA_RUNTIME/repos/libero_plus"
export CLEAN_REPO="$VLA_RUNTIME/repos/libero_clean"
export SERVER_ENV="$VLA_RUNTIME/server"
export SIM_ENV="$VLA_RUNTIME/sim"
export MAGICK_PREFIX="$VLA_RUNTIME/native-7.1.1/squashfs-root/usr"
export UV_CACHE_DIR="$VLA_RUNTIME/cache/uv"
export UV_PYTHON_INSTALL_DIR="$VLA_RUNTIME/python"
export OPENPI_DATA_HOME="$VLA_RUNTIME/cache/openpi"
export HF_HOME="$VLA_RUNTIME/cache/huggingface"
export XDG_CACHE_HOME="$VLA_RUNTIME/cache/xdg"
export NUMBA_CACHE_DIR="$VLA_RUNTIME/cache/numba"
export MPLCONFIGDIR="$VLA_RUNTIME/cache/matplotlib"
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
UV="$ROOT/.tools/uv"
sim_python() {
    MAGICK_HOME="$MAGICK_PREFIX" LD_LIBRARY_PATH="$MAGICK_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
        "$SIM_ENV/bin/python" "$@"
}
