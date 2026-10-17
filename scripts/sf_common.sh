#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export SF_REPO="$VLA_RUNTIME/repos/spatial_forcing"
export SF_ENV="$VLA_RUNTIME/sf_train"
export SF_HOME="$VLA_RUNTIME/training"
export PYTHONPATH="$ROOT/src:$SF_REPO/openpi-SF/src"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
cd "$ROOT"

export TORCHINDUCTOR_CACHE_DIR="$VLA_RUNTIME/cache/torchinductor"
export TRITON_CACHE_DIR="$VLA_RUNTIME/cache/triton"
