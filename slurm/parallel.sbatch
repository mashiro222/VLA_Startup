#!/usr/bin/env bash
# Each array member has one GPU and runs its clean shard, then its plus shard.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@90
set -euo pipefail
: "${SLURM_JOB_ID:?Requires a Slurm allocation}"
: "${SLURM_ARRAY_TASK_ID:?Requires a Slurm job array}"
source "$SLURM_SUBMIT_DIR/scripts/common.sh"
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# Give workers separate writable compilation/plot caches to avoid collisions.
export MPLCONFIGDIR="$VLA_RUNTIME/cache/matplotlib-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
export NUMBA_CACHE_DIR="$VLA_RUNTIME/cache/numba-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
exec python3 scripts/parallel.py worker "$1" --index "$SLURM_ARRAY_TASK_ID" \
    --require-smoke --clean-smoke "${2:-smoke-clean-001}" --plus-smoke "${3:-smoke-plus-001}"
