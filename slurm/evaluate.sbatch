#!/usr/bin/env bash
# Resources and paid account are passed explicitly by scripts/submit.sh.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@90
set -euo pipefail
: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation}"
source "$SLURM_SUBMIT_DIR/scripts/common.sh"
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
exec python3 scripts/run_node.py "$1"
