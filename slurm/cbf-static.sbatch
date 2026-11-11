#!/usr/bin/env bash
#SBATCH --signal=B:TERM@90
set -euo pipefail
: "${SLURM_JOB_ID:?Requires a Slurm allocation}"
: "${SLURM_ARRAY_TASK_ID:?Requires an array unit}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .runtime/arena/bin/python -m cbf_static.runtime "$1" --index "$SLURM_ARRAY_TASK_ID"
