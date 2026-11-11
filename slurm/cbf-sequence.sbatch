#!/usr/bin/env bash
#SBATCH --signal=B:TERM@90
set -euo pipefail
: "${SLURM_JOB_ID:?Requires a Slurm allocation}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .runtime/arena/bin/python scripts/cbf_sequence.py run "$1" --stage "$2"
