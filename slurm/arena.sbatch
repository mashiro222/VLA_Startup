#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@90
set -euo pipefail
: "${SLURM_JOB_ID:?Requires a Slurm allocation}"
cd "$SLURM_SUBMIT_DIR"
exec python3 scripts/arena.py worker "$1" --smoke-run "${2:-arena-smoke-001}"
