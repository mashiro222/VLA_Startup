#!/usr/bin/env bash
#SBATCH --time=04:00:00
set -euo pipefail
source scripts/sf_common.sh
calibration="$(realpath -- "$1")"
# Only the unfinished batch-2 cases; retain the existing batch-1 measurements.
# Leave ten minutes for the final checkpoint, inference check, and report.
export SF_DEADLINE=$(( $(date +%s) + 13800 ))
exec 9>"$calibration/batch2-retry.lock"
flock -n 9
"$SF_ENV/bin/python" scripts/calibrate_mb2.py "$calibration"
