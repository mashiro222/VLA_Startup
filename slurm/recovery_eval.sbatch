#!/usr/bin/env bash
set -euo pipefail
source scripts/sf_common.sh
spec="$1"
python3 -m sf_recovery.workflow check "$spec"
local_root="/tmp/vla-recovery-eval-${UID}-${SLURM_JOB_ID}"
[[ ! -L "$local_root" ]]
mkdir -p -m 700 "$local_root"
trap 'rm -rf -- "$local_root"' EXIT
python3 -m sf_recovery.stage "$spec" "$local_root" --runtime-only
export PYTHONPATH="$local_root/site:$PYTHONPATH"
"$SF_ENV/bin/python" -m sf_recovery.evaluate "$spec" "$local_root"
