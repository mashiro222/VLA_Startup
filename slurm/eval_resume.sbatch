#!/usr/bin/env bash
set -euo pipefail
source scripts/sf_common.sh
spec="$1"
python3 -m eval_resume.workflow check "$spec"
local_root="$(mktemp -d /tmp/vla-baseline-eval-${UID}-${SLURM_JOB_ID}-XXXXXX)"
trap 'rm -rf -- "$local_root"' EXIT
python3 -m eval_resume.runtime worker "$spec" "$local_root"
