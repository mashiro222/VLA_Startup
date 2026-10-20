#!/usr/bin/env bash
set -euo pipefail
# Optional manual staging avoids importing an interpreter from a stalled shared
# filesystem. This bounded wait never submits another job.
if [[ "${SF_AUDIT_STAGED:-0}" == 1 ]]; then
    audit_local="/tmp/vla-sf-audit-${UID}-${SLURM_JOB_ID}"
    [[ ! -L "$audit_local" ]]
    mkdir -p -m 700 "$audit_local"
    echo "CPU audit ready for manual staging on $(hostname): $audit_local"
    deadline=$(( $(date +%s) + 1800 ))
    until [[ -f "$audit_local/ready" ]]; do
        [[ $(date +%s) -lt "$deadline" ]] || { echo 'CPU staging deadline exceeded'; exit 1; }
        sleep 2
    done
    export PYTHONPATH="$audit_local/code/src"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$audit_local/env/bin/python" -u -m sf_recovery.audit \
        runs/arena-sf-train-full-002/plan.json \
        runs/arena-sf-train-full-002/sf/step-00003000.pt \
        reports/sf-recovery-2026-10-20/checkpoint-audit.json \
        --local-file "$audit_local/env/step-00003000.pt"
fi
source scripts/sf_common.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# A one-off CPU checkpoint restore, not a GPU training smoke or a controller.
"$SF_ENV/bin/python" -u -m sf_recovery.audit \
    runs/arena-sf-train-full-002/plan.json \
    runs/arena-sf-train-full-002/sf/step-00003000.pt \
    reports/sf-recovery-2026-10-20/checkpoint-audit.json
