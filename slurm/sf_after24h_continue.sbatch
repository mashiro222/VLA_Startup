#!/usr/bin/env bash
#SBATCH --signal=B:USR1@300
set -euo pipefail
source scripts/sf_common.sh
campaign="$1"
arm="$2"
export SF_DEADLINE=$(( $(date +%s) + 85800 ))
"$SF_ENV/bin/python" -m sf_followup.after24h check "$campaign"
plan=$("$SF_ENV/bin/python" -m sf_followup.after24h path "$campaign" "$arm" continue)
"$SF_ENV/bin/python" -m sf_followup.continuation migrate "$plan" "$arm"
export SF_FOLLOWUP_PLAN="$plan" SF_FOLLOWUP_ARM="$arm"
export SF_MASTER_ADDR
SF_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export SF_MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
srun --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash scripts/sf_multinode_worker.sh &
worker_pid=$!
trap 'kill -USR1 "$worker_pid" 2>/dev/null || true' USR1 TERM
wait "$worker_pid" || { code=$?; if [[ "$code" -gt 128 ]]; then wait "$worker_pid"; else exit "$code"; fi; }
# Stop here. Any further allocation requires another explicit manual submission.
