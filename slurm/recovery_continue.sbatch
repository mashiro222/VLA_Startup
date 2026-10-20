#!/usr/bin/env bash
#SBATCH --signal=B:USR1@300
set -euo pipefail
source scripts/sf_common.sh
spec="$1"
export SF_RECOVERY_SPEC="$spec"
export SF_DEADLINE=$(( $(date +%s) + 85800 ))
export SF_RECOVERY_LOCAL="/tmp/vla-recovery-${UID}-${SLURM_JOB_ID}"
[[ ! -L "$SF_RECOVERY_LOCAL" ]]
mkdir -p -m 700 "$SF_RECOVERY_LOCAL"
python3 -m sf_recovery.workflow check "$spec"
python3 -m sf_recovery.stage "$spec" "$SF_RECOVERY_LOCAL" --runtime-only
export PYTHONPATH="$SF_RECOVERY_LOCAL/site:$PYTHONPATH"
plan=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["continuation"]+"/plan.json")' "$spec")
"$SF_ENV/bin/python" -m sf_followup.continuation migrate "$plan" sf
export SF_MASTER_ADDR
SF_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export SF_MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
srun --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash scripts/recovery_worker.sh &
worker_pid=$!
trap 'kill -USR1 "$worker_pid" 2>/dev/null || true' USR1 TERM
wait "$worker_pid" || { code=$?; if [[ "$code" -gt 128 ]]; then wait "$worker_pid"; else exit "$code"; fi; }
