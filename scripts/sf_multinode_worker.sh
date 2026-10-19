#!/usr/bin/env bash
set -euo pipefail
source scripts/sf_common.sh
"$SF_ENV/bin/python" -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=2 --node_rank="$SLURM_PROCID" \
    --rdzv_backend=c10d --rdzv_endpoint="$SF_MASTER_ADDR:$SF_MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" --max_restarts=0 \
    -m sf_training.train --plan "$SF_FOLLOWUP_PLAN" --arm "$SF_FOLLOWUP_ARM" --resume &
worker_pid=$!
trap 'pkill -USR1 -P "$worker_pid" 2>/dev/null || true' USR1 TERM
wait "$worker_pid" || { code=$?; if [[ "$code" -gt 128 ]]; then wait "$worker_pid"; else exit "$code"; fi; }
