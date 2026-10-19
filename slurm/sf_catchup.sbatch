#!/usr/bin/env bash
#SBATCH --signal=B:USR1@300
set -euo pipefail
source scripts/sf_common.sh
export SF_DEADLINE=$(( $(date +%s) + 85800 ))
plan="$1"
arm="$2"
"$SF_ENV/bin/python" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node=2 -m sf_training.train --plan "$plan" --arm "$arm" --resume --stop-step 5000 &
worker_pid=$!
trap 'pkill -USR1 -P "$worker_pid" 2>/dev/null || true' USR1 TERM
wait "$worker_pid" || { code=$?; if [[ "$code" -gt 128 ]]; then wait "$worker_pid"; else exit "$code"; fi; }
