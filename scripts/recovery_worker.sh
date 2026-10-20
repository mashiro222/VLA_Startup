#!/usr/bin/env bash
set -euo pipefail
source scripts/sf_common.sh
local_root="$SF_RECOVERY_LOCAL"
[[ ! -L "$local_root" ]]
mkdir -p -m 700 "$local_root"
diagnostics=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["continuation"]+"/diagnostics")' "$SF_RECOVERY_SPEC")
mkdir -p "$diagnostics"
finish() {
    local rc=$?
    shopt -s nullglob
    for file in "$local_root"/rank-*-phase.json; do
        cp -- "$file" "$diagnostics/$(basename "$file" .json)-${SLURM_JOB_ID}.json" || true
    done
    # Only remove this job's node-local cache, never shared checkpoints or inputs.
    if [[ "$local_root" == "/tmp/vla-recovery-${UID}-${SLURM_JOB_ID}" && ! -L "$local_root" ]]; then
        rm -rf -- "$local_root"
    fi
    return "$rc"
}
trap finish EXIT
python3 -m sf_recovery.stage "$SF_RECOVERY_SPEC" "$local_root"
export PYTHONPATH="$local_root/site:$PYTHONPATH"
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=4096
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$diagnostics/nccl-${SLURM_JOB_ID}-"
"$SF_ENV/bin/python" -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=2 --node_rank="$SLURM_PROCID" \
    --rdzv_backend=c10d --rdzv_endpoint="$SF_MASTER_ADDR:$SF_MASTER_PORT" \
    --rdzv_conf=join_timeout=3600,read_timeout=120 \
    --rdzv_id="$SLURM_JOB_ID" --max_restarts=0 \
    -m sf_recovery.train "$SF_RECOVERY_SPEC" "$local_root" &
worker_pid=$!
trap 'pkill -USR1 -P "$worker_pid" 2>/dev/null || true' USR1 TERM
wait "$worker_pid" || { code=$?; if [[ "$code" -gt 128 ]]; then wait "$worker_pid"; else exit "$code"; fi; }
