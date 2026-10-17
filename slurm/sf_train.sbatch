#!/usr/bin/env bash
#SBATCH --signal=B:USR1@300
set -euo pipefail
source scripts/sf_common.sh
run_id="$1"
mode="$2"
if [[ "$mode" == smoke ]]; then export SF_DEADLINE=$(( $(date +%s) + 10200 )); else export SF_DEADLINE=$(( $(date +%s) + 85800 )); fi
plan="$ROOT/runs/$run_id/plan.json"
if [[ "$mode" != smoke ]]; then "$SF_ENV/bin/python" -m sf_training.workflow verify-smoke "$run_id" "$3"; fi
# Reserve this path across both arms; conversion is atomic and never done on login nodes.
exec 8>"$SF_HOME/base-conversion.lock"
flock 8
"$SF_ENV/bin/python" -m sf_training.convert
flock -u 8
worker_pid=''
forward_stop() { if [[ -n "$worker_pid" ]]; then pkill -USR1 -P "$worker_pid" 2>/dev/null || true; fi; }
trap forward_stop USR1 TERM
run_worker() {
    "$SF_ENV/bin/python" -m torch.distributed.run --standalone --nnodes=1 \
        --nproc_per_node="$SLURM_GPUS_ON_NODE" -m sf_training.train --plan "$plan" "$@" &
    worker_pid=$!
    wait "$worker_pid" || { code=$?; if [[ "$code" -gt 128 ]]; then wait "$worker_pid"; else return "$code"; fi; }
    worker_pid=''
}
if [[ "$mode" == smoke ]]; then
    for arm in baseline sf; do
        run_worker --arm "$arm" --stop-step 2
        run_worker --arm "$arm" --resume --stop-step 3
    done
    "$SF_ENV/bin/python" -m sf_training.workflow finish-smoke "$run_id"
else
    resume=()
    if compgen -G "$ROOT/runs/$run_id/$mode/step-*.pt" > /dev/null; then resume=(--resume); fi
    run_worker --arm "$mode" "${resume[@]}"
fi
"$SF_ENV/bin/python" -m sf_training.report "$run_id"
