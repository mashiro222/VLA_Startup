#!/usr/bin/env bash
# One array, at most four one-GPU workers. Only --submit calls sbatch.
set -euo pipefail
source "$(dirname "$0")/common.sh"
batch_id="${1:?Usage: bash scripts/submit_parallel.sh BATCH_ID [--submit]}"
mode="${2:---dry-run}"
[[ $# -le 2 ]] || { printf 'Too many arguments\n' >&2; exit 2; }
[[ "$batch_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 2
[[ "$mode" == --dry-run || "$mode" == --submit ]] || exit 2
source "$ROOT/config/pace.env"
if [[ -f "$ROOT/config/local.env" ]]; then source "$ROOT/config/local.env"; fi
case "${VLA_GPU_TYPE:-}" in
    '') ;;
    a100) PARTITION=gpu-a100; GRES=gpu:a100:1 ;;
    h100) PARTITION=gpu-h100; GRES=gpu:h100:1 ;;
    *) printf 'VLA_GPU_TYPE must be a100 or h100.\n' >&2; exit 2 ;;
esac
case "$PARTITION:$GRES" in
    gpu-a100:gpu:a100:1|gpu-h100:gpu:h100:1) ;;
    *) printf 'Use a matching A100 or H100 partition and one GPU per worker.\n' >&2; exit 2 ;;
esac
if [[ -n "${VLA_AFTERANY_JOB_ID:-}" && ! "$VLA_AFTERANY_JOB_ID" =~ ^[0-9]+$ ]]; then
    printf 'VLA_AFTERANY_JOB_ID must be a numeric Slurm array job ID.\n' >&2
    exit 2
fi
python3 "$ROOT/scripts/preflight.py"
python3 "$ROOT/scripts/parallel.py" check "$batch_id"
workers="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["workers"])' "$ROOT/runs/$batch_id/batch.json")"
remaining="$(python3 "$ROOT/scripts/parallel.py" remaining "$batch_id")"
if [[ -z "$remaining" ]]; then
    printf 'All workers already complete. Merge reports; no submission needed.\n'
    exit 0
fi
mkdir -p "$ROOT/logs"
dependency=()
if [[ -n "${VLA_AFTERANY_JOB_ID:-}" ]]; then
    dependency=("--dependency=afterany:$VLA_AFTERANY_JOB_ID")
fi
cmd=(sbatch --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
    --gres="$GRES" --cpus-per-task="$CPUS" --mem="$MEMORY" --time="${TIME_LIMIT:-12:00:00}" \
    "${dependency[@]}" \
    --array="$remaining%$workers" --job-name="pi05-$batch_id" \
    --chdir="$ROOT" --output="$ROOT/logs/$batch_id-%A_%a.out" \
    "$ROOT/slurm/parallel.sbatch" "$batch_id" \
    "${CLEAN_SMOKE_RUN:-smoke-clean-001}" "${PLUS_SMOKE_RUN:-smoke-plus-001}")
printf 'Submission command: '
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "$mode" == --submit ]]; then
    python3 "$ROOT/scripts/parallel.py" check "$batch_id" --require-smoke \
        --clean-smoke "${CLEAN_SMOKE_RUN:-smoke-clean-001}" --plus-smoke "${PLUS_SMOKE_RUN:-smoke-plus-001}"
    # Avoid charging a second copy of the same batch. Do not resubmit while any
    # old array member is pending, running, or completing.
    active="$(squeue -h -u "$(id -un)" -n "pi05-$batch_id" -o '%i')"
    if [[ -n "$active" ]]; then
        printf 'Batch already has active jobs: %s\n' "$active" >&2
        exit 2
    fi
    cd "$ROOT"
    if "${cmd[@]}"; then
        :
    else
        code=$?
        printf 'Submission reply failed. A job may still exist; do not retry before checking squeue.\n' >&2
        squeue -u "$(id -un)" -n "pi05-$batch_id" || true
        exit "$code"
    fi
else
    printf 'Dry run only. No job submitted. Workers verify the selected benchmarks smoke runs before loading models.\n'
fi
