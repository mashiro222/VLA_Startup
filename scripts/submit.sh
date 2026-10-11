#!/usr/bin/env bash
# Printing the command is the default. Only an explicit --submit calls sbatch.
set -euo pipefail
source "$(dirname "$0")/common.sh"
run_id="${1:?Usage: bash scripts/submit.sh RUN_ID [--submit]}"
mode="${2:---dry-run}"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { printf 'Invalid run ID\n' >&2; exit 2; }
[[ "$mode" == --dry-run || "$mode" == --submit ]] || { printf 'Unknown mode\n' >&2; exit 2; }
source "$ROOT/config/pace.env"
if [[ -f "$ROOT/config/local.env" ]]; then source "$ROOT/config/local.env"; fi
run_dir="$ROOT/runs/$run_id"
python3 "$ROOT/scripts/preflight.py" --run-dir "$run_dir"
profile="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "$run_dir/manifest.json")"
case "$profile" in
    smoke-*) time_limit=00:45:00 ;;
    pilot-plus) time_limit=04:00:00 ;;
    full-clean) time_limit=24:00:00 ;;
    full-plus) time_limit=24:00:00 ;;
    *) printf 'Invalid profile\n' >&2; exit 2 ;;
esac
mkdir -p "$ROOT/logs"
cmd=(sbatch --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
    --gres="$GRES" --cpus-per-task="$CPUS" --mem="$MEMORY" --time="${TIME_LIMIT:-$time_limit}" \
    --job-name="pi05-$run_id" --chdir="$ROOT" --output="$ROOT/logs/$run_id-%j.out" \
    "$ROOT/slurm/evaluate.sbatch" "$run_id")
printf 'Submission command: '
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "$mode" == --submit ]]; then
    cd "$ROOT"
    "${cmd[@]}"
else
    printf 'Dry run only. No job submitted. Add --submit when ready.\n'
fi
