#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
run_id="${1:?Usage: bash scripts/report.sh RUN_ID}"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 2
PYTHONPATH="$ROOT/src" python3 -m vla_startup.summarize "$ROOT/runs/$run_id"
sim_python "$ROOT/scripts/report.py" "$run_id"
