#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
profile="${1:?Usage: bash scripts/prepare.sh PROFILE RUN_ID}"
run_id="${2:?Supply a stable run ID; reuse it to resume}"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { printf 'Invalid run ID\n' >&2; exit 2; }
run_dir="$ROOT/runs/$run_id"
mkdir -p "$run_dir" "$ROOT/logs"
if [[ -f "$run_dir/manifest.json" ]]; then
    python3 - "$run_dir/manifest.json" "$profile" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['profile'] == sys.argv[2], 'Run ID already belongs to a different profile'
PY
else
    PYTHONPATH="$ROOT/src" python3 -m vla_startup.manifest \
        --profile "$profile" --output "$run_dir/manifest.json"
fi
python3 "$ROOT/scripts/preflight.py" --run-dir "$run_dir"
printf 'Prepared %s. Preview submission: bash scripts/submit.sh %s\n' "$run_id" "$run_id"
