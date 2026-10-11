#!/usr/bin/env bash
# Run on a login node after reviewing reports/RUN_ID. Uploads only that report.
set -euo pipefail
source "$(dirname "$0")/common.sh"
run_id="${1:?Usage: bash scripts/publish_results.sh RUN_ID}"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 2
cd "$ROOT"
bash scripts/report.sh "$run_id"
python3 - "$run_id" <<'PY'
from pathlib import Path
import sys
for p in (Path('reports')/sys.argv[1]).rglob('*'):
    if p.is_file() and p.stat().st_size > 50*1024**2:
        raise SystemExit(f'Report too large for routine GitHub backup: {p}')
PY
# Avoid accidentally committing unrelated staged work.
if ! git diff --cached --quiet; then
    printf 'Commit or unstage existing staged changes first.\n' >&2
    exit 2
fi
git add -- "reports/$run_id"
if git diff --cached --quiet; then
    printf 'No report changes to commit.\n'
else
    git commit -m "Save evaluation report $run_id"
fi
git push origin HEAD
