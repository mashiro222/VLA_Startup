#!/usr/bin/env bash
# Explicit GitHub backup of reports; never pushes checkpoints, scenes, or videos.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
run_id="${1:?Usage: bash scripts/publish_arena.sh ARENA_RUN_ID}"
[[ "$run_id" =~ ^arena-[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 2
[[ $# -eq 1 ]] || exit 2
git diff --cached --quiet || { printf 'Commit existing staged changes first.\n' >&2; exit 2; }
bash scripts/arena.sh report "$run_id"
python3 - "$run_id" <<'PY'
from pathlib import Path
import sys
for p in (Path('reports') / sys.argv[1]).rglob('*'):
    if p.is_file() and p.stat().st_size >= 50 * 1024 * 1024:
        raise SystemExit(f'Report too large for this backup script: {p}')
PY
git add -- "reports/$run_id"
if ! git diff --cached --quiet; then git commit -m "Back up Arena report $run_id"; fi
git push origin main
