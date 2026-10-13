#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
python3 - <<'PY'
import subprocess
import sys
sys.path.insert(0, 'src')
from arena_startup.runtime import environment
raise SystemExit(subprocess.run(
    ['.runtime/arena/bin/python', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_arena.py', '-v'],
    env=environment()).returncode)
PY
