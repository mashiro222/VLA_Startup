#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT"
python3 scripts/bootstrap.py
mkdir -p "$VLA_RUNTIME" "$UV_CACHE_DIR" "$OPENPI_DATA_HOME"
# Both Python environments are created and maintained by uv.
# Runtime setup performs no training, inference, or Slurm allocation.
if [[ ! -x "$SIM_ENV/bin/python" ]]; then
    "$UV" venv --python 3.10 "$SIM_ENV"
fi
if [[ ! -f requirements/sim.lock ]]; then
    "$UV" pip compile requirements/sim.in --python "$SIM_ENV/bin/python" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match --output-file requirements/sim.lock
fi
"$UV" pip sync --python "$SIM_ENV/bin/python" requirements/sim.lock \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
"$SIM_ENV/bin/python" scripts/patch_sim.py
# PYTHONPATH selects exactly one LIBERO implementation at run time. Avoid upstream
# editable-install namespace problems and never install both as site packages.
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$SERVER_ENV" \
    "$UV" sync --project "$OPENPI_REPO" --python 3.11 --frozen --no-dev
mkdir -p "$VLA_RUNTIME/locks"
"$UV" pip freeze --python "$SERVER_ENV/bin/python" > "$VLA_RUNTIME/locks/server.freeze.txt"
"$UV" pip freeze --python "$SIM_ENV/bin/python" > "$VLA_RUNTIME/locks/sim.freeze.txt"
python3 scripts/configure.py
printf 'Setup finished. Next: bash scripts/download.sh\n'
