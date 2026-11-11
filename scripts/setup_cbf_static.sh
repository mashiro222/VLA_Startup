#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.tools/uv pip install --python .runtime/arena/bin/python --target .runtime/cbf_deps --no-deps -r requirements/cbf-static.txt
.tools/uv pip install --python .runtime/server/bin/python --target .runtime/cbf_server_deps -r requirements/cbf-static-server.txt
python3 scripts/download_cbf_checkpoint.py
