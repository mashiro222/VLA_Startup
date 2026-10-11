#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
sim_python "$ROOT/scripts/download.py" "$@"
