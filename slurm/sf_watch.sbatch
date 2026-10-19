#!/usr/bin/env bash
set -euo pipefail
source scripts/common.sh
export PYTHONPATH="$ROOT/src"
exec python3 -m sf_followup.control monitor "$1"
