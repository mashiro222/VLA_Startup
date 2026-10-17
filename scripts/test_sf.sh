#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/sf_common.sh"
"$SF_ENV/bin/python" -m pytest tests/sf_training -q "$@"
