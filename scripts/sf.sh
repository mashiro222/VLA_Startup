#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/sf_common.sh"
exec "$SF_ENV/bin/python" -m sf_training.workflow "$@"
