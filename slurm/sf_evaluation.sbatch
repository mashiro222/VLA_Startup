#!/usr/bin/env bash
set -euo pipefail
source scripts/sf_common.sh
exec "$SF_ENV/bin/python" -m sf_followup.evaluation worker "$1"
