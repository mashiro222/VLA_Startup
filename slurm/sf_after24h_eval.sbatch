#!/usr/bin/env bash
set -euo pipefail
source scripts/sf_common.sh
campaign="$1"
arm="$2"
"$SF_ENV/bin/python" -m sf_followup.after24h check "$campaign"
run=$("$SF_ENV/bin/python" -m sf_followup.after24h path "$campaign" "$arm" eval)
"$SF_ENV/bin/python" -m sf_followup.evaluation worker "$run"
"$SF_ENV/bin/python" -m sf_followup.after24h compare "$campaign"
