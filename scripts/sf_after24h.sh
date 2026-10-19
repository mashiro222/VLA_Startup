#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/sf_common.sh"
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash scripts/sf_after24h.sh {baseline|sf} {eval|continue} [--submit]" >&2
    exit 2
fi
case "$1:$2" in
    baseline:eval|baseline:continue|sf:eval|sf:continue) ;;
    *) echo "Expected baseline/sf and eval/continue" >&2; exit 2 ;;
esac
if [[ $# -eq 3 && "$3" != --submit ]]; then
    echo "Only --submit is accepted as the optional third argument" >&2
    exit 2
fi
exec python3 -m sf_followup.after24h submit \
    "$ROOT/runs/arena-after24h-20261019-001/campaign.json" "$@"
