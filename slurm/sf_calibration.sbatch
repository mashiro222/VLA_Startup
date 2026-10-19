#!/usr/bin/env bash
set -euo pipefail
source scripts/sf_common.sh
calibration="$1"
export SF_DEADLINE=$(( $(date +%s) + 3300 ))
for micro in 1 2; do
    for arm in baseline sf; do
        "$SF_ENV/bin/python" -m torch.distributed.run --standalone --nnodes=1 \
            --nproc_per_node=2 -m sf_training.train \
            --plan "$calibration/mb$micro/plan.json" --arm "$arm"
    done
done
"$SF_ENV/bin/python" -m sf_followup.calibration summarize "$calibration"
