"""Numerical and coverage gates for the real-model training smoke."""

import json
import math
from sf_training.normalization import MAX_FLOW_MSE, MAX_NORMALIZED_ABS

SMOKE_STEPS = 32
SMOKE_RESUME_STEP = 16


def validate_metrics(path, steps=SMOKE_STEPS):
    records = {r["step"]: r for r in map(json.loads, path.read_text().splitlines())}
    if set(records) != set(range(1, steps + 1)):
        raise ValueError("Smoke metrics do not cover every required update")
    rows = list(records.values())
    for row in rows:
        if not all(math.isfinite(row[k]) for k in ("loss", "flow_mse", "alignment", "grad_norm", "target_abs_max")):
            raise ValueError("Nonfinite smoke metrics")
        if not 0 <= row["flow_mse"] <= MAX_FLOW_MSE or row["target_abs_max"] > MAX_NORMALIZED_ABS:
            raise ValueError("Extreme finite loss or normalized targets in smoke")
    rare = sum(r["rare_rotation_samples"] for r in rows)
    if rare < 1:
        raise ValueError("Smoke did not exercise sparse rotation samples")
    return dict(updates=len(rows), max_flow_mse=max(r["flow_mse"] for r in rows),
                target_abs_max=max(r["target_abs_max"] for r in rows), rare_rotation_samples=rare)
