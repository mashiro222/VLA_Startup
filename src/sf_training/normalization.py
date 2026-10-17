"""Training-only quantile bounds, sparse-coordinate fallback, and exact inverse.

Keep nondegenerate quantiles unchanged. For a collapsed quantile interval, use
the observed training range; constant coordinates receive a unit-width interval.
No values are clipped and no rare rotations are discarded.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
from sf_training.utils import sha, write_json

METHOD = "exact-frame-weighted-quantiles-clamped-chunks-safe-range-v2"
EPS = 1e-6
MAX_NORMALIZED_ABS = 100.0
MAX_FLOW_MSE = 1000.0


def safe_bounds(stats, minimum, maximum):
    result = copy.deepcopy(stats)
    low = np.asarray(stats.get("empirical_q01", stats["q01"]), np.float64)
    high = np.asarray(stats.get("empirical_q99", stats["q99"]), np.float64)
    minimum, maximum = np.asarray(minimum, np.float64), np.asarray(maximum, np.float64)
    if not (low.shape == high.shape == minimum.shape == maximum.shape):
        raise ValueError("Normalization dimensions differ")
    if not np.isfinite([low, high, minimum, maximum]).all():
        raise ValueError("Nonfinite normalization statistics")
    if np.any(high < low) or np.any(minimum > low) or np.any(high > maximum):
        raise ValueError("Invalid quantile ordering")
    fallback = high - low <= EPS
    result.update(empirical_q01=low.tolist(), empirical_q99=high.tolist())
    low, high = np.where(fallback, minimum, low), np.where(fallback, maximum, high)
    constant = high - low <= EPS
    center = (low + high) / 2
    low, high = np.where(constant, center - .5, low), np.where(constant, center + .5, high)
    result.update(q01=low.tolist(), q99=high.tolist(), minimum=minimum.tolist(),
                  maximum=maximum.tolist(), fallback_dimensions=np.flatnonzero(fallback).tolist(),
                  constant_dimensions=np.flatnonzero(constant).tolist())
    bounds(result)
    return result


def bounds(stats):
    low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    if low.ndim != 1 or low.shape != high.shape or not np.isfinite([low, high]).all():
        raise ValueError("Invalid normalization bounds")
    if np.any(high - low <= EPS):
        raise ValueError("Degenerate normalization interval; regenerate safe training statistics")
    return low, high


def normalize(x, stats):
    low, high = bounds(stats)
    x = np.asarray(x, np.float32)
    if x.shape[-1] != len(low):
        raise ValueError("Normalization input dimension differs")
    value = (x - low) / (high - low + EPS) * 2 - 1
    if not np.isfinite(value).all() or np.max(np.abs(value)) > MAX_NORMALIZED_ABS:
        raise ValueError("Nonfinite or extreme normalized training target/state")
    return value


def unnormalize(x, stats):
    """Inverse of this experiment's transform, including the denominator epsilon."""
    low, high = bounds(stats)
    x = np.asarray(x, np.float32)
    if x.shape[-1] != len(low) or not np.isfinite(x).all():
        raise ValueError("Invalid normalized policy output")
    value = (x + 1) / 2 * (high - low + EPS) + low
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite unnormalized policy output")
    return value


def audit_arrays(arrays, stats):
    report = {}
    for key, values in arrays.items():
        value = normalize(values, stats[key])
        restored = unnormalize(value, stats[key])
        if not np.allclose(restored, values, atol=2e-6, rtol=2e-6):
            raise ValueError("Normalization inverse failed: " + key)
        report[key] = dict(frames=len(values), normalized_abs_max_by_dimension=np.abs(value).max(0).tolist(),
                           roundtrip_abs_error=float(np.abs(restored - values).max()),
                           fallback_dimensions=stats[key].get("fallback_dimensions", []))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repair", action="store_true")
    a = ap.parse_args()
    root = Path(os.environ["VLA_ROOT"])
    home = root / ".runtime/training"
    path = home / "norm_stats.json"
    document = json.loads(path.read_text())
    packed = home / "arena_l0/packed"
    marker = json.loads((packed / "complete.json").read_text())
    if document["source_manifest"] != marker["source_manifest"]:
        raise ValueError("Statistics and packed data identities differ")
    arrays = {k: np.load(packed / (k + ".npy"), mmap_mode="r") for k in ("state", "actions")}
    for key in arrays:
        if sha(packed / (key + ".npy")) != marker["files"][key + ".npy"]["sha256"]:
            raise ValueError("Packed numerical data checksum differs: " + key)
    if any(len(v) != document["frames"] for v in arrays.values()):
        raise ValueError("Frame count differs")
    if a.repair and document["method"] != METHOD:
        if document["method"] != "exact-frame-weighted-quantiles-clamped-chunks-v1":
            raise ValueError("Unsupported source statistics")
        old_hash = sha(path)
        # Preserve the original artifact for historical runs and diagnosis.
        archived = home / "normalization_archive" / (old_hash + ".json")
        archived.parent.mkdir(exist_ok=True)
        if not archived.exists():
            archived.write_bytes(path.read_bytes())
        document["norm_stats"] = {k: safe_bounds(document["norm_stats"][k], v.min(0), v.max(0)) for k, v in arrays.items()}
        document.update(method=METHOD, previous_norm_stats_sha256=old_hash)
        audit_arrays(arrays, document["norm_stats"])
        write_json(path, document)
    if document["method"] != METHOD:
        raise ValueError("Legacy statistics; run with --repair before preparing a new run")
    report = dict(method=METHOD, norm_stats_sha256=sha(path),
                  packed_manifest_sha256=sha(packed / "complete.json"),
                  arrays=audit_arrays(arrays, document["norm_stats"]))
    write_json(home / "normalization_audit.json", report)
    write_json(root / "reports/preparation/sf-normalization-audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
