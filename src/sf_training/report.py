"""Local training tables/curves. Loss curves are not safety evaluation results."""

import argparse
import csv
import json
import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sf_training.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    a = ap.parse_args()
    root = Path(os.environ["VLA_ROOT"])
    run = root / "runs" / a.run_id
    out = root / "reports" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run / "plan.json").read_text())
    batch = plan["recipe"]["global_batch"]
    summaries = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for arm in ("baseline", "sf"):
        file = run / arm / "metrics.jsonl"
        if not file.exists():
            continue
        # Resume can repeat logged but uncheckpointed steps. Keep the last record for each step.
        records = {r["step"]: r for r in map(json.loads, file.read_text().splitlines())}
        rows = [records[k] for k in sorted(records)]
        with (out / (arm + "-training.csv")).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        measured = rows[1:] or rows
        seconds = sum(r["seconds"] for r in measured) / len(measured)
        summaries.append(
            dict(
                arm=arm,
                completed_steps=rows[-1]["step"],
                mean_seconds_per_step=seconds,
                global_batch=batch,
                gpus=plan["gpus"],
                projected_60000_steps_at_global32_hours=seconds
                * 60000
                / 3600
                * 32
                / batch,
                peak_allocated_gib=max(r["peak_allocated_gib"] for r in rows),
                peak_reserved_gib=max(r["peak_reserved_gib"] for r in rows),
            )
        )
        for ax, key, title in zip(
            axes,
            ("flow_mse", "alignment", "seconds"),
            ("Flow matching MSE", "Spatial alignment loss", "Seconds / optimizer step"),
        ):
            ax.plot([r["step"] for r in rows], [r[key] for r in rows], label=arm)
            ax.set_xlabel("Optimizer step")
            ax.set_title(title)
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training-curves.png", dpi=150)
    plt.close(fig)
    write_json(
        out / "training-summary.json",
        dict(
            results=summaries,
            caveat="Short smoke timing is a preliminary extrapolation, excluding queue time, conversion and checkpoint I/O. Projection scales linearly from the measured global batch to 32 at the same GPU count; this is approximate and is not a multi-GPU speedup claim.",
        ),
    )
    if summaries:
        with (out / "training-summary.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0]))
            w.writeheader()
            w.writerows(summaries)
    print(out)


if __name__ == "__main__":
    main()
