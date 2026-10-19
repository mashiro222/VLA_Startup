# 5k evaluation / 20k continuation preparation

Prepared on 2026-10-18 (America/New_York). This directory contains preparation and submission evidence, not new model results.

- Calibration job: **14171051**, two A100 40 GB GPUs, one-hour allocation, account `gts-jli741-paid`; pending when checked.
- Handoff job: **14171052**, one CPU, 2 GB system RAM, no GPU; running and writing campaign state when checked.
- Original baseline **14143587** continues; original SF **14143602** remains queued. Their frozen source, recipe, dependencies and Slurm requests were not modified.
- **18 tests passed**: checkpoint preservation, migration invariants, replayable input RNG, training/inference preprocessing, strict step/arm identity, submission idempotency, GPU-cap dependencies and existing communication regression checks.

The paired initial evaluation uses exactly 5,000 updates, global batch 32, and 300 episodes per arm across StaticObstacles/DynamicObstacles L0/L1/L2. After successful calibration and completion of initial training, continuation starts from the matched 5k checkpoints and targets 20k, using two nodes with two A100s each, micro-batch 2 and accumulation 4. Four-GPU jobs have not yet been submitted; the controller submits them only after their prerequisites are met.

Actual batch-2 GPU memory/throughput, trained-policy Arena inference/EGL, and four-GPU cross-node execution are **not yet verified**. Scheduler test-only accepted all requests; its candidate dates are provisional and are not measured execution results. Baseline/SF allocation times and throughput can differ.

See [operational protocol](../../docs/run/microbatch-and-20k.md), [frozen campaign](campaign.json), [submission receipts](calibration.json), [CPU handoff receipt](monitor.json), [validation snapshot](validation.json), and [test results](tests.xml).

Large model/checkpoint files, video and scenario assets remain on scratch and are not included here.
