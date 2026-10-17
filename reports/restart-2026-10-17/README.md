# Training restart and Arena evaluation — 2026-10-17

The first formal training runs were stopped after confirming degenerate action normalization. Jobs 14127636 (baseline) and 14127637 (SF) are cancelled; their checkpoints are excluded from the corrected comparison. The earlier three-update validation is insufficient evidence for the corrected training pipeline.

The new statistics retain nondegenerate quantiles and replace collapsed intervals with the observed training range, with a unit-width interval for constant dimensions. No action clipping or sample removal is applied. All 361,736 frames pass normalization and inverse checks: maximum action magnitude after normalization is approximately 5, and maximum action roundtrip error is below 6e-8. A scan of every action coordinate also covers every value that can appear in a clamped action chunk.

The corrected formal run is `arena-sf-train-full-002`. GPU training smoke is **explicitly skipped** for this restart; this is not recorded as a GPU validation pass. Both arms start from pi05_base in fresh directories, each requesting two A100 GPUs. Frozen input identity, normalized input range, finite gradients and a maximum flow MSE guard still apply at runtime. Submission receipts record `gpu_smoke=explicitly_skipped` and contain no smoke dependency.

An optional expanded smoke implementation remains available for future validation: global batch 32, 32 updates per arm, process restart at update 16, sparse rotation coverage and inference checks. It is not submitted for this restart.

## Evaluation

Arena diagnostic smoke `arena-smoke-003`, job 14126130_0: 15/15 episodes completed, SR 0%, mean CC 33.33, no infrastructure errors. Predicate attribution is recorded for all episodes. Low SR is not an infrastructure failure and does not by itself establish poor safety.

The matching `arena-balanced1500-001` evaluation is submitted as job array **14143493**, with up to four independent one-A100 workers, 8 hours each. It covers 75 tasks × 20 initial states = 1,500 episodes. All jobs use account `gts-jli741-paid`.

## Evidence

- [Original normalization defect](../training-audit-2026-10-17/README.md)
- [Full-dataset repaired normalization audit](../preparation/sf-normalization-audit.json)
- [Replay of the first 32 batches](../preparation/sf-sparse-batch-replay.json)
- [Training protocol and monitoring](../../docs/arena-sf-training.md)
- [Evaluation smoke results](../arena-smoke-003/summary.md)

The corrected training code has not yet been GPU-validated; no new GPU smoke is submitted. CPU tests and a normalization audit do not substitute for that validation. Submission receipts and frozen plans are archived here after preparation/submission; active logs, checkpoints and data stay in scratch.

## Confirmed submissions

| Experiment | Job ID | GPUs | Gate |
|---|---|---|---|
| Corrected baseline | 14143587 | 2 A100 | GPU training smoke explicitly skipped |
| Corrected SF | 14143602 | 2 A100 | GPU training smoke explicitly skipped |
| pi05_libero Safety balanced1500 | 14143493 | Up to 4 independent A100 workers | Matching evaluation smoke completed |

Training requests have a 24-hour limit and no smoke dependency. Total simultaneous demand is at most 8 GPUs. A 24-hour allocation is not a guarantee of completing all 60,000 updates; interrupted runs save checkpoints and can be resumed using the commands in the training documentation.

CPU validation: initial suite 43 passed; subsequent targeted normalization/submission checks 23 passed, with overlapping test coverage. See [validation record](validation.json) and [targeted JUnit results](../preparation/sf-restart-tests.xml). Both the original and corrected normalization evidence are retained; no model weights or dataset assets are included in this archive.

The training plan, repaired normalization metadata and all three submission receipts are in this directory. Live status is available from Slurm and the run directories; this is a submission snapshot, not a claim that GPU training has completed.
