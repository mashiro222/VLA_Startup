# Paired 5k evaluation: 900 episodes per arm

**Historical preparation; superseded by the [manual-only workflow](../../docs/run/after-24h-followup.md). The CPU controllers are stopped and no follow-up GPU jobs were submitted under this plan.**

The evaluation budget increased before any evaluation job was submitted: 30 task configurations × 30 distinct initial states = **900 episodes per arm**, or **1,800** across baseline and SF. The original 300-episode manifests remain archived and are superseded for this comparison.

| Scope | Task configurations | Episodes per arm |
|---|---:|---:|
| StaticObstacles, L0/L1/L2 | 15 | 450 |
| DynamicObstacles, L0/L1/L2 | 15 | 450 |
| Total | 30 | 900 |

Each suite/level has five tasks and 150 episodes. All 30 task files contain 50 available initial states; evaluation uses indices 0–29, with identical task IDs and episode seeds across arms. The checkpoint remains exactly 5,000 optimizer updates at global batch 32. Episode horizon remains 300 simulation steps. Each single-GPU evaluation now requests a maximum of 24 hours; this is a resource limit, not a runtime prediction.

The CPU-only controller **14171052** was stopped and replaced by **14171526** to load the revised frozen plans. Training jobs **14143587 / 14143602** and calibration **14171051** were preserved. The controller requests one CPU, 2 GB RAM and no GPU. Workflow GPU concurrency remains at most eight; evaluation allocations finish before the SF four-GPU continuation starts. This does not cap independent jobs manually submitted outside the workflow.

- [Budget validation](budget-validation.json): counts, matched schedules and plan identities.
- [Baseline plan](baseline-plan.json) / [SF plan](sf-plan.json): full evaluation manifests.
- [Campaign](campaign.json): revised controller manifest.
- [Replacement controller receipt](monitor-900.json): actual CPU job submission.
- [Regression checks](budget-tests.xml): **7 passed**, covering 900 distinct task/initial-state pairs, balanced suite/level coverage, insufficient initial states, missing task detection, requested wall time and evaluation-to-training dependencies.
- [Execution guide](../../docs/run/microbatch-and-20k.md): control flow and the eight-GPU cap.

These are preparation and scheduling records, not completed evaluation results. Real inference/EGL and four-GPU throughput remain subject to their scheduled jobs.
