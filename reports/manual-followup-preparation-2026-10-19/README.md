# Manual follow-up jobs after the initial 24-hour training allocations

Prepared four independent manual commands: baseline evaluation, SF evaluation, baseline continuation, and SF continuation. **No follow-up job has been submitted.** The former CPU controllers are stopped and their CLI startup is disabled.

Each arm uses its own final complete checkpoint after its original allocation ends. Evaluation covers 900 episodes per arm, with identical task/initial-state schedules. Continuation uses four A100 GPUs, micro-batch 2 and accumulation 4, for exactly 15,000 additional optimizer updates. The final target is determined from that arm's checkpoint; it is not fixed at 20,000. There is no total GPU cap, no cross-arm dependency, no polling controller, and no automatic resubmission.

- [Prepared configuration](campaign.json): original jobs, manual-only policy and frozen inputs.
- [Four command previews](previews.txt): account and resource requests; no sbatch submission was executed.
- [Validation](validation.json): zero new jobs and the verified budget.
- [Regression tests](tests.xml): **39 passed**, including final-checkpoint selection, different per-arm update counts, default preview, single-task submission, duplicate prevention, manual resume without extending the target, initial-job state checks, and policy transport.
- [Manual runbook](../../docs/run/after-24h-followup.md): exact commands and output paths.

Original training jobs 14143587 / 14143602 and calibration 14171051 remain unchanged. Real GPU inference/EGL and four-GPU training are not established by these CPU/transport checks; the existing batch-2 calibration remains a separate runtime check.
