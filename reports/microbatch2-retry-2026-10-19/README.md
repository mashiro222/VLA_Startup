# Batch 2 calibration retry — 2026-10-19

The one-hour calibration job `14171051` timed out after completing both micro-batch 1 controls (12 updates each, finite inference). Neither micro-batch 2 case produced training records. This is incomplete validation, with no observed OOM.

Job **14179857**, submitted at **10:13 EDT**, runs only the unfinished micro-batch 2 cases: baseline and SF, 12 updates each. It requests **2 × A100 40GB**, 16 CPUs, 96 GB host memory, and **4 hours** under `gts-jli741-paid` / `inferno`. Global batch remains 32 (2 GPUs × micro-batch 2 × accumulation 8). The job exits when both checks finish; its internal deadline leaves ten minutes for saving and reporting. Initial Slurm state: `PENDING (Priority)`.

Existing micro-batch 1 evidence is read and reused without retraining. Completed micro-batch 2 cases are skipped; partial output is rejected for inspection instead of being overwritten. The original calibration plans remain frozen; the new sbatch command and script override the old one-hour allocation limit. This test does not measure four-GPU cross-node communication.

- [Preparation, exact command, script hashes and reused evidence hashes](preparation.json)
- [Confirmed Slurm submission receipt](submission/batch2-only.json)
- [45 CPU/transport regression checks](tests.xml)
- [21 focused retry/checkpoint checks](retry-checkpoint-tests.xml)
- [Retry implementation](../../scripts/calibrate_mb2.py) and [Slurm entry point](../../slurm/microbatch2_retry.sbatch)

The final-checkpoint tests cover both 4,875 and 6,200 updates, including an older 5,000-update file. Evaluation selects the actual final complete checkpoint, and continuation targets that step plus 15,000. These are synthetic selection tests, not measured final training steps.

Baseline job `14143587`, SF job `14143602`, their source plan and frozen training implementation are unchanged. No evaluation, continuation or controller jobs were submitted. GPU batch-2 validation remains pending. Successful completion will write the four-case calibration report to `reports/arena-microbatch-20261018-001/` and its local run directory.
