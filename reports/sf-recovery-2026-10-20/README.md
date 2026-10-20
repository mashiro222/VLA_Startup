# SF recovery and checkpoint retention

The interrupted SF run saved update 3,000 before an NCCL all-reduce timeout. Its last logged update was 3,460. The precise cause is unresolved; the next 32 training samples decoded successfully and all recorded losses were finite.

| Evidence | Content |
|---|---|
| [Failure analysis](failure-analysis.json) | Original job, finite losses, collective sequence and limits of the diagnosis |
| [NCCL excerpt](nccl-failure-excerpt.txt) | Original timeout messages |
| [Data replay](update-3461-data-replay.json) | The next update's exact sample indices and CPU decoding checks |
| [Baseline retention](baseline-retention.json) | Only retention increased; baseline job 14217601 was not resubmitted |
| [Previous manifests](baseline-retention-before/) | Configuration snapshots before increasing retention |
| [Regression tests](tests.xml) | 53 passing CPU checks, including checkpoint restoration, state migration and 18k retention |
| [Checkpoint restore](checkpoint-audit.json) | CPU job 14229891 restored 955 trainable tensors, 947 Adam states and both ranks' CPU RNG states; GPU execution remains untested |

The recovery protocol is SF 3,000 → 18,000 updates with four A100 GPUs, micro-batch 2 and accumulation 4. Its independent 900-episode evaluation uses the same task/initial-state pairs as the existing baseline 5,952-step evaluation. Those interim results have unequal training update counts. A future comparison should use both 18,000-step checkpoints, which the retention settings preserve.

The baseline continuation still targets 20,952 updates. The 18,000-step checkpoint is an intermediate saved checkpoint, not a newly submitted baseline job.

Submitted on 2026-10-20 at 17:05 EDT: SF continuation **14230443**, SF evaluation **14230444**. Both were pending for priority at 17:06. Baseline continuation **14217601** and evaluation **14199187** retain their original job IDs. See [submission receipts](submission/), [prepared configuration](preparation.json), and [Slurm resource records](slurm-jobs.txt).

Runtime weights, datasets and node-local caches are excluded from GitHub. See the [detailed recovery protocol](../../docs/run/sf-recovery-2026-10-20.md).
