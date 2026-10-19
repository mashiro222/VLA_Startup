# Training progress and checkpoint audit — 2026-10-18

- [Snapshot](snapshot.json): baseline at update 360, recent 100-update mean 13.505 seconds, no extreme/nonfinite flow loss; SF pending at the recorded check. This is an intermediate observation, not a completed-training result.
- [Scheduling](scheduling.json): test-only four-GPU comparisons and subsequent two-GPU queue estimate. No new GPU job submitted; neither existing job cancelled or modified.
- [Checkpoint tests](checkpoint-tests.xml): 8 CPU checks passed. Real trainer control flow and AdamW/checkpoint operations use a tiny CPU model; GPU construction is replaced. Covers deadline, signal, atomic-write failure, optimizer/RNG restore, and batch partitioning. This does not establish real multi-GPU checkpoint I/O reliability.
- [Environment repair](environment-repair.json): one installed PaliGemma source file contained 85 extra null bytes and failed import/hash validation. Restored atomically from the unchanged pinned source to the exact expected hash. The corrupt copy remains locally archived; cause not established. Full local plan verification and policy/teacher imports subsequently passed, without loading model weights.
- [Validation](validation.json): unchanged training-source hashes, restored transformer-overlay hashes, and successful checks.
- [Storage](storage.json): scratch quota and file-count headroom for checkpoints.

The baseline and SF retain their original two-A100, 24-hour allocations. The speed observation motivates later scaling assessment; the four-GPU continuation is not implemented or submitted. Full experimental context is maintained in the [chronological experiment log](../../docs/results-overview.md); shutdown and resume limits are in the [checkpoint guide](../../docs/run/training-checkpoints.md).
