# Baseline 5,952-step evaluation recovery

The previous allocation, **14199187**, failed on a 600-second policy-response deadline after **341/900** durable episodes. The interrupted episode reached simulator step 95. No server inference stack was captured; the underlying stall cause remains unconfirmed.

Completed episodes: **275 successes / 341**, SR **80.65%**, mean CC **24.74**. All completed episodes are from Static obstacles; these are partial results, not the final 900-episode score.

The retry preserves the original manifest and journal prefixes and evaluates the remaining **559 episodes: Static 109 + Dynamic 450**. It uses one A100, 8 CPUs, 96 GB host memory and a 24-hour allocation under `gts-jli741-paid`. Policy inputs are staged locally with checksum verification. Transport failures permit at most three worker attempts within this allocation; task failures and model errors are not retried. No new jobs are submitted automatically.

Existing jobs **14217601**, **14230443**, **14230444** and their frozen inputs remain unchanged.

Submitted retry: **14252522**, initial state `PENDING (Priority)`. [Submission receipt](submission.json). The allocation evaluates only the remaining 559 episodes; the original 341 records remain in the same run directory.

Validation: **34 tests passed**, including **22 recovery/socket tests**. Eight unchanged native-simulator integration tests were excluded after shared-library import waits; they are not counted as passed. Original policy environment imports and a local diagnostic CPU-library copy also stalled in `cl_sync_io_wait` on the login node. Only those local diagnostic processes were stopped. The recovery stages the policy CUDA and simulator CPU runtimes separately and retains strict environment/checkpoint validation at GPU startup.

- [Failure evidence](failure-analysis.json)
- [Regression tests](tests.xml)
- [Focused recovery/socket tests](recovery-tests.xml)
- [Frozen completed prefixes and remaining episode keys](resume.json)
- [Local validation limitation](local-validation-limit.json)
- [Recovery protocol and commands](../../docs/run/baseline-eval-resume-2026-10-21.md)

Real GPU/EGL recovery is pending execution; CPU and socket tests do not establish that the original hardware/runtime stall cannot recur.
