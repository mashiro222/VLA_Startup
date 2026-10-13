# full-plus

Full coverage; differences are descriptive, not a pass/fail replication test. LIBERO-Plus community protocol is incomplete.

Completed: 10030 / 10030
Successes: 8477

| Group | Completed / expected | Successes | Success rate (completed) |
|---|---:|---:|---:|
| libero_spatial | 2402 / 2402 | 2158 | 89.84% |
| libero_object | 2518 / 2518 | 2244 | 89.12% |
| libero_goal | 2591 / 2591 | 2086 | 80.51% |
| libero_10 | 2519 / 2519 | 1989 | 78.96% |
| Background Textures | 1076 / 1076 | 1035 | 96.19% |
| Robot Initial States | 1550 / 1550 | 1154 | 74.45% |
| Camera Viewpoints | 1599 / 1599 | 1163 | 72.73% |
| Language Instructions | 1537 / 1537 | 1314 | 85.49% |
| Sensor Noise | 1601 / 1601 | 1389 | 86.76% |
| Objects Layout | 1525 / 1525 | 1321 | 86.62% |
| Light Conditions | 1142 / 1142 | 1101 | 96.41% |

| Group | Ours | Reference | Difference (pp) |
|---|---:|---:|---:|
| Camera Viewpoints | 72.73% | 67.30% | +5.43 |
| Robot Initial States | 74.45% | 78.80% | -4.35 |
| Language Instructions | 85.49% | 86.30% | -0.81 |
| Light Conditions | 96.41% | 95.80% | +0.61 |
| Background Textures | 96.19% | 93.80% | +2.39 |
| Sensor Noise | 86.76% | 87.30% | -0.54 |
| Objects Layout | 86.62% | 86.40% | +0.22 |
| Overall (micro) | 84.52% | 84.20% | +0.32 |

Reference: https://github.com/sylvestf/LIBERO-plus/issues/57

Parallel execution: Each shard starts the policy RNG at [0, 0], as in the serial evaluator. Its stream is then saved and resumed independently. This is not the same continuous random stream as a single serial full run.
