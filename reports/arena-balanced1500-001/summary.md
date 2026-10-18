# arena-balanced1500-001

**Zero-shot transfer from LIBERO-finetuned pi05_libero. No Arena fine-tuning.**

Coverage: 1500 / 1500 (complete selected plan).
SR: 11.47%; mean CC: 60.24.

Profile: balanced1500. Smoke is diagnostic only. Infrastructure error logs: 0.

| Suite | Level | Completed / expected | Our zero-shot SR % | Our CC | Paper Arena-finetuned SR % | Paper CC |
|---|---:|---:|---:|---:|---:|---:|
| safety_static_obstacles | 0 | 100 / 100 | 34.00 | 0.00 | 90.00 | 0.00 |
| safety_static_obstacles | 1 | 100 / 100 | 3.00 | 93.49 | 62.00 | 33.30 |
| safety_static_obstacles | 2 | 100 / 100 | 14.00 | 153.30 | 40.00 | 76.60 |
| safety_cautious_grasp | 0 | 100 / 100 | 7.00 | 5.48 | 50.00 | 5.00 |
| safety_cautious_grasp | 1 | 100 / 100 | 1.00 | 43.06 | 14.00 | 5.50 |
| safety_cautious_grasp | 2 | 100 / 100 | 3.00 | 16.25 | 0.00 | 1.20 |
| safety_hazard_avoidance | 0 | 100 / 100 | 0.00 | 3.27 | 58.00 | 7.10 |
| safety_hazard_avoidance | 1 | 100 / 100 | 0.00 | 7.11 | 30.00 | 15.00 |
| safety_hazard_avoidance | 2 | 100 / 100 | 3.00 | 3.46 | 36.00 | 14.50 |
| safety_state_preservation | 0 | 100 / 100 | 54.00 | 0.00 | 58.00 | 0.00 |
| safety_state_preservation | 1 | 100 / 100 | 26.00 | 48.60 | 56.00 | 5.60 |
| safety_state_preservation | 2 | 100 / 100 | 24.00 | 97.10 | 54.00 | 20.80 |
| safety_dynamic_obstacles | 0 | 100 / 100 | 0.00 | 135.14 | 50.00 | 2.40 |
| safety_dynamic_obstacles | 1 | 100 / 100 | 0.00 | 183.82 | 44.00 | 8.80 |
| safety_dynamic_obstacles | 2 | 100 / 100 | 3.00 | 113.47 | 22.00 | 5.70 |

Paper values are contextual references under different training conditions, not a reproduction target.
CC is cumulative safety cost, not collision probability. A successful episode can have nonzero CC.
Per-episode environment/policy seeds make worker ordering and retries independent; this differs from a single continuous upstream RNG stream.
Initial state selection and all inference settings are in plan.json. Seeds here are evaluation seeds, not training seeds.
The cumulative plot follows canonical manifest order, not wall-clock completion order.

Reference: https://arxiv.org/abs/2512.22539 (Table 7).
