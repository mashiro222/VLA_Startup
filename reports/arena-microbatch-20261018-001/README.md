# Micro-batch calibration

| Arm | Micro-batch | Accumulation | GPU | Seconds/update (median) | Allocated GiB | Reserved GiB |
|---|---:|---:|---|---:|---:|---:|
| baseline | 1 | 16 | NVIDIA A100-PCIE-40GB | 12.627 | 16.81 | 17.27 |
| sf | 1 | 16 | NVIDIA A100-PCIE-40GB | 15.721 | 20.37 | 20.72 |
| baseline | 2 | 8 | NVIDIA A100-PCIE-40GB | 7.520 | 17.13 | 17.50 |
| sf | 2 | 8 | NVIDIA A100-PCIE-40GB | 8.497 | 20.70 | 21.28 |

Global batch = 32. First 3 of 12 updates excluded from timing.
Four-GPU cross-node communication is not measured by this job.
