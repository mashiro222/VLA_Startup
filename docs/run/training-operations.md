# Arena LoRA / SF：运行与恢复

实验结果、异常与调整决定统一见[实验记录](../results-overview.md)。本页保存环境重建、提交和监控命令；配方与代码导航见[训练说明](../arena-sf-training.md)。

## 环境与准备

从仓库根目录执行；所有大文件保存在 scratch 下 `.runtime/training/`，不上传 GitHub。已有 Arena 评测环境保持隔离。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
bash scripts/setup_sf.sh
.runtime/server/bin/python scripts/sf_assets.py
source scripts/sf_common.sh
"$SF_ENV/bin/python" -m sf_training.pack
"$SF_ENV/bin/python" -m sf_training.normalization --repair
bash scripts/test_sf.sh
```

依赖由 uv 维护，锁文件为 `requirements/sf_train.lock`。只在 `.runtime/sf_train` 中安装 SF 要求的 transformers 4.53.2 overlay。下载清单固定 HF revision、GCS generation、文件大小和 SHA256/MD5/CRC32C；所有文件校验后才生成 `assets_verified.json`。

Parquet 原始图像采用有界并行读取，数值与索引直接写入预分配磁盘数组，CPU 预处理不加载 PyTorch。按原 episode 顺序打包成随机读取文件，避免每取一帧都重新解压整个 episode。实际样本的打包前后等价性单独核对；数值、图像、语言 token 和轨迹末尾动作片段均须一致。归一化只使用训练示范，并按实际 clamped action chunk 统计。`pi05_base` 参数转换在首次训练验证的已分配节点上执行，不在登录节点运行大模型。转换检查全部参数名和 shape；唯一显式补零的是从不参与动作推理的 expert LM vocabulary head。

## 训练验证与正式提交

训练 smoke 不是性能 benchmark。它验证真实模型加载、DDP、两组 loss/梯度、冻结教师、checkpoint 保存、进程重启后的 optimizer/projector/RNG 恢复，以及两组最终的 10 步动作推理。两组各执行 16 次更新，退出进程，然后恢复至第 32 次；global batch 与正式训练均为 32。逐 batch 记录 `target_abs_max` 和 `rare_rotation_samples`；任何退化区间、非有限值、归一化输入绝对值超过 100 或 flow MSE 超过 1,000 都触发错误，不会只检查 NaN/Inf。阈值是发现异常的宽松保护线，不是收敛或安全性判据。

```bash
# 重建时使用新的 run ID；当前 run 已存在时不要重复 prepare/submit
bash scripts/sf.sh prepare full arena-sf-train-full-002 --gpus 2

# 本次方案：显式跳过 GPU smoke，直接提交正式两组
bash scripts/sf.sh submit arena-sf-train-full-002 --arm baseline --skip-smoke --submit
bash scripts/sf.sh submit arena-sf-train-full-002 --arm sf --skip-smoke --submit
```

去掉 `--submit` 只打印命令。账户固定为 `gts-jli741-paid`。`--skip-smoke` 只适用于 full，不能同时使用 `--smoke-run` 或 `--after-smoke`；回执明确记录 `gpu_smoke=explicitly_skipped`。省略此开关时，默认仍要求匹配的训练 smoke。两个正式回执分别为 `runs/arena-sf-train-full-002/submission/baseline.json` 和 `sf.json`。如果提交响应不确定，流程停止且不自动重试；先核对回执和 Slurm，避免重复任务。

可选增强 smoke 的重建方式：先准备新的 smoke/full 两个 run（相同 GPU 数），再执行 `bash scripts/sf.sh submit SMOKE_RUN --arm smoke --then-full-run FULL_RUN --submit`。此路径仅在验证通过后创建两组任务，且等待 smoke 分配释放后启动；不是本次正式重启所用方式。

如果同名作业已提交，不要再次执行上述提交命令。回执在 `runs/<run_id>/submission/`；网络超时也可能已经成功提交，必须先查询队列。正式训练由于时限停止且旧作业已退出时可续训：

```bash
bash scripts/sf.sh submit arena-sf-train-full-002 --arm baseline \
  --skip-smoke --resume --submit
bash scripts/sf.sh submit arena-sf-train-full-002 --arm sf \
  --skip-smoke --resume --submit
```

恢复要求配置、代码、数据、环境和 GPU 数一致。不要在 pending/running 作业期间修改训练代码。需改参数时新建 run，重新验证。


## 2026-10-17 重启归档

以下为提交时记录，排队状态不代表当前状态。

### 运行状态与归一化修复（2026-10-17）

旧训练验证 `14127618` 完成了三步训练、恢复与推理，但未覆盖稀疏旋转动作。正式 baseline `14127636` 与 SF `14127637` 在约 44% 的更新中出现极大有限 loss：动作第 4、5 维的 q01/q99 同为 0，少量非零旋转被放大到百万量级。这两份作业已取消，其 checkpoint 不用于修复后的训练或性能比较。[异常诊断](../../reports/training-audit-2026-10-17/README.md)保留原始证据。

修复后的正式计划为 `arena-sf-train-full-002`，两组各申请同一节点 **2 张 A100、16 CPU、96 GB 主存**，账户 **`gts-jli741-paid`**。两组正式训练仍分别从官方 `pi05_base` 开始，不加载旧 run 的 checkpoint。

本次正式重启显式使用 `--skip-smoke`，完成 CPU 回归、全数据归一化审计和稀疏 batch 回放后直接提交。**修复版 GPU 训练尚未验证，不能将跳过 smoke 记作验证通过。** 两组无 smoke 依赖；运行时仍检查冻结代码/数据身份、归一化输入范围及 loss/梯度的有限性，并在 flow MSE 超过保护阈值时退出。

可选的增强 smoke 已实现：正式 global batch 32，每组 32 次 optimizer update；第 16 步保存并重启，再恢复至第 32 步。只有稀疏旋转覆盖、全部数值检查和两组推理通过后才生成标记。可通过 `--then-full-run` 在验证成功后自动提交两组，当前重启不使用此路径。

| 本次任务 | Job ID | GPU 资源 | GPU smoke 依赖 |
|---|---|---|---|
| LoRA baseline | `14143587` | 同节点 2 × A100 | 无；显式跳过 |
| LoRA + SF | `14143602` | 同节点 2 × A100 | 无；显式跳过 |
| pi05_libero 1,500 条评测 | `14143493` | 最多 4 个单 A100 worker | 已完成匹配的评测 smoke |

```bash
squeue -j 14143587,14143602,14143493 -r -o '%.20i %.10T %.40j %.10M %R'
sacct -j 14143587,14143602,14143493 --format=JobID%20,JobName%40,State,Elapsed,ExitCode
# 日志在作业启动后创建；-F 可等待文件出现。
tail -F logs/arena-sf-train-full-002-baseline-14143587.out
tail -F logs/arena-sf-train-full-002-sf-14143602.out
```

2026-10-17 22:25 EDT 状态：baseline `14143587` 与 SF `14143602` 均为 **PENDING / Priority**，尚无训练日志或 loss。1,500 条 `pi05_libero` 评测已完成，结果见[独立评测报告](arena-results-2026-10-17.md)；这些 SR/CC 不属于待运行的 baseline/SF。

运行和提交回执保存在 `runs/<run_id>/submission/`，本次归档见 [修复与提交记录](../../reports/restart-2026-10-17/README.md)。旧的三步验证不再作为当前训练的有效依据。训练 smoke 与 `pi05_libero` 的 Arena 评测 smoke 是独立流程。


## 汇总、监控与导出

```bash
source scripts/sf_common.sh
"$SF_ENV/bin/python" -m sf_training.report arena-sf-train-full-002
squeue -u gliu386 -r -o '%.20i %.10T %.36j %.12M %R'
sacct -u gliu386 -S today --format=JobID%20,JobName%36,State,Elapsed,ExitCode
```


导出模型应在有足够 CPU 内存的计算节点内执行，不在登录节点加载完整模型：

```bash
"$SF_ENV/bin/python" -m sf_training.export \
  --plan runs/arena-sf-train-full-002/plan.json --arm sf \
  --checkpoint runs/arena-sf-train-full-002/sf/step-00060000.pt \
  --output .runtime/training/export/arena-sf-step60000
```
