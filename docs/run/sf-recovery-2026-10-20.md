# SF 同步超时、checkpoint 恢复与 18k 对照

**后续状态（2026-10-22）：** 评测 `14230444` 已在 PyTorch 导入阶段因本地运行库缺少 `libcusparseLt.so.0` 失败，有效 episode 为 0；续训 `14230443` 仍排队，且复用同一存在遗漏的运行库复制函数。当前未修复。[最新定位与影响范围](progress-2026-10-22.md)。下文提交状态为 10/20 的历史记录。

## 故障与证据

原两卡 SF 作业 `14143602` 于 2026-10-20 13:32:36 EDT 失败，运行 17:32:07。训练记录到第 3,460 次更新，最后落盘的是 `step-00003000.pt`。后续 460 次更新没有保存，不纳入恢复后的累计步数。

失败发生于 NCCL 梯度 `ALLREDUCE`。两 rank 最后完成的 collective 序号都是 238777；rank 1 已排入 238842，rank 0 仍停在 238777。因此 rank 1 等待 rank 0 进入后续 collective。日志没有 rank 0 的 Python 堆栈，也未启用 NCCL flight recorder，不能由此确认具体根因。全部 3,460 条 loss/梯度记录均为有限值，没有 OOM 记录；最后 100 步 flow MSE 中位数 0.01309，对齐损失中位数 0.03788。

第 3,461 次更新对应的 32 个样本已在 CPU 上回放，图像解码与所有数值检查通过，最长单样本读取约 0.29 秒。后续登录节点检查观察到 `cl_sync_io_wait`，进一步确认该节点每用户内存上限为 4 GiB；完整 checkpoint 恢复引发大量内存回收并以 137 退出。这个登录节点问题不能作为原训练作业存储故障的证据。原训练请求 96 GB 主机内存，记录的峰值约 31.4 GiB，日志没有 OOM 记录。

[故障分析](../../reports/sf-recovery-2026-10-20/failure-analysis.json) · [原始报错摘录](../../reports/sf-recovery-2026-10-20/nccl-failure-excerpt.txt) · [故障后下一批数据回放](../../reports/sf-recovery-2026-10-20/update-3461-data-replay.json)

## 恢复验证与实现

`src/sf_recovery/audit.py` 检查 checkpoint 的原计划、arm、步数、基座身份、全部可训练参数名称/形状/类型、权重及 Adam 一二阶矩有限性、Adam 步数和两 rank 随机状态，并通过原 `checkpoint.restore` 函数恢复参数与优化器。原始第一步梯度记录中有 8 个未使用参数，允许它们没有 Adam 状态；其他缺失状态仍拒绝。CPU 检查不包含 GPU forward、EGL 或四卡通信验证。

登录节点的临时 PyTorch 2.7.1 CPU 环境完成了张量检查，但完整恢复超过其 4 GiB 用户内存上限，不能记为恢复通过。首次 24 GB CPU 验证 `14229255` 在共享盘运行库读取等待中达到时限，其节点随后被 PACE 标为 DRAIN，原因是节点健康检查超时；绑定该节点的重试 `14229813` 在排队阶段撤销。这些后续基础设施问题不等于已确认原训练失败的根因。

改用节点本地 Python 3.11.17、PyTorch 2.7.1 CPU、相同源码和经过 SHA256 校验的 checkpoint 副本后，CPU 验证 **14229891** 于 **16:59:22 EDT** 正常完成。检查恢复了 **955 个可训练张量、947 份 Adam 状态、两个 rank 的 CPU 随机状态**；原记录中 8 个未使用参数没有 Adam 状态，符合预期。2,849 个权重/动量张量有限，真实恢复检查耗时约 32.8 秒。CUDA 随机状态和 GPU forward 仍需真实 GPU 运行验证。该作业不训练、不占 GPU，也不提交后续任务。[完整恢复报告](../../reports/sf-recovery-2026-10-20/checkpoint-audit.json)。

53 项 CPU 回归检查通过，涵盖恢复、有限值检查、未使用参数、两卡到四卡状态迁移和 18k 保留规则。

恢复入口增加以下措施：

- 原始 3,000 步 checkpoint 固定到独立目录，避免后续保留策略删除它。
- 两节点分别将 packed 数据、基座权重、教师、恢复权重及 PyTorch/CUDA 运行库复制到本地 `/tmp` 后再训练；已知清单的文件检查 SHA256。约需每节点 50–60 GB 临时磁盘，正常退出时清理本次缓存。
- 使用原训练循环及原模型，只替换输入文件路径，并记录每个 rank 的读取、forward、backward 进入阶段。超时启用 NCCL flight recorder 和 Python 堆栈；异常退出不尝试保存可能不同步的权重。
- 保留原 NCCL collective 超时阈值。两节点启动 rendezvous 最多等待一小时，以覆盖不同节点的输入准备时间。
- SF 的 LayerNorm 开关、对齐层、α、归一化、学习率及 LoRA 配方沿用原 checkpoint 的配置；没有同时更换算法设置。

相关诊断选项见 [PyTorch NCCL 环境变量](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html) 和 [Flight Recorder 说明](https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/)。这些是针对可疑 I/O 路径的缓解和定位措施，尚不能记为原故障已经在真实四卡环境下复现并修复。

## 续训与评测协议

2026-10-20 **17:05 EDT** 已提交下列独立作业，账户均为 `gts-jli741-paid`，时限均为 24 小时。17:06 检查为 `PENDING (Priority)`，尚无新训练或评测结果。

| 实验 | 作业号 | 资源与范围 |
|---|---|---|
| SF 续训 | **14230443** | 两节点各 2 张 A100；3,000 → 18,000 步 |
| SF checkpoint 评测 | **14230444** | 1 张 A100；3,000 步权重，900 episodes |

[提交回执与配置](../../reports/sf-recovery-2026-10-20/submission/)记录实际资源请求；没有新增 baseline 评测、依赖链或总 GPU 并发上限。

SF 从 3,000 步追加 15,000 步，累计目标 **18,000**。申请两个节点、每节点两张 A100：`4 GPUs × micro-batch 2 × accumulation 4 = global batch 32`。保留 Adam 状态和学习率进度，不重新 warmup；两卡到四卡的随机流按既有迁移规则重建，不保证逐位重放。单次最长 24 小时，达到目标或接近时限时保存退出；不自动提交下一段。

新增 SF 3,000 步的 **900 episodes** 单卡评测：Static / Dynamic obstacles 各 450 条，30 个任务配置各 30 个不同初态，与现有 baseline 5,952 步评测的任务、初态、seed、horizon 和推理设置一致。输出 SR、平均 CC、终止原因、逐谓词 cost 归因和图表。

Baseline 3,000 步曾被“保留最近两份”策略清理，现有 5,000 / 5,952 步。此次不新增 baseline 评测，也不把不同训练步数伪装成等步数对照。初步比较为 **baseline 5,952 vs SF 3,000**；后续采用两组 **18,000 vs 18,000** 的 checkpoint 才是等更新次数对照。各阶段硬件/梯度累积不同仍需报告；单个 seed 的结果不等于已确认的普遍提升。

## Baseline 18,000 步保留

Baseline 续训 `14217601` 的作业号、排队位置、4 卡配置、5,952 步起点和 20,952 步最终目标均保留，只增加 checkpoint 保留数量。每 1,000 步保存一次，因此会生成 18,000 步权重。

保留上限设为本段追加步数加 2，即 15,002。这是对 checkpoint 文件数量的上界，不是实际生成数量：正常每 1,000 步保存，本段约生成 15–17 份；即使每次更新都是分配退出点，也不会在本段中删掉 18,000 步。完整 checkpoint 每份约 5.2–5.3 GiB，需要保留相应 scratch 空间，不上传 GitHub。

迁移兼容检查仅放宽“可以增加保留数量”，仍禁止减少保留量或更改学习率、模型和数据设置。现有 baseline 评测 `14199187` 的 checkpoint 和评测内容未变；其冻结清单只刷新了该兼容检查模块的哈希及计划摘要，避免依赖文件更新导致已排队作业拒绝启动。[调整前快照](../../reports/sf-recovery-2026-10-20/baseline-retention-before/) · [修改范围检查](../../reports/sf-recovery-2026-10-20/baseline-retention.json)

## 文件与命令

| 文件 | 作用 |
|---|---|
| `src/sf_recovery/audit.py` | 真实 checkpoint 的 CPU 完整性与恢复检查 |
| `src/sf_recovery/workflow.py` | 明确绑定 SF 3,000 步，准备和手动提交续训/评测 |
| `src/sf_recovery/stage.py` | 节点本地缓存与文件校验 |
| `src/sf_recovery/train.py` | 原训练循环的本地路径和逐 rank 诊断适配 |
| `src/sf_recovery/evaluate.py` | 现有评测器使用本地 policy-server 运行库 |
| `slurm/recovery_continue.sbatch`、`scripts/recovery_worker.sh` | 两节点四卡恢复训练 |
| `slurm/recovery_eval.sbatch` | 单卡 900-episode 评测 |
| `tests/test_sf_recovery.py` | 保留策略、缓存、恢复选择、梯度和提交范围回归检查 |

准备通过后，配置文件位于 `runs/arena-sf-recovery-20261020-001/recovery.json`，提交回执位于同目录 `submission/`。每条提交入口都有重复提交保护；再次运行已提交入口只返回现有 job ID，不创建额外作业。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
source scripts/sf_common.sh
python3 -m sf_recovery.workflow check runs/arena-sf-recovery-20261020-001/recovery.json
# 默认只预览。正式提交需要 --submit；已有回执时不会重复提交。
python3 -m sf_recovery.workflow submit runs/arena-sf-recovery-20261020-001/recovery.json continue
python3 -m sf_recovery.workflow submit runs/arena-sf-recovery-20261020-001/recovery.json eval
# 两组评测完成后可手动生成初步比较；该命令不提交任何作业。
python3 -m sf_recovery.workflow compare runs/arena-sf-recovery-20261020-001/recovery.json
```

训练输出：`runs/arena-sf-recovery-20261020-001-sf-plus15k/sf/`；逐 rank 诊断在其上一级 `diagnostics/`。评测日志/视频在 `runs/arena-sf-recovery-20261020-001-sf-eval900/`，表图在同名 `reports/`，初步比较在 `reports/arena-sf-recovery-20261020-001-comparison/`。启动时间以 Slurm 实时调度为准，不保证在 10 月 22 日前完成。
