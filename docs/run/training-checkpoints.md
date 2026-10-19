# 训练 checkpoint 与资源调整

实验结果与调整决定统一见[实验记录](../results-overview.md)。本页说明当前 `arena-sf-train-full-002` 的保存机制、验证方法和扩卡限制。2026-10-18 的决定为：baseline 与 SF 保留现有两卡作业，各运行本次 24 小时分配。

## 24 小时结束时保存什么

当前脚本与训练循环已经实现，无需修改运行中的源码：

1. 每 **1,000 optimizer updates** 保存一次，只由 rank 0 写入。
2. Slurm batch 启动后 **23 小时 50 分钟**触发内部 deadline；完成当前更新后，各 rank 汇总停止标志并保存，即使未到 1,000 步边界也保存。
3. Slurm 还配置了硬时限前 5 分钟的 `USR1` 通知。batch 将信号转发给训练进程，作为另一条退出保存路径。
4. 保存包含所有可训练参数（含 SF projector）、AdamW 状态、已完成步数、配置身份和各 rank 的 RNG。冻结 base 不重复写入，恢复仍依赖同一份 base 权重和归一化统计。
5. 先写 `.tmp`，`fsync` 后原子改名为 `step-XXXXXXXX.pt`；只有完整写入后才清理旧文件，保留最近两份完整 checkpoint。最后写 `status.json`。

对应代码：[Slurm 脚本](../../slurm/sf_train.sbatch)、[训练循环](../../src/sf_training/train.py)、[checkpoint 保存恢复](../../src/sf_training/checkpoint.py)。

Baseline `14143587` 于 10/18 20:07:39 EDT 启动，硬时限约为 **10/19 20:07:40**；内部退出保存预计在 **19:57 左右**触发。SF `14143602` 尚在排队，其 24 小时从实际启动开始计算。内部 deadline 包含初始化时间，实际训练不足 24 小时。

这是尽量在正常时限前安全保存的机制，不保证断电、文件系统故障或强制 SIGKILL 后还能保存正在计算的更新；这些情况恢复到最近完整文件。当前正常运行无需发送停止信号。

## 检查保存结果

所有命令从仓库根目录执行，均不提交新作业：

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
sacct -j 14143587,14143602 --format=JobID%20,State,Elapsed,ExitCode
ls -lh runs/arena-sf-train-full-002/baseline/step-*.pt
ls -lh runs/arena-sf-train-full-002/sf/step-*.pt
cat runs/arena-sf-train-full-002/baseline/status.json
cat runs/arena-sf-train-full-002/sf/status.json
```

在首次 1,000 步或退出保存前，`step-*.pt` 不存在是正常的。`status.json` 在本次训练进程退出时生成；运行中通过 `metrics.jsonl` 查看进度。如果 `stop_reason=wall_time_budget` 且 `completed=false`，说明本次分配正常保存退出，但还没完成目标 60,000 步。`COMPLETED` 是 Slurm 进程退出状态，不等于训练预算完成。

精确续训应使用完整的 `.pt`，不能使用 `.tmp`。同配置、同 GPU 数续训命令在[运行手册](training-operations.md)，应先确认原作业结束，避免同时写同一 run。当前不会自动提交下一个计费作业。

10/18 的空间检查显示当前使用约 179.3 GiB，个人限额 15 TiB，文件数 664,520 / 1,000,000，未发现 checkpoint 空间不足；[存储快照](../../reports/training-progress-2026-10-18/storage.json)。

文件保存在 scratch，不上传 GitHub。长期保留还需将 checkpoint、对应 `plan.json`、归一化统计及源码版本备份到组内持久存储；仅备份本仓库不能恢复未备份的训练权重。

## 保存路径验证

[CPU 回归测试](../../tests/sf_training/test_shutdown.py)执行真实训练循环、AdamW、checkpoint 写入与恢复，使用小模型替代 GPU 模型：分别覆盖 deadline、停止信号，以及写入失败时已有完整文件不被覆盖。它验证退出保存的控制流程，不等同于已验证真实多 GPU 文件系统保存。既有测试另覆盖同配置 optimizer/RNG 精确恢复，以及 1/2/4 rank 的全局 batch 划分。

[测试记录](../../reports/training-progress-2026-10-18/checkpoint-tests.xml)。这 8 项 CPU 检查已通过。训练源码、配置、数据保持原样；检查时发现一个安装的 PaliGemma 文件含 85 个额外空字节，已从固定上游恢复到计划要求的 SHA256，未改变版本或配方。详见[修复记录](../../reports/training-progress-2026-10-18/environment-repair.json)。

## 两卡到四卡的边界

后续已新增隔离的显式迁移入口，按两组相同 5k checkpoint 迁移到两节点四卡、每卡 batch 2；原入口的严格检查保留。实施方式与尚待 GPU 验证的边界见[5k / 20k 流程](microbatch-and-20k.md)。以下描述原始训练入口本身的限制。

当前 DDP 每张卡持有完整模型；两卡训练的运行中进程不能热加两张 GPU。四卡是新分配下恢复训练，不是把两个独立训练任务合并。

当前恢复严格校验 plan hash、模型组、base 身份和 world size。因此**现有实现不能直接把两卡 checkpoint 放到四卡计划里恢复**。后续扩卡需要完成以下工作后才能提交：

- 显式校验源/目标模型、数据、归一化、LoRA/SF 配方及全局 batch 一致，保留原 optimizer 与更新步数。
- GPU 数从 2 改为 4 时，global batch 仍为 32，micro-batch 仍为 1，累积步数从 16 改为 8；不重置学习率进度。
- 为新增 rank 定义可追溯的随机流，记录迁移来源；DDP 求和与随机数消费改变后不保证逐位相同。
- 新配置使用隔离源码快照和新 run，不覆盖仍在运行的冻结代码；验证恢复与一次真实更新。

本次没有实现或提交四卡迁移，也未更改两个现有作业。公平比较最终仍使用相同训练步数、数据协议和评测计划，不将相同墙钟时长当作相同训练预算。

## 2026-10-18 GPU 与排队快照

账户为 `gts-jli741-paid`，QOS 为 `inferno`，候选请求均为 24 小时。下表来自 Slurm 查询和 `sbatch --test-only`，后者不分配 GPU、不创建真实运行作业。

| 方案 | 21:36—21:40 EDT 查询结果 | 解释 |
|---|---|---|
| 现有 SF：同节点 2 A100 | 候选 10/19 08:40 | 保留现有作业与排队资格 |
| 同节点 4 A100 | `Requested node configuration is not available` | 本次申请被拒；原因尚未完全定位，不能解释成整个 A100 分区不可用 |
| 同节点 4 H100 | 候选 10/27 15:04 | 资源试算可接受；并非确定开机时间，也不是训练验证 |
| 2 节点 × 2 A100 | 候选 10/30 22:16 | 还需要新增多节点启动；当前 trainer 的正式入口只支持单节点 |

当时 13 台 A100 节点中 12 台各 2 卡，仅一台列出 8 卡；4 卡单节点请求可用节点范围很小。H100 列出的节点卡数更多，但节点占用和队列影响实际分配。[PACE 官方节点介绍](https://pace.gatech.edu/2023/03/07/new-compute-nodes-on-the-phoenix-cluster/)也说明了两卡 A100 节点的申请方式；本次实际资源状态以 Slurm 快照为准。

21:50 再查时，两卡 SF 的候选已提前到 10/19 06:40，说明这些日期会变化；[原始结构化快照](../../reports/training-progress-2026-10-18/scheduling.json)保留两次检查。

**选择依据：** 获得最终结果的时间包含排队与训练两部分。四 H100 的计算潜力值得评测，但本项目没有其训练吞吐实测，不能承诺倍速；当前长队列抵消了换卡的吸引力。因此先保留两卡运行，得到 24 小时 checkpoint 和更稳定的吞吐数据，再重新比较资源。

当前 baseline 约 13.5–14 秒/update，60k 步约 9.4–9.7 天；四 A100 理想两倍加速约 4.7–4.9 天，仅是理想参考，不是实测预测。SF 的 VGGT 教师增加计算，需单独测量。将每卡 micro-batch 从 1 增大也可能减少累积开销，但需分别验证 baseline/SF 的显存和吞吐，本次不修改此参数。
