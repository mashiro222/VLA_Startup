# 24 小时训练后的手动评测与追加训练

## 实验配置

Baseline 和 SF 分别完成当前的 **24 小时训练分配**后，使用各自正常退出时保存的最后一份完整 checkpoint，手动选择提交对应评测或续训。没有常驻 CPU 控制器，没有自动检测后提交，没有自动续提，也没有工作流总 GPU 上限。

| 任务 | 配置 | 目标 |
|---|---|---|
| baseline 评测 | 1 张 A100 | 900 episodes，SR、CC、失败原因及 cost 归因 |
| SF 评测 | 1 张 A100 | 相同 900 个任务/初态组合及指标 |
| baseline 续训 | 4 张 A100，每卡 micro-batch 2，累积 4 | 从自己的最终 checkpoint 追加 15,000 optimizer updates |
| SF 续训 | 4 张 A100，每卡 micro-batch 2，累积 4 | 从自己的最终 checkpoint 追加 15,000 optimizer updates |

全局 batch 为 **4 × 2 × 4 = 32**。四卡训练采用两个节点、每节点两张 A100，不限制 40/80 GB。模型、数据、归一化、优化器、学习率曲线和各组训练目标保持不变；迁移保留训练参数、AdamW 状态和累计步数，四个 rank 的随机流按固定规则重建。

两组独立提交，评测与续训之间没有作业依赖。四个任务可以重叠，合计申请 10 张 GPU，实际启动由 Slurm 调度；每个四卡 DDP 作业需要自己的四张卡同时分配。

续训目标是 `本次24小时最终步数 + 15000`。例如最后为 4,875 步，则目标为 19,875；最后为 3,650，则目标为 18,650。示例不是实测结果。不补齐到 5,000，不回退到较早的定期 checkpoint，不固定累计 20,000 步。

## 手动提交命令

先查看原始作业：baseline **14143587**、SF **14143602**、现有 batch 2 校准 **14171051**。后续任务尚未提交。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
squeue -u gliu386 -o '%.18i %.36j %.10T %.10M %.24b %R'
sacct -j 14143587,14143602,14171051 --format=JobID,State,ExitCode,Elapsed
```

哪组原始训练正常结束，就可以手动提交那组的任务，无需等待另一组：

```bash
# baseline 正常完成后：
bash scripts/sf_after24h.sh baseline eval --submit
bash scripts/sf_after24h.sh baseline continue --submit

# SF 正常完成后：
bash scripts/sf_after24h.sh sf eval --submit
bash scripts/sf_after24h.sh sf continue --submit
```

去掉 `--submit` 只打印对应的一条 sbatch 命令，不提交作业。四个命令分别只提交指定任务，不会顺带提交另一组或另一阶段。计费账户固定为 `gts-jli741-paid`。

手动提交时检查该组初始 Slurm 作业已正常完成，并从最终 `status.json` 读取准确步数，检查对应的是最新完整 checkpoint。续训还检查已运行的 batch 2 校准结果；没有新增训练 smoke。校准未通过时不能把 micro-batch 2 当作可用配置，但不阻止单卡评测。

同一组的评测与续训绑定同一份保留的 checkpoint。提交记录用于防止重复提交；若同一任务已经排队或运行，重复执行命令会返回现有作业号。提交超时或状态不明时先检查队列，不盲目重复申请。

## 运行时限与手动续跑

每份评测和续训分配最长 **24 小时**，这是作业时限，不是预计运行时长。评测完成 900 条或续训达到目标即可提前结束。

续训接近时限时保存完整状态并退出，**不会自行提交下一段**。如果追加 15,000 步尚未完成，再次手动执行同一组的 `continue --submit` 命令即可从新 checkpoint 恢复；目标仍是最初选定 checkpoint 的步数加 15,000，不会每次再增加 15,000。评测若正常保存了部分 episode，同一手动命令可在原计划上继续；已完成的 episode 不重复计算。异常或无进展退出应先检查日志。

## 评测范围与结果

**2 个 suite × 3 个难度等级 × 每级 5 个任务 × 30 个不同初态 = 每组 900 episodes**，两组共 1,800 条。每个 suite 450 条，每个 suite 的每个等级 150 条。每份任务文件有 50 个初态，固定取索引 0–29，两组任务、初态与 episode seed 一致。

保持每个 episode 最长 300 步、10 次 flow 积分、每次执行前 5 个动作的推理配置。每组单独输出 SR、平均 CC、分任务/难度表图、失败原因和逐谓词 cost 归因。两组评测均完成后，最后完成的评测进程生成联合表图；这只汇总结果，不提交新作业。

两组使用相同的初始训练分配时限，但实际更新步数可能不同。结果表明确记录 `baseline_step`、`sf_step`，不能标为等 optimizer update 数的对照。一次训练 seed 和早期 checkpoint 的结果不代表收敛性能。

## 文件与输出

| 文件 | 用途 |
|---|---|
| `scripts/sf_after24h.sh` | 四个手动提交入口，默认预览 |
| `slurm/sf_after24h_eval.sbatch` | 单卡评测作业，共用于 baseline / SF |
| `slurm/sf_after24h_continue.sbatch` | 四卡续训作业，共用于 baseline / SF |
| `src/sf_followup/after24h.py` | 手动提交时的最终权重选择、追加步数、重复提交检查 |
| `src/sf_followup/continuation.py` | 两卡到四卡的状态迁移 |
| `src/sf_followup/evaluation.py` | 训练权重推理、SR/CC 与诊断表图 |
| `scripts/sf_multinode_worker.sh` | 两节点同步 DDP 启动 |
| `tests/test_sf_after24h.py` | 最终 checkpoint、手动提交及预算回归测试 |

本轮配置位于 `runs/arena-after24h-20261019-001/campaign.json`。各组首次手动提交时生成 `baseline-selection.json`、`sf-selection.json`，记录实际起始步数、目标步数及计划哈希；`pinned-final/` 保留初始训练的最终 checkpoint，`submission/` 保存实际提交回执。这些记录尚未生成时，表示尚未手动提交该组后续任务。

- 评测逐条日志与视频：`runs/arena-after24h-20261019-001-{baseline,sf}-eval/`。
- 每组评测表图：同名 `reports/` 目录。
- 联合表图：`reports/arena-after24h-20261019-001-comparison/`。
- 续训：`runs/arena-after24h-20261019-001-{baseline,sf}-plus15k/`，该目录下对应 arm 保存 loss、运行状态与 checkpoint。

```bash
# 首次手动提交后查看绑定的 checkpoint、目标与作业号：
cat runs/arena-after24h-20261019-001/*-selection.json
cat runs/arena-after24h-20261019-001/submission/*.json
```

## 协议修订

此前固定 5k / 累计 20k / 总并发 8 卡的自动方案已停用。CPU 控制器 **14171052 / 14171526** 已停止，其期间没有提交评测或四卡续训，旧启动入口已禁用。原两卡训练与 batch 2 校准保持原作业和排队位置。

修订采用实际 24 小时结束的权重，保留手动选择提交时机，去除固定步数门槛、两组互等、总卡数限制及自动提交。完整实验时间线见[实验记录](../results-overview.md)。
