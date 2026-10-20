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

续训目标是 `本次24小时最终步数 + 15000`。例如最后为 6,200 步，则评测使用 6,200 步权重，续训目标为 21,200；最后为 4,875，则目标为 19,875。示例不是实测结果。不补齐或截断到 5,000，不回退到较早的定期 checkpoint，不固定累计 20,000 步。

训练超过 5,000 步后仍继续运行，直到本次分配的内部 deadline（启动后约 23 小时 50 分钟）触发保存退出。即使最终步数不是 1,000 的倍数，也保存该步完整状态，随后写入 `status.json`。评测入口要求原作业正常结束，并核对最终状态的步数与最新完整 `.pt` 文件一致；缺失时会报错，不会静默改用 5,000 步或更早的文件。测试覆盖最终 4,875 和 6,200 步两种情况。

## 手动提交命令

原始训练作业为 baseline **14143587**、SF **14143602**。Batch 2 补测 **14179857** 已于 10/20 03:15 EDT 通过，见[batch 2 补测](#batch-2-补测)。Baseline 评测 **14199187** 和四卡续训 **14217601** 已分别于 10/19、10/20 提交，均使用初始训练最终 **5,952 步 checkpoint**。SF 保持当前两卡训练，其评测与续训尚未提交。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
squeue -u gliu386 -o '%.18i %.36j %.10T %.10M %.24b %R'
sacct -j 14143587,14143602,14179857,14199187,14217601 --format=JobID,State,ExitCode,Elapsed
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

## Baseline 5,952 步评测

训练 **14143587** 于 10/19 20:00:26 EDT 正常结束（`COMPLETED`，退出码 `0:0`），最终保存 `step-00005952.pt`。权重在 `runs/arena-after24h-20261019-001/pinned-final/baseline/` 通过硬链接保留，评测绑定该文件。

评测作业 **14199187**：1 张 A100、8 CPUs、96 GB 主机内存、24 小时时限，账户 `gts-jli741-paid`。Static / Dynamic obstacles 各 450 episodes；初始状态为 `PENDING (Priority)`。900 episodes 是计划数量，提交时尚无评测结果。

```bash
squeue -j 14199187 -o '%.18i %.10T %.12M %R'
sacct -j 14199187 --format=JobID,State,ExitCode,Elapsed
# 作业启动并创建日志后：
tail -f logs/sf-24h-eval-baseline-14199187.out
```

逐 episode 输出位于 `runs/arena-after24h-20261019-001-baseline-eval/`；表图完成后保存到同名 `reports/` 目录。[提交回执](../../reports/arena-after24h-20261019-001-baseline-eval/submission.json)和[checkpoint / 任务覆盖检查](../../reports/arena-after24h-20261019-001-baseline-eval/preparation.json)已归档。

10/19 本次提交仅包含 baseline 评测。10/20 batch 2 验证通过后，baseline 四卡续训另行提交，见下一节；该评测的 checkpoint、任务集合与队列保持不变。

## Baseline 四卡追加训练

作业 **14217601** 于 **10/20 10:13 EDT** 提交，从保留的 **5,952** 步 checkpoint 追加 **15,000** 次 optimizer updates，累计目标 **20,952**。配置为 **2 个节点 × 每节点 2 张 A100 × 每卡 batch 2 × 梯度累积 4 = global batch 32**；每节点 16 CPUs、96 GB 主机内存，账户 `gts-jli741-paid`，单次时限 24 小时。

迁移保存权重、AdamW 状态和累计步数，沿用原学习率曲线，不重新 warmup。四 rank 的随机流重新确定，因此不保证与两卡逐位一致。批处理只启动本次分配；若 24 小时内未到 20,952 步，保存后退出，后续需手动提交，目标不改变。已完成两卡 batch 2 检查，跨节点通信与实际四卡速度仍需本作业验证。

```bash
squeue -j 14217601 -o '%.18i %.10T %.12M %R'
sacct -j 14217601 --format=JobID,State,ExitCode,Elapsed
# 作业启动并创建日志后：
tail -f logs/sf-24h-continue-baseline-14217601.out
```

运行输出和 checkpoint：`runs/arena-after24h-20261019-001-baseline-plus15k/baseline/`。[提交回执](../../reports/arena-after24h-20261019-001-baseline-plus15k/submission.json) · [配置检查](../../reports/arena-after24h-20261019-001-baseline-plus15k/preparation.json)。SF 初始训练和 baseline 评测保持原作业，未设置二者与本次续训之间的依赖。

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

此前固定 5k / 累计 20k / 总并发 8 卡的自动方案已停用。CPU 控制器 **14171052 / 14171526** 已停止，其期间没有提交评测或四卡续训，旧启动入口已禁用。原两卡训练保持原作业和排队位置。

修订采用实际 24 小时结束的权重，保留手动选择提交时机，去除固定步数门槛、两组互等、总卡数限制及自动提交。完整实验时间线见[实验记录](../results-overview.md)。

## Batch 2 补测

首次校准作业 **14171051** 于 10/19 00:51 EDT 达到一小时时限，两组 batch 1 各完成 12 次更新并通过有限动作推理检查；batch 2 没有训练记录，不能视为通过，也没有 OOM 证据。

补测作业 **14179857** 于 10/19 10:13 EDT 提交，初始状态为 `PENDING (Priority)`。仅执行 batch 2 的 baseline 和 SF，各 12 次更新，使用 **2 张 A100 40GB × micro-batch 2 × 累积 8 = global batch 32**。作业时限改为 **4 小时**，预留十分钟保存和汇总，完成即退出。四卡续训的累积次数仍为 4，此补测不测量四卡跨节点通信。原校准计划保留以匹配已有结果；本次实际时限由新的 Slurm 提交参数和批处理脚本设置为四小时。

入口是 `slurm/microbatch2_retry.sbatch` 和 `scripts/calibrate_mb2.py`。已完成的 batch 1 记录只读复用；若 batch 2 某组已有完整结果，也跳过该组。遇到不完整输出时停止并要求检查，不覆盖或重复追加日志。成功后将原 batch 1 和新 batch 2 合并为四行校准表，写入 `runs/arena-microbatch-20261018-001/calibration.json`，报告同步到同名 `reports/` 目录。补测不会提交评测或正式续训。

提交回执和 CPU 检查见 [batch 2 补测记录](../../reports/microbatch2-retry-2026-10-19/README.md)。

### 补测结果（10/20）

**14179857** 于 03:15 EDT 以 `COMPLETED / 0:0` 结束，两组 batch 2 各 12 步及推理检查通过。两卡 batch 2 的中位耗时为 baseline **7.52 秒/步**、SF **8.50 秒/步**，相对 batch 1 分别为 **1.68 倍 / 1.85 倍吞吐**；对应峰值已分配显存 **17.13 / 20.70 GiB**。这是短程两卡测试，未验证四卡跨节点吞吐。[校准表与 JSON](../../reports/arena-microbatch-20261018-001/README.md)。

## 10/20 恢复与保留更新

Baseline 续训仍为 `14217601`，新增保留本段全部 checkpoint 的设置，确保 18,000 步不会被后续清理。SF 原作业因同步超时退出，恢复与单独的 900-episode 评测使用[独立恢复入口](sf-recovery-2026-10-20.md)，不走本页要求初始作业正常完成的提交入口。
