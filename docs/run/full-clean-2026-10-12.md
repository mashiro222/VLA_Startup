# 2026-10-12：full-clean 的提交与检查

> 阶段性技术归档；实验结果、调整原因与最新记录统一见[实验记录](../results-overview.md)。文中的排队状态仅对应原检查时间。

## 2026-10-15 更新

旧补跑 `13988230_2` 于 08:26:46 开始，08:41:54 以 `FAILED / 1:0` 退出，日志为 `TimeoutError: Policy startup timed out`，仍触发 900 秒模型启动限制，0 个新增 episode。完整 clean 仍缺第 2 片的 500 条，已完成部分 1,446/1,500（96.40%）。这不是 Arena 的 WebSocket 故障。第 2 片暂不继续补跑，结果保留为部分覆盖。下面 10 月 12/13 日的排队信息是历史记录，不能当作当前状态。最新总览见 [results-overview.md](../results-overview.md)。

本地证据：`logs/fullclean-4gpu-001-13988230_2.out` 与 `sacct -j 13988230 --format=JobID,State,ExitCode,Start,End,Elapsed`。

## 2026-10-13 08:23 EDT 历史检查

| 评测 | 完成数 | 成功数 | 当前状态 |
|---|---:|---:|---|
| Full-plus | 10,030 / 10,030 | 8,477（84.52%） | 四片均 COMPLETED，ExitCode 0:0 |
| Full-clean | 1,500 / 2,000 | 1,446（已完成部分的 96.40%） | 0、1、3 片完成；补跑第 2 片仍 PENDING (Priority) |

Plus 最后一片于 **10 月 12 日 23:06:16 EDT** 结束，每片实际用时约 13 小时 35 分至 14 小时 20 分。合并检查通过：10,030 条唯一 episode，无缺失、重复或程序错误记录；报告文件 SHA256 校验通过。[完整汇总](../../reports/fullplus-4gpu-001-full-plus/summary.md)包含 CSV 表、逐条记录及两张 PNG 图。累计成功率图按合并后的任务清单顺序绘制，不是四片同时运行时的墙钟时间曲线。

本次 Plus 的 84.52% 比[社区报告的 84.2%](https://github.com/sylvestf/LIBERO-plus/issues/57)高 0.32 个百分点。这是描述性比较：该社区结果不是官方排行榜，公开协议不完整；本次每个分片也各自从相同初始随机状态开始，不能宣称与单进程连续随机流完全一致。各扰动类别仍有差异，详见汇总表。

Clean 原数组第 2 片 `13974531_2` 在 10 月 12 日 16:25:03 以 `1:0` 退出，直接错误为 `TimeoutError: Policy startup timed out`（900 秒启动上限），尚未完成任何 episode。日志没有给出明确 CUDA/OOM 证据，启动变慢的根因尚未确认。10 月 12 日 18:09:44 单独提交第 2 片补跑，作业号为 **`13988230_2`**，资源为 1 张 H100、8 CPU、64 GB、8 小时，账户 `gts-jli741-paid`。截至 10 月 13 日 08:23 EDT 检查，该补跑尚未启动，启动时间为 `N/A`；后续状态见本文顶部更新。其它三片在 H100 上已正常完成，每片约 58–75 分钟；补跑实际耗时仍需以运行结果为准。

已完成的 Clean 分片也分别备份为 [shard 00](../../reports/fullclean-4gpu-001-full-clean-shard-00/summary.md)、[shard 01](../../reports/fullclean-4gpu-001-full-clean-shard-01/summary.md)、[shard 03](../../reports/fullclean-4gpu-001-full-clean-shard-03/summary.md)。它们只是部分结果，等剩余 500 条完成后才能合并和比较 full-clean 成绩。

## 10 月 12 日提交与调度历史

`fullplus-4gpu-001` 的 Slurm 数组编号为 `13967217`：4 个 A100 worker，评测 10,030 个 Plus episode。`fullclean-4gpu-001` 于 2026-10-12 08:26 EDT 提交为数组 **`13974531`**：4 个 H100 worker，每个 1 张 H100、8 CPU、64 GB，评测原版 LIBERO 的 2,000 个 episode，每片 500 个。提交后核对四片均为 `PENDING (Priority)`、`Dependency=(null)`。`ArrayTaskThrottle=4` 只是同时运行的上限，各分片可以分别启动，不要求同时拿到四张卡。两批使用不同的 `runs/`、`logs/` 和 `reports/` 目录，不会覆盖结果。两批作业的总并发上限为 8 张 GPU；full-clean **不设置**等待 full-plus 的作业依赖。两批都使用付费账户 `gts-jli741-paid`、`inferno` QoS。

2026-10-12 08:13 EDT 的 `sbatch --test-only` 比较记录在 [clean-queue-probe-2026-10-12.json](../../reports/preparation/clean-queue-probe-2026-10-12.json)。当时新申请的 H100 1/2/4 worker 均显示首个候选约 10 月 15 日 06:09；A100 则约 10 月 16 日 17:40。**这不是预约时间，也不代表四个成员会同时启动**。已有 full-plus 的候选时间从 10 月 12 日变到 10 月 13 日，说明估计会随调度重算。4 个 H100 是当时被测试的有效选项中较早的候选，选 2 个 H100 并没有显示更早排到，反而会让每卡承担更多 episode。

同日约 08:35 EDT 的实时 `sinfo -N` 在 `gpu-h100` 分区列出 4 台节点、31 张 H100（7+8+8+8），其中 7 卡节点状态为 `inval`；其它 3 台、24 张处于可调度节点，但已有作业占用。新作业试算的 2×H100 与 4×H100 首个候选同为约 10 月 15 日 07:12，4×A100 则约 10 月 16 日 12:30。现有作业的 `squeue --start` 一度显示 `N/A`，它表示没有得到当前估计，**不表示永远排不到**；[Slurm 官方 FAQ](https://slurm.schedmd.com/faq.html)说明待排作业并不一定都有 backfill 启动预测。

08:48 EDT 进一步核对 `scontrol show node`：H100 的配置总数其实是 32（4×8）；异常节点只上报 7 张，因 `gres/gpu count reported lower than configured (7 < 8)` 被标记为 `DRAIN+INVALID_REG`，所以前述 31 是注册信息，不能当作物理配置总数。其余 3 台上的 24 张当时全部已分配。full-clean 作业还排除了节点 `atl1-1-03-010-1-0`，因此这 24 张也不能都视为该作业的候选资源；其它两个正常候选节点上的 16 张当时均占用。

同次读取的 PACE 调度配置包含 `bf_min_age_reserve=21600`：[Slurm 文档](https://slurm.schedmd.com/slurm.conf.html#OPT_bf_min_age_reserve)说明，待排且可运行不足 6 小时的作业不会获得未来资源预留。这很可能解释了 08:26 提交的 full-clean 为什么暂时没有启动预测；约 14:26 后才满足此年龄条件，但不保证那时能生成预测或开始执行。**该规则不禁止作业在有可用空位时提前启动**，也与模型能否在 H100 上推理无关，后者需等实际运行验证。

full-plus 的数组成员 0、1 已于当天 08:45:50 EDT 开始运行，实际 Slurm Job ID 分别为 `13974684`、`13974685`（仍可用 `13967217_0`、`13967217_1` 查询）。两者各自获得一张不同 UUID 的 A100 80GB，日志通过 smoke gate；08:48 检查时仍在模型启动阶段，尚无完成的 episode。成员 2、3 继续排队。这验证了数组可以先启动部分成员，无需等待四张卡一起到齐。

原版总量少于 Plus，但不能简单将时间按 2,000/10,030 换算：不同任务的 horizon、场景构建时间及成功/失败步数不同。A100 smoke 的 Slurm 记录显示峰值 GPU 显存约 9.2–9.4 GiB；这说明模型不需要 A100 的全部显存，不是 full 全量或其它 GPU 的兼容保证。L40S 有 48 GB 且支持 BF16，理论上可试，但本环境没有在 L40S 上验证 JAX 推理与 EGL 渲染；当天 L40S 的队列试算比 H100 更晚。提交时 H100 尚未验证，随后三个 full-clean 分片已正常完成。如果作业发生启动、CUDA 或 EGL 错误，先看日志并停止重复提交。单个 H100 worker 正常结束前最多运行 8 小时；这个时限不是预测用时。

## 作业查询与提交记录

在 PACE 终端查询现有作业。若 `squeue` 查询失败，应先核实作业状态，避免重复提交：

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
squeue -r -u gliu386 -o '%.18i %.12P %.32j %.12T %.12M %R'
python3 scripts/parallel.py status fullplus-4gpu-001
python3 scripts/parallel.py status fullclean-4gpu-001
```

检查准备好的 clean 计划和提交命令；这两条不申请 GPU：

```bash
python3 scripts/parallel.py check fullclean-4gpu-001 --require-smoke
VLA_GPU_TYPE=h100 TIME_LIMIT=08:00:00 bash scripts/submit_parallel.sh fullclean-4gpu-001
```

**以下为历史提交命令。第 2 片补跑已于 10 月 15 日失败，当前没有继续补跑计划。** 该入口用于未完成分片的续跑，会检查同名活动数组，并只请求未完成的分片；提交通信超时后需先核实队列，避免重复提交。

```bash
VLA_GPU_TYPE=h100 TIME_LIMIT=08:00:00 bash scripts/submit_parallel.sh fullclean-4gpu-001 --submit
```

若需限制两批作业的总并发为 4 张卡，可设置 `VLA_AFTERANY_JOB_ID=13967217`，让 clean 在 Plus 整个数组结束后才变为可运行；此次并行配置未设置该依赖。[Slurm 数组依赖说明](https://slurm.schedmd.com/job_array.html#job-dependencies)确认 `afterany` 等待该数组全部成员结束，不要求它们全部成功。

可用以下命令看作业和日志。`13974531` 是初次 full-clean 数组 ID，`13988230` 是第 2 片补跑 ID；`--test-only` 输出的数字不是正式作业 ID。

```bash
squeue -r -u gliu386 -o '%.18i %.32j %.12T %.12M %R'
sacct -j 13974531,13988230 --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
ls -lt logs/fullclean-4gpu-001-*.out
python3 scripts/parallel.py status fullclean-4gpu-001
```

若作业失败或超时，先查看每个数组成员的 `logs/fullclean-4gpu-001-<job>_<index>.out` 和对应 `runs/fullclean-4gpu-001-full-clean-shard-0<index>/errors.jsonl`，不要把未完成的结果当 full 成绩。等旧数组所有成员停止、找出错误原因后，用相同 batch ID 续交；提交入口会跳过已完成分片。不要修改现有批次冻结的模型代码、配置、任务清单或 worker 数；需要更改时创建新 batch ID。

## 报告与备份

四个分片完成后，在登录节点合并（不申请 GPU）：

```bash
python3 scripts/parallel.py merge fullclean-4gpu-001
python3 scripts/parallel.py merge fullplus-4gpu-001
```

第二条只在 full-plus 四片全部完成时运行。报告分别在 `reports/fullclean-4gpu-001-full-clean/` 和 `reports/fullplus-4gpu-001-full-plus/`，其中 `summary.csv` 是汇总表，`episodes.csv` 是逐条记录，`cumulative_success.png` 是累计成功率折线图，`success_by_group.png` 是分组柱状图；`summary.md` 标出完成覆盖率。合并会拒绝缺失或重复 episode。审查报告后，把小型结果上传 GitHub：

```bash
bash scripts/publish_results.sh fullclean-4gpu-001-full-clean
bash scripts/publish_results.sh fullplus-4gpu-001-full-plus
```

GitHub 仅备份代码、文档和小型报告；模型 checkpoint、LIBERO 场景资产、视频、环境、原始作业日志都被 `.gitignore` 排除。模型运行路径和参数看 [代码地图](../code-map.md)，benchmark 组成和分数比较限制看 [实验说明](../experiment.md)，完整四卡设计与作业查询看 [overnight.md](overnight.md)。
