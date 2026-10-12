# 2026-10-12：full-clean 的提交与检查

当前 `fullplus-4gpu-001` 已由用户提交为 Slurm 数组 `13967217`：4 个 A100 worker，评测 10,030 个 Plus episode。`fullclean-4gpu-001` 已准备并通过本地检查：4 个 H100 worker，每个 1 张 H100、8 CPU、64 GB，评测原版 LIBERO 的 2,000 个 episode，每片 500 个。两批使用不同的 `runs/`、`logs/` 和 `reports/` 目录，不会覆盖结果。用户本次允许最多 8 张 GPU 重叠运行；full-clean 因而**不添加**等待 full-plus 的作业依赖。两批都使用付费账户 `gts-jli741-paid`、`inferno` QoS。

2026-10-12 08:13 EDT 的 `sbatch --test-only` 比较记录在 [clean-queue-probe-2026-10-12.json](../reports/preparation/clean-queue-probe-2026-10-12.json)。当时新申请的 H100 1/2/4 worker 均显示首个候选约 10 月 15 日 06:09；A100 则约 10 月 16 日 17:40。**这不是预约时间，也不代表四个成员会同时启动**。已有 full-plus 的候选时间从 10 月 12 日变到 10 月 13 日，说明估计会随调度重算。4 个 H100 是当时被测试的有效选项中较早的候选，选 2 个 H100 并没有显示更早排到，反而会让每卡承担更多 episode。

原版总量少于 Plus，但不能简单将时间按 2,000/10,030 换算：不同任务的 horizon、场景构建时间及成功/失败步数不同。A100 smoke 的 Slurm 记录显示峰值 GPU 显存约 9.2–9.4 GiB；这说明模型不需要 A100 的全部显存，不是 full 全量或其它 GPU 的兼容保证。L40S 有 48 GB 且支持 BF16，理论上可试，但本环境没有在 L40S 上验证 JAX 推理与 EGL 渲染；当天 L40S 的队列试算比 H100 更晚。H100 也尚未通过真实 smoke，若作业发生启动、CUDA 或 EGL 错误，先看日志并停止重复提交。单个 H100 worker 正常结束前最多运行 8 小时；这个时限不是预测用时。

## 状态与提交

在 PACE 终端运行，先看已有作业。`squeue` 失败时不要猜测它是否存在，也不要重复提交：

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

**只有在确认 full-clean 没有活动作业后**，下面这条才真正提交计费数组。本次用户已特别授权 Codex 代为提交这一份 full-clean；若 Slurm 控制器故障导致 Codex 未提交成功，可以由用户执行它。提交入口会检查同名活动数组，并只请求未完成的分片；遇到通信超时先查队列，不要盲目重试。

```bash
VLA_GPU_TYPE=h100 TIME_LIMIT=08:00:00 bash scripts/submit_parallel.sh fullclean-4gpu-001 --submit
```

若将来改回总共最多 4 张卡，可额外设置 `VLA_AFTERANY_JOB_ID=13967217`，让 clean 在 Plus 整个数组结束后才变为可运行；本次为尽快出结果，不设置它。[Slurm 数组依赖说明](https://slurm.schedmd.com/job_array.html#job-dependencies)确认 `afterany` 等待该数组全部成员结束，不要求它们全部成功。

可用以下命令看作业和日志；将 `CLEAN_JOB_ID` 换成实际返回的数字。`--test-only` 输出的数字不是正式作业 ID。

```bash
squeue -r -u gliu386 -o '%.18i %.32j %.12T %.12M %R'
sacct -j CLEAN_JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
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

GitHub 仅备份代码、文档和小型报告；模型 checkpoint、LIBERO 场景资产、视频、环境、原始作业日志都被 `.gitignore` 排除。模型运行路径和参数看 [代码地图](code-map.md)，benchmark 组成和分数比较限制看 [实验说明](experiment.md)，完整四卡设计与作业查询看 [overnight.md](overnight.md)。
