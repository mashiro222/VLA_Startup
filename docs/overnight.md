# 四张 A100 优先运行 full-plus

本页记录 full-plus 设计、排队试算与通用四卡操作。**最新作业状态和 full-clean H100 方案请先看 [2026-10-12 提交与检查说明](full-clean-2026-10-12.md)。**下面的脚本不自动接续 smoke、不取消任何作业；默认仅预览，只有 `--submit` 才创建计费作业。

2026-10-11 的 Plus smoke 作业 `13966714` 正常退出（0:0），14 分 32 秒，七类共 7/7 条有效记录、6 次成功、1 次任务失败，没有程序异常。见 [真实报告](../reports/smoke-plus-001/summary.md)和 [检查记录](validation.md)。四卡 full-plus 计划 `fullplus-4gpu-001` 已准备并通过 smoke gate，用户随后提交为 Slurm 数组 `13967217`。

## 查看 gliu386 的全部作业

当前排队、运行和收尾中的作业，展开数组中的每个 worker：

```bash
squeue -r -u gliu386 -o '%.18i %.12P %.32j %.12T %.12M %.6D %R'
```

每 10 秒刷新；Ctrl+C 只退出查看，不取消作业：

```bash
watch -n 10 "squeue -r -u gliu386 -o '%.18i %.32j %.12T %.12M %R'"
```

今天的全部记录，包括已经完成或取消的作业：

```bash
sacct -u gliu386 -S today --format=JobID%22,JobName%34,State%22,ExitCode,Elapsed,AllocTRES%75,MaxRSS
```

换成 `-S 2026-10-01` 可以查询更早的记录；可查询范围由集群记账保留期决定。仅看每个作业一行，可以加 `-X`；需要 `MaxRSS` 时保留 `.batch` 等步骤行，作业总行的内存数据常常为空。

| 输出 | 含义 |
|---|---|
| `PENDING` / `PD` | 尚未启动；`Resources` 是等待资源，`Priority` 是等待调度优先级 |
| `RUNNING` / `R` | 正在执行；还需看 episode 日志确认业务进度 |
| `COMPLETED` + `ExitCode=0:0` | 程序正常退出；机器人成功率另看报告 |
| `FAILED` / `TIMEOUT` / `OUT_OF_MEMORY` | 程序失败、达到时限、内存超限，需看日志 |
| `CANCELLED` | 作业被取消 |
| `Elapsed` | 开始运行后的累计墙钟时间，不含排队时间 |
| `AllocTRES` | 分配的 CPU、GPU、主机内存等；`billing` 是调度器计费资源单位，不是美元或实际花费 |
| `MaxRSS` | 记账记录的峰值主机内存，不是 GPU 显存 |
| 字段末尾 `+` | 列宽不够，内容被截断；例如 `.ba+` 是 `.batch` 的截断 |

提交出现 `Socket timed out on send/recv operation` 时，调度器可能已创建作业，只是客户端没有收到回复。先用 `squeue` 查询，不能立即反复提交。提交入口现在会检查同名活动作业，并在提交回复失败后提示检查队列；这无法代替对短暂通信故障的人工判断。

## PACE 是否支持四卡

2026-10-11 的实际 Slurm 查询结果：

- 账户 `gts-jli741-paid` 的用户关联允许 `inferno`；该用户关联未显示额外的 GPU 数限制。
- `inferno` 的 `MaxTRESPU` 为 `cpu=6000,gres/gpu=32`。
- `gpu-a100` 分区 `State=UP`，13 个节点共 32 张 A100，最大作业时限 3 天。
- `MaxArraySize=1001`，本方案只有 4 个数组成员。
- 对四成员数组、每成员 1 张 A100 / 8 CPU / 64 GB 的请求执行 `sbatch --test-only` 返回成功；这不会创建作业。

这些检查说明资源请求可被接受，不表示有四张空闲卡、保证同时启动或保证账户 credit 充足。此次 12、8、4 小时时限的试算都给出约 **10 月 13 日晚**的启动估计。队列会变化，不能承诺 10 月 11 日当晚启动或完成；`--test-only` 输出中的编号是试算信息，不是已提交的计费作业。

10 月 11 日 21:10–21:13（EDT）的额外比较如下，均为 12 小时时限、每个 worker 一张卡，原始请求与响应在 [queue-probe-2026-10-11.json](../reports/preparation/queue-probe-2026-10-11.json)。

| GPU | CPU/worker | 数组并发上限 | 当时返回的启动估计 |
|---|---:|---:|---|
| A100 | 8 | 1 / 2 / 4 / 8 | 均为 10 月 13 日 22:38 |
| H100 | 8 | 4 | 10 月 14 日 07:04 |
| L40S | 4 | 4 | 10 月 18 日 18:27 |
| H200 | 8 | 4 | 10 月 24 日 06:24 |

L40S 最初用 8 CPU 被分区的 4:1 CPU/GPU 上限拒绝；表中是改为 4 CPU 后的有效试算。八卡仅用于比较，执行脚本仍限制最多四卡。数组试算只返回一个候选启动估计，**不是所有成员同时启动的承诺**；不能从相同的输出推断增减卡数不影响排队。H100/H200/L40S 尚未验证本环境的模型、渲染与真实速度，当前没有证据表明切换会更早完成。

21:25 再比较四个 A100 worker 的 12 / 24 小时时限，两者当时均返回 **10 月 13 日 23:55**。这也说明估计会随队列变化。因为没有观察到 24 小时时限的额外排队代价，下面推荐将此次上限设为 24 小时，减少中途超时后重新排队的机会；并不保证实际启动时间相同。

复查限额和分区：

```bash
sacctmgr -nP show qos where name=inferno format=Name,MaxTRESPU,MaxSubmitPU
scontrol show partition gpu-a100
```

参考：[Slurm 作业数组](https://slurm.schedmd.com/job_array.html)、[sacct](https://slurm.schedmd.com/sacct.html)、[sbatch 的 test-only](https://slurm.schedmd.com/sbatch.html)。

## 四卡具体跑什么

这个数组最多 4 个 worker 同时执行，只跑 full-plus。每个 worker 申请 **1 张** A100，可以位于不同节点，无需一台机器同时空出 4 张卡。不要把 `GRES` 改成 4，那会变成每个 worker 申请 4 张卡。2026-10-12 用户另行批准一份最多 4 张 H100 的 full-clean 数组与它重叠运行，因此两份合计可达 8 张 GPU。

| Worker | full-plus | 每个 worker 的资源 |
|---|---:|---|
| 0 | 2,507 episodes | 1 A100、8 CPU、64 GB |
| 1 | 2,508 episodes | 同上 |
| 2 | 2,507 episodes | 同上 |
| 3 | 2,508 episodes | 同上 |
| 合计 | 10,030 | 同时最多 4 A100、32 CPU、256 GB |

按任务 horizon 分配负载，同一 clean 任务的 50 个初始状态放在同一个 worker 上，避免重复创建场景。分片互斥、并集等于完整测试集。真实耗时仍受场景创建、成功/失败步数、存储和排队影响。

脚本默认每个 worker 最多运行 **12 小时**；下面的推荐命令显式设置 **24 小时**，完成后会立即退出。这是上限，不是预计完成时间。四个 worker 全部用满 12 / 24 小时分别为 48 / 96 GPU·小时上限，不是实际 credit 报价；提前结束按实际资源使用记账，续交会增加用量。各 worker 的计时从自己获得资源后开始，故整个数组可能跨越更长的实际时间。

仅作时间敏感性估算：若 Plus 每个变体连同建场景平均 10 秒，四卡均衡运行约需 7 小时；平均 20 秒则约 14 小时，还要加启动和排队。原版 smoke 的四条记录不足以估计全部 Plus。不要把原单卡脚本的 24 小时时限理解成 full 一定需要或一定能在 24 小时内完成。

全量完成时间取决于最晚完成的 worker：`max(每片排队时间 + 模型启动 + 该片全部场景/rollout/汇总耗时)`。四卡只并行独立任务，不会让单个 episode 加速；环境创建、CPU 图像扰动和共享存储也有耗时。七条 smoke 覆盖七类扰动各一个 spatial 代表，不等于全量 10,030 个变体均无 bug，也不能充分代表其它 suite 的长任务。smoke 无程序错误后可以直接开始 full，后续异常仍会单独记录，不能悄悄当作机器人失败或完整成绩。

此次 Plus smoke 的传感器用例耗时 48.62 秒，其中模型推理仅 1.51 秒；其它已预热的五个用例约 4.78–9.34 秒，首个用例 73.88 秒且包含首次推理编译。因此直接换更快 GPU 不会按算力比例缩短全部耗时。当前可执行建议是在四卡上限内直接分摊 full-plus，等它完成再跑 clean；不能根据七条样本承诺一晚完成。

用于量级判断的粗算：此次运行配置写出前的启动约 668 秒，配置到最后一条记录约 196 秒。后者减去约 60 秒首次推理编译，再机械外推 `10030 × (196−60) / 7 / 4 / 3600 ≈ 13.5 小时`，尚不含排队和每片启动。这**不是可靠预测区间**：七条都来自 spatial，类别比例、扰动强度、环境构造开销和其它 suite 的 horizon 都不同。它说明 12 小时上限可能不足，不能将“一晚跑完”作为承诺。

## 运行命令

在 PACE 终端执行。使用新的 `fullplus-4gpu-001` 批次；旧 `overnight-4gpu-001` 是 clean 后接 plus 的方案，且冻结了旧代码，不用于此次提交。前两条脚本命令均不提交作业：

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
python3 scripts/parallel.py prepare fullplus-4gpu-001 --workers 4 --benchmarks plus
TIME_LIMIT=24:00:00 bash scripts/submit_parallel.sh fullplus-4gpu-001
```

smoke-plus 结束并检查报告后，用户已经执行了一次 full-plus 的计费提交。下面保留提交命令作为复现或超时续跑参考；活动数组存在时不要重复提交：

```bash
TIME_LIMIT=24:00:00 bash scripts/submit_parallel.sh fullplus-4gpu-001 --submit
```

plus-only 提交入口要求 Plus smoke 有 7/7 条完整记录，并且没有未审查的基础设施错误；clean-only 检查原版 4/4，both 则检查两者。任务本身未成功是有效评测结果，不要求 smoke 成功率 100%。尚未跑完会拒绝提交 full。没有作业依赖或自动接续机制。

如果希望控制单次作业时长，也可以选择 8 小时（或改为 12 小时）上限，在预览和提交时保持一致：

```bash
TIME_LIMIT=08:00:00 bash scripts/submit_parallel.sh fullplus-4gpu-001
TIME_LIMIT=08:00:00 bash scripts/submit_parallel.sh fullplus-4gpu-001 --submit
```

缩短时限不能保证更早排到卡，并可能需要续跑。不要另外重复提交一份单卡 full-plus。原版 `fullclean-4gpu-001` 已提交为 H100 数组 `13974531`；用户此次允许两份数组总计最多 8 张 GPU 同时运行，查看方式见[最新说明](full-clean-2026-10-12.md)。

`--benchmarks both` 保留原来的每片 clean 后接 plus 模式，需要两套 smoke 都通过；它会推迟 full-plus 的完成，不是当前的优先方案。

## 查看进度、续跑、汇总

```bash
python3 scripts/parallel.py status fullplus-4gpu-001
squeue -r -u gliu386 -o '%.18i %.32j %.12T %.12M %R'
ls -lt logs/fullplus-4gpu-001-*.out
```

数组日志命名为 `logs/fullplus-4gpu-001-数组编号_成员编号.out`；逐条结果保存在 `runs/fullplus-4gpu-001-full-plus-shard-00/` 到 `shard-03/` 四个分片目录里。每个分片停止后会尝试输出自己的表格与图。

如果时限到了且还有缺失记录，等旧数组所有成员彻底停止后，使用相同 batch ID 再执行提交入口。它只提交仍未完成的 worker，跳过已完成的分片，并按已有 episode/RNG 状态续跑。不要修改已开始批次的评测代码、配置、worker 数或 manifest；需要修改请使用新批次。

full-plus 全部完成后，在登录节点合并：

```bash
python3 scripts/parallel.py merge fullplus-4gpu-001
```

报告目录：

- `reports/fullplus-4gpu-001-full-plus/`

包含 `summary.md`、`summary.csv`、`episodes.csv`、成功率柱状图和累计成功率折线图。合并会核对全部任务、初始状态、配置与代码指纹，拒绝缺失和重复记录；保留分片来源和原始指纹，不能把 plus 的部分结果当全量成绩。

确认报告后上传小型结果（不会上传模型、场景或视频）：

```bash
bash scripts/publish_results.sh fullplus-4gpu-001-full-plus
```

## 协议与代码位置

分片沿用模型、图像、动作、任务数、horizon 和环境 seed。每片从原评测器默认的模型 RNG `[0, 0]` 开始，再独立记录和恢复随机流；因此**不是单卡串行执行的同一条连续随机序列**，不能宣称动作或成功率逐位一致。完整报告会明确标注这一点，公开成绩只作描述性比较。合并目录仅用于报告，不能当作串行续跑目录；续跑使用原分片。

- `scripts/submit_parallel.sh`：独立提交入口、并发上限、smoke 和重复提交检查。
- `slurm/parallel.sbatch`：每个数组成员的资源环境与启动方式。
- `scripts/parallel.py`：任务分片、配置冻结、状态、按所选 benchmark 执行、结果合并。
- `tests/test_parallel.py`：分片覆盖和错误合并拒绝测试。
- `tests/test_submission.py`：使用假的 Slurm 命令测试默认不提交、smoke gate、重复作业、超时不重试。

并行功能已经做本地测试与调度器校验；四卡 full 运行结果尚待作业完成。真实单卡结果见 [smoke-clean 报告](../reports/smoke-clean-001/summary.md)和 [smoke-plus 报告](../reports/smoke-plus-001/summary.md)。
