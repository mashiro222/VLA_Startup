# Arena Safety full：代码复查与资源配置

**2026-10-15 后续检查：数组四个成员均已在 10 月 14 日凌晨运行后失败（退出码 1:0），错误为 WebSocket `keepalive ping timeout`，完成 0/2,250。Smoke 也遇到同一错误。** 详情见 [失败诊断](../reports/arena-diagnostics-2026-10-15/summary.md)。本文下方保留提交前复查及提交当时的历史信息，不表示作业仍在排队。

2026-10-13 的 full 评测采用独立提交配置，未以 smoke 完成为前提；资源方案为 **最多 4 张 A100**。**16:03:18 EDT 已提交 full 数组 `14020266`，成员 `_0` 至 `_3`，每成员 1 张 A100、8 CPU、64 GB、12 小时时限，账户 `gts-jli741-paid`、QoS `inferno`。** 原 smoke `14016239_0` 没有修改或取消。

提交后核对 `ArrayTaskThrottle=4`、`Dependency=(null)`；四个成员均为 `PENDING (Priority)`，候选时间 `N/A`，已完成 0/2,250，尚无运行结果。提交命令与校验摘要见 [实际提交记录](../reports/preparation/arena-full-submission-2026-10-13.json)。这与下方历史 `--test-only` 的候选日期是不同信息；实际启动仍由调度器决定。

## 复查结论与范围

提交前静态复查未发现阻塞问题；后续运行暴露的心跳超时未被当时测试覆盖。原有 10 项测试和新增直接提交入口的 8 项测试分别通过；检查了初始化、图像双轴翻转和缩放、状态向量、每次执行 5 步动作、300 步 horizon、SR/CC 判定、超时终态代价和 HazardAvoidance 的 0.05 缩放。与固定上游 OpenPI evaluator 的相应逻辑一致。完整覆盖为 75 个 Safety 任务 × 3 个评测 seed × 10 次，共 **2,250 episodes**，不是整个 Arena 的全部类别。

确认项目使用原 `pi05_libero` 权重及 normalization；上游版本、依赖、权重校验标记/文件大小、75 个任务定义和初始状态可读取。三个已有计划的文件哈希仍有效；full 的 2,250 条 episode key 及生成的随机种子没有重复。

测试覆盖包括：基础设施异常不计为模型失败、超时不误算成功、成功可以有安全代价、终态代价不重复累计、断点日志尾部恢复、跨进程任务锁、部分结果不伪装完整报告、重复提交拒绝、sbatch 回复不明时禁止盲目重试。当前 scratch 是带 `flock` 选项的 Lustre；多节点运行仍待真实作业验证。

**未验证的部分仍是首次 Arena GPU 推理、EGL 图像渲染、完整场景构建和真实机器人闭环。测试使用模拟环境接口，不能替代 full 的运行验证或用于估计成功率。** 依赖导入的 robosuite 可选机器人/IK、Gym 和 matplotlib 提示未使检查失败；本实验使用 Panda 与 numpy 1.26。

本次沿用已记录的独立 episode RNG 协议及每条 episode 新建环境，非逐位复现上游连续 RNG 流。未额外强制上游启动器中的 XLA/cuBLAS deterministic flags；相同 episode seed 不等于跨 GPU/驱动逐位一致。对论文的分数只能做训练条件不同的背景比较。

## 2026-10-13 约 15:13 EDT 的队列试算

使用 `sbatch --test-only`，**没有创建实际作业**。账户 `gts-jli741-paid`、QoS `inferno`，数组 `0-3%4`，每成员 1 GPU、8 CPU、64 GB。

| GPU | 每 worker 时限 | 候选启动时间（EDT） |
|---|---|---|
| A100 | 12 小时 | 10 月 19 日 22:29:12 |
| H100 | 12 小时 | 10 月 22 日 16:28:35 |
| A100 | 8 小时 | 10 月 19 日 22:29:12 |
| H100 | 8 小时 | 10 月 22 日 16:28:35 |

15:19 EDT 使用最终独立提交入口再次试算 12 小时方案：A100 仍为 **10 月 19 日 22:29:12**，H100 调整为 **10 月 22 日 16:25:00**。原始输出和检查摘要见 [复查记录 JSON](../reports/preparation/arena-full-review-2026-10-13.json)。

这是新请求的当前估计，不是预约、最早保证时间或四个成员全部启动时间，可以提前或延后。试算只返回一个候选，无法据此算出四张卡全部开始的时刻。原 smoke 自己的候选仍为 `N/A`。约 15:05 的资源快照中 A100 分区 32 张卡全部已分配；H100 正常节点的 24 张全部已分配，另一个配置 8 卡的节点因只注册 7 卡处于 `DRAIN+INVALID_REG`。

当时选定的方案为 **最多 4 张 A100、每 worker 12 小时**：排队试算优于 H100，缩短至 8 小时没有改善估计。H100 可能缩短模型推理，但 Arena 还包括 CPU 仿真、场景重建和渲染，现无同一批 Arena episode 的 A100/H100 对照，不能给出可靠倍速。之前的 H100 clean 与 A100 plus 任务不同，不构成硬件速度对照。

## 并行机制与运行时间估算

四个成员分别申请一张卡，可以分别开始，不需要同一台节点凑齐四张。每个 worker 持续领取一个未完成的“任务 × seed”单元，每单元 10 条；完成后继续领取。单元不是预先永久分给某一张 GPU。只有一张排到时也会继续处理工作。

失败 worker 的未完成记录可以在后续续跑时恢复；不会自动增加计费作业或自动重试提交。其它 worker 已经跳过一个被锁定的单元后，不保证在失败释放锁时回头领取，因此最后仍需检查 2,250 条完整覆盖。

尚无 Arena 实测速率。假设四卡同时持续工作、平均每条耗时包含环境重建及推理，则计算部分大致如下（仅算术情景，不是预测）：

| 平均每条耗时 | 四卡完成 2,250 条约需 |
|---|---|
| 15 秒 | 2.34 小时 |
| 30 秒 | 4.69 小时 |
| 60 秒 | 9.38 小时 |

还需加模型加载/JIT、排队及 worker 启动不齐的时间；12 小时是每 worker 的终止上限，不是承诺耗时。部分卡迟迟不启动时，先开始的卡仍会推进剩余任务。到时限未完成可按原计划续跑。

## 提交入口与运行记录

原 `scripts/arena.py` / `slurm/arena.sbatch` 仍要求 smoke 通过，且被冻结的模型和评测代码未改动。`scripts/arena_direct.py` / `slurm/arena-direct.sbatch` 用于首轮不依赖 smoke 的 full 提交。`runs/arena-full-001/direct-launch.json` 记录提交配置、smoke 验证豁免、原计划哈希和两个入口文件的哈希；提交与 worker 启动均执行校验，该记录同时进入报告 provenance。

以下为历史方案的预览命令，**均不提交**。实际运行采用 A100 配置：

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
python3 scripts/arena_direct.py submit arena-full-001 --gpu a100
python3 scripts/arena_direct.py submit arena-full-001 --gpu h100
```

A100 配置通过 `--submit` 提交，返回作业号 `14020266`；H100 配置未提交。该 full 与独立 smoke 并行运行，没有完成依赖。上述入口与命令保留为历史记录；后续实验使用匹配配置的 smoke 验证条件，详见 [运行说明](arena-safety.md)。

之后仍可用统一的检查和报告入口：

```bash
bash scripts/arena.sh check arena-full-001
squeue -r -j 14020266 -o '%.20i %.12T %.24S %R'
sacct -j 14020266 --format=JobID%22,State,ExitCode,Elapsed
squeue -r -u gliu386 -o '%.20i %.12P %.32j %.12T %.12M %R'
bash scripts/arena.sh report arena-full-001 --partial
# 全部完成后：
bash scripts/arena.sh report arena-full-001
```

作业启动后日志为 `logs/arena-full-001-14020266_0.out` 至 `_3.out`；例如 `tail -F logs/arena-full-001-14020266_0.out` 可以等待第 0 个成员的日志出现。没有完成 episode 时，报告命令会拒绝生成无数据报告，这是预期行为。

结果与错误保存在本地，报告含表格与图；权重、场景资产和视频保留在本地，不使用 W&B。复核测试入口为 `bash scripts/test_arena.sh`；该命令不申请 GPU。

参考：[Slurm test-only](https://slurm.schedmd.com/sbatch.html#OPT_test-only)、[作业数组的并发上限](https://slurm.schedmd.com/job_array.html)、[固定版本 Arena evaluator](https://github.com/PKU-Alignment/VLA-Arena/blob/babe582ebffc82b979b77964a7e56417d02f63a4/vla_arena/models/openpi/evaluator.py)。
