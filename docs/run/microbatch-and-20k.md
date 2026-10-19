# 固定 batch 的吞吐测试与 5k / 20k 配对实验

**历史方案，已于 2026-10-19 停用。** 当前改为各组在原 24 小时分配结束后使用最终 checkpoint，分别评测 900 条并以四卡追加 15,000 步；不设置总卡数上限，不使用常驻 CPU 控制器。参见[当前执行流程](after-24h-followup.md)。下文保留准备过程与历史作业记录，旧启动入口已禁用。

## 预算与调整原因

第一轮固定两组各 **5,000 optimizer updates**，之后从同一步数续训 **15,000 updates**，累计 **20,000**。全局 batch 始终为 **32**，训练数据、优化器、学习率曲线和模型/SF 配方保持一致。第一轮每组评测 StaticObstacles、DynamicObstacles 的 L0/L1/L2：**2 个 suite × 3 个等级 × 5 个任务 × 30 个初态 = 900 episodes**，两组共 **1,800** 条。

这里的 30 指包含难度等级的任务配置，不是 30 种完全不同的语言目标。每个 suite 450 条，每个 suite 的每个等级 150 条；每个任务文件提供 50 个初态，固定取索引 0–29。两组使用相同任务、初态与 episode 随机种子。由原先 10 次增加到 30 次重复，是为了提高每个任务的评测稳定性；不增加训练 seed 数，也不改变训练或任务覆盖。900-episode 评测每组申请一张卡、最长 24 小时，这是作业时限而非预计耗时。

原两卡 baseline 约需 14 秒/update，60k 约 9.7 天纯训练。5k 对应约 19.4 小时 baseline 计算，不含排队/初始化/评测；SF 需额外运行教师，不能假定一天也达到 5k。10/22 前优先取得 5k 配对结果，20k 的完成日期仍取决于实际吞吐和排队。

| 阶段 | GPU | 每卡 micro-batch | 累积次数 | Global batch |
|---|---|---:|---:|---:|
| 现有初步训练 | 2 × A100，允许 40/80 GB | 1 | 16 | 32 |
| 校准对照 | 2 × A100，指定 40 GB | 1、2 分别测量 | 16、8 | 32 |
| 后续续训 | 2 节点 × 每节点 2 张 A100，允许 40/80 GB | 2 | 4 | 32 |

更大的 micro-batch 增加每轮前后向工作量，同时可能提高硬件利用率。应比较实际秒/update、samples/second 和峰值显存，不能把累积次数的 `16/4` 直接当作四倍加速。保持 batch 32 是控制实验变量；较小 batch 并不必然降低效果。

## 自动执行链

入口为独立的 `src/sf_followup/`。被冻结的 `src/sf_training/` 与现有 baseline `14143587`、SF `14143602` 保持原样，原作业仍运行各自的 24 小时。

1. 两张 **40 GB A100** 分别测量 baseline/SF 的 micro-batch 1、2，每项运行 12 个完整更新，排除前 3 步计时。包含真实 optimizer 状态、DDP、有限梯度/损失检查及动作推理；校准权重不用于实验比较。batch 2 失败时，四卡续训不会自动放行。
2. 轻量 CPU Slurm 作业每分钟检查完整 checkpoint，通过硬链接独立保留 3k/5k；原训练的“最近两份”清理不会删除保留副本。模型不上传 GitHub。
3. 初步训练正常到达时限但不足 5k 时，在原作业退出后用原两卡、micro-batch 1 配置补齐到 5k，保留 optimizer/LR。异常退出、数值问题或缺失 checkpoint 不自动重试。
4. 两组准确的 5k 文件均可用后，各提交一个单卡 900-episode 评测。保持既有图像/动作约定、300 步 horizon、10 步 flow 积分、每次执行前 5 个动作，输出失败原因与逐谓词 cost。首批 episode 同时验证真实推理/EGL；基础设施错误不计作任务失败。
5. 初步训练/补齐均结束且校准通过后，baseline 四卡续训可与两份单卡评测并行，最多 6 卡。SF 四卡续训等待两份评测分配结束，之后两组训练合计最多 **8 卡**。账户始终为 `gts-jli741-paid`。
6. 每份四卡分配最多 24 小时，正常保存退出且未到 20k 时继续提交同配置续训，错误不自动重试。CPU 控制作业每次最多 72 小时，正常未完成工作会续接控制作业。

四卡训练需同时分配两台各有两张卡的节点，是同步 DDP，不是四个独立评测 worker。跨节点 NCCL 通信的实际性能尚未验证。启动、梯度及数值检查在正式分配内执行，没有单独增加多节点 smoke 队列。

### 控制作业与总并发

控制作业只申请 **1 CPU、2 GB 内存、0 GPU**，每分钟检查保存状态、保留完整 checkpoint，在前置条件满足后提交评测、补齐或续训，并生成配对结果汇总。实际 checkpoint 由训练进程写出；控制器负责保留和调度。它运行在 Slurm 节点上，不依赖本地电脑或 VS Code 保持连接。

总并发上限 **8 GPU** 适用于这套流程中同时获得资源的测试、训练、评测作业之和，不是每个作业都用 8 张，也不是必须凑齐 8 张才能启动。排队中的请求不占 GPU。该规则由工作流的阶段条件和 Slurm 依赖实现，不是对账户内另行手动提交作业的全局限额。

| 可同时运行的阶段 | GPU 合计 |
|---|---:|
| 初始 baseline 2 + 初始 SF 2 + 显存/吞吐校准 2 | 6 |
| 上述阶段 + 两份单卡评测（两组 5k 均已保留时） | 8 |
| baseline 四卡续训 + baseline 单卡评测 + SF 单卡评测 | 6 |
| baseline 四卡续训 + SF 四卡续训（评测分配均已结束） | 8 |

SF 四卡续训使用 `afterany` 等待尚未结束的评测分配，目的是释放两张评测卡、避免出现 4+4+1+1=10 张的重叠；评测是否成功仍单独校验。每个四卡 DDP 作业需要自己的四张卡同时到位，两个四卡作业可以分别启动。

## 迁移与比较边界

两组均从 **5,000 步**迁移，保留策略/LoRA/projector、AdamW moments、更新步数和原 LR 曲线，校验 base、数据、归一化、环境和配方。GPU 数和 micro-batch 改变后，按固定公式重建所有 rank 的随机流，两组规则相同，记录在 `migration.json`；不声称与两卡继续训练逐位等价。后续同四卡配置恢复仍还原完整 RNG。10k/15k/20k 也独立保留。

结果属于 Arena 微调后的短预算比较，不是 `pi05_libero` zero-shot。L0 属于训练任务分布，L1/L2 衡量难度泛化。单训练 seed、早期 checkpoint 不足以证明收敛；测试结果不用于选择最有利的训练步数或 SF 系数。

## 文件与输出

| 文件 | 用途 |
|---|---|
| `src/sf_followup/calibration.py` | 显存、吞吐对照与通过条件 |
| `src/sf_followup/checkpoints.py` | 独立保留完整 checkpoint |
| `src/sf_followup/evaluation.py` | 训练权重服务、5k 身份检查、Arena rollout、表图与配对汇总 |
| `src/sf_followup/continuation.py` | 兼容性校验、优化器与随机流迁移 |
| `src/sf_followup/control.py` | 阶段衔接、8 卡上限、幂等提交与错误停止 |
| `slurm/sf_continue.sbatch`、`scripts/sf_multinode_worker.sh` | 两节点四卡 torchrun |
| `tests/test_sf_followup.py` | 保留/迁移、预处理、随机流、配对预算及提交回归测试 |

- `runs/arena-followup-20261018-001/campaign.json`：流程计划；`state.json`：状态；`submission/`：真实提交回执。
- `runs/arena-microbatch-20261018-001/`：测试日志；完成后在同名 `reports/` 目录生成显存和加速表。
- `runs/arena-sf-5k-baseline-900-002/`、`runs/arena-sf-5k-sf-900-002/`：900-episode 逐条日志、视频和诊断；小型汇总位于同名 `reports/`。原 `*-001` 是未提交的 300-episode 旧计划，不用于本轮结果。
- `reports/arena-followup-20261018-001-5k-comparison/`：两组均完成后生成 `comparison.csv`、`comparison.png`。
- `runs/arena-sf-train-20k-mb2-001/`：续训 loss、迁移记录及 checkpoint。

## 查看与控制

2026-10-18 23:27 EDT 已提交：校准 **14171051**（排队）、控制作业 **14171052**（运行）。原训练保持运行/排队，四卡续训尚未提交，待先决条件满足后自动提交。[提交与 18 项测试证据](../../reports/followup-preparation-2026-10-18/README.md)。`state.json` 在控制作业首次运行后生成，其中等待校准的提示属于正常状态。

随后在评测尚未提交时，将预算改为每组 900 条，并用 CPU 控制作业 **14171526** 替换 **14171052**，使控制器读取重新冻结的评测计划。只停止了旧 CPU 控制器；原训练 **14143587 / 14143602** 和校准 **14171051** 的作业与排队位置保持不变。新增预算与提交依赖相关检查 **7 项通过**。[900-episode 计划与替换记录](../../reports/followup-evaluation-900-2026-10-18/README.md)。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
squeue -u gliu386 -o '%.18i %.32j %.10T %.10M %.28b %R'
cat runs/arena-followup-20261018-001/state.json
cat runs/arena-followup-20261018-001/submission/calibration.json
cat runs/arena-followup-20261018-001/submission/monitor-900.json
```

启动入口默认只预览；已提交后无需重复操作：

```bash
PYTHONPATH=src python3 -m sf_followup.control start \
  runs/arena-followup-20261018-001/campaign.json
# 添加 --submit 才提交；已有回执复用，结果未知时拒绝自动重试。
```

停止自动后续提交：从 `submission/monitor*.json` 找到当前控制作业，执行 `scancel 实际作业号`。这不会取消已经提交的训练/评测，各作业单独显示在 `squeue` 中。不要原样输入占位符 `JOB_ID`。

训练环境保持冻结。评测专用通信依赖放在 `.runtime/sf_followup_deps/`，由独立 lock 和文件 SHA256 约束，不安装进现有训练环境。
