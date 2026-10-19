# 固定 batch 的吞吐测试与 5k / 20k 配对实验

## 预算与调整原因

第一轮固定两组各 **5,000 optimizer updates**，之后从同一步数续训 **15,000 updates**，累计 **20,000**。全局 batch 始终为 **32**，训练数据、优化器、学习率曲线和模型/SF 配方保持一致。第一轮每组评测 StaticObstacles、DynamicObstacles 的 L0/L1/L2：30 个任务 × 10 个初态 = **300 episodes**，两组共 600 条。

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
4. 两组准确的 5k 文件均可用后，各提交一个单卡 300-episode 评测。保持既有图像/动作约定、300 步 horizon、10 步 flow 积分、每次执行前 5 个动作，输出失败原因与逐谓词 cost。首批 episode 同时验证真实推理/EGL；基础设施错误不计作任务失败。
5. 初步训练/补齐均结束且校准通过后，baseline 四卡续训可与两份单卡评测并行，最多 6 卡。SF 四卡续训等待两份评测分配结束，之后两组训练合计最多 **8 卡**。账户始终为 `gts-jli741-paid`。
6. 每份四卡分配最多 24 小时，正常保存退出且未到 20k 时继续提交同配置续训，错误不自动重试。CPU 控制作业每次最多 72 小时，正常未完成工作会续接控制作业。

四卡训练需同时分配两台各有两张卡的节点，是同步 DDP，不是四个独立评测 worker。跨节点 NCCL 通信的实际性能尚未验证。启动、梯度及数值检查在正式分配内执行，没有单独增加多节点 smoke 队列。

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
- `runs/arena-sf-5k-baseline-001/`、`runs/arena-sf-5k-sf-001/`：逐 episode 日志、视频和诊断；小型汇总位于同名 `reports/`。
- `reports/arena-followup-20261018-001-5k-comparison/`：两组均完成后生成 `comparison.csv`、`comparison.png`。
- `runs/arena-sf-train-20k-mb2-001/`：续训 loss、迁移记录及 checkpoint。

## 查看与控制

2026-10-18 23:27 EDT 已提交：校准 **14171051**（排队）、控制作业 **14171052**（运行）。原训练保持运行/排队，四卡续训尚未提交，待先决条件满足后自动提交。[提交与 18 项测试证据](../../reports/followup-preparation-2026-10-18/README.md)。`state.json` 在控制作业首次运行后生成，其中等待校准的提示属于正常状态。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
squeue -u gliu386 -o '%.18i %.32j %.10T %.10M %.28b %R'
cat runs/arena-followup-20261018-001/state.json
cat runs/arena-followup-20261018-001/submission/calibration.json
cat runs/arena-followup-20261018-001/submission/monitor.json
```

启动入口默认只预览；已提交后无需重复操作：

```bash
PYTHONPATH=src python3 -m sf_followup.control start \
  runs/arena-followup-20261018-001/campaign.json
# 添加 --submit 才提交；已有回执复用，结果未知时拒绝自动重试。
```

停止自动后续提交：从 `submission/monitor*.json` 找到当前控制作业，执行 `scancel 实际作业号`。这不会取消已经提交的训练/评测，各作业单独显示在 `squeue` 中。不要原样输入占位符 `JOB_ID`。

训练环境保持冻结。评测专用通信依赖放在 `.runtime/sf_followup_deps/`，由独立 lock 和文件 SHA256 约束，不安装进现有训练环境。
