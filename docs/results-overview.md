# 实验记录

按实验推进时间排列，统一记录结果、故障、调整原因和后续决定。时间使用 America/New_York（EDT）；队列与训练进度为注明时间的快照。迁移评测使用官方 `pi05_libero`，LoRA / SF 配对训练从 `pi05_base` 初始化，两类结果分开解释。

| 主要实验 | 覆盖 / 预算 | 结果或状态 |
|---|---:|---|
| LIBERO-Plus | 10,030 / 10,030 episodes | SR **84.52%** |
| LIBERO clean | 1,500 / 2,000 episodes | 已完成部分 SR **96.40%**；停止补跑缺失分片 |
| Arena Safety balanced1500 | 1,500 / 1,500 episodes | SR **11.47%**，平均 CC **60.24** |
| Arena LoRA / LoRA+SF | 5k 初步评测；后续累计 20k | baseline 正在训练，SF 排队；准备 batch 2 测量与配对评测，尚无训练后 SR/CC |

## 2026-10-11：LIBERO 环境与 smoke

`pi05_libero` 完成原版 smoke **4/4 成功**，Plus 七类扰动 smoke **6/7 成功**，均无基础设施错误。Plus 的机器人初态用例达到 horizon 后记为任务失败。权重加载、JIT、EGL、动作闭环及视频抽查通过；少量 smoke 不用于估计正式分数。

**后续决定：** 扩展到完整 Plus 和 clean 对照，用四个独立单卡 worker 分片。证据：[clean smoke](../reports/smoke-clean-001/summary.md)、[Plus smoke](../reports/smoke-plus-001/summary.md)、[环境与检查归档](run/validation.md)。

## 2026-10-12—15：LIBERO-Plus 完成，clean 保留部分覆盖

Plus 完成 **10,030/10,030**，成功 **8,477 条（84.52%）**。分组成功率：背景 96.19%、光照 96.41%、噪声 86.76%、布局 86.62%、语言 85.49%、机器人初态 74.45%、相机视角 72.73%。相机视角与机器人初态是当前较弱的扰动类别，尚不能归因于特定网络模块。

社区自报参考值为 84.2%，本次高约 0.32 个百分点；该值不是官方排行榜，协议可能不同，见[比较边界](experiment.md#公开结果对照)。

Clean 已完成 **1,500/2,000**，成功 **1,446 条（96.40%）**。缺失分片首次及补跑均在模型启动 900 秒限制下退出，没有新增有效 episode；其余三个 H100 分片成功运行，不能据此认定 H100 不兼容。**调整原因与决定：** 重复启动超时且根因未明，停止补跑这一片，保留部分覆盖结果，不报告完整 clean 分数。此结果与完整 Plus 也不是严格配对对照。

**Plus 的表和图：** [汇总](../reports/fullplus-4gpu-001-full-plus/summary.md)、[分组 CSV](../reports/fullplus-4gpu-001-full-plus/summary.csv)、[逐 episode CSV](../reports/fullplus-4gpu-001-full-plus/episodes.csv)、[分组图](../reports/fullplus-4gpu-001-full-plus/success_by_group.png)、[累计成功率](../reports/fullplus-4gpu-001-full-plus/cumulative_success.png)。累计曲线按任务顺序排列，不是训练曲线。

Clean：[分片 00](../reports/fullclean-4gpu-001-full-clean-shard-00/summary.md)、[01](../reports/fullclean-4gpu-001-full-clean-shard-01/summary.md)、[03](../reports/fullclean-4gpu-001-full-clean-shard-03/summary.md)。[详细故障与运行归档](run/full-clean-2026-10-12.md)。

## 2026-10-13—15：Arena 首轮通信失败与修复

首轮 smoke **0/15**、full **0/2,250**：所有 worker 均在首个动作返回前退出，WebSocket 心跳超时，没有有效模型成功率。

**调整原因与修复：** 仿真初始化/同步调用期间的心跳处理不可靠；连接推迟到场景就绪，关闭本机短心跳，保留请求与 episode deadline，并补充通信回归测试。旧计划保留审计，修复代码使用新 run，不合并旧数据。

证据：[失败诊断](../reports/arena-diagnostics-2026-10-15/summary.md)、[异常明细](../reports/arena-diagnostics-2026-10-15/errors.jsonl)、[修复与测试](run/arena-transport-fix-2026-10-15.md)。提交试算等背景材料见[首轮技术归档](run/arena-full-review-2026-10-13.md)。

## 2026-10-16：Arena smoke 跑通，增加失败与 cost 诊断

`arena-smoke-002` 完成 **15/15**，SR **0%**，平均 CC **41.05**，全部达到 300 步上限；通信故障未复现。录像、图像和动作接口检查未发现足以解释全部失败的明确适配错误，但不能排除所有实现问题。

**调整原因与决定：** 少量零成功样本不足以概括迁移性能；新增终止原因、逐谓词 cost、接触段与持续步数记录。正式评测改为固定 **1,500 episodes**，覆盖全部 75 项 Safety 任务、每项 20 个不同初态，以控制规模并保留分组覆盖。协议先固定，不因已观察成绩挑选样本。

证据：[smoke 结果](../reports/arena-smoke-002/summary.md)、[录像与接口审查](run/arena-smoke-audit-2026-10-16.md)。定义见[诊断协议](arena-diagnostics.md)。

## 2026-10-17：训练归一化异常，停止并从 base 重启

首轮 LoRA baseline / LoRA+SF 的三步训练验证通过了保存、恢复与推理，却没有覆盖稀疏旋转动作。正式训练约 **44% 更新出现极大有限 loss**，最高约 **2.8×10⁹**：动作第 4、5 维的 q01=q99=0，稀疏非零值被近零分母放大至百万量级。

**中断原因：** 归一化数值异常使训练结果无效。两组旧任务于约 19:32 停止，旧 checkpoint 不用于修复后的训练或对照。

**修复与重启：** 退化分位数范围回退到训练数据实际范围，恒定维度使用单位宽度；不裁剪动作、不删除稀疏样本。完成 361,736 帧归一化/逆变换审计和正式 batch 回放，增加极大有限 loss 检查。`arena-sf-train-full-002` 两组重新从 `pi05_base` 开始，各 2 张 A100；这次显式跳过新的 GPU smoke，不能将其记为通过。

证据：[异常分析与曲线](../reports/training-audit-2026-10-17/README.md)、[全数据审计](../reports/preparation/sf-normalization-audit.json)、[稀疏 batch 回放](../reports/preparation/sf-sparse-batch-replay.json)、[重启配置归档](../reports/restart-2026-10-17/README.md)。配方与源码见[训练说明](arena-sf-training.md)，命令见[运行手册](run/training-operations.md)。

## 2026-10-17：Arena 诊断 smoke 与 1,500 条评测完成

新记录器的 `arena-smoke-003` 完成 **15/15**，SR **0%**，平均 CC **33.33**，逐谓词诊断完整且无基础设施错误。随后固定 `pi05_libero` 的 `arena-balanced1500-001` 于 22:00:40 全部完成：**172/1,500 成功，SR 11.47%，平均 CC 60.24**。

| Safety suite | 成功 / 完成 | SR | 平均 CC |
|---|---:|---:|---:|
| Static obstacles | 51 / 300 | 17.00% | 82.26 |
| Dynamic obstacles | 3 / 300 | 1.00% | 144.14 |
| Cautious grasp | 11 / 300 | 3.67% | 21.60 |
| Hazard avoidance | 3 / 300 | 1.00% | 4.61 |
| State preservation | 104 / 300 | 34.67% | 48.57 |

1,328 条失败全部达到 300 步上限；这是终止条件，不是抓取/规划失败的完整物理原因。1,500 条记录均有 cost 分解且总和一致，无通信、渲染或计分错误。Static / Dynamic 的夹爪接触分别贡献平均 CC **67.13 / 139.67**；CC 累积谓词激活和终态代价，不等于独立碰撞次数。成功但有代价 52 条，失败但零代价 264 条，SR 与安全性必须分开分析。

**后续决定：** 固定这套任务、初态和指标用于训练后 baseline/SF 对照；不再以 15 条 smoke 的零成功率代表整体。当前完成的是 LIBERO 权重的迁移参照，并非 Arena 微调模型结果。

[完整报告](../reports/arena-balanced1500-001/summary.md) · [SR/CC 图](../reports/arena-balanced1500-001/safety_by_level.png) · [cost 构成图](../reports/arena-balanced1500-001/cost_attribution.png) · [逐 episode 表](../reports/arena-balanced1500-001/episodes.csv) · [失败原因](../reports/arena-balanced1500-001/failure_modes.csv) · [详细分解与运行归档](run/arena-results-2026-10-17.md)

## 2026-10-18：修复后训练正常推进，速度促使重新评估资源

21:42 EDT 快照：baseline 已完成 **360/60,000 updates**，最新 FM loss **0.0430**，最近 100 步中位数 **0.0461**；已覆盖 207 个稀疏旋转样本，未发现非有限或超过保护阈值的 loss。实际设备为 **2 × A100 PCIe 40 GB**，每卡峰值已分配显存约 **16.79 GiB**。SF 仍在等待资源，尚无修复后的 SF loss。一次 rendezvous 心跳警告后 baseline 继续更新，未形成训练退出。

**调整动机：** 先前约 14 秒/update，60,000 步线性外推约 **9.7 天纯计算**；当前最近 100 步约 13.5 秒，对应约 9.4 天。两者均不含排队和保存开销，说明现有预算无法一晚完成。四卡即使理想翻倍也仍约需 4.7–4.9 天；SF 另有教师计算，不能照搬 baseline 的耗时。

**本次决定：** 保留 baseline 与 SF 现有两卡作业，各运行本次 24 小时分配，不取消、不重新提交。Baseline 预计于 **10/19 19:57 左右**进入保存退出流程，Slurm 硬时限约 **20:07**。SF 的 24 小时从实际启动计算。每 1,000 步及临近时限保存，保留最近两份完整 checkpoint；到时停止不代表完成 60,000 步。

**扩卡计划与限制：** 后续评估从 checkpoint 续训到四卡。当前恢复校验要求 GPU 数一致，尚不能仅把 `--gpus 2` 改成 4；需显式迁移优化器、步数和随机状态并验证。保持 global batch=32，四卡时每卡累积从 16 降为 8。改变卡数后的随机流与规约顺序不保证逐位一致。

21:36—21:40 的只读试算中，两卡 SF 候选 10/19 08:40；四 H100 候选 10/27；单节点四 A100 请求被拒，跨两节点四 A100 候选 10/30。它们是可变排队估计，不是预约。**因此本次保留较早的两卡队列，四卡尚未提交。** 当前只实现单节点训练，跨节点方案还需分布式启动支持；未把试算通过写成运行验证。

[进度快照](../reports/training-progress-2026-10-18/snapshot.json) · [checkpoint 机制、检查与资源比较](run/training-checkpoints.md)。恢复后的训练成果仍需独立 Arena 评测；loss 正常不代表安全性已经改善。

### 同日检查：退出保存与安装文件修复

新增 **8 项 CPU 检查通过**：覆盖内部 deadline、停止信号触发的非定期 checkpoint、模型/优化器恢复、原子写入失败保护及全局 batch 划分；不把这些检查当作真实 GPU 存储验证。

检查中另发现安装环境的 `modeling_paligemma.py` 比冻结源码多出 **85 个空字节**，触发 `SyntaxError`，SHA256 不匹配。固定上游源码完整，因此仅原子替换这个安装文件，恢复到原计划要求的哈希；损坏副本保留在本地。**修复原因：** 避免排队中的 SF 在导入或计划校验时失败。未更改模型配方、依赖版本、训练步数或现有作业，文件损坏的根因尚未确定。随后同计划的完整本地校验与 policy/teacher 导入均通过。这与此前动作归一化异常是两个独立问题。

证据：[退出保存测试](../reports/training-progress-2026-10-18/checkpoint-tests.xml)、[安装文件修复记录](../reports/training-progress-2026-10-18/environment-repair.json)。

## 2026-10-18：固定 5k 初步评测，测量 batch 2，续训目标调整为 20k

**调整原因：** 两卡 micro-batch 1 的实测吞吐使 60k 预算过长；baseline 峰值显存约 16.8 GiB，因此测量更大的 micro-batch，而不按累积次数直接推断倍速。

**决定：** 原两卡作业保持本次 24 小时，独立保留 3k/5k checkpoint。两组同为 5k 时评测 StaticObstacles / DynamicObstacles 的 L0/L1/L2，每组 300 episodes，输出 SR、CC、失败原因及 cost 分解。不足 5k 的正常时限退出作业按原配置补齐。

下一阶段从配对 5k checkpoint 续训 15k，累计 **20k**。候选为两节点四张 A100、每卡 micro-batch 2、累积 4 次，global batch=32，不限定 40/80 GB。独立申请两张 40 GB A100，分别对 baseline/SF 测量 batch 1/2；通过后才放行续训。初步评测与后续训练按最多 8 卡安排，10/22 前优先取得 5k 结果，不承诺 20k 的固定完成日期。

新的入口与冻结训练代码分离，包括 PyTorch 权重的 Arena 服务、严格步数/归一化检查、配对表图及提交衔接。18 项 CPU/通信回归测试通过，不能替代真实 GPU/EGL、多节点通信或吞吐结果。23:27 EDT，校准作业 **14171051** 已提交并排队，CPU 控制作业 **14171052** 已运行；四卡训练尚未提交。[执行说明](run/microbatch-and-20k.md) · [提交与验证证据](../reports/followup-preparation-2026-10-18/README.md)。

## 阅读与证据索引

[模型与 benchmark 开发指南](research-guide.md) · [LIBERO/Plus 协议](experiment.md) · [Arena 协议](arena-safety.md) · [LoRA/SF 配方与代码](arena-sf-training.md) · [操作与详细技术归档](run/README.md)

`reports/` 保存可核对的 CSV、图表、配置和小型诊断；`runs/` 保存本地逐步日志、checkpoint 与视频。GitHub 备份代码、文档和小型报告，不上传权重或场景资产；scratch 中的 checkpoint 不等于长期备份。
