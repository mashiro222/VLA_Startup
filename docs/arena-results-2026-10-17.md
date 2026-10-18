# VLA-Arena Safety：pi05_libero 迁移评测结果

## 完成情况

固定 LIBERO 微调 checkpoint `pi05_libero`，无 Arena 微调。`arena-balanced1500-001` 覆盖 5 个 suite × 3 个 level × 5 个任务 × 20 个初始状态，共 **1,500 episodes**。全部完成，成功 **172/1,500（11.47%）**，平均累计安全代价 **CC=60.24**，无基础设施错误。所有 episode 均有逐谓词 cost 记录，分项之和与原始 CC 一致。

数组作业 `14143493` 的 4 个单 A100 worker 均为 `COMPLETED / 0:0`。2026-10-17 20:34:42 EDT 开始分配首个 worker，22:00:40 全部结束，从首个启动到最后一个结束约 1 小时 26 分钟；各 worker 用时约 1 小时 21–23 分钟。计费账户为 `gts-jli741-paid`。

## 分组结果

每个 suite 含 300 条，每个 level 含 100 条。

| Suite | 成功 / 完成 | SR | 平均 CC |
|---|---:|---:|---:|
| Static obstacles | 51 / 300 | 17.00% | 82.26 |
| Dynamic obstacles | 3 / 300 | 1.00% | 144.14 |
| Cautious grasp | 11 / 300 | 3.67% | 21.60 |
| Hazard avoidance | 3 / 300 | 1.00% | 4.61 |
| State preservation | 104 / 300 | 34.67% | 48.57 |
| **总体** | **172 / 1,500** | **11.47%** | **60.24** |

这是完整的 balanced1500 协议结果，不是旧 2,250 条 full 协议，也不是 Arena 微调模型的论文复现。之前 15 条 smoke 的 SR=0% 不能替代此结果或用于预测整体 SR。

## 失败原因与代价来源

全部 **1,328 条失败**均以 `horizon_exhausted` 结束，即达到 300 个策略控制步仍未完成任务；未出现通信、渲染或计分一致性错误。此标签描述终止条件，不能单凭它断言抓取、规划或放置失败的物理原因。

Static 和 Dynamic 的平均 CC 分解如下；每列按对应 suite 的全部 300 条计算。

| 原始谓词 | Static CC 贡献 | Dynamic CC 贡献 |
|---|---:|---:|
| `CheckGripperContact`（夹爪接触） | 67.13 | 139.67 |
| `InContact`（对象接触） | 11.27 | 1.88 |
| `Fall`（掉落） | 3.87 | 2.60 |
| **总 CC** | **82.26** | **144.14** |

分项显示这两个 suite 中夹爪接触贡献最大。CC 累积接触持续步数和终态代价，不是独立碰撞事件的次数；显示值四舍五入后可能存在 0.01 的加总差异。

| 任务结果 | 零 CC | 正 CC |
|---|---:|---:|
| 成功 | 120 | 52 |
| 失败 | 264 | 1,064 |

低 SR 与安全代价应分别分析：264 条失败的 CC 为零，不能将不完成任务解释为安全能力提高。后续 baseline/SF 比较应使用相同任务、初态和计分规则，同时保留 SR、CC、成功/失败分层及接触持续步数。

## 表格、图与原始记录

- [完整结果与逐 level 比较](../reports/arena-balanced1500-001/summary.md)
- [SR/CC 曲线](../reports/arena-balanced1500-001/safety_by_level.png)
- [代价构成图](../reports/arena-balanced1500-001/cost_attribution.png)
- [逐 episode 表](../reports/arena-balanced1500-001/episodes.csv)
- [失败原因表](../reports/arena-balanced1500-001/failure_modes.csv)
- [逐谓词归因](../reports/arena-balanced1500-001/cost_components.csv)
- [成功/失败与代价分层](../reports/arena-balanced1500-001/outcomes.csv)

`reports/arena-balanced1500-001/` 还保存冻结计划、GPU/代码来源、压缩逐 episode 记录及 SHA256 清单。视频、逐步轨迹和图像样本保存在 `runs/arena-balanced1500-001/units/`，不上传模型或场景资产。
