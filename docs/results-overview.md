# 实验结果与未完成项目总览

检查日期：2026-10-17，时区 America/New_York（EDT）。作业实时变化需查看 Slurm。已有实测模型均为官方 LIBERO 微调后的 **`pi05_libero`**，Arena LoRA / Spatial Forcing 训练配置与状态见[训练说明](arena-sf-training.md)。

## 实验结果

| 实验 | 有效完成 / 计划 | 成功 / 有效完成 | 结果说明 | 详细文件 |
|---|---:|---:|---|---|
| LIBERO smoke-clean | 4 / 4 | 4 / 4 | 链路检查，不是正式分数 | [报告](../reports/smoke-clean-001/summary.md) |
| LIBERO-Plus smoke | 7 / 7 | 6 / 7 | 七类代表用例，无程序异常 | [报告](../reports/smoke-plus-001/summary.md) |
| **LIBERO-Plus full** | **10,030 / 10,030** | **8,477 / 10,030 = 84.52%** | 本次完整鲁棒性评测结果 | [完整报告](../reports/fullplus-4gpu-001-full-plus/summary.md) |
| LIBERO clean full（部分） | 1,500 / 2,000 | 1,446 / 1,500 = 96.40% | 只代表已完成样本，不能称 full 最终分数 | [片 00](../reports/fullclean-4gpu-001-full-clean-shard-00/summary.md)、[片 01](../reports/fullclean-4gpu-001-full-clean-shard-01/summary.md)、[片 03](../reports/fullclean-4gpu-001-full-clean-shard-03/summary.md) |
| Arena 首轮 smoke | 0 / 15 | N/A | 通信故障，没有有效成功率 | [失败诊断](../reports/arena-diagnostics-2026-10-15/summary.md) |
| Arena 首轮 full | 0 / 2,250 | N/A | 四个 worker 都在首个动作返回前退出 | 同上 |
| Arena 修复后 smoke `arena-smoke-002` | 15 / 15 | 0 / 15 = 0% | 通信正常；全部达到 300 步上限，平均 CC 41.05；仅为 smoke 样本 | [结果](../reports/arena-smoke-002/summary.md)、[录像与接口诊断](arena-smoke-audit-2026-10-16.md) |
| Arena 归因 smoke `arena-smoke-003` | 15 / 15 | 0 / 15 = 0% | 逐谓词记录覆盖 15/15；均达 300 步，平均 CC 33.33，无基础设施错误 | [结果](../reports/arena-smoke-003/summary.md)、[终止与代价](../reports/arena-smoke-003/diagnostics.md) |

Clean 第 2 片原作业 `13974531_2` 在模型启动 900 秒限制下失败；补跑 `13988230_2` 于 **10 月 15 日 08:26:46–08:41:54** 再次以相同 `Policy startup timed out` 退出，没有新增有效 episode。启动超时的根因尚未确定；另外三片 H100 已运行成功，现有证据不足以认定 GPU 不兼容。该分片暂不继续补跑，clean 结果保留为部分覆盖。此问题与 Arena 的 WebSocket 心跳故障不同。历史记录与日志路径见 [clean 运行记录](full-clean-2026-10-12.md)。

## Plus 的表和图

结果目录：`reports/fullplus-4gpu-001-full-plus/`。

| 文件 | 用途 |
|---|---|
| [summary.md](../reports/fullplus-4gpu-001-full-plus/summary.md) | 实验结果汇总 |
| [summary.csv](../reports/fullplus-4gpu-001-full-plus/summary.csv) | 按 suite、类别、难度的表，导入 Excel/Numbers |
| [episodes.csv](../reports/fullplus-4gpu-001-full-plus/episodes.csv) | 10,030 条逐任务结果、prompt、步数、耗时 |
| [success_by_group.png](../reports/fullplus-4gpu-001-full-plus/success_by_group.png) | 分组成功率柱状图 |
| [cumulative_success.png](../reports/fullplus-4gpu-001-full-plus/cumulative_success.png) | 累计成功率折线；不是训练 loss，也不是墙钟进度 |
| `manifest.json`、`run_config.json`、`provenance-*.json`、`sha256.json` | 任务清单、版本/参数、来源和校验 |

按扰动类别，成功率为：背景 96.19%、光照 96.41%、噪声 86.76%、布局 86.62%、语言 85.49%、机器人初态 74.45%、相机视角 72.73%。在本评测协议下，该 checkpoint 对相机视角和机器人初态变化更敏感。这一观察来自分组统计，尚不能归因于特定网络模块。

总体 84.52% 比社区自报的 84.2% 高约 0.32 个百分点；该参考值来自社区 issue，不是官方排行榜。同名 checkpoint 不保证环境、prompt、随机流等协议完全一致，细节见 [比较边界](experiment.md#公开结果对照)。Clean 的部分 96.40% 和 Plus 的完整 84.52% 不构成严格配对的鲁棒性对照。

## Arena 实验与诊断记录

- [首轮失败总结](../reports/arena-diagnostics-2026-10-15/summary.md)：五个 worker 的时间、状态和共性故障。
- [结构化诊断](../reports/arena-diagnostics-2026-10-15/summary.json)与 [原始异常](../reports/arena-diagnostics-2026-10-15/errors.jsonl)：可追溯证据。
- [修复与测试说明](arena-transport-fix-2026-10-15.md)：改动、回归测试和新 smoke 的具体查看命令。
- [Arena 协议](arena-safety.md)：训练条件、五类 × 三级、SR/CC、与论文结果的比较限制。
- [Smoke 录像与接口诊断](arena-smoke-audit-2026-10-16.md)：15 条失败的可见行为、分级 CC、接口检查和证据局限。
- [1,500 条评测与 cost 明细](arena-diagnostics.md)：新初态协议、终止原因、逐谓词 cost 与接触持续时间。新记录器已通过 `arena-smoke-003` 的单 A100 运行验证。
- [LoRA / Spatial Forcing 对照设计](arena-spatial-forcing-plan.md)：从同一 `pi05_base` 出发的两组训练；[训练实现及提交记录](arena-sf-training.md)列出配置、验证与恢复方式。

本地原始文件：

| 路径 | 内容 |
|---|---|
| `runs/arena-smoke-001/plan.json`、`runs/arena-full-001/plan.json` | 原失败实验的冻结计划 |
| `runs/arena-smoke-002/plan.json` | 修复后的新计划；不合并到旧 run |
| `runs/arena-*/units/*/episodes.jsonl` | 完成的有效 episode，每条刷盘 |
| `runs/arena-*/units/*/errors.jsonl` | 运行时错误，不算入模型失败分母 |
| `runs/arena-*/workers/*.json`、`*-server.log` | 作业/GPU/版本来源和模型服务日志 |
| `logs/arena-*.out` | Slurm stdout/stderr |
| `runs/arena-*/units/*/videos/*.mp4` | 实际轨迹录像，本地保留 |

`reports/arena-smoke-003/` 包含 15/15 条逐谓词诊断，新增 `failure_modes.csv`、`cost_components.csv`、`cost_by_category.csv`、`cost_attribution.png`。均在 horizon 结束只说明终止条件，不能直接确定物理失败原因。

`reports/arena-smoke-002/` 已包含汇总、逐 episode 表、终止原因及 `safety_by_level.png`、`cumulative_success.png`。旧 smoke 没有逐谓词记录，诊断覆盖率标为 0；不从录像编造 cost 分解。

## 配对训练状态

2026-10-17：Arena LoRA baseline `14127636` 与 LoRA+SF `14127637` 已提交，各 2 张 A100。两组从 `pi05_base` 初始化，均依赖短训练验证 `14127618` 成功后启动，当前没有训练后评测成绩。配置、检查证据与监控命令见[训练说明](arena-sf-training.md)。

## 总结

迁移评测使用官方 LIBERO 微调的 π0.5 固定 checkpoint。LIBERO-Plus 的 10,030 条测试已全部完成，成功率为 84.52%；七类扰动中，相机视角和机器人初态变化下的表现相对较弱。原版 LIBERO 对照完成 1,500/2,000 条，已完成部分成功率为 96.40%，完整评测尚未完成。

VLA-Arena Safety 用于评估同一 checkpoint 的 zero-shot 迁移表现。首轮实验因通信超时未产生有效结果；修复后单卡 smoke 完成 15 条，SR 0%、平均 CC 41.05。现有证据更符合任务迁移不足，但不能排除全部适配问题，也不能用低 SR 单独证明安全性差。后续采用固定 1,500 条协议，并对 static/dynamic 的接触与终态代价进行分解；LoRA 与 LoRA+SF 为独立的后续对照实验。

实际录像截图见 [场景图像](../reports/presentation-2026-10-15/README.md)。Arena 初始截图仅验证场景渲染，不代表动作执行或任务完成。小型报告随仓库备份；checkpoint、场景资产和视频保留在本地。
