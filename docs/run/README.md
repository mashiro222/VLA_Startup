# 操作与技术归档

**完整实验时间线、结果、错误和调整原因：[实验记录](../results-overview.md)。** 本目录保存运行命令和详细证据说明；历史队列状态只对应文中时间，不是当前作业状态。

| 内容 | 文档 |
|---|---|
| 24 小时后手动评测、四卡各追加 15,000 步 | [当前后续实验流程](after-24h-followup.md) |
| SF 同步超时、3k 恢复、900-episode 评测与两组 18k 保留 | [2026-10-20 恢复记录](sf-recovery-2026-10-20.md) |
| Baseline 评测 600 秒通信超时、保留 341 条并补跑 559 条 | [2026-10-21 评测恢复](baseline-eval-resume-2026-10-21.md) |
| 已停用：固定 5k / 20k、总并发 8 卡 | [旧流程归档](microbatch-and-20k.md) |
| 24 小时保存、恢复检查、扩卡限制与本次资源试算 | [训练 checkpoint](training-checkpoints.md) |
| LoRA / SF 环境、提交、同配置续训与导出 | [训练运行手册](training-operations.md) |
| Arena 环境、评测提交、结果与记录器重建 | [Arena 运行手册](arena-evaluation.md) |
| LIBERO/Plus 环境、单卡运行、报告与备份 | [LIBERO 运行手册](libero-evaluation.md) |
| LIBERO 四卡分片与查询命令 | [四卡评测手册](overnight.md) |
| LIBERO 初始环境检查与 smoke | [检查归档](validation.md) |
| Clean 资源选择、缺失分片与启动超时 | [2026-10-12 clean 归档](full-clean-2026-10-12.md) |
| Arena 首轮 full 代码复查和排队试算 | [2026-10-13 归档](arena-full-review-2026-10-13.md) |
| Arena 通信故障修复与测试 | [2026-10-15 归档](arena-transport-fix-2026-10-15.md) |
| Arena smoke 录像、图像、动作与 cost 审计 | [2026-10-16 归档](arena-smoke-audit-2026-10-16.md) |
| Arena 1,500 条分组结果与 cost 明细 | [2026-10-17 详细报告](arena-results-2026-10-17.md) |

模型、训练目标、任务与指标的长期说明保留在 `docs/`；可核对的数值、图和机器可读证据保留在 `reports/`。
