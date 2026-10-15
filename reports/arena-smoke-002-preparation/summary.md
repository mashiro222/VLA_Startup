# Arena 修复后的单卡 smoke：准备与提交记录

2026-10-15 10:15 EDT 提交：`arena-smoke-002`，Slurm **`14074038_0`**。首次核对为 `PENDING (Priority)`，尚无 episode，也没有有效成功率。

| 项目 | 值 |
|---|---|
| 账户 / QoS | `gts-jli741-paid` / `inferno` |
| 资源 | 1 张 A100、8 CPU、64 GB、2 小时时限 |
| 覆盖 | 5 类 Safety × 3 levels × task 0 × 1 条 = 15 |
| Checkpoint | 现有官方 LIBERO 微调 `pi05_libero`，无 Arena 微调 |
| 修复代码 | commit `32c5152` |
| CPU 检查 | 25 项 Arena 测试通过；服务端环境的 6 项真实通信回归也通过；75 项任务/import 检查通过 |
| GPU 验证 | 等待此新 smoke 实际完成，不以 CPU 测试代替 |
| Full | 尚未准备或提交新 full 作业 |

本目录备份 `plan.json`、`submission.json`、`checks.json`。旧 smoke/full 仍保留原始失败记录；新运行不续接旧计划。这个目录是准备记录，不是模型成绩报告。

[修复说明与查看命令](../../docs/arena-transport-fix-2026-10-15.md) · [结果总览](../../docs/results-overview.md) · [阅读导航](../../docs/research-guide.md)
