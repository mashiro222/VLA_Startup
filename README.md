# π0.5 × LIBERO-Plus / Arena Safety

本仓库包含 π0.5 在 LIBERO、LIBERO-Plus 和 VLA-Arena Safety 上的闭环评测，以及从 `pi05_base` 开始的 Arena LoRA / Spatial Forcing 配对训练。迁移评测使用官方 `pi05_libero`；训练与迁移结果分开报告。

**所有实验记录统一见 [docs/results-overview.md](docs/results-overview.md)**，按时间记录结果、错误、修复和调整原因。详细提交命令与技术归档位于 [docs/run/](docs/run/README.md)。

| 内容 | 入口 |
|---|---|
| 实验记录、结果与表图 | [实验记录](docs/results-overview.md) |
| 模型、training/inference 参数、benchmark 定义、图像导出 | [开发指南](docs/research-guide.md) |
| 从 image / instruction 到动作；action expert、Flow Matching 与微分方程基础 | [π0.5 原理学习文档](docs/learning/05-pi05-flow-matching.md) |
| LIBERO 四类任务、Plus 扰动与参考协议 | [实验说明](docs/experiment.md)、[代码地图](docs/code-map.md) |
| Arena Safety 任务、SR/CC 与迁移比较边界 | [Arena 协议](docs/arena-safety.md) |
| 失败原因与 cost 归因 | [诊断协议](docs/arena-diagnostics.md) |
| LoRA / SF 假设、数据、损失、参数冻结与代码 | [对照设计](docs/arena-spatial-forcing-plan.md)、[训练说明](docs/arena-sf-training.md) |
| 24 小时 checkpoint、同配置恢复与扩卡限制 | [checkpoint 说明](docs/run/training-checkpoints.md) |
| 24 小时后手动评测、四卡各追加 15,000 步 | [后续实验流程](docs/run/after-24h-followup.md) |
| 环境搭建、提交、监控与报告导出 | [操作目录](docs/run/README.md) |
| 环境、Slurm、VS Code 问题 | [故障排查](docs/troubleshooting.md) |

PACE 工作目录：`/storage/scratch1/1/gliu386/VLA_Startup`。VS Code 建议仅打开该项目目录。

Python 环境由 uv 管理；上游版本、依赖和资产校验信息随仓库保存。结果输出为本地表格和图，不使用 W&B。GitHub 保存代码、说明与小型报告；模型、训练数据、场景、环境和视频保留在本地，不上传。

已有实验无需重复提交。重建或开展新实验时，从对应运行手册开始；修改被冻结的代码、参数或数据后使用新的 run ID。
