# 提交前检查记录

检查日期：2026-10-11，PACE 登录节点。准备阶段未提交 Slurm 作业，未进行训练、GPU 推理或真实 benchmark rollout。下列准备检查不能代替 GPU smoke。

| 检查 | 实际结果 |
|---|---|
| 环境安装 | uv 两套环境已完成；server Python 3.11.17、sim Python 3.10.22 |
| OpenPI 导入 | CPU 模式成功导入模型配置、policy factory、WebSocket server；JAX 0.5.3 |
| 模型配置 | `pi05=True`、action horizon 10、内部 action dim 32、`discrete_state_input=False` |
| 输入输出 adapter | 本地 tokenizer、norm stats 可读取；小型测试输入产生 200 个 prompt token 位置，动作 adapter 输出 `(10, 7)`；未加载神经网络权重 |
| 仿真导入 | 原版和 Plus 的 benchmark、OffScreenRenderEnv 导入成功；未创建或渲染场景 |
| 初始状态读取 | 原版和 Plus 的第一个 spatial 任务均读取到 `(50, 92)` 数组 |
| Wand 原生库 | ImageMagick 7.1.1-28 成功加载，并完成 16×16 小图的 motion blur 检查 |
| robosuite 日志 | 已用版本约束、可重复的补丁解决共享 `/tmp/robosuite.log` 权限冲突 |
| 测试 | 仿真 Python 下 13 项测试全部通过，包括真实 CSV/PNG 导出路径的临时单元测试 fixture |
| 脚本静态检查 | Python 编译和全部 shell/Slurm 脚本 `bash -n` 通过 |
| 任务清单 | 4 / 7 / 56 / 2,000 / 10,030 次五种计划生成成功；所有选中任务 BDDL 与初始状态路径存在 |
| 上游源码 | 三个固定 commit 均匹配，跟踪源码无改动 |
| Checkpoint | 全部 16 个对象通过固定 generation、大小、CRC32C 校验；tokenizer SHA256 通过 |
| 场景资产 | 6,395,849,578 字节 ZIP 的 SHA256 通过；448,799 个文件解压和 CRC 检查完成 |
| Git 排除规则 | checkpoint、场景 ZIP、运行日志、local.env 均匹配 ignore；仅小型代码与文档加入版本控制 |

测试中的 fixture 是临时合成输入，只用于检查汇总与图文件格式，运行后删除；它不属于任何模型评测成绩，未发布到 `reports/`。

## 交付确认

场景资产已完整校验、解压；最终 `preflight.py --imports` 检查通过。`smoke-clean-001` 和 `smoke-plus-001` 的任务清单已准备好，提交命令只做了 dry-run。

实际预览：[原版 smoke](../reports/preparation/submit-smoke-clean-dry-run.txt)、[Plus smoke](../reports/preparation/submit-smoke-plus-dry-run.txt)。两份均使用 `gts-jli741-paid` / `gpu-a100` / `inferno` / 1 张 A100 / 8 CPU / 64 GB / 45 分钟。未执行 `--submit`。

补充检查：登录节点启动保护会拒绝没有 Slurm allocation 的评测进程；MP4 编码器可以生成并删除小型测试视频。运行中报告导出会固定同一份 episode 快照，已用“导出期间追加记录”的测试验证 CSV、Markdown 与压缩 journal 的一致性。独立 preflight 的缓存路径也已统一到 scratch。

准备阶段的版本、任务数量、资产校验和检查元数据保存在 `reports/preparation/`，不是 benchmark 结果。

尚未验证：计算节点的驱动/CUDA 可用性、完整权重加载、显存占用、JIT 编译、EGL 实际渲染、闭环任务成功率、GPU 时间和 credit 消耗。必须由用户在本地说明中的显式 `--submit` 命令启动 smoke 后确认。

本地详细检查日志在忽略的 `.runtime/` 中：`server-import-check.log`、`model-adapter-check.log`、`sim-import-check.log`、`clean-import-check.log`、`tests.log`、`download.log`。源码重建、任务协议和检查步骤保存在 GitHub；大文件不上传。
