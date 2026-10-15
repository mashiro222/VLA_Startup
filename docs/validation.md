# 提交前检查记录

2026-10-15 Arena 通信修复、25 项测试和新单卡 smoke 的记录见 [专门说明](arena-transport-fix-2026-10-15.md)。所有实验现状见 [结果总览](results-overview.md)；下文保留最初 LIBERO 准备阶段的检查。

## 后续实际 GPU 验证与并行准备

`smoke-clean-001`，Slurm Job `13965730`，在 2026-10-11 正常结束：`COMPLETED`、`ExitCode=0:0`、运行 13 分 02 秒、1 张 A100 / 8 CPU / 64 GB，记账 MaxRSS 为 `26506992K`（约 25.3 GiB 主机内存）。4/4 episode 均成功，缺失为 0；真实结果在 [reports/smoke-clean-001](../reports/smoke-clean-001)。这验证了原版完整权重加载、JIT、EGL 和闭环运行，不能用四条记录推断全量成功率，也尚不能替代 Plus smoke。

`smoke-plus-001`，Slurm Job `13966714`，在 2026-10-11 21:07:39–21:22:11（EDT）运行并正常结束：`COMPLETED`、`ExitCode=0:0`、14 分 32 秒、1 张 A100 / 8 CPU / 64 GB，MaxRSS `25483756K`（约 24.3 GiB 主机内存）。七类扰动各有一条完整记录，6 次成功，Robot Initial States 用例达到 220 步上限后记为任务失败；没有 `errors.jsonl`。真实表格、图、压缩记录和校验和见 [reports/smoke-plus-001](../reports/smoke-plus-001)。

Plus 两段视频分别有 126 / 220 帧、10 fps、224×224；抽查首末画面正常，成功用例最终把碗放入盘子，失败用例未完成放置。报告 SHA256、运行时记录的评测器源码指纹均已核对，`parallel.py check fullplus-4gpu-001 --require-smoke` 通过。说明七个代表性用例已跑通，不保证其它 10,023 个变体和其它 suite 均无异常；smoke 的 85.71% 不能当作 full-plus 分数。

由文件时间和逐条记录复核：启动至 `run_config.json` 约 668 秒，配置到最后一条记录约 196 秒；七条 episode 内部计时总和 153.86 秒（不包含外部环境构造），其中推理 77.31 秒。首条包含首次推理编译，不能用整份 smoke 的 14 分 32 秒直接按 episode 数线性放大全量耗时。

后续四卡方案使用四个单 GPU 数组成员，full-clean 分为每片 500 episodes，full-plus 分为 2507/2508/2507/2508。支持 `--benchmarks plus/clean/both`，当前优先方案只跑 full-plus。原版与 Plus 评测器本身未因分片而修改；每片独立恢复 RNG，合并明确保留来源。准备和提交只做预览与 Slurm `--test-only`，未执行四卡计费提交。具体操作与排队限制见 [overnight.md](overnight.md)。

并行功能加入后，仿真 Python 下 **22 项测试全部通过**，包括 plus-only 计划不含 clean、分片覆盖、拒绝不完整/重复/不同配置的合并、CSV/PNG 生成，以及使用假 Slurm 命令的提交保护测试。shell 语法和 Python 编译检查通过；真实 smoke-clean 报告文件 SHA256 全部核对通过。测试未创建任何真实 Slurm 作业。

## 初始准备阶段记录

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

## 初始准备阶段验证范围

场景资产已完整校验、解压；最终 `preflight.py --imports` 检查通过。`smoke-clean-001` 和 `smoke-plus-001` 的任务清单已准备好，提交命令只做了 dry-run。

实际预览：[原版 smoke](../reports/preparation/submit-smoke-clean-dry-run.txt)、[Plus smoke](../reports/preparation/submit-smoke-plus-dry-run.txt)。两份均使用 `gts-jli741-paid` / `gpu-a100` / `inferno` / 1 张 A100 / 8 CPU / 64 GB / 45 分钟。未执行 `--submit`。

补充检查：登录节点启动保护会拒绝没有 Slurm allocation 的评测进程；MP4 编码器可以生成并删除小型测试视频。运行中报告导出会固定同一份 episode 快照，已用“导出期间追加记录”的测试验证 CSV、Markdown 与压缩 journal 的一致性。独立 preflight 的缓存路径也已统一到 scratch。

准备阶段的版本、任务数量、资产校验和检查元数据保存在 `reports/preparation/`，不是 benchmark 结果。

尚未验证：计算节点的驱动/CUDA 可用性、完整权重加载、显存占用、JIT 编译、EGL 实际渲染、闭环任务成功率、GPU 时间和 credit 消耗。这些项目需要通过显式 `--submit` 提交的 GPU smoke 验证。

本地详细检查日志在忽略的 `.runtime/` 中：`server-import-check.log`、`model-adapter-check.log`、`sim-import-check.log`、`clean-import-check.log`、`tests.log`、`download.log`。源码重建、任务协议和检查步骤保存在 GitHub；大文件不上传。
