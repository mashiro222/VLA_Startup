# π0.5 × LIBERO-Plus / Arena Safety：PACE 评测

**下午汇报先看 [结果总览](docs/results-overview.md)，再看 [研究阅读导航](docs/research-guide.md)。** 后者集中解释文件分工、training / inference 的修改位置、benchmark 测试内容、以及怎样生成具体测试的图像。

使用官方 LIBERO 微调的 `pi05_libero` checkpoint，不训练，不使用 W&B。结果保存在本地表格和图中；GitHub 备份代码、说明及小型报告，不上传模型权重、场景资产或视频。

**2026-10-15 当前结果：** LIBERO-Plus 完整完成，8,477 / 10,030 成功（84.52%）；clean 完成 1,500 / 2,000，已完成部分为 96.40%。Clean 第 2 片的旧补跑 `13988230_2` 今天再次模型启动超时，没有新增结果，按用户决定暂不补跑。见 [结果总览与表图](docs/results-overview.md)。

**Arena：** 首轮 smoke/full 因心跳超时全部退出，没有有效成功率；已修复并通过 25 项回归检查。10 月 15 日 10:15 EDT 已提交新的 **单 A100 smoke `14074038_0` / `arena-smoke-002`**，账户 `gts-jli741-paid`，15 条、2 小时时限，初次核对为 `PENDING (Priority)`。**尚未验证 GPU 闭环，本轮没有重新提交 full。** 查看 [修复、新作业与测试记录](docs/arena-transport-fix-2026-10-15.md)和 [Arena 运行说明](docs/arena-safety.md)。

| 想做什么 | 阅读入口 |
|---|---|
| 准备汇报、找结果和未完成实验总结 | [docs/results-overview.md](docs/results-overview.md) |
| 看懂代码；改训练/推理参数；看具体测试；导出图片 | [docs/research-guide.md](docs/research-guide.md) |
| 理解 LIBERO/Plus 的闭环、任务与比较协议 | [docs/experiment.md](docs/experiment.md)、[docs/code-map.md](docs/code-map.md) |
| 理解 Arena 五类安全任务、SR/CC、zero-shot | [docs/arena-safety.md](docs/arena-safety.md) |
| 直接取用真实场景截图 | [reports/presentation-2026-10-15/README.md](reports/presentation-2026-10-15/README.md) |
| 查看全部 Slurm 作业、LIBERO 四卡方案 | [docs/overnight.md](docs/overnight.md) |

下方保留 LIBERO 初始搭建和运行手册；已有完成的 run 不需要重新提交。Arena 使用独立入口，见上方链接。

## 1. 当前目录与环境

PACE 工作目录：`/storage/scratch1/1/gliu386/VLA_Startup`。GitHub 备份：[mashiro222/VLA_Startup](https://github.com/mashiro222/VLA_Startup)。

用 VS Code 阅读时，通过 **File → Open Folder** 只打开上面这个项目目录，不要把整个 home 或 scratch 作为工作区。本仓库的 `.vscode/settings.json` 会减少大目录监听和 Python 索引；若反复 reconnect，见 [VS Code 连接排查](docs/troubleshooting.md#vscode-reconnect)。

| 内容 | 位置 | 管理方式 | 上传 GitHub |
|---|---|---|---|
| 自己的评测代码、配置、文档 | `src/`、`scripts/`、`config/`、`docs/` | Git | 是 |
| OpenPI 模型环境 | `.runtime/server/`，Python 3.11 | uv + 上游锁文件 | 否 |
| LIBERO 仿真环境 | `.runtime/sim/`，Python 3.10 | uv + `requirements/sim.lock` | 否 |
| 上游代码 | `.runtime/repos/` | `config/sources.json` 中固定 commit | 否，保存可重建的版本信息 |
| ImageMagick 原生库 | `.runtime/native-7.1.1/` | 官方 AppImage 解包、SHA256 校验 | 否 |
| checkpoint、场景 | `.runtime/checkpoints/`、`.runtime/repos/libero_plus/` | 专门下载脚本、固定版本及校验和 | **否** |
| 原始运行日志、视频 | `runs/`、`logs/` | 本地 | 否 |
| 小型评测报告 | `reports/RUN_ID/` | CSV/JSON/Markdown/PNG/压缩 journal | 运行后按需备份 |

uv 管理 Python 和依赖；模型权重与场景是数据资产，由 `download.sh` 下载。最终流程不依赖 conda。Wand 所需 ImageMagick 是原生共享库，不是 uv 能独立替代的 Python 包。OpenPI 上游依赖中可能安装 wandb，但本项目没有初始化 W&B，且显式禁用其运行模式。

新机器或 scratch 被清除后，在登录节点重建：

```bash
cd /storage/scratch1/1/gliu386
git clone https://github.com/mashiro222/VLA_Startup.git
cd VLA_Startup
bash scripts/setup.sh
bash scripts/download.sh
python3 scripts/preflight.py --imports
```

当前已经准备过的目录不需要重新 clone。安装与下载可重复执行，完成的下载会校验后复用。权重约 12.4 GB，场景 ZIP 约 6.4 GB；另需解压空间、源码、两套环境与下载缓存。不要仅按这两个文件的体积预留空间。下载不需要训练数据集。

## 2. 第一份计费作业如何提交

先进入仓库。以下前两条脚本命令只准备和预览，不申请资源：

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
bash scripts/prepare.sh smoke-clean smoke-clean-001
bash scripts/submit.sh smoke-clean-001
```

确认预览使用 `--account=gts-jli741-paid --partition=gpu-a100 --qos=inferno --gres=gpu:a100:1`。默认 8 CPU、64 GB 内存，smoke 时限 45 分钟。**下面这条才会真的提交计费作业，由你执行：**

```bash
bash scripts/submit.sh smoke-clean-001 --submit
```

记录返回的 Job ID。原版 smoke 完成后再检查 Plus 的七种扰动：

```bash
bash scripts/prepare.sh smoke-plus smoke-plus-001
bash scripts/submit.sh smoke-plus-001
# 准备开始下一份计费作业时执行：
bash scripts/submit.sh smoke-plus-001 --submit
```

smoke 的通过条件是模型与环境成功运行、所有选定 episode 都有有效成功/失败记录、没有基础设施错误。不能要求少量 episode 的成功率接近完整 benchmark，也不能把 smoke 分数当作论文结果。前两个 episode 默认保存视频。

## 3. 从 smoke 到完整评测

| Profile | 内容 | Episode 数 | 默认时限 | 用途 |
|---|---|---:|---|---|
| `smoke-clean` | 原版四个 suite 各一个任务 | 4 | 45 分钟 | 验证模型、环境、相机和动作 |
| `smoke-plus` | spatial 中七类扰动各一个变体 | 7 | 45 分钟 | 验证 Plus 资产与扰动实现 |
| `pilot-plus` | 每个 suite、每类扰动选两个不同基础任务 | 56 | 4 小时 | 检查覆盖并测量速度 |
| `full-clean` | 原版 40 个任务，每个 50 次 | 2,000 | 24 小时 | 对照 OpenPI 官方结果 |
| `full-plus` | 所有 10,030 个变体，每个一次 | 10,030 | 24 小时 | 完整 Plus 鲁棒性评测 |

时限是作业上限，**不是运行耗时承诺**。pilot 可以帮助估计速度，但不是必须步骤。当前按 full-plus 优先，Plus smoke 检查通过后已提交[四卡 full-plus 数组](docs/overnight.md#运行命令)；full-clean 的最新安排见[2026-10-12 说明](docs/full-clean-2026-10-12.md)。七条 smoke 只能验证代表性用例，不能保证全量无异常。下面保留单卡 pilot/full 的备选流程，不要与四卡数组重复提交。

```bash
bash scripts/prepare.sh pilot-plus pilot-plus-001
bash scripts/submit.sh pilot-plus-001
# 检查预览后，由你执行
bash scripts/submit.sh pilot-plus-001 --submit

# pilot 完成并检查报告后，分别准备完整运行
bash scripts/prepare.sh full-clean full-clean-001
bash scripts/submit.sh full-clean-001
bash scripts/prepare.sh full-plus full-plus-001
bash scripts/submit.sh full-plus-001
# 以上两份完整作业尚未提交；各自加 --submit 才会提交。
```

粗略时间估计：`预计剩余小时 ≈ 剩余 episode × pilot 平均 episode 秒数 / 3600`，并预留模型启动、JIT、任务切换时间。不同任务 horizon、失败率会影响耗时，56 次 pilot 也不一定代表全部变体。按组内当期 PACE 计费规则换算 credit；本仓库不编造价格。2026-10-12 用户允许 full-plus 与 full-clean 两份四卡数组并行，最多同时 8 张 GPU。

修改资源请写入忽略的 `config/local.env`。仅改时限可用：

```bash
TIME_LIMIT=08:00:00 bash scripts/submit.sh full-plus-001
# 确认后在同一命令末尾加 --submit
```

## 4. 查看进度、中断和续跑

```bash
squeue -u "$USER"
# 将 JOB_ID 换成 sbatch 返回的数字
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,MaxRSS
tail -f logs/smoke-clean-001-JOB_ID.out
# 模型启动详细日志位于 runs/smoke-clean-001/server-JOB_ID.log
```

评测每完成一个 episode 就追加并刷盘到 `runs/RUN_ID/episodes.jsonl`，异常单独记到 `errors.jsonl`。超时、模型/渲染错误不会被悄悄统计为任务失败。作业正常结束或收到可处理的终止信号后自动汇总；强制杀死时可手工重建报告。

主动停止用 `scancel JOB_ID`。确认旧作业已结束后，重新执行相同 run ID 的 `submit.sh ... --submit` 即续跑；已完成 episode 不重复计数。保存了 JAX 和 NumPy RNG 状态，未完成 episode 重做。相同 run ID 不允许更改 checkpoint、评测代码或参数；需要更改实验时用新 ID。跨硬件、驱动或依赖修改不保证逐位一致。

## 5. 本地表格和图，以及小文件备份

```bash
bash scripts/report.sh smoke-clean-001
# 查看报告后，备份这一份小型结果到 GitHub：
bash scripts/publish_results.sh smoke-clean-001
```

`report.sh` 不提交作业，不上传文件；`publish_results.sh` 才会 Git commit/push 该报告目录。计算节点不会自动访问 GitHub。

| 文件 | 含义 |
|---|---|
| `reports/RUN_ID/episodes.csv` | 逐 episode：任务、扰动类型、成功、步数、耗时、真实 prompt |
| `summary.csv` / `summary.md` / `summary.json` | 总体、suite、扰动类别、难度统计，明确 completed/expected/missing |
| `cumulative_success.png` | 随完成 episode 数变化的累计成功率折线，**不是训练 loss 曲线** |
| `success_by_group.png` | 各扰动类别或原版 suite 成功率柱状图 |
| `episodes.jsonl.gz` | 压缩原始记录，保留 RNG 等审计信息 |
| `manifest.json`、`run_config.json`、`provenance-*.json` | 任务清单、版本、参数、作业与 GPU 信息 |
| `sha256.json` | 该报告的小文件校验和 |

无完成记录时不生成假数据图。未完成计划时显示已完成样本的成功率，完整分数留空。只有完整 profile、完整覆盖、默认 seed、prompt 和控制参数才列出参考值差异。折线受固定任务顺序影响，不代表模型在学习或不同 episode 独立同分布。

模型、场景 ZIP、环境、视频均被 `.gitignore` 排除。GitHub 可恢复代码和已备份报告，但不能恢复尚未备份的 scratch 日志；因此长任务的中间报告也可手工备份。恢复环境重新下载即可。

## 6. 学习与比较

先看 [docs/code-map.md](docs/code-map.md)：本次新增文件、参数修改入口、模型源码阅读顺序，以及 benchmark 的全部测试清单在哪里。常用控制参数集中在 `config/evaluation.json`。

按 [docs/experiment.md](docs/experiment.md) 阅读一次闭环中 model、benchmark、环境和汇总器各自的作用，以及原版/Plus 的参考分数和比较限制。实现入口是 `src/vla_startup/evaluate.py`，模型入口是 `src/vla_startup/server.py`。

本地检查（不启动模型，不申请 GPU）：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_results.py -v
python3 scripts/preflight.py --imports
# Arena 的独立环境与真实本机通信回归：
bash scripts/test_arena.sh
```

常见环境和作业问题见 [docs/troubleshooting.md](docs/troubleshooting.md)。
