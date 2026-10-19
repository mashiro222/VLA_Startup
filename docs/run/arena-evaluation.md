# Arena 评测：环境与运行操作

实验结果、调整原因与状态统一见[实验记录](../results-overview.md)。以下保留阶段性命令，已结束的 run 不要重复提交。

## 2. 环境与依赖

| 内容 | 本地位置 | 用途 |
|---|---|---|
| Arena 源码与资产 | `.runtime/repos/vla_arena/` | 固定 commit `babe582ebffc82b979b77964a7e56417d02f63a4` |
| Arena 仿真环境 | `.runtime/arena/` | uv 管理，Python 3.11；独立于旧仿真环境 |
| 模型服务 | `.runtime/server/` | 复用已有 OpenPI/JAX 环境 |
| 模型权重及 normalization | `.runtime/checkpoints/pi05_libero/` | 复用已校验权重，不下载新模型 |
| Arena 配置路径 | `.runtime/config/arena/` | 隔离旧模块的 `LIBERO_CONFIG_PATH`，不写 home 中的 `.libero` |
| 原始记录、日志、视频 | `runs/arena-*/`、`logs/arena-*.out` | 留在本地，不进 GitHub |
| 小型报告 | `reports/arena-*/` | CSV、JSON、Markdown、PNG、压缩 episode journal |

仿真依赖锁在 `requirements/arena.lock`。robosuite 1.5.1、MuJoCo 3.5.0、bddl 3.6.0 对齐本次固定上游版本；Torch CPU 仅用于初始状态文件及客户端，模型仍在原 JAX GPU 服务中推理。安装的是仿真与客户端所需依赖，不安装其它模型或训练工具链。无需 conda、Docker、sudo 或 W&B。

新目录重建时，先按 README 重建现有 OpenPI 模型环境及 checkpoint，再执行：

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
bash scripts/setup_arena.sh
bash scripts/arena.sh preflight --imports
```

当前机器已经安装，无需为了阅读说明重装。Arena 资产随固定源码 checkout 一起下载，当前 assets 目录有 2,928 个文件、875,080,252 bytes；此外还有源码、初始状态及 Git 对象占用。`git fsck` 与干净工作树检查用于验证这份源码/资产快照。GPU 渲染、动作控制和推理仍必须经 smoke 作业验证。

本地准备记录见 [arena-readiness-2026-10-13.json](../../reports/preparation/arena-readiness-2026-10-13.json)，全部任务的名称、指令与初始状态数量见 [arena-task-catalog.csv](../../reports/preparation/arena-task-catalog.csv)。目前 25 项计分/断点/并发/提交/通信测试已通过；可用 `bash scripts/test_arena.sh` 重跑。它使用模拟环境和真实本机 WebSocket，不加载模型权重或申请 GPU。详见修复记录。

## 3. 当前 smoke 与后续计划

2026-10-16 更新：`arena-smoke-002` 已完成，15/15、SR 0%、平均 CC 41.05，通信超时未复现。[录像/接口审查](arena-smoke-audit-2026-10-16.md)和[分项诊断协议](../arena-diagnostics.md)记录其局限。当前后续计划采用 1,500 条，不提交旧 full。

| Profile | 覆盖 | 单元数 | 默认 worker 数 / 时限 |
|---|---:|---:|---|
| smoke | 每类每级选 task 0 × 1 = 15 | 15 | 1 / 2 小时 |
| pilot | 75 × 10 × 1 seed = 750 | 75 | 4 / 8 小时 |
| **balanced1500** | **75 × 20 个初态 × 1 seed = 1,500** | **75** | **4 / 8 小时** |
| full | 75 × 10 × 3 seeds = 2,250 | 225 | 4 / 12 小时 |

历史运行 `arena-smoke-002` 已结束。提交命令保留作记录，**不要重复提交**：

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
# 2026-10-15 已执行：
# bash scripts/arena.sh prepare smoke arena-smoke-002 --workers 1
# bash scripts/arena.sh submit arena-smoke-002 --gpu a100 --smoke-run arena-smoke-002 --submit

# 现在只查进度：
squeue -r -j 14074038 -o '%.20i %.12T %.24S %R'
sacct -j 14074038 --format=JobID%22,State,ExitCode,Elapsed
bash scripts/arena.sh check arena-smoke-002
tail -F logs/arena-smoke-002-14074038_0.out
```

Smoke 通过要求：15 条有效 episode、没有基础设施错误，不要求 15 次模型都成功。完成后运行 `bash scripts/arena.sh report arena-smoke-002`。未完成但已有记录可加 `--partial`；0 条记录不生成成功率图。

新的 1,500 条计划以匹配配置的 smoke 完成端到端验证为前提。新记录器更改了冻结代码，需要新的 smoke；以下命令用于准备与提交预览，**不申请 GPU 资源**：

```bash
bash scripts/arena.sh prepare smoke arena-smoke-003 --workers 1
bash scripts/arena.sh submit arena-smoke-003 --smoke-run arena-smoke-003 --gpu a100
bash scripts/arena.sh prepare balanced1500 arena-balanced1500-001
bash scripts/arena.sh submit arena-balanced1500-001 --smoke-run arena-smoke-003 --gpu a100
```

上述为历史准备命令，smoke003 与 balanced1500 已完成。重建使用新的 run ID；显式添加 `--submit` 才创建新作业。旧 `arena-full-001` 不用新代码续跑。

每个 Arena 单元为一项任务的一个评测 seed，full 单元含 10 条 episode。worker 完成一个单元后动态领取下一个，只排到一张卡也能继续领取。它不是 LIBERO 旧四卡方案的固定四分之一分片。GPU 资源选项为 `--gpu a100` 或 `--gpu h100`，每个 worker 各申请一张；时限用 `--time-limit` 覆盖。Arena 的 H100 路径尚未验证，GPU/时限选择不保证排队速度。

## 4. 断点续跑与结果保存

每个 episode 完成后立即写入并 fsync。单元使用文件锁，多个 worker 不会同时写同一单元；worker 被终止时锁释放。下次续跑跳过已完成记录，截断未写完整的最后一行，重新执行中断的 episode。已经排队或运行的同名任务会阻止重复提交。

若 sbatch 回复超时，会保留 `runs/<run>/submission/unknown.json`，并阻止盲目重试。先用 `squeue` / `sacct` 确认是否已提交成功；仅在确定没有该任务时再处理标记，不能仅凭报错重复提交。脚本不会自动取消其它作业。

旧数组已全部结束、错误已检查且没有修改冻结代码时，使用同一提交命令续跑。每个 worker 启动前核对冻结文件；如果需要修改评测代码、任务定义或关键参数，使用新 run ID，不把改变后的数据接到原成绩中。无需修改环境的临时排队或时限问题可原计划续跑。基础设施异常会记录到单元的 `errors.jsonl` 并使 worker 非零退出，不计入模型失败。

后续 1,500 条 episode 完成后，在登录节点合并、出图，不申请 GPU：

```bash
bash scripts/arena.sh report arena-balanced1500-001
```

不完整时默认拒绝生成完整报告。需要观察中途结果时可明确使用：

```bash
bash scripts/arena.sh report arena-balanced1500-001 --partial
```

输出目录 `reports/arena-balanced1500-001/`，新增终止原因和分项 cost 表详见 [diagnostics](../arena-diagnostics.md)：

- `summary.md` / `summary.csv`：各 suite × level 的 SR、CC、覆盖数、带代价的成功数和论文参考。
- `by_seed.csv`：所用**评测** seed 的分组结果；balanced1500 仅使用 seed 7，不是三个训练 seeds 的误差条。
- `episodes.csv` / `episodes.jsonl.gz`：逐条可核对记录，含初始状态索引、种子、动作步数与计时。
- `safety_by_level.png`：五类任务随 L0→L2 的成功率及 CC 折线图。
- `cumulative_success.png`：按任务清单顺序排列的累计成功率，不是时间进度曲线。
- `plan.json`、`provenance/`、`sha256.json`：配置、版本/作业来源与文件校验；若有历史异常，保留错误记录。

未来审查结果后，以下命令会明确把该小型报告提交并推送到 GitHub：

```bash
bash scripts/publish_arena.sh arena-balanced1500-001
```

它只暂存指定 `reports/` 子目录；不上传 checkpoint、场景、环境、视频或原始服务日志。模型任务失败属于正常结果，不应删除失败记录来提高分数。


## 准备与验证

2026-10-17：单 A100 作业 `14126130_0` / `arena-smoke-003` 已完成 15/15 条，SR 0%、平均 CC 33.33，无基础设施错误；逐谓词记录覆盖 15/15。结果见[汇总](../../reports/arena-smoke-003/summary.md)与[终止原因](../../reports/arena-smoke-003/diagnostics.md)。`arena-balanced1500-001` 已于 2026-10-17 提交，作业数组 `14143493`，最多 4 个单 A100 worker，账户 `gts-jli741-paid`。该评测于 22:00:40 EDT 完成 1,500/1,500 条，SR 11.47%、平均 CC 60.24，无基础设施错误；[结果与归因](arena-results-2026-10-17.md)包含 static/dynamic 分项。以下保留重建步骤；已有 run 不要重复 prepare。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
bash scripts/test_arena.sh
bash scripts/arena.sh prepare smoke arena-smoke-003 --workers 1
bash scripts/arena.sh prepare balanced1500 arena-balanced1500-001 --workers 4

# 以下只打印提交预览，不联系 sbatch。
bash scripts/arena.sh submit arena-smoke-003 --gpu a100 --smoke-run arena-smoke-003
bash scripts/arena.sh submit arena-balanced1500-001 --gpu a100 --smoke-run arena-smoke-003
```

新 instrumentation 的 GPU 验证来自 `arena-smoke-003`；旧 `arena-smoke-002` 的冻结配置不同。上述命令不含 `--submit`；显式添加该开关才会提交 1,500 条计划。

生成结果：`bash scripts/arena.sh report arena-balanced1500-001`。未完成时显式加 `--partial`；基础设施错误与模型失败分开。所有任务和失败记录都保留，不根据观测结果选择性删样本。
