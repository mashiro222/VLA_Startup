# VLA-Arena Safety：固定 pi05_libero 的 zero-shot 评测

本轮使用现有 `gs://openpi-assets/checkpoints/pi05_libero/`，不加载 Arena 作者微调的权重，不训练。它是“LIBERO 微调模型向 Arena 的 zero-shot 迁移”，不是未经任务微调的 `pi05_base`。本轮范围只有 5 类 Safety、L0/L1/L2 共 75 个任务，不包含干扰、组合泛化、长时序等其它类别。

环境、脚本和计划已准备；**2026-10-13 14:39 EDT 已按用户授权提交 smoke：`14016239_0`，账户 `gts-jli741-paid`、1 张 A100、8 CPU、64 GB、2 小时时限。** 提交后首次检查为 `PENDING (Priority)`，尚无 Arena 成绩，也未验证真实 GPU 推理和 EGL 渲染。Pilot/full 未提交；已有 Clean/Plus 的环境、权重及作业不因本轮准备而改变。

**后续选择更新：用户要求代码复查后直接提交 full，不再等待独立 smoke。独立入口已准备，等待用户选 A100/H100 后才提交；原 smoke 未动。** 见 [full 复查、四卡队列比较与提交入口](arena-full-review-2026-10-13.md)。下文保留的普通入口仍强制要求 smoke 完成。

## 1. π0.5 在论文中表现如何？

用户提供的 [VLA-Arena 论文](https://arxiv.org/abs/2512.22539)附录 A、表 7（PDF 页码 14）给出了 π0.5 结果。下面是**在 Arena L0-L 上专项微调后的模型**，不是本次 zero-shot 权重的预期成绩：

| 安全类别 | L0 成功率 | L1 成功率 | L2 成功率 | L0 / L1 / L2 的 CC |
|---|---:|---:|---:|---|
| 静态避障 StaticObstacles | 90% | 62% | 40% | 0 / 33.3 / 76.6 |
| 安全抓取 CautiousGrasp | 50% | 14% | 0% | 5.0 / 5.5 / 1.2 |
| 危险区域规避 HazardAvoidance | 58% | 30% | 36% | 7.1 / 15.0 / 14.5 |
| 状态保持 StatePreservation | 58% | 56% | 54% | 0 / 5.6 / 20.8 |
| 动态避障 DynamicObstacles | 50% | 44% | 22% | 2.4 / 8.8 / 5.7 |
| **五类等权平均（由表中数值计算）** | **61.2%** | **41.2%** | **30.4%** | 不将跨类 CC 汇总当统一安全分 |

解释：它在较难任务上仍有明显短板。静态避障 L2 虽有 40% 成功率，但 CC 达 76.6，说明任务完成与安全是两回事；安全抓取 L2 成功率为零，低 CC 也可能来自未能有效执行。其它能力方面，表 7 的长时序任务成功率从 L0 的 85% 降至 L1/L2 的 0%；空间关系组合则是 62% → 24% → 6%。不能把这些条件下的数字直接作为 `pi05_libero` 的 zero-shot 成绩。

本项目最终报告会将 **Our zero-shot SR/CC** 和 **Paper Arena-finetuned SR/CC** 分列，明确训练条件不同。它们是背景对照，不是误差容限或“复现通过”的判据。

对本次 zero-shot 的定性预期：LIBERO 中学到的抓取、放置能力可能迁移到部分 Arena 任务，但新场景、物体、细致抓取及动态避障要求可能让成绩明显下降，个别任务组出现零成功并不意外。目前没有同一 checkpoint 在本协议下的实测依据，不能承诺总体非零，也不宜给出数值预测。15 条 smoke 每类每级只有一次；即使 0/15，也不能据此断定全量成功率为零。须结合视频、动作、计分和错误日志，区分未完成任务与接口错误；低 CC 也可能只是模型没有有效行动。

## 2. 安装了什么？

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

本地准备记录见 [arena-readiness-2026-10-13.json](../reports/preparation/arena-readiness-2026-10-13.json)，全部任务的名称、指令与初始状态数量见 [arena-task-catalog.csv](../reports/preparation/arena-task-catalog.csv)。计分/断点/并发领取/提交边界的 10 项测试已通过；可用 `bash scripts/test_arena.sh` 重跑，它使用可控模拟接口，不加载模型权重或申请 GPU。

## 3. 已准备的计划与提交命令

| Profile | Run ID | 任务 × 每任务次数 | 任务单元数 | 最多同时用卡 | 每 worker 时限 |
|---|---|---:|---:|---:|---|
| smoke | `arena-smoke-001` | 每类每级选 1 个任务 × 1 = **15** | 15 | 1 | 2 小时 |
| pilot | `arena-pilot-001` | 75 × 10 × 1 seed = **750** | 75 | 4 | 8 小时 |
| full | `arena-full-001` | 75 × 10 × 3 seeds = **2,250** | 225 | 4 | 12 小时 |

三个计划互相独立；pilot 是可选，不会并入 full。每个任务单元是一项任务的一个评测 seed，full 中含 10 个 episodes。同一 worker 完成一个单元后领取下一单元，不再固定承包总任务的四分之一。只排到一张卡也能不断领取后续任务；更多 worker 启动后共同处理剩余任务。

**Smoke 的 `--submit` 已执行一次，返回 `14016239`；不要重复提交。下方 pilot/full 的 `--submit` 尚未执行。**

下面只准备文件或预览，不向 Slurm 提交请求，重复准备相同计划是安全的：

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
bash scripts/arena.sh prepare smoke arena-smoke-001
bash scripts/arena.sh prepare pilot arena-pilot-001
bash scripts/arena.sh prepare full arena-full-001
bash scripts/arena.sh submit arena-smoke-001
```

预览应包含 `--account=gts-jli741-paid --partition=gpu-a100 --qos=inferno --gres=gpu:a100:1`、8 CPU、64 GB，smoke 仅一个数组成员。**本轮已经执行过以下提交命令，保留作记录：**

```bash
bash scripts/arena.sh submit arena-smoke-001 --submit
```

本轮作业的查询命令（`tail -F` 可以等日志文件在作业启动后创建；按 Ctrl-C 只退出查看）：

```bash
squeue -r -j 14016239 -o '%.20i %.12T %.24S %R'
sacct -j 14016239 --format=JobID%22,State,ExitCode,Elapsed
bash scripts/arena.sh check arena-smoke-001
tail -F logs/arena-smoke-001-14016239_0.out
```

运行后检查：

```bash
squeue -r -u gliu386 -o '%.20i %.32j %.12T %.12M %R'
sacct -u gliu386 -S today --format=JobID%22,JobName%32,State,ExitCode,Elapsed
bash scripts/arena.sh status arena-smoke-001
ls -lt logs/arena-smoke-001-*.out
```

最后一条列出真实日志文件名，随后对那一个实际文件使用 `tail -f`，不要输入字面值 `JOB_ID`。Slurm 的 `COMPLETED` 只代表进程退出状态，还要核对 episode 覆盖及 `errors.jsonl`。

Smoke 的通过条件是 15 个 episode 全部产生有效成功/失败与 cost 记录，且无基础设施错误；不要求模型 15 次全成功。此时生成报告：

```bash
bash scripts/arena.sh report arena-smoke-001
```

检查 smoke 报告和视频后，可选择直接跑 full。两条分开，前者仅预览，后者提交：

```bash
bash scripts/arena.sh submit arena-full-001
bash scripts/arena.sh submit arena-full-001 --submit
```

若想先看 750 条：将上面 run ID 改为 `arena-pilot-001`，不必同时跑 pilot 与 full。提交 pilot/full 时会强制验证匹配配置的 Arena smoke；旧 LIBERO smoke 不能代替它。

选 H100 或调整时限的语法如下，第一条仍仅预览：

```bash
bash scripts/arena.sh submit arena-full-001 --gpu h100 --time-limit 08:00:00
# 确认后才在同一命令末尾加 --submit
```

默认使用 A100。此前的 LIBERO 实验证明模型能在 H100 上运行，但 Arena 的 H100 渲染仍需实际验证。时限是上限，不是运行时间预测。尚无 Arena 实测秒数，不承诺一晚结束。

## 4. 断点续跑与结果保存

每个 episode 完成后立即写入并 fsync。单元使用文件锁，多个 worker 不会同时写同一单元；worker 被终止时锁释放。下次续跑跳过已完成记录，截断未写完整的最后一行，重新执行中断的 episode。已经排队或运行的同名任务会阻止重复提交。

若 sbatch 回复超时，会保留 `runs/<run>/submission/unknown.json`，并阻止盲目重试。先用 `squeue` / `sacct` 确认是否已提交成功；仅在确定没有该任务时再处理标记，不能仅凭报错重复提交。脚本不会自动取消其它作业。

旧数组已全部结束、错误已检查且没有修改冻结代码时，使用同一提交命令续跑。每个 worker 启动前核对冻结文件；如果需要修改评测代码、任务定义或关键参数，使用新 run ID，不把改变后的数据接到原成绩中。无需修改环境的临时排队或时限问题可原计划续跑。基础设施异常会记录到单元的 `errors.jsonl` 并使 worker 非零退出，不计入模型失败。

所有 episode 完成后，在登录节点合并、出图，不申请 GPU：

```bash
bash scripts/arena.sh report arena-full-001
```

不完整时默认拒绝生成完整报告。需要观察中途结果时可明确使用：

```bash
bash scripts/arena.sh report arena-full-001 --partial
```

输出目录 `reports/arena-full-001/`：

- `summary.md` / `summary.csv`：各 suite × level 的 SR、CC、覆盖数、带代价的成功数和论文参考。
- `by_seed.csv`：三个**评测** seeds 的分组结果，不是三个训练 seeds 的误差条。
- `episodes.csv` / `episodes.jsonl.gz`：逐条可核对记录，含初始状态索引、种子、动作步数与计时。
- `safety_by_level.png`：五类任务随 L0→L2 的成功率及 CC 折线图。
- `cumulative_success.png`：按任务清单顺序排列的累计成功率，不是时间进度曲线。
- `plan.json`、`provenance/`、`sha256.json`：配置、版本/作业来源与文件校验；若有历史异常，保留错误记录。

未来审查结果后，以下命令会明确把该小型报告提交并推送到 GitHub：

```bash
bash scripts/publish_arena.sh arena-full-001
```

它只暂存指定 `reports/` 子目录；不上传 checkpoint、场景、环境、视频或原始服务日志。模型任务失败属于正常结果，不应删除失败记录来提高分数。

## 5. 计分和 zero-shot 比较边界

SR 是任务是否完成。CC 是上游安全谓词在整个轨迹上的累计代价，越低越好，不是碰撞概率。默认保持上游 `safety=false` 所对应的普通任务成功判定；CC 独立报告。需要看“成功但违规”的情况，而非把两个指标混成一个分数。

保留的上游协议：外部+腕部相机，渲染 256、模型输入 224，图像双轴翻转；模型每次预测 10 步，执行前 5 步后重规划；10 步静置不计分，Safety 的任务 horizon 为 300；控制接口不另加动作归一化。超时需补算最终状态代价，HazardAvoidance 的累计代价乘 0.05，环境返回的 `info['success']` 用于区别成功和环境终止。

明确记录的实验选择：

- normalization 来自原 `pi05_libero` checkpoint，没有用 Arena 数据重新计算。
- 默认与当前上游 YAML 一样，每任务使用第一个预存初始状态；10 次重复仍有不同的策略采样种子。可以在新实验里改为 `episode_idx`，但必须标出协议差异。
- 三个评测 seed 标签为 7/17/27。每条 episode 从“suite、level、task、seed、episode”生成稳定的独立随机状态，且每条新建环境。这样 worker 数量、执行顺序和断点续跑不改变已定义的 episode；它不同于上游串行运行的连续 RNG 流，也不承诺跨 GPU 型号逐位一致。
- 没有开启额外视觉或语言扰动，没有加安全控制器。这是直接测试现有策略，不是增强策略。
- 源码版本与论文发布时的环境可能有差异，因此最终数字是本次固定版本协议下的迁移评测，不能视为作者训练条件的严格复现。
- 当前启动器没有额外强制上游 evaluator 的 XLA/cuBLAS deterministic flags，固定 seed 不承诺跨硬件或驱动逐位一致。

## 6. 看哪些文件、在哪里改参数？

| 想了解或修改 | 文件 |
|---|---|
| 任务类别、levels、seeds、重复次数、horizon、重规划步数、默认卡数/时限、论文参考分数 | `config/arena.json` |
| 固定依赖与独立环境安装 | `requirements/arena.in`、`requirements/arena.lock`、`scripts/setup_arena.sh` |
| 准备计划、状态检查、提交预览、smoke gate | `scripts/arena.py`，入口 `scripts/arena.sh` |
| GPU 资源与作业入口 | `slurm/arena.sbatch`、`scripts/arena.py` 的 `submit()` |
| 模型服务启动与关闭 | `src/arena_startup/runtime.py`；复用 `src/vla_startup/server.py` |
| 观测如何进入模型、action chunk 如何执行、CC 如何累计 | `src/arena_startup/evaluate.py` |
| 任务领取、断点记录、episode seed 和计分函数 | `src/arena_startup/core.py` |
| 表格、图片与原始记录导出 | `src/arena_startup/report.py` |
| 计分、异常、并发领取、断点与禁止自动提交测试 | `tests/test_arena.py` |
| 上游任务名称与分级 | `.runtime/repos/vla_arena/vla_arena/vla_arena/benchmark/vla_arena_suite_task_map.py` |
| 各任务具体场景、目标与安全条件 | 上游 `vla_arena/vla_arena/bddl_files/safety_*/level_*/` |
| 安全谓词定义和执行 | 上游 `envs/predicates/`、`envs/bddl_base_domain.py` |
| π0.5 模型内部 | `.runtime/repos/openpi/src/openpi/models/pi0.py`、`pi0_config.py`；详见 [原代码地图](code-map.md) |

提交后不要改被冻结的文件。新参数请用新 run ID，按新协议重新完成 smoke。关于这两个安全 benchmark 的完整比较，见 [选型说明](safety-benchmark-comparison.md)。
