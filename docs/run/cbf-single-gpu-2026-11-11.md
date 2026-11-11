# Static Obstacles GT-CBF：独立单卡验证与正式评测（2026-11-11）

状态：两份独立作业已提交：**验证 15488897，正式 15488898**。尚无本实验的真实评测结果。

模型、动作适配、C0/C1/C2、SR/CC 和诊断定义沿用[实验说明](../cbf-static-obstacles.md)。不训练模型，不增加 gamma 扫描。

## 运行安排

| 作业 | 资源 | 时限 | 条目 | 调度依赖 |
|---|---|---|---|---|
| 验证 | 1 × A100 80GB、8 CPU、64 GiB RAM | 2 小时 | 10 条标定后接 40 条 smoke | 无 |
| 正式 | 1 × A100 80GB、8 CPU、64 GiB RAM | 24 小时 | 3,000 条 L1/L2 后接 1,000 条 L0 | 无 |

计费账户为 `gts-jli741-paid`，QOS 为 `inferno`。两份作业同时提交、独立排队，可同时占用两张 GPU。没有 CPU 控制器，没有 `afterok` 或自动提交后续作业。

验证合并到同一分配，避免标定与 smoke 分别排队。正式也合并成一次分配，使用一个策略服务完成 4,000 条，避免 40 个分片各自启动和加载权重。正式作业若先启动，会独立完成必要的 10 条标定；若已有合格标定则复用。不等待验证作业结束，不用验证成功率筛选是否开始正式评测。未通过的尺度拟合仍会明确停止当前作业。

额外标定、smoke 不计入正式结果。完整实验总执行数为 4,050 或 4,060；正式样本始终是同一批 4,000 条。

## 时限与调度依据

11 月 11 日 13:24 EST，相同 24 小时、单 GPU、8 CPU、64 GiB 请求的 `sbatch --test-only` 结果：A100 80GB 候选为 11 月 17 日 18:19 EST，H200 为 11 月 18 日 14:27 EST。试算不提交作业，候选时间可提前或延后，不代表正式作业承诺。

本实验尚无实测完整耗时。24 小时是申请预算，不保证所有任务都完成；不申请分区允许的最长 72 小时。完整 episode 逐条保存，未完成 episode 不计分，任何后续续投均须显式执行。

## 文件与检查

- [提交和运行入口](../../scripts/cbf_sequence.py)，[工作流](../../src/cbf_static/sequence.py)，[Slurm 脚本](../../slurm/cbf-sequence.sbatch)。
- [回归测试](../../tests/cbf_static/test_sequence.py)：正式 4,000 条与原主实验/对照逐条相同；无调度依赖；正式先启动时自行标定；已存在标定时复用；标定失败不进入后续计算；提交超时不重复提交；低 SR 不被误判为程序故障。
- 本地运行目录：`runs/cbf-single-gpu-20261111-001/`。
- 提交回执：`submission.json`；验证日志：`validation-JOB_ID.out`；正式日志：`full-JOB_ID.out`。
- 验证：`calibration/calibration.json`、`smoke/summary/README.md`、`validation-passed.json`。
- 正式：`full/summary/README.md`，原始 episode 位于 `full/results/`。中途可运行 `python3 scripts/cbf_static.py report runs/cbf-single-gpu-20261111-001/full` 生成明确标记 PARTIAL 的报告。

模型、场景、完整轨迹和录像仅写本地存储，不上传 GitHub。汇总表图可在产生后另行备份。

## 提交与验证

本次 23 项测试通过，包含真实 MuJoCo 物理但假策略的集成检查及 8 项工作流回归。真实 GPU checkpoint 恢复、EGL 和策略行为以作业结果为准。两份真实提交均返回成功作业号。

[提交回执](../../reports/cbf-single-gpu-20261111-001/submission.json) · [检查清单](../../reports/cbf-single-gpu-20261111-001/readiness.json) · [Slurm 资源与依赖核对](../../reports/cbf-single-gpu-20261111-001/slurm-verified.json)

```bash
squeue -j 15488897,15488898 -o "%.18i %.25j %.12T %.12M %.24S %R"
sacct -j 15488897,15488898 --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES%50
```
