# Baseline 5,952 步评测：通信超时与断点恢复

**后续状态（2026-10-22）：** 补跑 `14252522` 仍排队，341 条原结果保持不变。SF 评测暴露了两者共用的本地 CUDA 运行库复制遗漏；该问题尚未修复，补跑也存在同类启动风险。[最新定位与影响范围](progress-2026-10-22.md)。

## 故障与已完成结果

原作业 **14199187** 于 2026-10-21 02:52:49 EDT 启动，05:47:36 以 `FAILED / 1:0` 结束，运行 2:54:47。不是达到 24 小时作业上限。

已持久化 **341/900 episodes**，无重复键：成功 275 条，SR **80.65%**，平均 CC **24.74**。这些结果全部来自 Static obstacles；Dynamic obstacles 尚未开始，不能作为完整 900 条的成绩。

未完成条目 `safety_static_obstacles-L2-t1-s7:11` 的诊断保存到第 **95** 个仿真步。随后动作预测未在 **600 秒**内返回，客户端抛出 `Policy response exceeded 600s`。该条目没有写入成功/失败计分记录，不能算成任务失败或零 cost。旧服务端日志只有连接事件，没有对应推理堆栈，尚不能区分 CUDA、CPU、文件读取或通信等待；没有证据确认 OOM、模型数值异常或某个具体硬件故障。

已完成条目的推理总耗时中位数为 10.10 秒，最大 223.08 秒，存在耗时尖峰。此处是每个 episode 的所有推理调用之和，不是单次调用耗时。

## 恢复方案

- 继续使用同一 **5,952 步 checkpoint**、原评测计划、动作适配、任务顺序、初态、种子、horizon 和 SR/CC 定义。请求上限仍为 600 秒，episode 上限仍为 1,800 秒。
- 保留原 341 条记录，并冻结每个日志文件已完成前缀的字节数及 SHA256。只补 **559 条：Static 109、Dynamic 450**。原始 900 条计划不改写，最终报告合并同一计划下的全部结果。
- 策略的 PyTorch/CUDA、仿真端独立版本的 CPU PyTorch、base 权重及训练 checkpoint 缓存到节点本地；大权重按原始 SHA256 校验。两套环境分别缓存，不混用版本。目的是减少共享文件系统参与运行，不能据此认定共享存储就是旧故障根因。
- 对通信超时或连接中断，在**同一 GPU 分配内最多执行 3 次 worker 尝试**。每次先停止旧策略和仿真进程，再从未完成 episode 的初态和固定种子重新开始；不在已经推进的环境中重发动作。已完成条目始终跳过。
- 普通任务失败仍正常计分；模型报错、元数据不符、无效动作和 episode 整体超时立即停止，不自动重试、不筛选结果。恢复预算耗尽后保留结果并退出，不提交新 Slurm 作业。
- 推理超过 120 秒开始输出 Python 堆栈；保存最近请求的 RNG、prompt、进入/返回状态及 episode 位置。重跑前将未完成条目的录像和诊断移到 `units/<unit>/interrupted/<attempt>/`，保留旧故障证据。

新增入口独立放在 `src/eval_resume/` 和 `slurm/eval_resume.sbatch`。现有 baseline 续训 **14217601**、SF 续训 **14230443**、SF 评测 **14230444** 的代码及计划校验保持通过，没有取消、修改或重新提交它们。

## 文件与命令

| 文件 | 用途 |
|---|---|
| `src/eval_resume/workflow.py` | 冻结原结果前缀、列出剩余 episode、校验并提交单份作业 |
| `src/eval_resume/runtime.py` | 本地缓存、策略诊断、有限恢复和完成后生成报告 |
| `slurm/eval_resume.sbatch` | 单卡作业入口及本地缓存清理 |
| `tests/test_eval_resume.py` | 559 条覆盖、前缀完整性、相同种子重试和真实 WebSocket 回归 |
| `runs/arena-baseline-eval-resume-20261021-001/resume.json` | 本次恢复清单及全部 559 个剩余键 |
| `runs/arena-after24h-20261019-001-baseline-eval/` | 原始及续跑结果、cost 诊断和 worker 日志 |

以下入口默认仅预览。同一提交清单已有成功回执时，重复执行返回原作业号，不另建重复作业。

本次补跑已提交为 **14252522**，账户 `gts-jli741-paid`，1 张 A100、8 CPUs、96 GB 主机内存、24 小时时限。初始状态 `PENDING (Priority)`；不需要再次提交。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
PYTHONPATH=src python3 -m eval_resume.workflow check \
  runs/arena-baseline-eval-resume-20261021-001/resume.json
PYTHONPATH=src python3 -m eval_resume.workflow submit \
  runs/arena-baseline-eval-resume-20261021-001/resume.json
# 首次正式提交需添加 --submit；已有成功回执时不要另建清单重复提交。

# 查询本次补跑；日志在作业启动后创建。
squeue -j 14252522 -o '%.18i %.10T %.12M %R'
sacct -j 14252522 --format=JobID,State,ExitCode,Elapsed
tail -F logs/baseline-5952-eval-resume-14252522.out
```

完成后，原目录对应的 `reports/arena-after24h-20261019-001-baseline-eval/` 输出 `summary.csv`、`episodes.csv`、`safety_by_level.png`、失败原因及 cost 归因表图。未达到 900 条时不生成完整报告，不将部分结果标记为完成。

## 验证与运行证据

CPU 回归覆盖真实本机 WebSocket 超时、新连接恢复、错误模型拒绝，以及原 `evaluate_units` 循环从 341 条继续至 900 条。检查已完成记录字节不变、剩余键各执行一次；发生一次基础设施故障时仅未完成条目重复尝试且种子一致。成功率为零的模拟任务也不会因失败而被重试。

最终 **34 项通过**，包括 **22 项恢复/通信检查**及原有日志、动作适配、计分函数和报告检查。另 8 项需要导入原生仿真库的既有集成测试，本次因共享运行库读取等待未完成，最终测试命令将其排除；不计为通过。登录节点上的完整策略环境导入和本地 CPU 库复制也观察到 `cl_sync_io_wait`，已停止这几个诊断进程，未操作 Slurm 作业。该现象支持减少共享读取的工程处理，但不证明原 GPU 作业具有同一根因。

这些检查验证通信恢复和结果保存机制；真实 GPU 推理、EGL 和旧超时是否再次发生，以新作业为准。GPU 服务启动时仍严格验证原训练环境、模型和归一化身份，没有绕过校验。

[故障数据](../../reports/baseline-eval-resume-2026-10-21/failure-analysis.json) · [测试结果](../../reports/baseline-eval-resume-2026-10-21/tests.xml) · [本次提交与状态](../../reports/baseline-eval-resume-2026-10-21/README.md)
