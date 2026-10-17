# Arena 失败原因、cost 归因与 1,500 条评测

## 实验范围

`balanced1500`：5 个 Safety suite × 3 个 level × 5 个任务 × 20 个初始状态 × 1 个评测 seed 标签 = **1,500 episodes / checkpoint**。每个任务使用预存初态索引 0–19；全部 75 个任务各有 50 个初态。它是固定的有覆盖协议，不是从全部 50 个初态中随机抽样。Static 与 Dynamic 各 300 条、每个 level 各 100 条。

旧 `full` 是 2,250 条；保留用于历史记录，本实验采用 `balanced1500`。该规模可以估计 SR/CC 及分组差异，不能预先保证某个结论或统计功效。`pi05_libero` 是 LIBERO 微调后的迁移参照；主要的 SF 因果比较必须是同一起点、同配置的 Arena LoRA baseline 与 Arena LoRA+SF。

所有比较模型固定同一任务、初始状态、episode seed、horizon、动作块长度、重规划步数与计分版本。改变模型身份时另建 run；当前服务仅支持 `pi05_libero` JAX，训练后的 PyTorch/LoRA 权重接入尚需实现。

## 终止原因与物理失败原因

| 字段/标签 | 精确定义 |
|---|---|
| `horizon_exhausted` | 执行 300 步仍未成功，环境未提前 done |
| `environment_timeout` | 环境返回 timeout；与外部 300 步上限分开 |
| `environment_done_without_success` | 环境结束但未满足成功条件 |
| `safety_filter_rejected` | 达到任务目标但被显式开启的 safety success filter 拒绝；默认不开启 |
| `task_success` | 环境目标成功，独立于安全过滤 |
| `final_goal_predicates` | 环境最后一次实际计算的目标条件及真假值 |
| `errors.jsonl` | 通信/渲染/模型/计分不一致等基础设施异常，不当作任务失败 |

“超过 300 步”是可复现的终止原因，不是“为什么没做好”的完整因果解释。抓错物体、没有抓稳、放置偏差等需要结合视频、状态和目标谓词；当前没有把这些自动猜测写成 ground truth 标签。帧级动作/状态足够支持后续显式规则或人工标注。

## CC 与接触计数

`diagnostics.py:PredicateObserver` 包装上游 `_eval_predicate`、`_check_cost`、`_check_final_cost` 和 `_check_success`，记录原调用结果；不重复求值、不替换原计分公式、不改变接触后停动的副作用。每步、每次最终代价和整个 episode 都验证分解之和等于原始 cost；不一致直接报错。

| 输出 | 含义 |
|---|---|
| `predicate` | 原 BDDL 条件，保留对象名与部位/距离参数 |
| `raw_step_cost` | temporal 谓词的逐步原始代价贡献 |
| `raw_terminal_cost` | 终态谓词的贡献，可能在环境 done 或外部 horizon 计入 |
| `cc` | 应用 suite 缩放后的贡献；所有项之和等于 episode CC |
| `active_steps` | 该 temporal 条件为非零的控制步数 |
| `onsets` | 按该谓词计算的 false→true 起始次数；起始时就接触记为一次 |
| `first_step` / `last_step` | 首次/末次激活的策略步编号，从 1 开始 |
| `predicate_value_sum` | 原谓词整数值之和；例如 Spilled 可能一次返回多个颗粒 |

`CheckGripperContact` 归入 gripper contact，`InContact` 归入 object contact；Fall、Spilled、距离阈值各自保留。`onsets` 是 **20 Hz 观测到的谓词接触段数量**，不是接触力峰值或 MuJoCo 子步的独立物理碰撞次数。同一事件可激活多个谓词，合计次数不去重。持续 100 步的一次接触可能产生 100 分；如果两个条件同时激活也可能产生 200 分。可在同类谓词内部报告 `active_steps / 20` 秒，但跨谓词求和可能重叠。

原 smoke 没有这些逐项记录。历史报告将诊断覆盖率标为 0，不虚构分解，也不以 0 填补缺失项。

## 文件与图表

| 文件 | 内容 |
|---|---|
| `runs/<run>/units/<unit>/episodes.jsonl` | 正式 SR/CC、步数、终止原因、全部 cost component 与目标谓词汇总 |
| `.../diagnostics/<episode>/steps.jsonl.gz` | 每步执行动作、前后状态、目标真假、逐项 cost、累计 CC；末行记录终态 cost |
| `.../diagnostics/<episode>/policy-call-000.npz` 等 | 有录像 episode 的前 3 次调用：raw/policy 双相机图、8 维状态、预测动作块、调用前 RNG |
| `.../diagnostics/<episode>/steps.partial.jsonl.gz` | 中途失败留下的非完整 trace，不算有效 episode |
| `.../videos/<episode>.mp4` | 模型前视图录像；保存数由 `videos_per_worker` 控制 |
| `reports/<run>/summary.csv` | suite × level 的 SR/CC，与不同训练条件的论文参考分列 |
| `failure_modes.csv` | 按 suite/level 分组的终止原因数量 |
| `outcomes.csv` | 成功/失败 × 零/正 cost 四格、平均步数、每 100 控制步累计代价 |
| `cost_components.csv` | 每个 episode、每个 BDDL 谓词的归因 |
| `cost_by_category.csv` | suite/level/类别的 CC、onsets、active steps 汇总 |
| `cost_attribution.png` | 实测有逐项记录时才生成的 cost 构成堆叠图 |
| `diagnostics.md` | 终止原因与诊断覆盖说明 |

`cc_per_100_control_steps` 只是辅助暴露量指标：包含终态罚分，不是碰撞概率或瞬时风险。主结果始终保留原 SR/CC、成功与失败分层和轨迹长度，避免把“更早结束或不行动”当成安全改进。

## 准备与验证

2026-10-17：单 A100 作业 `14126130_0` / `arena-smoke-003` 已完成 15/15 条，SR 0%、平均 CC 33.33，无基础设施错误；逐谓词记录覆盖 15/15。结果见[汇总](../reports/arena-smoke-003/summary.md)与[终止原因](../reports/arena-smoke-003/diagnostics.md)。`arena-balanced1500-001` 尚未提交。以下保留重建步骤；已有 run 不要重复 prepare。

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
