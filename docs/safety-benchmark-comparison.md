# π0.5 的安全 benchmark 选择：VLA-Arena 与 LIBERO-Safety

核对日期：2026-10-13。依据用户提供的两篇 PDF，以及当天官方 GitHub / Hugging Face 的源码和文件清单。此文保留选型时的审查结果。**后续用户已选择 VLA-Arena Safety：独立环境、资产和评测脚本已准备，未提交作业；见 [运行说明](arena-safety.md)。** 没有下载两者作者的专项模型权重；LIBERO-Safety 环境仍未安装。现有 Clean 补跑没有修改。

## 1. 当前模型是什么，能否叫 zero-shot？

本项目一直使用 OpenPI 官方 `gs://openpi-assets/checkpoints/pi05_libero/`，配置名 `pi05_libero`。它从 π0.5 base 在原版 LIBERO 上微调而来，官方标记为 30k 步 finetuned checkpoint；我们没有再训练。[官方说明](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md#results)

| 用法 | 应如何描述 |
|---|---|
| `pi05_libero` → 原版 LIBERO | LIBERO 微调模型在对应评测集上的测试 |
| 同一 `pi05_libero` → LIBERO-Plus | 无 Plus 专项训练的扰动鲁棒性评测 |
| 同一 `pi05_libero` → Arena / Safety | **相对于目标 benchmark 的 zero-shot 迁移评测** |
| 作者的 Arena / Safety 专项权重 → 对应 benchmark | 专项微调模型的推理评测；我们不训练，但权重已经训练过 |

推荐实验表述：**固定一个 LIBERO-finetuned π0.5 checkpoint，在不同 benchmark 上进行评测，不进行目标 benchmark 的额外训练。** 不能写成未经任务微调的 `pi05_base`，也不能声称模型从未见过任何相关物体、任务或场景：两个框架与 LIBERO 共享部分仿真基础与任务结构，预训练数据的完整重叠情况也未核实。

固定权重还应保留该 checkpoint 的 observation/action normalization、图像方向、状态编码及动作约定。不要为了跑新环境换用其训练集计算的 normalization statistics；必要的输入输出接口适配应记录。协议和参数应在正式测试前固定。

## 2. 两篇论文都跑过 π0.5

### VLA-Arena

用户提供版本的**附录 A、表 7（PDF 页码 14）**包含 π0.5，不能只看主文表 2（主文的早期模型列表是 π0 / π0-FAST 等）。标准实验在 Arena 的 **L0-L 数据上微调**，再测 L0/L1/L2。代码中的 `pi05_vla_arena` 也明确使用 Arena L0-L 数据；不是我们现有 `pi05_libero` 的成绩。

π0.5 安全部分的示例，SR 为百分数，CC 为累计安全代价、越低越好：

| 类别 | L0 SR | L1 SR | L2 SR | L0/L1/L2 CC |
|---|---:|---:|---:|---|
| StaticObstacles | 90 | 62 | 40 | 0 / 33.3 / 76.6 |
| CautiousGrasp | 50 | 14 | 0 | 5.0 / 5.5 / 1.2 |
| HazardAvoidance | 58 | 30 | 36 | 7.1 / 15.0 / 14.5 |
| StatePreservation | 58 | 56 | 54 | 0 / 5.6 / 20.8 |
| DynamicObstacles | 50 | 44 | 22 | 2.4 / 8.8 / 5.7 |

高任务成功率与高安全代价可以同时出现；低代价也可能来自不行动或提前失败，所以必须联合解释 SR 与 CC。

有公开作者权重：[VLA-Arena/pi05-vla-arena-finetuned](https://huggingface.co/VLA-Arena/pi05-vla-arena-finetuned)。当天确实可列出 `params/` 和 normalization 文件，并非只有占位链接；没有 model card，不能仅凭仓库名确认论文全部训练种子与实验设置。若只推理，未来只需取参数和对应 normalization，不需要下载其中的 `train_state/`。

### LIBERO-Safety

用户 PDF 的**主文表 3（页码 11）**包含 π0.5 的四个物理安全 suite，各有 L0/L1/L2。论文在该 benchmark 的 L0/L1 安全数据上微调，L2 留作泛化；附录 C.2 / 表 A.5 说明 π0.5 微调 30k 步。表 3 示例：安全抓取 AAG 的 L0/L1/L2 成功率为 **78.7 / 59.3 / 35.3%**，人机交互 HRI 为 **84.7 / 88.7 / 83.3%**。

有公开作者权重：[LIBERO-Safety/pi05_libero_safety](https://huggingface.co/LIBERO-Safety/pi05_libero_safety)，包含 `params/` 与 `assets/lerobot/norm_stats.json`，同样没有 model card。

**该论文还报告 zero-shot 对照**：附录 C.3 / 表 A.10（页码 38）的 π0.5 zero-shot 为 **SR 0.7%、CR 75.7%**。但该处没有明确该行使用的 checkpoint URI、与我们 `pi05_libero` 的关系，以及逐 suite 覆盖明细。它提示低成功率是需要考虑的可能性，不能作为我们应复现到 0.7% 的依据。

语义安全 SSR 是另一条评测轨道：表 4 使用 RoboBrain / RynnBrain 判断指令安全性与是否拒绝，并非 π0.5 闭环动作测试。当前 π0.5 动作接口不能直接输出同定义的文本拒绝率；若自行定义动作层面的拒绝，需要另立协议，不能冒充论文 SSR 复现。

## 3. 对当前实验的优缺点

| 比较项 | VLA-Arena | LIBERO-Safety |
|---|---|---|
| 主要定位 | 综合能力：安全、干扰、组合泛化、长时序 | 专门测物理安全及独立的语义安全推理 |
| 适合 π0.5 的安全任务 | 5 类 × 3 级 × 5 个任务 = **75** | 4 类 × 3 级 × 5 个任务 = **60**；另有 15 个 SSR 任务 |
| 安全内容 | 静态/动态避障、安全抓取、危险区域规避、物体状态保持 | 安全抓取、人机交互、桌面避障、手与物体组合的三维避障 |
| 核心指标 | 任务 SR 与累计违规代价 CC 分开报告 | 无违规完成的 SR、碰撞率 CR；辅以时间和平滑度 |
| 优点 | 现成 uv/OpenPI/WebSocket 入口；可先只测安全部分，再扩到全 170 任务 | 安全问题更集中，人机接触、力阈值和复杂几何更丰富；直接报告安全成功率 |
| 代价/局限 | 全套包含许多非安全任务；任务失败与语言泛化不足也会拉低分数；CC 受时长、提前失败及谓词定义影响 | 当前公开代码需要补齐符合论文的 rollout/终止/计分逻辑；SSR 不能直接纳入 π0.5 动作评测；物理仿真和人类代理不代表真实人机安全 |
| 当前代码可用性判断 | 固定 `pi05_libero` 接入路线直接，尚待 GPU smoke | 可以接入，但需要更多协议核对和适配，尚不能称为开箱即用 |

两个 benchmark 的 SR 定义与任务分布不同，不能把它们与 LIBERO-Plus 的 84.52% 直接相减并解释成“安全能力下降了多少”。应按 suite/难度分析，并区分正常未完成、违规终止和程序异常。

## 4. 实际源码核对与接入风险

### VLA-Arena：已有入口，但默认配置不是 π0.5

核对版本：`babe582ebffc82b979b77964a7e56417d02f63a4`。

- [OpenPI 评测 YAML](https://github.com/PKU-Alignment/VLA-Arena/blob/babe582ebffc82b979b77964a7e56417d02f63a4/vla_arena/configs/evaluation/openpi.yaml)：默认选择 π0 LoRA，不能原样启动后当成 π0.5。固定现有权重时需选 `train_config_name: pi05_libero`，指定本地 checkpoint，或连接现有 OpenPI server。
- [训练/模型配置](https://github.com/PKU-Alignment/VLA-Arena/blob/babe582ebffc82b979b77964a7e56417d02f63a4/vla_arena/models/openpi/src/openpi/training/config.py)：同时存在 `pi05_libero` 和 `pi05_vla_arena`；前者用于我们的迁移实验，后者配作者专项权重。
- [评测器](https://github.com/PKU-Alignment/VLA-Arena/blob/babe582ebffc82b979b77964a7e56417d02f63a4/vla_arena/models/openpi/evaluator.py)：输入是外部/腕部图像、末端状态及 prompt，经 WebSocket 返回 action chunk，与我们现有接口一致。默认 replan=5，普通任务 horizon=300，长时序高级任务=600，不能照搬旧 LIBERO 各 suite horizon。
- YAML 的 `safety: false` 不等于关闭安全指标：评测器仍累加 `info['cost']`；开关会改变成功判定条件。要按 SR/CC 的论文协议固定设置，不可凭名称直接改成 true。
- 默认 `init_state_selection_mode: first`；论文的重复次数/种子与初始状态选择要一并记录。评测器会捕获 episode 异常并返回失败，因此正式接入时必须独立记录基础设施错误，不能把异常静默计入模型失败。
- 官方已提供 [uv-only 指南](https://github.com/PKU-Alignment/VLA-Arena/blob/babe582ebffc82b979b77964a7e56417d02f63a4/docs/finetuning_and_evaluation.md)，无需引入 conda。官方 README 给出的任务资产量约 850 MB（与 Safety 的资产不是同一份）。

### LIBERO-Safety：需要明确安全终止和成功标记

核对版本：`19ec8df23eedfbb9265bafd3e56495fcebfcfcd0`。

- [任务注册表](https://github.com/LIBERO-SAFETY/LIBERO-Safety/blob/19ec8df23eedfbb9265bafd3e56495fcebfcfcd0/libero/libero/benchmark/vla_safety_task_map.py)确认上述 5×3×5 个任务结构，物理四类可独立选择。
- [环境代码](https://github.com/LIBERO-SAFETY/LIBERO-Safety/blob/19ec8df23eedfbb9265bafd3e56495fcebfcfcd0/libero/libero/envs/bddl_base_domain.py)的 `step()` 用 `_check_success()` 设置 `done`，把 `_check_constraint(done)` 的字典放进 `info['cost']`；该函数在 `done=True` 时跳过约束评估。检查的环境入口本身没有因违规自动终止，与论文“任何安全违规立即终止、算失败”的协议不能直接画等号。
- 因此需在新 runner 中验证每个违规谓词及碰撞标记、违规终止逻辑、同一步达成目标且违规的优先级，并区分 collision / 其它违规 / timeout / model failure / infrastructure error。不能仅将 `done=True` 当作安全成功。这是代码与论文协议的接入缺口，不是已经验证该 benchmark 无法使用。
- 官方 README 没有完整的 π0.5 推理命令；源码树里未发现独立的 π0.5 rollout 入口，存在的是原 LIBERO 训练/评测结构。需要适配现有 server/client 与它的 level/task API。
- 包名仍叫 `libero`，并要求它自己的 `third_party/robosuite-1.4`。应建独立 uv 仿真环境和 `LIBERO_CONFIG_PATH`，不能覆盖现有 Clean/Plus 环境。
- [公开资产](https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets/tree/main)的 `assets.zip` 元数据大小为 **10,670,353,443 bytes（约 10.67 GB）**，还需解压空间；无须下载训练示范数据来执行零样本评测。
- 论文主文说三组随机种子，表 3 caption 又称三组 training seeds；公开单份 checkpoint 不足以严格复现跨训练种子的误差条。我们可以报告固定 checkpoint 的评测种子波动，但需明确其含义。

上述是静态审查，不是 GPU smoke 通过证明。两者正式计费前都应先用少量不同类别/难度任务验证图像、动作、渲染和计分。

## 5. 建议的选择和规模

**若本周目标是先跑通、固定权重看差异，优先 VLA-Arena 的 Safety 子集。** 它的接入路径更明确，75 个任务已足以展示成功率与安全代价之间的差异。若重点是人机交互或违规即失败，则优先 LIBERO-Safety 的四个物理 suite，但先补齐上节的评测协议。

建议分两层，而非立即申请整套全量：

| 阶段 | Arena Safety | LIBERO-Safety Physical |
|---|---:|---:|
| 接口 smoke：每类每级选一个任务、各一次 | 15 episodes | 12 episodes |
| 固定 checkpoint 的 1 个评测 seed：每任务 10 次 | 750 | 600 |
| 固定 checkpoint 的 3 个评测 seeds：每任务每 seed 10 次 | 2,250 | 1,800 |

Arena 全部 170 任务同样按 3×10 次计算为 5,100 episodes，暂不必全部做。上述数量是根据任务数计算的计划，不是承诺与所有论文统计协议完全一致；尤其 Safety 的 training-seed 歧义需标注。

继续复用现有 `pi05_libero` 时不需要再下载模型。预计可沿用单卡推理、多个独立单卡 worker 并行的方式；论文中训练使用的多卡数量不是推理硬性要求。尚无这两个环境的 PACE 实测速度，不拿 Plus/Clean 的耗时直接推算全量结束时间。

我们自己的结果与作者专项微调结果应分列为 **LIBERO checkpoint 的迁移评测** 和 **目标 benchmark 专项微调参考**。若以后想测“安全训练带来的提升”，应把同一任务集上的两份权重作为受控比较，并固定评测设置。

## 6. 后续阅读入口

- 模型与 normalization：本仓库 `config/sources.json`、`config/evaluation.json`、[现有代码地图](code-map.md)。
- Arena：[论文](https://arxiv.org/abs/2512.22539)、附录 A 表 7、H.4 重复次数；[官方仓库](https://github.com/PKU-Alignment/VLA-Arena)的任务套件表、评测 YAML、OpenPI evaluator。
- Safety：[论文](https://arxiv.org/abs/2606.23686)、3.2 任务分类、4.1 评测协议、表 3 与表 4、附录 A 安全约束、C.3 表 A.10 零样本对照；[官方仓库](https://github.com/LIBERO-SAFETY/LIBERO-Safety)。
- 作者权重已核对 revision：Arena `31bbab4bcdeee70c99797ef50c083d24f1def528`；Safety `e66d84bfec17770384550593e39975f7e21e3463`。仅检查元数据，尚未下载。

选型后的实施入口见 [Arena Safety 运行说明](arena-safety.md)。本说明中的规模尚未提交为 Slurm 作业。
