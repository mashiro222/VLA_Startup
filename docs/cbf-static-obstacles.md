# π0.5 + GT CBF：Static Obstacles 卡住诊断

本实验评估：冻结的 Arena 微调策略经过局部 CBF 动作过滤后，是否减少碰撞，却增加无法完成任务的卡住现象。重点比较 L1 的单障碍场景与 L2 的双障碍场景。实验不训练模型。

## 1. 实验身份与边界

模型使用官方 [VLA-Arena/pi05-vla-arena-finetuned](https://huggingface.co/VLA-Arena/pi05-vla-arena-finetuned)，固定 revision `31bbab4bcdeee70c99797ef50c083d24f1def528`。它与此前 `pi05_libero`、本仓库训练至 5,952 步的 baseline 均不同。官方配置为 `pi05_vla_arena`，权重元数据不含 LoRA 参数；直接加载完整推理参数，不再添加或训练 LoRA。

只下载 `params/`、`assets/` 和 checkpoint 元数据，合计 12,439,994,645 bytes（约 12.44 GB / 11.59 GiB），不下载优化器和 `train_state/`。文件及 SHA-256 见 [checkpoint.json](../config/cbf_static/checkpoint.json)。归一化明确读取该 checkpoint 的 `assets/VLA_Arena_L0_L_lerobot_openpi/VLA_Arena/norm_stats.json`，不回退到 LIBERO 统计量。

Arena 固定为 `babe582ebffc82b979b77964a7e56417d02f63a4`，沿用已有环境。服务端使用该仓库内的 OpenPI 实现及其 `pi05_vla_arena` 配置。预处理与官方 evaluator 一致：两路 RGB、双轴翻转、256 渲染后 resize/pad 至 224、8 维 proprioception、7 维动作、预测 10 步、执行前 5 步后重新观察；flow sampling 为 10 步。每个 episode 先静置 10 步，再至多执行 300 个控制步，控制频率 20 Hz。SR 使用原始任务成功，不以 CC 阈值改写成功。

本实现是 **VLSA 启发的 GT、末端球体、平移 CBF 诊断**，不是 VLSA 或 VLPSA 完整复现：

| 方法 | 几何与过滤范围 | 本实验关系 |
|---|---|---|
| [VLSA / AEGIS](https://arxiv.org/abs/2512.11891)，Sec. IV | 视觉定位障碍、点云椭球、末端椭球及辅助分离状态 | 借鉴运行时最小修改 QP；替换为 GT 几何和简化点—膨胀椭球 CBF |
| [VLPSA](https://arxiv.org/abs/2609.22462)，Sec. III | Poisson 场、关节速度层、多部位采样、抓取后增加物体保护点 | 本次未实现这些机制；机械臂连杆和被抓物体只诊断，不纳入约束 |
| 本实验 | MuJoCo collision geoms，夹爪中心球体，只修改三维平移 | 用于定位过滤器与策略不协调、几何保守性及低层跟踪误差 |

结果不能称为“全身安全保证”，也不能仅凭“低碰撞、高超时”断言所有失败均由 CBF 引起。两个条件同时失败时，原因记为未决。

## 2. 条件、配对与 episode 数

- **C0：** 原策略动作直接进入 `env.step`，包括控制器正常的动作缩放/限幅。
- **C1：** 静置结束时，选择中心到“初始夹爪位置→目标物体中心”线段最近的障碍，整个 episode 固定选择它；障碍当前姿态仍逐步更新。
- **C2：** 同时约束所有非目标障碍。

目标和容器依据 BDDL `obj_of_interest` 与 placement goal 解析；障碍集合取 movable objects 减去目标与容器，并与 cost predicates 中的对象交叉核验，不把桌面当作避障对象。L0 没有障碍，因此只跑 C0/C2。L1 中 C1/C2 实际是同一约束，是额外的配对一致性检查。

任务分别为 Apple、Lemon、Mango、Onion、Tomato，每级每任务 50 个官方初始状态。正式比较用 **50 个状态 × 2 个采样种子 = 100 episodes/任务/条件**。条件之间的环境与 JAX 采样 seed 完全相同；seed 不含条件名、gamma、运行名或 GPU 编号。不同轨迹导致后续观测不同，因此配对不代表动作序列一直相同。

L1/L2 除障碍数量外，布局、目标容器及初始化分布也可能不同。因此跨级曲线表示 benchmark 等级之间的关联；不能仅凭它把变化归因于“多了一个障碍”。同一级内的 C0/C1/C2 配对才是本实验直接控制的比较。

| Profile | 内容 | Episodes | 独立单卡作业单元 |
|---|---|---:|---:|
| `calibration` | L0，5 任务 × 2 状态，C0 | 10 | 1 |
| `smoke` | Apple：L1/L2 各 5 状态 × C0/C1/C2；L0 5 状态 × C0/C2 | 40 | 1 |
| `main` | L1/L2，5 任务 × 100 × 3 条件 | **3,000** | 30，每单元 100 episodes |
| `sanity` | L0，5 任务 × 100 × C0/C2 | **1,000** | 10，每单元 100 episodes |
| `gamma`（可选） | L2，5 任务 × 100 × C1/C2 × gamma 0.1/0.6 | **2,000** | 20，每单元 100 episodes |

建议主实验与完整 L0 对照共 **4,000 个正式 episodes**；加标定与 smoke 共 **4,050 次执行**。若增加 gamma 扫描，共 **6,050 次执行**。扫描复用主实验的 C0 和 gamma=0.3，不重复提交这些条件。Smoke 和标定数据不并入正式统计；其中重复状态是接口检查，不是额外独立样本。

数量、任务、状态与采样种子在 [experiment.yaml](../config/cbf_static/experiment.yaml) 的 `profiles` 中调整。所有配置修改后必须生成新 run，已生成的 plan 不修改。

## 3. CBF 与动作尺度

每个物体的碰撞 mesh vertices 经 MuJoCo geom→world→object-body 转换后，在物体坐标系中拟合 MVEE，Khachiyan 容差为 `1e-3`。有限迭代误差通过再次扩大包络消除，确保所有输入顶点均被包含。非 mesh 基本几何用保守包围盒顶点，平面、高度场等不支持的可移动障碍类型直接报错。每步依据当前 body pose 把椭球变回世界坐标系。

夹爪代理为实际 `grip_site` 中心的球，半径 0.05 m，另加 0.01 m margin。该半径是实验代理参数，并未证明覆盖 Panda 的所有夹爪姿态；真实夹爪/连杆接触另外统计。

原说明中的“各半轴加球半径”并不能普遍保证包住椭球与球的 Minkowski 和。本实现采用保守外包络。设原椭球形状为 \(Q=A^{-1}\)，膨胀半径为 \(s\)：

\[
Q_{out}=(1+\beta)Q+(1+1/\beta)s^2I,\quad
\beta=s\sqrt{3/\operatorname{tr}(Q)}.
\]

它由支持函数上的平方不等式得到，在球形特例恰好等于半径相加。椭球包络会填满杯子等物体的凹陷，可能导致代理几何阻挡实际可行路径。这是需要与录像、初始 `h` 一起解释的保守性来源。

\[
h_i(p)=(p-c_i)^TA_i(p-c_i)-1,\qquad \nabla h_i=2A_i(p-c_i).
\]

控制器已核对为 OSC_POSE delta：平移输入范围 [-1,1]，配置输出范围 ±0.05 m，默认命令坐标系为 robot base。代码读取 base→world 旋转 \(R\)，模型近似为 \(p_{t+1}\simeq p_t+kRu_t\)。每个控制步求解：

\[
\min_u\|u-u_{nom}\|_2^2\quad
\text{s.t.}\quad k\nabla h_i^TRu\ge-\gamma h_i,\quad -1\le u_j\le1.
\]

默认 gamma=0.3，是**离散控制步**参数，不是论文连续时间系数的直接照搬。使用 OSQP，不加 safety slack。旋转和夹爪命令逐元素保持不变。无障碍或 C0 不作额外限幅，保证 L0 C0/C2 接口一致。

`k=0.05` 只是控制器配置尺度，**不是实测实际位移**。正式运行前用 L0 标定作业记录动作和实际位移，在控制器坐标系中拟合 k；排除接触、抓取、大旋转和接近零的动作，并用交替的留出样本报告相对 RMSE、各轴尺度和激励样本数。阈值见 YAML `calibration`。拟合不通过时不会自动替换参数继续正式评测，应先分析模型误差；脚本不会自动提交后续作业。

即使标定通过，OSC 跟踪、惯性、旋转耦合和物体运动仍使上述关系只是近似。记录每步 `prediction_error_m`、`h_after`；不把理想离散点模型的 CBF 推导当作真实机器人的严格安全证明。初始 h<0、几何移动导致约束不可行、solver failure 都独立记录。求解失败时平移为零，但保持旋转/夹爪命令；日志明确记录零平移是否满足当前线性约束，不能把回退叫作安全动作。

## 4. 碰撞、CC 与卡住标签

官方 **CC 完整保留**，由已有 `PredicateObserver` 观察原本执行的 predicate 调用，包含 horizon 时的 terminal cost，不额外重复调用 `_check_cost`。静态任务的 gripper contact、target-object contact 与 fall 均可在 JSON 中追踪。

物理接触与 CC 分开统计。代码在每个 MuJoCo physics substep 后采集 contact pair，再按控制步合并，避免只看 20 Hz 最后一个状态而漏掉短暂接触。保存 geom、body、对象名和首次/末次仿真时间。主碰撞定义是**机器人任意部分或当前抓持的目标物体，与指定非目标障碍接触**，不把正常抓取、桌面支撑、目标容器接触都算作违约。

- 分开标记 gripper、robot link、held object；一条 episode 可以命中多个类别，部件比例之和不要求为 100%。
- 另存全部 target object↔obstacle 接触，即使该物体当时未被抓住，便于与官方 `InContact` CC 对照。
- held object 用 Robosuite 双指接触抓取判定，取控制步前后任一满足；这是可审计的状态判据，不是完美的抓持真值。释放后发生的目标物体接触仍在额外字段中保留。
- 静置结束的初始接触单列。主 collision 标签包含初始违约，因此不能把这些 episode 的全部碰撞归因于后续策略或过滤器。

对每条 timeout episode 的末尾 3 秒（60 控制步），检查末端位置距离窗口均值的最大值是否 ≤0.03 m，同时实际夹爪两指位置范围变化 ≤0.005 m。满足即标记 stuck；这个规则对 C0 同样适用。`stuck_window_start_step` 是用于判定的**尾窗口起点**，不是已证明的物理卡住首次发生时间。

其余诊断包括：

- safe stuck / collision stuck；首次碰撞在尾窗口前还是窗口内。
- near/far stuck：尾窗口 subgoal 距离中位数阈值 0.10 m。未抓持时用夹爪到目标物体中心，抓持时用夹爪到容器中心；抓持切换可能造成距离跳变。
- filter-associated stuck：尾窗口中平移修正 `k * ||u_exec-u_nom|| > 0.005 m` 的步占比 >0.5。
- timeout 但不 stuck：wandering timeout，并分别统计是否发生碰撞，另存轨迹长度与净位移。
- 配对 success→stuck：相同任务/状态/种子，C0 成功而过滤组 stuck。输出每个完整配对的比例及在 C0 成功子集中的条件比例。
- 配对 rescue：C0 碰撞，过滤组成功且无碰撞；双方失败只记未决。

同时输出 `W∈{2,3,5}s × r∈{0.02,0.03,0.05}m` 的阈值敏感性结果。成功的 C0 也检查相同 stationary window，不加 timeout 前提，以便发现“正常结束时静止”造成的假阳性，而不是用定义上的零误判率自证有效。

## 5. 文件导航

| 文件 | 内容 |
|---|---|
| [config/cbf_static/experiment.yaml](../config/cbf_static/experiment.yaml) | episode 数、配对种子、gamma、半径、阈值、时限与视频策略 |
| [scripts/cbf_static.py](../scripts/cbf_static.py) | 准备、提交预览、显式提交、进度、汇总、测试入口 |
| [geometry.py](../src/cbf_static/geometry.py) | MVEE、保守膨胀、h 与梯度 |
| [filter.py](../src/cbf_static/filter.py) | 逐步 CBF-QP，坐标转换，失败回退 |
| [scene.py](../src/cbf_static/scene.py) | 障碍识别、GT 几何、控制器断言、物理接触监测 |
| [rollout.py](../src/cbf_static/rollout.py) | 真实 policy→filter→env.step 路径、官方 CC、录像与轨迹 |
| [server.py](../src/cbf_static/server.py) | 官方模型配置、checkpoint 归一化、配对 JAX RNG |
| [metrics.py](../src/cbf_static/metrics.py) | 卡住分类、单位换算、k 标定 |
| [report.py](../src/cbf_static/report.py) | CSV、配对分析、Wilson 区间、图与 Markdown |
| [runtime.py](../src/cbf_static/runtime.py) / [Slurm](../slurm/cbf-static.sbatch) | 一个 GPU policy server + 一个 simulator；超时退出与完整 episode 续跑 |
| [tests/cbf_static](../tests/cbf_static) | 数学、接触、配对、真实物理但假策略的集成测试 |
| [准备记录](run/cbf-static-preparation-2026-11-11.md) | 已验证内容、未验证内容与本地准备路径 |

既有训练、评测脚本和 checkpoint 不被此实验修改。本实验没有 fine-tuning 代码或训练作业。

## 6. 环境与准备命令

以下命令以仓库现有 uv 环境为基础：`.runtime/server` 为 JAX/OpenPI，`.runtime/arena` 为仿真，新增 OSQP 和必要导入依赖放在独立 target 目录。新 checkout 需先按原仓库说明建立基础环境和 Arena 资产。模型和场景均只保存在本地，不纳入 Git。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
bash scripts/setup_cbf_static.sh
python3 scripts/cbf_static.py test
python3 scripts/cbf_static.py test --scene-audit
```

下载命令可续传并重新校验 SHA-256。测试中的假策略/黑色图像只存在于临时目录；它们验证数据管道，不能证明 GPU 策略成功率或 EGL 渲染正确。

## 7. 手动提交顺序

下面 `submit` **不带 `--submit` 都仅打印命令**。实际计费命令必须由操作者显式增加 `--submit`。没有 CPU 控制器、没有自动串联提交或取消作业。

### 7.1 标定：单 GPU

```bash
python3 scripts/cbf_static.py prepare calibration runs/cbf-calibration-001
python3 scripts/cbf_static.py submit runs/cbf-calibration-001 --gpu a100 --parallel 1 --hours 3
# 确定 GPU 后，在上一条末尾加 --submit 才会提交。
```

如选 H100，把 `--gpu a100` 改为 `--gpu h100`。标定完成会生成 `runs/cbf-calibration-001/calibration.json`，必须先确认 `passed: true` 并检查 k 与拟合残差。

### 7.2 配对 smoke：单 GPU，40 episodes

```bash
python3 scripts/cbf_static.py prepare smoke runs/cbf-smoke-001 \
  --calibration runs/cbf-calibration-001/calibration.json
python3 scripts/cbf_static.py submit runs/cbf-smoke-001 --gpu a100 --parallel 1 --hours 4
python3 scripts/cbf_static.py status runs/cbf-smoke-001
python3 scripts/cbf_static.py report runs/cbf-smoke-001
```

检查真实图像、动作方向、CC 对账、solver failure、预测误差、L0 C0/C2 和 L1 C1/C2 配对一致性。需要人为抽查卡住录像，不能只看测试通过。

### 7.3 正式主实验和 L0 对照

```bash
python3 scripts/cbf_static.py prepare main runs/cbf-main-001 \
  --calibration runs/cbf-calibration-001/calibration.json
python3 scripts/cbf_static.py prepare sanity runs/cbf-l0-001 \
  --calibration runs/cbf-calibration-001/calibration.json
python3 scripts/cbf_static.py submit runs/cbf-main-001 --gpu a100 --parallel 4 --hours 12
python3 scripts/cbf_static.py submit runs/cbf-l0-001 --gpu a100 --parallel 4 --hours 12
```

`--parallel 4` 只控制**这一个数组**最多同时运行四个单卡单元；两个数组同时这样提交，可能合计八张卡，不存在隐藏的全局卡数限制。数组单元独立启动，不需要四卡同节点，也不需要等齐四张。可以改成 1、2 或其他选定并发。GPU 类型和并发数由实际排队情况决定。

每个单元申请 **1 GPU、8 CPU、64 GiB 主机内存**。这不是分布式模型推理，多卡只是同时跑不同 episodes。12 小时是 wall-time 上限，不是测出的运行时长；标定和 smoke 后使用报告中的 `wall_seconds`、`inference_seconds`、`filter_seconds` 估算总 GPU-hours 与并发耗时，排队时间另外计算。当前没有实际测得的本实验 A100/H100 显存峰值或速度，不能据此承诺某晚跑完。

中断后同一 run 可以重新提交指定单元，只跳过已有完整 `episode.json` 的 episodes；发生错误的半条轨迹重做，不计作任务失败：

```bash
python3 scripts/cbf_static.py status runs/cbf-main-001
python3 scripts/cbf_static.py submit runs/cbf-main-001 --gpu a100 --parallel 2 --hours 12 --indices 2,5
```

不要在 `sbatch` 反馈网络超时后立即反复重试；先用 `squeue -u gliu386` 核对是否已有作业。重复运行的同一单元有文件锁，但不能代替检查已排队的重复付费请求。

### 7.4 可选 gamma 扫描及汇总

```bash
python3 scripts/cbf_static.py prepare gamma runs/cbf-gamma-001 \
  --calibration runs/cbf-calibration-001/calibration.json
python3 scripts/cbf_static.py submit runs/cbf-gamma-001 --gpu a100 --parallel 4 --hours 12
python3 scripts/cbf_static.py report runs/cbf-main-001 --extra runs/cbf-l0-001
# 扫描完成后，可在 --extra 后继续添加 runs/cbf-gamma-001。
```

如果运行目录已经存在，使用新的编号；不要覆盖已准备或运行的 plan。准备脚本记录源文件哈希，提交和运行时拒绝源文件悄悄改变。修代码后可新建 run，并明确决定是否迁移已完成结果，不能把不同实现混成同一实验。

## 8. 输出与阅读顺序

每个 episode 位于 `runs/<run>/results/L<level>/<task>/<condition>/g<gamma>/seed<repeat>/init<index>/`：

- `episode.json`：原子完成记录；包含 SR、CC、终止原因、卡住分类、接触部件和 solver failures。
- `trace.npz`：每步 EE pose、夹爪状态、原动作/执行动作、所有障碍 h、选择约束的 slack/dual、介入量、subgoal 距离、控制器误差和接触分类。
- `contacts.jsonl.gz`：实际接触 body/geom、官方成本 predicate、solver 状态。
- `geometry.json`：目标、障碍、初始碰撞/h、MVEE、C1 固定选择与控制器坐标系。
- `front-initial.png`、`wrist-initial.png`：实际输入图像；`video.mp4`：各任务/等级/条件前十条，以及所有 C1/C2 stuck 录像。为了保留事后才能判定的 stuck，运行中先流式写临时录像，结束后按规则保留。

汇总入口为 `runs/<run>/summary/README.md`。优先看：

| 输出 | 用途 |
|---|---|
| `by-task.csv` / `macro-average.csv` | 每个任务与任务等权平均 SR、CAR、SSR、CC 和各类 timeout |
| `paired-summary.csv` / `paired-episodes.csv` | C0 成功→过滤组卡住，以及碰撞→安全成功的配对分析 |
| `paired-induced-stuck.png` | 障碍数与配对卡住率，Wilson 95% 区间 |
| `rates.png` | SR、CAR、safe stuck 对照 |
| `contact-components.csv/.png` | 夹爪、连杆、被抓物体接触比例；类别允许重叠 |
| `cost-predicates.csv/.png` | 官方 CC 的 gripper contact、object contact、fall 等谓词归因，与物理接触比例分开 |
| `process-*.png` / `process-examples.csv` | 距离、介入量、活跃约束数、名义/执行平移量，以及对应录像目录 |
| `sensitivity.csv` | 9 组卡住阈值敏感性；成功 C0 的尾窗口假阳性诊断 |
| `sanity-checks.json` | L0 C0/C2、L1 C1/C2 配对一致性 |
| `reference-comparison.csv` | C0 与参考 SR 的差值，超过 15 个百分点提示调查，不判定实现必错 |

只统计完成的 episodes，明确显示 observed/expected。未完成任务不可当零成功率，部分数据的 pooled rate 也不可冒充完整任务等权平均。两个采样种子复用同一初始状态，Wilson 区间仅为 episode 级描述区间，不能解释为独立场景抽样的严格置信范围。

参考 SR 来自 [FailBank](https://arxiv.org/html/2609.39820v1) 的基础策略结果：L1 为 90/83/77/72/85%，L2 为 8/0/89.3/81.3/74%。其初始化/采样协议和 Arena 版本与本实验不完全相同，仅用于发现明显接口问题；不直接把差值当作方法改进。该工作已讨论运行时 shield 与策略不协调，因此本实验的价值是做可审计的 GT 几何、接触部位和卡住分类诊断，不宜声称首次发现这一现象。

本地模型、轨迹和视频属于 scratch 数据，应另行备份到持久存储。GitHub 只保存代码、配置、说明以及后续小型汇总表和图。
