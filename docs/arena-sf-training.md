# Arena π0.5 LoRA / Spatial Forcing 训练

## 提交记录（2026-10-17）

01:00 EDT 已核对 Slurm 中的账户、资源和依赖。账户均为 **`gts-jli741-paid`**，分区 `gpu-a100`，每项申请同一节点的 2 张 A100、16 CPU、96 GB 主存。

| 任务 | Job ID | 时限 | 核对时状态 |
|---|---|---|---|
| 两组短训练验证 | `14127618` | 3 小时 | PENDING / Priority |
| 正式 LoRA baseline | `14127636` | 24 小时 | PENDING / Dependency |
| 正式 LoRA + SF | `14127637` | 24 小时 | PENDING / Dependency |

**两个正式任务已提交。** 二者均依赖 `afterok:14127618`，验证成功后自动进入资源调度，无需再次手工提交。GPU 正确性尚未确认；验证失败时正式任务不会启动。正式两组同时运行最多使用 4 张 A100。

已完成 3,054 个文件（55,017,991,583 bytes）校验、3,018 条 / 361,736 帧数据打包与归一化、包含轨迹边界的 5 个真实样本等价检查，以及 26 项 CPU 测试。初版预处理在积累全数据集列表时被系统终止；现已改为磁盘数组，并隔离 PyTorch 导入，完整预处理正常退出，实测峰值 RSS 为 872,800 KiB。模型初始化时不需要的 Inductor 后台编译池也已限制为单线程。

[准备检查](../reports/preparation/sf-validation.json)、[真实样本检查](../reports/preparation/sf-data-audit.json)、[Slurm 核对记录](../reports/training-submissions-2026-10-17/slurm-verified.json)与同目录内的两个冻结计划、三个提交回执保存于 GitHub。模型、数据、checkpoint 和运行日志保留在 scratch。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
squeue -j 14127618,14127636,14127637 -o '%.18i %.10T %.40j %.10M %R'
sacct -j 14127618,14127636,14127637 --format=JobID%18,JobName%40,State,Elapsed,ExitCode
# 日志在作业启动后创建；-F 可以等待文件出现。
tail -F logs/arena-sf-train-smoke-001-smoke-14127618.out
```

正式日志分别为 `logs/arena-sf-train-full-001-baseline-14127636.out` 和 `logs/arena-sf-train-full-001-sf-14127637.out`。上表是提交时快照，实时进度以 Slurm、训练日志及 `runs/arena-sf-train-full-001/<arm>/metrics.jsonl` 为准。

## 实验定义

两组均从官方 **`pi05_base`** 初始化，使用相同 Arena L0-L 示范、数据顺序、随机种子、可训练策略参数、优化器及训练预算。Baseline 只使用 flow-matching MSE；SF 增加冻结 VGGT 教师的视觉表示对齐损失。`pi05_libero` 只作为单独的迁移评测参照，不作为本次训练初始化。

Spatial Forcing 原实现面向 π0。本实验使用其 PyTorch OpenPI 分支支持的 `pi05=True` 路径，保留 π0.5 的 action expert adaRMS 和 timestep MLP。**这是 SF 在 π0.5 + Arena LoRA 上的适配实验，不是 SF 论文原实验的复现。**

Arena 论文 Appendix H.3.4 / Table 29 描述 π0 的 LoRA，表格写 30,000 步而正文写 60,000 步；不能由此断言论文 π0.5 的公布成绩采用同一 LoRA 配方。本实验以固定 Arena 源码中的 `pi05_vla_arena_low_mem_finetune` 为配置依据：60,000 步、global batch 32、AdamW、cosine decay、无 EMA。

来源与版本：

- [VLA-Arena](https://github.com/PKU-Alignment/VLA-Arena/tree/babe582ebffc82b979b77964a7e56417d02f63a4)，配置位于 `vla_arena/models/openpi/src/openpi/training/config.py`。
- [Spatial Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing/tree/372819b31ba9a2c8fc5989edc7d525cb187cecd5)，使用 `openpi-SF/src/openpi/models_pytorch/` 与 `src/vggt/`。
- [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) 官方 JAX→PyTorch 参数映射。
- [训练集](https://huggingface.co/datasets/VLA-Arena/VLA_Arena_L0_L_lerobot_openpi/tree/8c6cdcac185c7c97582a78f8ad8c2ef3757ce2c7)。

## 数据与损失

训练集共 **3,018 条示范、361,736 帧、60 个任务**，覆盖 Arena 多个能力域的 L0-L，并非仅 Safety。每个样本包含当前 front/wrist 两路 RGB、8 维 state、任务语言，以及从当前时刻开始的 10×7 动作。轨迹末尾重复最后一个动作，绝不跨 episode 拼接。没有另做 delta-action 转换。

图像由数据集已有方向直接缩放到 224×224；模拟器评测时的 180° 图像修正不能再应用于训练集。动作与 state 使用训练集 q01/q99 归一化，再补零到 32 维。遵循 Arena π0.5 配置，`discrete_state_input=false`；state 不被加入语言 token，π0.5 路径也不使用 π0 的连续 state projection。该选择在两组一致。

令动作 chunk 为 a，噪声 ε∼N(0,I)，t∼Beta(1.5,1)×0.999+0.001：

```
x_t = t * ε + (1 - t) * a
v_target = ε - a
L_FM = mean((policy(obs, x_t, t) - v_target) ** 2)
L_baseline = L_FM
L_SF = L_FM + 0.5 * mean(1 - cosine(projector(h_student), h_VGGT))
```

MSE 与 OpenPI 一样包含 32 个 padded action 坐标。SF 取 student `all_hidden_states[12]`（第 12 层输入，零基索引），取 VGGT 最后一层 patch features，去掉特殊 token 后插值到每相机 16×16 patch，并使用上游位置编码。仅两路真实相机进入教师；OpenPI 第三路缺失相机始终被 mask。

为了保持空间 token 对应，两个训练组统一禁用几何 crop/rotation，保留同一套 student color jitter；教师读取同位置、未做颜色扰动的图像。这是相对于 Arena 默认图像增强的一项明确改动。两组相同，因此主比较只改变 SF 目标和其辅助 projector。教师冻结且不参与优化；baseline 完全不加载教师。

## 参数与资源

| 项目 | 设置 |
|---|---|
| 全局 batch | 32，卡数改变时保持不变 |
| 每卡 micro-batch | 1 |
| 1 / 2 / 4 卡梯度累积 | 32 / 16 / 8 |
| 训练步数 | 每组 60,000 optimizer updates |
| 学习率 | warmup 1,000；峰值 2.5e-5；30,000 步降到 2.5e-6，之后保持 |
| AdamW | β=(0.9,0.95)，ε=1e-8，weight decay=1e-10，grad clip=1 |
| VLM / action expert LoRA | rank=16 / 32，alpha=16 / 32，dropout=0 |
| LoRA 初始化 | A、B 均 N(0,0.01)，遵循 Arena/OpenPI；注意不是 B=0 |
| 精度 | BF16 autocast；可训练参数、优化器主状态 FP32 |
| checkpoint | 每 1,000 步，保留最近两份；原子写入 |

LoRA 覆盖两个 transformer 的 attention q/k/v/o 和 MLP gate/up/down，共 252 个 projection。attention 使用与 OpenPI JAX einsum 对应的**每个 attention head 独立低秩更新**；不能用普通 flattened Linear 的单个 rank-r 矩阵冒充相同配置。

冻结原有 language model / action expert 参数，包括 embedding、norm 与 expert adaRMS；LoRA 参数可训练。vision tower、视觉 projector、action 输入输出 projection 和 time MLP 全量训练，遵循 Arena 的 freeze filter。SF 另外训练 alignment projector；VGGT 参数无梯度。真实模型的 meta 结构检查得到策略可训练参数 **466,957,072（约 4.67 亿）**，其中视觉编码器 412,442,352、LoRA 49,987,584，其余为视觉/动作/time 投影；SF 另有 alignment projector。这里不是“只训练几百万 adapter”的配置。可训练参数列表由 `reports/preparation/sf-policy-parameters.json` 和运行内 `parameters.json` 给出。

**默认正式方案：每组 2 张 A100，同一节点 DDP；两组并行时总共 4 张。** 2026-10-17 查询的 Phoenix A100 分区大部分节点每台 2 卡，只有一台列出 8 卡。单组 4 卡会受限于能容纳四卡的节点，不能把四个独立单卡 array 当作同步训练。此处选择 2 卡是根据节点布局和并行效率作出的初始方案，不是已测得的最优配置。

DDP 每卡仍持有完整策略、教师（SF 组）和相应优化状态，增加卡数不会把模型参数显存除以卡数。BF16、activation checkpointing、micro-batch=1 用于控制每卡显存。实际峰值和训练时长需要 GPU 数据；短 smoke 只提供粗略测量，不能保证 60k 步在一个 24 小时作业内结束。临近时限会保存 checkpoint，后续显式续训。

## 环境与准备

从仓库根目录执行；所有大文件保存在 scratch 下 `.runtime/training/`，不上传 GitHub。已有 Arena 评测环境保持隔离。

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
bash scripts/setup_sf.sh
.runtime/server/bin/python scripts/sf_assets.py
source scripts/sf_common.sh
"$SF_ENV/bin/python" -m sf_training.pack
bash scripts/test_sf.sh
```

依赖由 uv 维护，锁文件为 `requirements/sf_train.lock`。只在 `.runtime/sf_train` 中安装 SF 要求的 transformers 4.53.2 overlay。下载清单固定 HF revision、GCS generation、文件大小和 SHA256/MD5/CRC32C；所有文件校验后才生成 `assets_verified.json`。

Parquet 原始图像采用有界并行读取，数值与索引直接写入预分配磁盘数组，CPU 预处理不加载 PyTorch。按原 episode 顺序打包成随机读取文件，避免每取一帧都重新解压整个 episode。实际样本的打包前后等价性单独核对；数值、图像、语言 token 和轨迹末尾动作片段均须一致。归一化只使用训练示范，并按实际 clamped action chunk 统计。`pi05_base` 参数转换在首次训练验证的已分配节点上执行，不在登录节点运行大模型。转换检查全部参数名和 shape；唯一显式补零的是从不参与动作推理的 expert LM vocabulary head。

## 训练验证与正式提交

训练 smoke 不是性能 benchmark。它验证真实模型加载、DDP、两组 loss/梯度、冻结教师、checkpoint 保存、进程重启后的 optimizer/projector/RNG 恢复，以及两组最终的 10 步动作推理。两组各执行 2 次更新，退出进程，然后恢复执行第 3 次；global batch 为 4，正式训练为 32。

```bash
# 仅准备，无计费提交
bash scripts/sf.sh prepare smoke arena-sf-train-smoke-001 --gpus 2
bash scripts/sf.sh prepare full arena-sf-train-full-001 --gpus 2

# 提交短训练验证
bash scripts/sf.sh submit arena-sf-train-smoke-001 --arm smoke --submit

# 允许现在排队；afterok 依赖 + 运行时身份检查确保验证通过后才开始正式训练
bash scripts/sf.sh submit arena-sf-train-full-001 --arm baseline \
  --smoke-run arena-sf-train-smoke-001 --after-smoke --submit
bash scripts/sf.sh submit arena-sf-train-full-001 --arm sf \
  --smoke-run arena-sf-train-smoke-001 --after-smoke --submit
```

去掉 `--submit` 只打印命令。账户固定为 `gts-jli741-paid`。训练验证失败时，依赖它的正式作业不会启动。评测 `arena-smoke-003` 与训练验证是两个独立实验；评测成功不能替代训练验证。

如果同名作业已提交，不要再次执行上述提交命令。回执在 `runs/<run_id>/submission/`；网络超时也可能已经成功提交，必须先查询队列。正式训练由于时限停止且旧作业已退出时可续训：

```bash
bash scripts/sf.sh submit arena-sf-train-full-001 --arm baseline \
  --smoke-run arena-sf-train-smoke-001 --resume --submit
bash scripts/sf.sh submit arena-sf-train-full-001 --arm sf \
  --smoke-run arena-sf-train-smoke-001 --resume --submit
```

恢复要求配置、代码、数据、环境和 GPU 数一致。不要在 pending/running 作业期间修改训练代码。需改参数时新建 run，重新验证。

## 文件导航与结果

| 文件 | 用途 |
|---|---|
| `config/sf_training/recipe.json` | 修改步数、batch、LR、LoRA、SF λ 和 layer |
| `src/sf_training/model.py` | π0.5 forward、FM loss、VGGT 特征和对齐 |
| `src/sf_training/lora.py` | LoRA 注入、每 head 参数化、冻结范围 |
| `src/sf_training/data.py`、`preparation.py`、`pack.py` | IL 样本、action chunk、tokenization、归一化与磁盘打包 |
| `src/sf_training/train.py` | DDP、累积梯度、训练循环和诊断 |
| `src/sf_training/checkpoint.py` | policy trainable、projector、optimizer、各 rank RNG 保存恢复 |
| `src/sf_training/convert.py` | 官方 base checkpoint 转换和严格参数检查 |
| `src/sf_training/workflow.py`、`slurm/sf_train.sbatch` | 配置冻结、验证门槛、提交与恢复 |
| `src/sf_training/export.py` | 合并 LoRA，导出不需要 VGGT 的推理权重 |
| `tests/sf_training/` | CPU 单元测试与真实模型 meta 参数结构检查 |

`runs/<run_id>/<baseline或sf>/metrics.jsonl` 记录 FM loss、alignment loss、总 loss、梯度范数、学习率、每步秒数和显存。`status.json` 区分更新步数与真正完成 60k 步。`step-*.pt` 包含完整可训练状态，仍需要同一份冻结 base；这是有意避免每次保存重复的冻结权重。

```bash
source scripts/sf_common.sh
"$SF_ENV/bin/python" -m sf_training.report arena-sf-train-full-001
squeue -u gliu386 -r -o '%.20i %.10T %.36j %.12M %R'
sacct -u gliu386 -S today --format=JobID%20,JobName%36,State,Elapsed,ExitCode
```

汇总位于 `reports/<run_id>/`：`baseline-training.csv`、`sf-training.csv`、`training-summary.csv/json`、`training-curves.png`。loss 下降不等同于 Safety 改善；后续使用固定的 1,500-episode Safety 计划比较 SR、CC、成功/代价四象限，以及 static/dynamic 的失败原因、谓词 active steps 和 onset。不能根据某一组的结果另选更容易的初始化或提前停止另一组。

导出模型应在有足够 CPU 内存的计算节点内执行，不在登录节点加载完整模型：

```bash
"$SF_ENV/bin/python" -m sf_training.export \
  --plan runs/arena-sf-train-full-001/plan.json --arm sf \
  --checkpoint runs/arena-sf-train-full-001/sf/step-00060000.pt \
  --output .runtime/training/export/arena-sf-step60000
```

导出的 policy 保留同一归一化统计和配置；教师与 alignment projector 不进入动作推理。现有固定 `pi05_libero` 的 JAX 评测命令不能直接加载这种 PyTorch 训练产物，训练产物的 Arena 服务端接入需单独进行 smoke 验证后再启动比较评测。
