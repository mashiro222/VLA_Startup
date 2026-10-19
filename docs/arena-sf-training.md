# Arena π0.5 LoRA / Spatial Forcing 训练

实验进展、故障和调整决定统一见[实验记录](results-overview.md)；环境重建、提交与同配置恢复见[运行手册](run/training-operations.md)，24 小时保存机制和扩卡限制见[checkpoint 与资源说明](run/training-checkpoints.md)。

## 实验定义

2026-10-19 后续协议调整为：两组分别完成原两卡任务的 24 小时分配，使用各自最后保存的 checkpoint 评测 **900 episodes**，并分别以四卡、micro-batch 2、累积 4 **追加 15,000 步**，global batch 仍为 32。两组分别手动提交，不自动调度，不设置总卡数上限，不固定 5k/20k 步数；参见[后续流程](run/after-24h-followup.md)。下文 60k / micro-batch 1 表格描述最初冻结的上游参考配方。

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

图像由数据集已有方向直接缩放到 224×224；模拟器评测时的 180° 图像修正不能再应用于训练集。动作与 state 使用训练集 q01/q99 归一化，再补零到 32 维。若某维 q99−q01 ≤ 1e-6，则改用该维训练数据的实际最小值/最大值；恒定维度使用以该值为中心、宽度为 1 的区间。正常分位数维度保持原值，不裁剪动作，不丢弃稀疏旋转。原始分位数、实际范围和回退维度均保存在统计文件中。此回退是本实验的显式适配，两组完全一致，不应表述为严格复现 Arena 的原始统计实现。遵循 Arena π0.5 配置，`discrete_state_input=false`；state 不被加入语言 token，π0.5 路径也不使用 π0 的连续 state projection。该选择在两组一致。

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

DDP 每卡仍持有完整策略、教师（SF 组）和相应优化状态，增加卡数不会把模型参数显存除以卡数。BF16、activation checkpointing、micro-batch=1 用于控制每卡显存。实际峰值与吞吐观测统一见[实验记录](results-overview.md)；短 smoke 只提供粗略测量，不能保证 60k 步在一个 24 小时作业内结束。临近时限会保存 checkpoint，后续显式续训。

## 文件导航与结果

| 文件 | 用途 |
|---|---|
| `config/sf_training/recipe.json` | 修改步数、batch、LR、LoRA、SF λ 和 layer |
| `src/sf_training/model.py` | π0.5 forward、FM loss、VGGT 特征和对齐 |
| `src/sf_training/lora.py` | LoRA 注入、每 head 参数化、冻结范围 |
| `src/sf_training/data.py`、`preparation.py`、`pack.py` | IL 样本、action chunk、tokenization、归一化与磁盘打包 |
| `src/sf_training/normalization.py`、`validation.py` | 安全归一化、精确逆变换、全数据审计与 smoke 数值/覆盖门槛 |
| `src/sf_training/train.py` | DDP、累积梯度、训练循环和诊断 |
| `src/sf_training/checkpoint.py` | policy trainable、projector、optimizer、各 rank RNG 保存恢复 |
| `src/sf_training/convert.py` | 官方 base checkpoint 转换和严格参数检查 |
| `src/sf_training/workflow.py`、`slurm/sf_train.sbatch` | 配置冻结、验证门槛、提交与恢复 |
| `src/sf_training/export.py` | 合并 LoRA，导出不需要 VGGT 的推理权重 |
| `tests/sf_training/` | CPU 单元测试与真实模型 meta 参数结构检查 |

`runs/<run_id>/<baseline或sf>/metrics.jsonl` 记录 FM loss、alignment loss、总 loss、梯度范数、学习率、每步秒数和显存。`status.json` 区分更新步数与真正完成 60k 步。`step-*.pt` 包含完整可训练状态，仍需要同一份冻结 base；这是有意避免每次保存重复的冻结权重。

汇总位于 `reports/<run_id>/`：`baseline-training.csv`、`sf-training.csv`、`training-summary.csv/json`、`training-curves.png`。loss 下降不等同于 Safety 改善；后续使用固定的 1,500-episode Safety 计划比较 SR、CC、成功/代价四象限，以及 static/dynamic 的失败原因、谓词 active steps 和 onset。不能根据某一组的结果另选更容易的初始化或提前停止另一组。

导出前核对归一化 SHA256 与训练计划一致；旧 run 不能误用修复后的统计。导出的 policy 保留同一归一化统计和配置；教师与 alignment projector 不进入动作推理。现有固定 `pi05_libero` 的 JAX 评测命令不能直接加载这种 PyTorch 训练产物，训练产物的 Arena 服务端接入需单独进行 smoke 验证后再启动比较评测。

## 归一化审计与回归测试

[全训练集审计](../reports/preparation/sf-normalization-audit.json)覆盖 361,736 帧的全部 state/action 坐标。每个 action chunk 只会选择这些帧并在轨迹末尾重复最后一帧，因此此扫描也覆盖所有可能的训练动作值；统计仍按 clamped chunk 加权。审计检查数值数组 SHA256、逐维归一化最大绝对值，以及 normalize/unnormalize 往返误差。原统计保存在 `.runtime/training/normalization_archive/`。新计划同时冻结统计文件和审计文件的 SHA256。[前 32 个正式 batch 的回放](../reports/preparation/sf-sparse-batch-replay.json)包含 17 个带稀疏旋转的样本；原异常第 5 步的归一化目标最大绝对值为 1，32 维 padded 平方均值约 0.119。

训练与推理使用成对变换：`z = 2*(x-q01)/(q99-q01+1e-6)-1`，`x = (z+1)/2*(q99-q01+1e-6)+q01`。推理必须沿用训练时保存的有效 q01/q99；不能重新计算或换用 `pi05_libero` 的统计。`tests/sf_training/test_normalization.py` 覆盖低于 1% 的稀疏旋转、恒定维度、正常分位数保持、无裁剪往返、极大有限目标，以及 smoke 的逐步 loss 和稀疏样本门槛。`test_workflow.py` 验证 smoke 未通过时不会自动提交正式任务。
