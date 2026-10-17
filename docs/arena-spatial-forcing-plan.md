# π0.5 LoRA 与 Spatial Forcing：Arena 对照设计

状态：实验设计与代码审查；**没有开始训练，尚无可直接运行的 π0.5+SF+LoRA 训练入口**。当前已具备的是固定 `pi05_libero` 的 Arena 评测链路和扩展诊断实现。

## 假设与比较对象

假设：在相同模仿学习数据、优化预算及可训练策略参数下，使用冻结几何教师的中间表示对齐，能否减少 Static/Dynamic Obstacles 的接触代价，同时保持或提高任务成功率？SF 不直接学习安全约束；改善空间表示不保证避障，也不保证语义安全。

| 组别 | 初始化 | Arena 数据 | 训练损失 | 用途 |
|---|---|---|---|---|
| 迁移参照 | 官方 `pi05_libero` | 不再训练 | 无 | 描述 LIBERO→Arena 迁移差距 |
| 主 baseline | 同一份官方 `pi05_base` | 固定 L0-L 训练集 | flow-matching MSE | Arena LoRA 适配 |
| SF | 同一份官方 `pi05_base` | 与 baseline 相同 | 同一 MSE + λ × spatial alignment | 检验 SF 的增量作用 |
| 作者模型（可选） | Arena 已发布模型 | 作者训练条件 | 作者方法 | 外部参照，不替代主 baseline |

不以“证明基础模型很差”为采样或停止条件。低 zero-shot SR 可以反映任务迁移问题；安全缺陷需要依据明确违规谓词与 CC。LoRA+SF 对 LoRA baseline 的差异才是核心比较。

## 数据与协议

训练集为 `VLA-Arena/VLA_Arena_L0_L_lerobot_openpi`，固定 revision `8c6cdcac185c7c97582a78f8ad8c2ef3757ce2c7`：3,018 条示范、361,736 帧、60 个任务，覆盖 Arena 全部能力域的 L0-L，不是 Safety 专属数据集。采用这份完整训练集是为了与 Arena 配方对齐；若改用 Safety 子集，必须同时修改两组并重新计算统计。

每个 IL 样本是图像两路、8 维机器人状态、文本指令和未来 10 个 7 维动作。没有额外的安全分类标签。训练只用 L0；评测分别报告 L0、L1、L2，后两者作为难度泛化。训练/验证按示范轨迹划分，不把相邻帧随机拆成独立验证集；测试任务结果不用于挑 λ 或 checkpoint。

两组均使用 PyTorch、同一代码路径和精度，避免把 JAX/PyTorch 差异混入 SF 效果。策略初始化、LoRA 初始化、数据顺序、增强、动作噪声与评测清单配对固定；VGGT 使用独立随机流，避免教师初始化改变策略或采样随机数。

## 训练配方与实现检查

参考 Arena `pi05_vla_arena_low_mem_finetune` 的配置：60,000 updates、global batch=32、AdamW、warmup=1,000、peak LR=2.5e-5、cosine decay 到第 30,000 步的 2.5e-6，此后保持末值；EMA 关闭。PaliGemma attention/FFN 的 LoRA rank=16、alpha=16，action expert rank=32、alpha=32。全局 batch 是每卡 microbatch × GPU 数 × gradient accumulation；两组须相同。若先做短预算探索，两组都在相同步数评测，并明确标记未收敛。

上游 JAX LoRA freeze filter 冻结 LLM/action expert 的原参数，保留 LoRA 参数训练；它没有冻结所有非 LLM 参数，视觉编码器及动作投影等仍可训练。PyTorch 移植时必须逐参数核对模块对应关系与 trainable manifest，不能假设 PEFT 的默认设置就等价。SF projector 为额外训练模块；VGGT 固定权重、`eval()`、`no_grad()`，不进入 optimizer。

损失是 flow-matching 速度场回归，而不是把最终动作直接做平方误差：

```python
x_t = t * noise + (1 - t) * normalized_action_chunk
target_velocity = noise - normalized_action_chunk
velocity, visual_hidden = policy(observation, x_t, t)
loss_fm = mean((velocity - target_velocity) ** 2)
loss = loss_fm
if spatial_forcing:
    with no_grad():
        teacher_hidden = frozen_vggt(unaugmented_multiview_images)
    loss_sf = masked_alignment(projector(visual_hidden), teacher_hidden)
    loss = loss + lambda_sf * loss_sf
loss.backward()
optimizer.step()  # 仅显式声明的可训练参数；SF 分支还包括 projector
```

λ=0 baseline 与 λ=0.5 SF 可作为预先指定的首轮配对；0.5 来自 SF 示例，尚未证明适合 π0.5/Arena/LoRA。暂沿用 PaliGemma 中间层索引 12、VGGT 最后一层、bilinear 对齐；需要验证 π0.5 的 token/层索引、无效相机和 padding mask。数据增强引起的几何变换也必须保持教师/学生 token 对齐。

官方 SF 的 π0 路径在 commit `372819b31ba9a2c8fc5989edc7d525cb187cecd5` 下有以下接入工作：

1. `train_align_pytorch.py` 没有实现 Arena JAX 的 `freeze_filter`/LoRA 注入；直接复制配置名不足以得到 LoRA 训练。
2. π0.5 的 adaRMS 等分支需要前向、梯度与加载验证，示例本身是 π0。
3. λ=0 时上游仍计算 VGGT；应跳过教师分支，且确保策略的 flow-matching 路径与 SF 组一致。
4. 上游保存函数保存 policy 与 optimizer，但没有显式保存独立的 `align_projector` 参数；本实验须保存/恢复 projector、LoRA、归一化统计、优化器、scheduler 与全部 RNG，进行断点恢复等价测试。
5. 当前本项目 JAX 服务写死 `pi05_libero` 身份与 RNG；训练权重接入需增加 PyTorch/LoRA checkpoint 身份验证和随机状态控制。两组须使用同一推理实现。

这些是训练开始前的必要工程工作，不是已经验证通过的功能。训练耗时和显存须在统一 global batch 下实测，不由评测速度推算。

## 评测与主要指标

每个 checkpoint 使用 [balanced1500](arena-diagnostics.md)，全部模型配对同一任务/初态/seed。Static 与 Dynamic 是预先指定的主要分析对象，其余三类是补充分析，不按结果事后筛选 suite。

主要指标：SR、原 CC、成功且零 cost 的比例、四格 outcome 表。机制指标：gripper/object contact 的 CC 分量、谓词接触段起始次数、激活步数、Fall 终态罚分、达到 horizon 的比例、最后目标谓词、轨迹长度。所有失败 episode 仍计算 cost。

若 SF 降低 CC 但也降低 SR，或仅因更少接近物体而减少接触，不能解释为安全操控能力改善。若 SR 上升、接触持续时间和代价在相同/相近完成度下下降，才提供更直接证据；仍不能凭两组指标单独证明几何表示是唯一机制。

首轮一个训练 seed 可作为探索；关键结果建议至少两组配对训练 seeds。评测置信区间按任务/初态层级的配对重采样计算，不能把同任务的重复 rollout 当完全独立样本。L1/L2 测试集不参与 early stopping。最终 checkpoint 或训练验证集选出的 checkpoint 的选择规则应在运行前固定。

## 依据

- [Arena 固定训练配置](https://github.com/PKU-Alignment/VLA-Arena/blob/babe582ebffc82b979b77964a7e56417d02f63a4/vla_arena/models/openpi/src/openpi/training/config.py)
- [SF 固定 PyTorch 训练入口](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/372819b31ba9a2c8fc5989edc7d525cb187cecd5/openpi-SF/scripts/train_align_pytorch.py)
- [SF 模型与对齐损失](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/372819b31ba9a2c8fc5989edc7d525cb187cecd5/openpi-SF/src/openpi/models_pytorch/pi0_align_pytorch.py)
- [OpenPI LoRA freeze filter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)
