# Static Obstacles GT-CBF：准备记录（2026-11-11）

状态：**代码与本地检查已完成；未提交任何新作业；没有本实验的真实模型评测结果。**

设计、文件导航和全部运行命令见 [实验说明](../cbf-static-obstacles.md)。目标是分析官方 Arena 微调 π0.5 经局部 CBF 过滤后的任务完成、接触和卡住变化，不进行模型训练。

## 已完成

- 核对官方 `VLA-Arena/pi05-vla-arena-finetuned` 固定版本与推理文件清单。完整推理文件约 12.44 GB，逐文件 SHA-256 校验；优化器状态未下载。模型与资产均不上传 GitHub。
- 核对 `pi05_vla_arena`：action horizon 10、action dim 32、连续 state 输入；官方 stats 为 state 8 维、actions 7 维，采用 quantile normalization。官方 config、policy、server transport、tokenizer/transforms 的 CPU 导入和构造已验证；未在登录节点加载模型或执行推理。
- 独立增加 OSQP 依赖与官方模块需要的 pytest 导入依赖，没有改动已有 server/arena venv。
- 15 项测试通过：有限差分梯度、MVEE 包络、球膨胀包络、单障碍理想轨迹及 MuJoCo 无接触、双障碍可行间隙、坐标旋转、不可行回退、空障碍 identity、配对清单、卡住/碰撞标签、尺度拟合、完整记录与部分结果报告等。
- 对全部 15 个 Static 任务核对 BDDL 障碍数与 50 个初始状态；对 Apple L0/L1/L2 实例执行无渲染的真实物理、几何、抓取接口和 substep contact 检查。
- 真实物理集成测试使用明确的假策略及黑色测试图像，验证 C0/C2 的 rollout、官方成本对账、录像/轨迹写出、部分结果统计与绘图。测试临时文件不作为实验结果。
- 修复了报告在 rate=0/1 时由浮点误差产生负误差棒的问题；最终集成测试通过。

准备中曾出现一次额外 `faulthandler` 导入探针异常退出；其输出未用作通过依据。最终独立服务导入检查完成，配置、归一化和 tokenizer 构造均通过。GPU 初始化、显存及驱动适配仍由后续真实 smoke 检查。

## 证据

- [测试日志](../../reports/cbf-static-preparation/tests.log)
- [15 个任务与代表场景的几何审计](../../reports/cbf-static-preparation/scene-audit.json)
- [模型服务导入检查](../../reports/cbf-static-preparation/server-import.log)
- [准备清单与提交预览](../../reports/cbf-static-preparation/readiness.json)
- [checkpoint 固定版本和校验值](../../config/cbf_static/checkpoint.json)

## 本地准备目录

| Run | 内容 | 当前状态 |
|---|---|---|
| `runs/cbf-calibration-001` | 10 条 L0 C0，用于实际动作尺度标定 | 已准备，可在选定 GPU 后显式提交 |
| `runs/cbf-smoke-draft-001` | 40 条配对 smoke | 草案，取得合格标定后新建正式 run |
| `runs/cbf-main-draft-001` | 3,000 条 L1/L2 主实验 | 草案，参数与单元清单已生成 |
| `runs/cbf-l0-draft-001` | 1,000 条 L0 C0/C2 | 草案 |
| `runs/cbf-gamma-draft-001` | 2,000 条可选 L2 gamma 扫描 | 草案 |

已经生成的 calibration run 不必再次 `prepare`；其余正式 run 根据 `calibration.json` 中的实测 k 生成。草案默认 k=0.05 仅表示控制器名义输出尺度，提交入口会拒绝未带标定的非 calibration run。不会在后台自动标定、提交、续跑或取消其他作业。

每个正式数组单元为一个任务/等级/条件/全部采样种子，单元内部 100 episodes，申请一张 GPU、8 CPU、64 GiB 主机内存。并发数与 A100/H100 均为提交时显式参数，计费账户 `gts-jli741-paid`。各数组并发上限独立计算。

## 尚待实际 GPU 作业验证

1. checkpoint 恢复、JAX 编译、真实图像和 EGL 渲染。
2. L0 无过滤动作的实际尺度 k、留出样本误差与各轴一致性。
3. 40 条 smoke 的 L0 C0/C2 与 L1 C1/C2 配对一致性、动作方向、官方 CC 对账、solver 状态及录像。
4. 本实验实际单 episode 耗时、GPU 显存峰值、A100/H100 吞吐量。

尚未产生 SR、CC、卡住率或方法优劣结论。主实验含完整 L0 对照共 4,000 个正式 episodes；另有 10 条标定和 40 条 smoke，可选 gamma 扫描再增加 2,000 条。
