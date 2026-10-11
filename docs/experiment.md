# 实验协议与阅读路线

本项目使用作者已经在 LIBERO 上训练过的 `pi05_libero` checkpoint，不自行 fine-tune，也不称为未经任务训练的基础模型 zero-shot 实验。原版 LIBERO 检查任务表现，LIBERO-Plus 检查相机、机器人初始状态、语言、光照、背景、噪声和物体布局变化下的表现。

## 一次闭环发生什么

```mermaid
flowchart LR
  A[任务清单与初始状态] --> B[LIBERO / MuJoCo 环境]
  B --> C[两路图像、机器人状态、指令]
  C --> D[OpenPI 输入处理]
  D --> E[GPU 上的 π0.5]
  E --> F[10 个未来动作]
  F --> G[执行前 5 个动作]
  G --> B
  B --> H[成功或达到步数上限]
  H --> I[episode 记录与表图]
```

1. Benchmark 指定任务定义 BDDL、初始状态和语言。Plus 包装器根据变体名改变环境/观测，因此评测时必须传递上游的**逻辑变体文件名**；磁盘上的 BDDL 可以是去掉 `_view_...` 的基础文件。直接批量改名会改变实验。
2. 环境重置并加载指定初始状态，先执行 10 个等待动作让物体稳定。环境生成 256×256 的前视和腕部相机图像。
3. 与 OpenPI 示例一致，将两幅图像旋转 180°、保持比例缩放并补边到 224×224。状态接口是末端位置 3 维、轴角 3 维和夹爪位置 2 维，共 8 维。语言使用上游 `task.language`，实际传入内容写入日志。
4. OpenPI 的 policy adapter 执行图像/语言处理及 checkpoint 对应的归一化。π0.5 的视觉语言主干为动作预测提供条件；动作分支从噪声出发，按 flow matching 学到的向量场做 10 步 Euler 积分，输出一个动作片段。本项目调用作者实现，不重新实现网络。
5. 此配置的预测长度是 10。内部动作维度经过 adapter 后成为 LIBERO 的 7 维控制向量（末端运动及夹爪）；执行前 5 个动作，然后用新观测重新规划。这是闭环控制，不是给整段视频一次性打分。
6. 环境返回任务成功则结束，否则到 horizon 后记失败。运行时异常中止并单独保存，不能混入模型失败率。

具体配置有个容易误读的地方：锁定的 `pi05_libero` 设置 `discrete_state_input=False`，且 π0.5 不走 π0 的连续状态 token 分支。虽然客户端保留标准 8 维 state 接口，也不能据此断言该配置把它作为 state token 输入网络。请结合上游 `training/config.py`、`models/pi0.py` 和 `policies/libero_policy.py` 阅读。

同一 GPU 作业内运行两个进程：Python 3.11/OpenPI/JAX 提供本机 WebSocket policy 服务；Python 3.10/LIBERO/MuJoCo 执行环境并发送观测。服务只监听 `127.0.0.1`。隔离环境避免两套项目的依赖冲突；并不意味着申请两张 GPU。

## 固定协议

| 参数 | 本项目值 |
|---|---|
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero/` |
| 模型 backend | JAX，官方 Orbax 参数 |
| NumPy / 环境 seed | 7 |
| 模型初始 RNG | 与 OpenPI 默认 `jax.random.key(0)` 一致 |
| Flow 积分步数 / 预测长度 / 执行长度 | 10 / 10 / 5 |
| 图像 render / policy 分辨率 | 256 / 224 |
| 等待步数 | 10，不计入下列控制步数 |
| spatial / object / goal / libero_10 horizon | 220 / 280 / 300 / 520 |
| clean 完整协议 | 每 suite 10 个任务，每个前 50 个初始状态 |
| Plus 完整协议 | 每逻辑变体 1 次，使用上游返回的第 0 个初始状态 |
| 排序 | spatial → object → goal → libero_10；各 suite 按上游 task_id |
| 默认 prompt | 上游 `task.language`，不清洗文件名元数据 |

三个上游 commit、uv 版本和资产 SHA 固定在 `config/sources.json`。checkpoint 的每个 GCS object 固定 generation、长度、CRC32C；tokenizer 固定 SHA256；不是每次执行时下载浮动的 latest。sim 依赖由本仓库 lock 固定，OpenPI 依赖使用该 commit 自带的 `uv.lock`。

分类 JSON 的 `id` 从 1 开始，benchmark API 的 `task_id` 从 0 开始；本项目按完整任务**名称**关联，避免类别错位。完整 Plus 清单检查结果：

| Suite | 变体数 |
|---|---:|
| libero_spatial | 2,402 |
| libero_object | 2,518 |
| libero_goal | 2,591 |
| libero_10 | 2,519 |
| 总计 | 10,030 |

| 扰动类别 | 变体数 |
|---|---:|
| Camera Viewpoints | 1,599 |
| Robot Initial States | 1,550 |
| Language Instructions | 1,537 |
| Light Conditions | 1,142 |
| Background Textures | 1,076 |
| Sensor Noise | 1,601 |
| Objects Layout | 1,525 |

## 如何与公开结果比较

原版 LIBERO 官方 `pi05_libero` 参考：spatial 98.8%、object 98.2%、goal 98.0%、libero_10 92.4%，四项等权平均 96.85%。这些是**原版**分数。来源：[OpenPI 固定版本的 LIBERO 说明](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)。

Plus 仓库 issue #57 中，Skywalker-yqz 在 2026-07-13 自报使用同一 checkpoint：Camera 67.3%、Robot 78.8%、Language 86.3%、Light 95.8%、Background 93.8%、Noise 87.3%、Layout 86.4%，Overall 84.2%。这是**社区自报**，不是官方 π0.5 leaderboard 条目；未提供全部协议与逐变体记录，因此只能作描述性对照。来源：[原始报告](https://github.com/sylvestf/LIBERO-plus/issues/57)。

总体成功率按 `全部成功 episode / 全部预期 episode` 计算，完整 Plus 分母是 10,030；各类别样本量不同，不能把七个类别成功率简单平均当 Overall。汇总器另行提供 category macro，但明确分开。完整 clean 四个 suite 各 500 次，micro 与四 suite 等权均值相同。

没有足够依据规定“误差小于某个数就一定复现正确”。先核对 coverage、checkpoint、prompt、仿真版本、图像方向、horizon、replan、初始状态和 seed，再分析分数。共享基础任务的 Plus 变体并非独立随机样本，不能随意把普通二项置信区间解释成全面泛化保证。

本项目为 clean 和 Plus 使用同一套 MuJoCo 3.2.3 / robosuite 1.4.0 仿真依赖，与社区 issue #65 披露的组合一致；并未宣称与 OpenPI 最初产出官方 clean 分数时的所有依赖、硬件和采样过程完全相同。它是可追踪的对照实验；若要做严格复现，还需进一步核对作者的完整运行环境。

**已知 prompt 差异：**锁定版本会把部分非语言变体的文件名参数带入 `task.language`。社区在 Camera/object 子集上观察到清洗 prompt 后分数变化；这不等于整个 Plus 的固定增益。为避免暗中修改 benchmark，默认保留发布行为并记录实际 prompt。可用 `manifest --prompt-mode bddl` 创建单独的消融实验，但汇总器会禁用其默认参考比较。来源：[issue #65](https://github.com/sylvestf/LIBERO-plus/issues/65)。不要把自定义 prompt 的分数标成与默认协议严格可比。

## 阅读代码的顺序

1. `src/vla_startup/manifest.py`：任务怎样选择，类别怎样关联，episode 分母如何确定。
2. `src/vla_startup/evaluate.py`：reset → 初始状态 → 观测处理 → infer → env.step → success。
3. `src/vla_startup/server.py`：加载官方配置与 checkpoint，保存/恢复采样 RNG。
4. 上游 `openpi/src/openpi/policies/policy_config.py`、`policy.py`、`libero_policy.py`：归一化与网络输入输出。
5. 上游 `openpi/src/openpi/models/pi0.py` 的 `sample_actions`：动作噪声如何逐步变成控制序列。
6. Plus `libero/libero/envs/env_wrapper.py` 与 `benchmark/__init__.py`：扰动与初始状态如何生效。
7. `journal.py`、`summarize.py` 和 `scripts/report.py`：错误、缺失、续跑和统计如何处理。
