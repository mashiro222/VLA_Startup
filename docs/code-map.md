# 文件导航、参数修改与源码阅读

本地根目录是 `/storage/scratch1/1/gliu386/VLA_Startup`，以下路径相对于此目录。先完成默认 smoke，再一次只改变一个因素，用新 run ID 运行；保留默认实验作为对照。

## 优先看哪些文档

| 目的 | 文件 |
|---|---|
| 准备、提交、监控、续跑、备份的完整命令 | [README.md](../README.md) |
| 看本次做了什么，以及怎样调整和跟读代码 | 当前文件 `docs/code-map.md` |
| 闭环推理、benchmark 协议、公开结果的比较限制 | [experiment.md](experiment.md) |
| 哪些检查已实际完成，哪些要等 GPU smoke | [validation.md](validation.md) |
| 环境与作业排错 | [troubleshooting.md](troubleshooting.md) |
| 查看全部作业、手动提交四卡 full、合并报告 | [overnight.md](overnight.md) |

## 本次新增文件的分工

| 文件 | 负责什么 |
|---|---|
| [config/sources.json](../config/sources.json) | 源码 commit、checkpoint 地址、场景版本 |
| [config/checkpoint_files.json](../config/checkpoint_files.json) | 权重各对象的 generation、大小、CRC32C |
| [requirements/sim.in](../requirements/sim.in)、[sim.lock](../requirements/sim.lock) | 仿真依赖和完整锁文件 |
| [scripts/setup.sh](../scripts/setup.sh)、[bootstrap.py](../scripts/bootstrap.py) | uv、源码、Python 环境、原生 ImageMagick |
| [scripts/patch_sim.py](../scripts/patch_sim.py) | robosuite 日志改用每用户路径；不改物理仿真 |
| [scripts/download.py](../scripts/download.py) | 断点续传、checksum、安全解压 |
| [scripts/preflight.py](../scripts/preflight.py) | 提交前检查，不执行模型推理 |
| [src/vla_startup/manifest.py](../src/vla_startup/manifest.py) | 选任务、关联类别、确定初始状态和 episode 分母 |
| [src/vla_startup/evaluate.py](../src/vla_startup/evaluate.py) | 图像预处理、观测→动作→环境、成功判断 |
| [src/vla_startup/server.py](../src/vla_startup/server.py) | 官方模型加载、推理服务、采样 RNG 状态 |
| [src/vla_startup/journal.py](../src/vla_startup/journal.py) | 刷盘、残缺末行恢复、防重复和混用实验 |
| [src/vla_startup/summarize.py](../src/vla_startup/summarize.py) | coverage、成功率、micro/macro、比较条件 |
| [scripts/report.py](../scripts/report.py) | CSV、PNG、压缩记录和报告校验和 |
| [scripts/submit.sh](../scripts/submit.sh)、[slurm/evaluate.sbatch](../slurm/evaluate.sbatch) | 默认仅预览，显式 `--submit` 才提交 |
| [scripts/run_node.py](../scripts/run_node.py) | 同一 GPU 作业内管理两个进程、终止和汇总 |
| [scripts/submit_parallel.sh](../scripts/submit_parallel.sh)、[slurm/parallel.sbatch](../slurm/parallel.sbatch) | 独立四卡入口，每成员一张卡，默认仅预览 |
| [scripts/parallel.py](../scripts/parallel.py) | 分片、冻结配置、检查进度、合并完整结果 |
| [tests/](../tests) | 统计、续跑、路径、协议改变和导出测试 |
| [.gitignore](../.gitignore) | 排除权重、资产、环境、视频和运行目录 |

固定版本的 OpenPI、原版 LIBERO、LIBERO-Plus 跟踪源码保持原样。自写包装和 robosuite 日志补丁在本仓库中显式保存，可重建。

## 在哪里调参数

| 参数 | 修改入口 | 含义与限制 |
|---|---|---|
| 账户、GPU、CPU、内存、partition、QoS | 查看 `config/pace.env`，覆盖写到 `config/local.env` | 影响资源和计费，默认一张 A100 |
| 作业时限 | `TIME_LIMIT=08:00:00 bash scripts/submit.sh RUN_ID` | 再加 `--submit` 才提交，不改评测协议 |
| Flow 积分步数 | `config/evaluation.json` 的 `num_steps` | 默认 10，影响速度与动作预测 |
| 每次执行的动作数 | 同文件 `replan_steps` | 默认 5，限 1–10；预测长度仍为 checkpoint 的 10 |
| 等待、图像尺寸 | 同文件 `wait_steps`、`render_resolution`、`policy_resolution` | 默认 10、256、224；部分扰动假设原始尺寸，改后需验证 |
| 各 suite 控制步数上限 | 同文件 `horizons` | 改变失败判定时限，属于协议变更 |
| suite、任务、每任务次数 | `manifest.py` 的 `build()` 和 profile 分支 | 默认 clean 50 次、Plus 1 次；新 profile 同步 CLI、submit 和统计规则 |
| seed、prompt 来源 | manifest CLI 的 `--seed`、`--prompt-mode` | 下方有例子；实际 prompt 进入日志 |
| 视频数量 | `run_node.py` 传给 evaluator 的 `--videos` | 默认 smoke 前 2 个，其他 0；不上传视频 |
| 单 episode 墙钟超时 | `evaluate.py` 的 `--episode-timeout` 默认 1800 秒 | 超时记 error，不算模型失败 |
| 服务启动超时 | `run_node.py` 中的 900 秒 | 先看 server 日志再调整 |
| 模型结构、checkpoint、action horizon | `server.py`、上游 `training/config.py` 和下载配置 | 必须匹配权重、结构、归一化、adapter 与身份检查，不能只改一个数字 |

`prepare.sh` 把评测参数写进 run manifest。改配置请用新 run ID；preflight 会拒绝与当前配置不符的旧 manifest。不要手工修改 manifest 绕过哈希检查。默认参考比较要求默认参数、seed=7、默认 prompt、完整 full profile 和完整覆盖。改变这些参数仍有表图，但不自动标注默认参考差值。任意自定义代码修改的科学含义不能自动判断，需同时审查比较逻辑。

创建单独 seed/prompt 消融的例子（不提交）：

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
mkdir -p runs/prompt-check-001
PYTHONPATH=src python3 -m vla_startup.manifest \
  --profile smoke-plus --seed 11 --prompt-mode bddl \
  --output runs/prompt-check-001/manifest.json
python3 scripts/preflight.py --run-dir runs/prompt-check-001
bash scripts/submit.sh prompt-check-001
```

默认 `upstream` 保留发布行为，包括文件名元数据进入部分 prompt 的已知现象。`bddl` 属于单独实验，见 `experiment.md`；不要合并不同 run 的结果。

## 怎样跟读 model 的运行

先读本项目 `evaluate.py → server.py`，再打开 `.runtime/repos/openpi/` 下的文件。读取 `training/config.py` 和 checkpoint 辅助函数只是复用配置与参数加载代码，不会启动训练：

| 路径（相对于 OpenPI） | 入口与作用 |
|---|---|
| `src/openpi/training/config.py` | 搜索 `name="pi05_libero"` 查看模型与数据配置 |
| `src/openpi/policies/policy_config.py` | `create_trained_policy`：组合权重、norm stats、transforms |
| `src/openpi/policies/libero_policy.py` | 观测字段和动作的输入输出适配 |
| `src/openpi/policies/policy.py` | `Policy.infer`：转换、batch、RNG、采样、输出转换 |
| `src/openpi/models/pi0.py` | `sample_actions`：噪声、视觉语言条件、Euler 迭代 |
| `src/openpi/models/pi0_config.py` | 动作维度、长度、π0.5 与 state 配置 |
| `packages/openpi-client/src/openpi_client/image_tools.py` | 图像缩放补边与 uint8 转换 |
| `examples/libero/main.py` | 官方 rollout 示例，用于对照本项目实现 |

在线：[固定版本 OpenPI 源码](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi)。本项目 server 是加载和通信包装；神经网络实现需要跟读上游。

## Benchmark 包含哪些测试，在哪里看

本地 `.runtime/repos/libero_plus/` 下：

| 路径 | 内容 |
|---|---|
| `libero/libero/benchmark/libero_suite_task_map.py` | 四个 suite 的全部变体及顺序 |
| `libero/libero/benchmark/task_classification.json` | 名称对应的类别、难度、分类 ID |
| `libero/libero/benchmark/__init__.py` | suite 注册、任务访问、初始状态解析 |
| `libero/libero/bddl_files/` | 场景对象、语言和目标条件 |
| `libero/libero/init_files/` | 初始状态，不是训练演示数据 |
| `libero/libero/envs/env_wrapper.py` | 相机、机器人初态、噪声等扰动实现 |

原版对应 `.runtime/repos/libero_clean/`。在线：[固定版本 Plus benchmark 源码](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182/libero/libero/benchmark)。

小计划直接看 [config/plans/](../config/plans)，实际执行以 `runs/RUN_ID/manifest.json` 为准。全部 10,030 个变体可在登录节点列出：

```bash
PYTHONPATH=src python3 -m vla_startup.manifest \
  --profile full-plus --output .runtime/full-plus.json
python3 - <<'PY'
import json
from collections import Counter
m = json.load(open('.runtime/full-plus.json'))
print('Episode 数:', m['expected_episodes'])
print('Suite:', dict(Counter(t['suite'] for t in m['tasks'])))
print('类别:', dict(Counter(t['category'] for t in m['tasks'])))
for t in m['tasks'][:10]:
    print(t['suite'], t['task_id'], t['category'], t['name'])
PY
```

生成器同时检查实际 BDDL 和初始状态路径。用完整 name 关联分类，不要把分类表的一基 ID 当 API 的零基 task_id。
