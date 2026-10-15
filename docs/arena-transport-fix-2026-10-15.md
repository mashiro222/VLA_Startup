# Arena 通信超时修复与第二次 smoke

2026-10-15：已修复本机 WebSocket 心跳超时的触发条件，完成回归检查，按用户授权提交 **`arena-smoke-002` / `14074038_0`**。账户 `gts-jli741-paid`，1 张 A100、8 CPU、64 GB、时限 2 小时，覆盖 15 条 Safety episode。**本轮没有重新提交 full；真实 GPU 闭环是否通过仍以新 smoke 为准。**

## 原故障与证据

首轮 smoke `14016239_0` 和 full `14020266_0..3` 均在首个动作返回前退出，报 `1011 keepalive ping timeout`，有效记录分别为 0/15、0/2,250。[原始失败诊断](../reports/arena-diagnostics-2026-10-15/summary.md)和五条异常保留。

旧客户端在加载仿真依赖/初始化场景前就建立连接；上游服务器使用默认 20 秒 ping interval 和 20 秒 ping timeout。客户端原生调用若长时间持有 Python GIL，接收线程无法及时回复 Pong；服务器同步推理/JIT 也可能长时间占住事件循环。用真实独立服务器进程和持有 GIL 的原生暂停，已复现旧配置的 1011 关闭。该测试确认了可触发的机制；没有对原始 GPU 故障做逐个 native call 的性能剖析，因此不宣称已确定是哪一个库调用占用了全部停顿时间。

## 改了什么

- [transport.py](../src/arena_startup/transport.py)：在场景完成初始化、第一次需要动作时才连接；沿用 OpenPI msgpack、metadata、推理处理器和 health endpoint，关闭服务器短周期 ping。兼容 websockets 14.2 客户端；若使用 15.x，也显式关闭新加入的客户端心跳。
- [server.py](../src/arena_startup/server.py)：Arena 单独启动该服务，保留同一 `pi05_libero` 权重、norm stats、JAX 采样和 RNG 回传；metadata 增加传输协议版本，避免误连旧服务。LIBERO 原服务文件未改。
- [evaluate.py](../src/arena_startup/evaluate.py)：使用懒连接客户端，补充场景开始/就绪日志。
- [runtime.py](../src/arena_startup/runtime.py)：指向 Arena 服务，提供对应导入路径。
- [config/arena.json](../config/arena.json)：单次响应最多 600 秒；模型启动 1,800 秒、单 episode 1,800 秒、Slurm 2 小时仍限制卡死。连接握手和 metadata 各限 30 秒。请求出错不会静默重试或计成失败 episode。
- 历史 `check/status/report` 可在代码更新后读取旧记录，仍验证计划 checksum 和每行计划身份；真正执行/续跑依然严格检查冻结文件。旧 run 不能直接接入新实现。

本次没有改变任务、成功判定、安全 cost、动作采样步数或权重。代码修复提交为 `32c5152`。新计划 SHA256 为 `e3280c1c6e56c35c46d2e206bf079d8cebb07c58a7a928e7ad91dff5231c04e7`，原始文件在 `runs/arena-smoke-002/plan.json`，备份见 [新 smoke 记录](../reports/arena-smoke-002-preparation/summary.md)。

## 实际完成的检查

| 检查 | 结果与范围 |
|---|---|
| `bash scripts/test_arena.sh` | 25 项通过；覆盖计分、记录/续跑、锁、提交边界、模拟 rollout、真实 WebSocket |
| 6 项通信回归 | 旧心跳重现 1011；新协议跨原生停顿/慢推理；初始化不连接；请求 deadline；错权重身份；服务异常传播 |
| 服务端环境再跑通信测试 | websockets 15.0.1 下 6 项通过；Arena 仿真环境为 14.2 |
| `bash scripts/arena.sh preflight --imports` | 75 项任务及初始状态、checkpoint 标记/文件尺寸、固定上游版本和依赖导入通过 |
| Python 编译 / shell 语法 / diff 空白检查 | 通过 |
| 图像工具 | 真实旧录像成功导出 PNG；任务列表含 75 项；登录节点禁止新场景渲染的保护生效 |
| 新 smoke 的模型推理 + EGL + 15 条完整记录 | 提交后等待实际作业验证，不能用 CPU 测试代替 |

原来 18 项测试未覆盖真实连接横跨仿真冷启动的行为，这是之前漏检的原因。这次新增 6 项通信测试及 1 项历史计划检查，共 25 项。测试里的服务器使用假策略，验证通信，不代表 checkpoint 动作正确。

## 现在查看进度

```bash
cd /storage/scratch1/1/gliu386/VLA_Startup
squeue -r -j 14074038 -o '%.20i %.12T %.24S %R'
sacct -j 14074038 --format=JobID%22,State,ExitCode,Elapsed
bash scripts/arena.sh check arena-smoke-002
tail -F logs/arena-smoke-002-14074038_0.out
```

`tail -F` 可以等待作业启动后创建日志；Ctrl-C 仅退出查看。服务日志在 `runs/arena-smoke-002/workers/*-server.log`。`PENDING` 表示尚未运行；`COMPLETED` 还需核对 15 条记录和空错误列表。任务失败是正常模型结果，通信/渲染异常则不能当成模型失败。

全部完成后，以下命令只生成本地表格和图，不提交资源：

```bash
bash scripts/arena.sh report arena-smoke-002
```

未完成但已有有效记录可以加 `--partial`。0 条记录时拒绝生成成功率图。15/15 且无基础设施错误才算链路 smoke 通过；不要求 15 次任务全部成功。之后再决定是否准备新的 full；本轮不会使用旧 waiver 入口绕过 smoke，也没有自动提交 full 的依赖作业。
