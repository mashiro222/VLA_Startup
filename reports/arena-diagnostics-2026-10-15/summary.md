# Arena 首轮运行失败诊断：2026-10-15

后续已完成通信修复并提交新的单卡 smoke；见 [修复与回归测试](../../docs/arena-transport-fix-2026-10-15.md)。下文保留修复前的原始诊断，不作为新作业状态。

10 月 15 日上午检查：smoke 和 full 已在 10 月 14 日凌晨启动过，但所有 worker 均异常退出，**没有任何完成的 episode，不存在有效成功率或性能曲线**。

| 作业 | 开始—结束（10 月 14 日 EDT） | Slurm 状态 / 退出码 | 完成数 |
|---|---|---|---|
| smoke `14016239_0` | 01:10:10—01:26:02 | FAILED / 1:0 | 0 / 15 |
| full `14020266_0` | 02:22:29—02:37:56 | FAILED / 1:0 | 0 / 2,250（全体合计） |
| full `14020266_1` | 02:27:50—02:37:56 | FAILED / 1:0 | — |
| full `14020266_2` | 02:32:31—02:45:47 | FAILED / 1:0 | — |
| full `14020266_3` | 02:32:31—02:45:47 | FAILED / 1:0 | — |

五个 worker 的共同异常：

```text
websockets.exceptions.ConnectionClosedError:
received 1011 (internal error) keepalive ping timeout;
then sent 1011 (internal error) keepalive ping timeout
```

异常位于 `src/arena_startup/evaluate.py` 的首次 `client.infer(...)`，下层在 `self._ws.send(data)` 时发现连接已关闭；没有得到首个动作响应。权重和 checkpoint 内的 normalization 已加载，模型服务已监听本机端口。它不是已完成机器人试验的失败记录，不能把这些异常计成模型成功率 0%。

Smoke 保存的视频已实际解码检查：只有 1 帧，224×224 RGB，能看到机器人、苹果和盘子。因此至少该任务的初始场景渲染已执行；这仍不能证明模型生成动作或完整闭环已通过。

目前的定位：客户端在导入仿真依赖和构建场景之前就建立 WebSocket，日志显示首次请求前初始化耗时数分钟。固定 OpenPI 服务端没有配置 keepalive 参数，当前环境默认 `ping_interval=20`、`ping_timeout=20`。收到的关闭码表明服务端心跳超时；**具体是哪个长耗时环节阻塞了客户端响应，仍需复现确认**。客户端实际使用 `websockets==14.2`，不是 15.x 升级引起的新客户端心跳行为。

之前的 18 项测试覆盖计分、断点、调度入口和模拟 rollout，没有覆盖真实 WebSocket 连接跨越仿真冷启动的场景，漏掉了这个运行时问题。应补充通信延迟回归验证，修复连接建立时机和心跳超时处理后再运行；本次状态检查没有修改评测代码，也没有提交新的计费作业。旧运行的失败记录保留。

结构化检查见 [summary.json](summary.json)，五条原始异常见 [errors.jsonl](errors.jsonl)。本地完整日志在 `logs/arena-smoke-001-14016239_0.out`、`logs/arena-full-001-14020266_0.out` 至 `_3.out` 及各 run 的 `workers/*-server.log`。

参考：[websockets keepalive 与超时说明](https://websockets.readthedocs.io/en/stable/topics/keepalive.html)。本页只报告基础设施故障，不生成缺乏有效样本的成功率图。
