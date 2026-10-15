# 学习笔记：π0.5 推理评测流程

按讲解顺序记录，方便回查。每篇以记录当天的代码为准；之后代码有变动时，以源码为准。

| 篇 | 内容 | 记录日期 |
|---|---|---|
| [01 闭环总览](01-closed-loop-overview.md) | 流程图里每个方框的含义：BDDL、初始状态、MuJoCo、观测、WebSocket、π0.5、动作块、判分、报告 | 2026-10-15 |
| [02 推理闭环代码精读](02-inference-walkthrough.md) | evaluate.py、server.py、WebSocket、Policy.infer、sample_actions 逐段讲解（不含训练） | 2026-10-15 |
| [03 VLA-Arena](03-vla-arena.md) | Arena 是什么；Safety 五类 75 个任务；CBDDL；SR 与 CC 的计分规则；本项目的协议、代码与进展；为什么只测 Safety | 2026-10-15 |

项目自身的文档索引见 [research-guide.md](../research-guide.md)。
