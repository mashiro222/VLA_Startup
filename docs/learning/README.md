# 学习笔记：π0.5 训练、推理与评测

按讲解顺序记录，方便回查。每篇以记录当天的代码为准；之后代码有变动时，以源码为准。

| 篇 | 内容 | 记录日期 |
|---|---|---|
| [01 闭环总览](01-closed-loop-overview.md) | 流程图里每个方框的含义：BDDL、初始状态、MuJoCo、观测、WebSocket、π0.5、动作块、判分、报告 | 2026-10-15 |
| [02 推理闭环代码精读](02-inference-walkthrough.md) | evaluate.py、server.py、WebSocket、Policy.infer、sample_actions 逐段讲解（不含训练） | 2026-10-15 |
| [03 VLA-Arena](03-vla-arena.md) | Arena 是什么；Safety 五类 75 个任务；CBDDL；SR 与 CC 的计分规则；本项目的协议、代码与进展；为什么只测 Safety | 2026-10-15 |
| [04 Spatial Forcing](04-spatial-forcing.md) | 原文 π0 LoRA 实验与公开代码的对应关系；本项目 π0.5 的 LoRA 参数范围、VGGT 对齐、梯度路径和实验协议 | 2026-10-21 |
| [05 从图像与指令到动作：π0.5 与 Flow Matching](05-pi05-flow-matching.md) | 输入 token 与张量形状、action expert、训练与推理、ODE/Euler 基础、CFM 数学推导、论文与公开实现的区别、可运行演示与自测 | 2026-10-20 |

项目自身的文档索引见 [research-guide.md](../research-guide.md)。
