# 环境与作业排查

先看 `logs/RUN_ID-JOB_ID.out`，模型侧细节看 `runs/RUN_ID/server-JOB_ID.log`。无需 W&B 账号。

| 现象 | 处理 |
|---|---|
| preflight 报缺 checkpoint、资产或版本不符 | 登录节点运行 `bash scripts/setup.sh`、`bash scripts/download.sh`，再检查；不通过时不要提交 |
| `/tmp/robosuite.log` PermissionError | setup 自动运行受版本检查约束的 `patch_sim.py`，改为包含当前 UID 的临时日志；不改别人的文件 |
| Wand 找不到 MagickWand | 使用本仓库入口，检查 `.runtime/native-7.1.1/`；不要把 ImageMagick 的库路径全局注入模型进程 |
| 新版 ImageMagick 要求更高 GLIBC | 已固定可在当前 PACE 登录节点加载的 7.1.1-28；不自动升级 AppImage |
| LIBERO 询问配置路径 | 应用本仓库的 `.runtime/config/clean` 或 `plus`；setup 会生成，无需编辑 `~/.libero` |
| 变体 BDDL 名看起来没有对应物理文件 | Plus 有虚拟文件名，包装器负责解码；manifest 已检查实际 BDDL 和 init 文件 |
| CUDA 或 EGL 错误 | 必须在分配到 GPU 的 Slurm 作业内排查；默认单 A100、`MUJOCO_GL=egl`。preflight 导入通过不代表 EGL 已渲染成功 |
| JAX 第一次 infer 很慢 | 包含首次编译；查看 server 日志。不能用首个 episode 直接估计全量耗时 |
| 模型超出显存 / 启动超时 | 保存 server 日志与 GPU provenance，停止该作业后排查；不要直接大量重复提交 |
| walltime 到期或手动取消 | 等旧作业彻底结束，使用相同 run ID 再提交；不要改已开始实验的代码/参数 |
| summary 显示 missing | 尚未完整跑完；不要把 completed 子集成功率当全量成绩 |
| GitHub push 身份失效 | 在 PACE 终端重新 `gh auth login`，不要把 token 放进仓库或聊天 |

只读检查安装包和配置不创建 MuJoCo 场景、不加载模型。真正的 rollout 留给 GPU smoke 作业，避免在登录节点运行评测。

由于 `robosuite==1.4.0` 在多用户环境中默认写共享 `/tmp/robosuite.log`，本项目只修改它的日志文件位置。补丁随 setup 可重复应用并记录在 Git 中，不改物理仿真、动作或 benchmark 源码。

`episodes.jsonl` 以完整换行作为提交边界；被杀死时未写完的末行在续跑时截断重做。已有完整行损坏、重复任务、顺序变化或指纹不符会报错，不能默默跳过。备份的 `reports/RUN_ID/episodes.jsonl.gz` 可解压回新建的 `runs/RUN_ID/episodes.jsonl`，连同原 `manifest.json`、`run_config.json` 恢复；必须重建同版本环境和代码后再续跑。

官方场景 ZIP 除约 6.4 GB 下载量之外，还包含约 45 万个小文件、解压约 9 GB。在共享文件系统上，创建文件和目录会占用明显时间。下载脚本用四个 I/O 工作线程、有限待处理队列和父目录复用解压；不删减上游资产。解压中断后重新运行会检查已存在文件的大小及 ZIP CRC，复用完整文件并修复残缺文件。只有全部完成后才写完成标记。

Scratch 的自动清理周期由 PACE 当前政策决定。本仓库不依赖其永久保留；代码和已提交的小型报告存在 GitHub，大文件可按固定版本重新下载。视频默认不备份，若有研究价值需另外复制到组内持久存储。
