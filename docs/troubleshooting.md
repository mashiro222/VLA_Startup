# 环境与作业排查

<a id="vscode-reconnect"></a>

## VS Code 每隔几秒 reconnect

2026-10-11 的远端检查发现两类直接证据：

- `~/.vscode-server/data/logs/20261011T193953/remoteagent.log` 和 `20261011T194023/remoteagent.log` 记录文件监听器被 `SIGKILL`；`20261011T194036/remoteagent.log` 记录扩展进程被 `SIGKILL`，终端服务也异常退出。
- 当时 `login-phoenix-gnr-3` 上当前账号的 cgroup 限额 `memory.max=4294967296`（4 GiB），`memory.events` 为 `oom=26`、`oom_kill=16`。检查时用量约 3.0 GiB，同一账号下仍有两套 VS Code Server，以及 Codex 进程。限额由该节点上的账号进程共同使用，并非每个 VS Code 窗口各有 4 GiB。

这些证据强烈支持内存压力造成服务退出。`oom_kill` 是该 cgroup 的累计值，不代表 16 次都来自 VS Code；截图终端是 `gnr-1`，当前检查在 `gnr-3`，且无权读取内核日志，因此不能把每次断线都精确归因到同一个节点或进程。这里的 4 GiB 是实测值，不是对 PACE 所有节点的通用承诺。计费作业申请的 64 GB 内存不会增加登录节点的额度。

先按以下顺序处理，无需提交 GPU 作业：

1. 在 VS Code 中关闭多余的 PACE 远程窗口。重新连接后，用 **File → Open Folder** 打开 `/storage/scratch1/1/gliu386/VLA_Startup`。在 home 的 Explorer 中展开这个目录，不等于更换工作区；此时项目的 `.vscode/settings.json` 也不会作为工作区设置生效。
2. 仓库已加入监听和搜索排除项，跳过 `.runtime/`、`.tools/`、缓存和原始运行输出，并关闭 Pylance 全项目索引、只诊断打开的文件。这会减少扫描负担；对自行扫描文件的第三方扩展不保证有效。文件未删除，Explorer 仍可展开；上游源码仍可按 [code-map.md](code-map.md) 直接打开。默认全局搜索不包含 `.runtime/`，需要搜索上游源码时可临时关闭搜索面板的排除/忽略开关。以后主动修改 `.runtime/` 内源码时，要留意这些目录的文件变更监听被关闭。
3. 如果仍断线，在远程扩展列表中暂时禁用当前不用的 AI/语言扩展，特别是先对 **GitHub Copilot Chat** 做排查。上述日志还记录 `github.copilot-chat-0.42.3` 的 `PendingMigrationError: navigator is now a global in nodejs`；这是另一项扩展兼容性线索，不能单凭它认定 OOM 的来源。只查看文档、代码和表格无需启动全部扩展。
4. 普通窗口关闭后，远端服务可能继续保留。如果需要清理残留服务，先保存编辑、确认该 VS Code 的集成终端没有要保留的任务，再在命令面板执行 **Remote-SSH: Kill VS Code Server on Host...**，然后只重新打开一个项目窗口。此操作会中断对应服务的终端和依附其运行的任务，不能把它当作无影响的刷新操作。本次诊断没有替用户执行这一步。

在发生问题的那个 VS Code 终端里，可以只读检查该节点的用量与累计 OOM 计数：

```bash
hostname
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.current
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.events
ps -u "$(id -u)" -o pid,ppid,rss,etime,comm --sort=-rss | head -20
```

内存值单位为字节，`ps` 的 RSS 单位为 KiB。如果复现前后 `oom_kill` 增长，说明期间该账号又有进程因内存不足被结束。RSS 相加不等于 cgroup 用量，后者还包括文件缓存和内核内存。没有增长也不能排除网络或其他节点的问题。见 [Linux cgroup v2 文档](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html) 和 [VS Code 文件监听说明](https://github.com/microsoft/vscode/wiki/File-Watcher-Issues)。

如果减少内存占用后仍重连，再看 **View → Output → Remote - SSH** 的客户端日志。Phoenix 通用登录入口可能分配不同节点；VS Code 官方说明，两次 SSH 连接落到不同节点会导致服务/隧道不匹配。日志中的 `Unknown reconnection token` 也可能只是服务重启的后果，不能单凭它认定负载均衡故障。

macOS 可在**本机 Mac**（不是 PACE）的 `~/.ssh/config` 中，将下列选项合并到实际使用的 PACE Host 段；保留原有身份认证设置。如果平时连接别名，例如 `pace`，应编辑该别名对应的段：

```sshconfig
Host login-phoenix.pace.gatech.edu
    HostName login-phoenix.pace.gatech.edu
    User gliu386
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

然后在 Mac 终端运行 `mkdir -p ~/.ssh/sockets`，重新建立远程连接。`ControlMaster` 让多个 SSH 会话复用同一连接，减少节点不一致的问题；keepalive 不能解决内存不足。配置依据：[VS Code 官方 Remote-SSH 排查文档](https://code.visualstudio.com/docs/remote/troubleshooting#_connecting-to-systems-that-dynamically-assign-machines-per-connection)。本仓库不能修改你 Mac 上的配置。

## 评测环境与作业

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

`preflight.py --imports` 在共享存储冷缓存时可能需要数分钟读取大型依赖。它会打印当前导入阶段；环境导入本身不等于模型推理。检查器和作业入口都把 Matplotlib 等缓存放在项目的 `.runtime/cache/`，不要求 home 目录可写。

由于 `robosuite==1.4.0` 在多用户环境中默认写共享 `/tmp/robosuite.log`，本项目只修改它的日志文件位置。补丁随 setup 可重复应用并记录在 Git 中，不改物理仿真、动作或 benchmark 源码。

`episodes.jsonl` 以完整换行作为提交边界；被杀死时未写完的末行在续跑时截断重做。已有完整行损坏、重复任务、顺序变化或指纹不符会报错，不能默默跳过。备份的 `reports/RUN_ID/episodes.jsonl.gz` 可解压回新建的 `runs/RUN_ID/episodes.jsonl`，连同原 `manifest.json`、`run_config.json` 恢复；必须重建同版本环境和代码后再续跑。

官方场景 ZIP 除约 6.4 GB 下载量之外，还包含约 45 万个小文件、解压约 9 GB。在共享文件系统上，创建文件和目录会占用明显时间。下载脚本用四个 I/O 工作线程、有限待处理队列和父目录复用解压；不删减上游资产。解压中断后重新运行会检查已存在文件的大小及 ZIP CRC，复用完整文件并修复残缺文件。只有全部完成后才写完成标记。

Scratch 的自动清理周期由 PACE 当前政策决定。本仓库不依赖其永久保留；代码和已提交的小型报告存在 GitHub，大文件可按固定版本重新下载。视频默认不备份，若有研究价值需另外复制到组内持久存储。
