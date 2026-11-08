# TestLocalImage 性能提升版本

> 口径：`TestLocalImage` 单次耗时（CPU 预处理 + TRT 推理），5 次循环平均，单位 ms。
> 基线：初始版本（`267c4d8`，2026-10-31）≈ 85 ms。数据来源：commit message 与代码注释中的实测值。

| 版本 | 性能 | 提升原因 | 分析链接 |
|---|---|---|---|
| `28be0c4` 11-07 Optimize TRT deploy | ≈ 85 ms → **1.6 ms**（↓约 53×） | ① 固定 shape 1×3×224×224，消除 conv 回退 generic 核（主因）；② 引擎落盘缓存，跳过约 8 s 重建；③ context/stream/缓冲初始化一次复用 + pinned 异步传输 | （待补充） |

