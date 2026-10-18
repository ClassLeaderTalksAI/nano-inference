# 本地 Mac flavor：实现与审核

当前先完成 Apple Silicon 本地运行，再审核 Linux/CUDA 的 RunPod 部署。

后续已加入多租户队列、成本公平调度、配额、缓存 salt 和本地监控页面；详见
[多租户与本地监控](multi-tenant.zh-CN.md)。下文的 FIFO 路径描述未启用租户配置时的行为。
平台代码共用，推理运行时、模型格式、依赖锁和内存参数按 flavor 配置。
此前的 llama.cpp CPU 验证保留为参考工具；Mac 主运行时采用 vLLM Metal。

## 运行结构

```text
客户端 / 同一套 benchmark
            |
       Gateway :8000
       请求校验、有限 FIFO 队列、超时、取消
       Round Robin / Least In-flight / Prefix Affinity
            |
       HTTP + SSE
         /      \
 vLLM Worker A   vLLM Worker B
   :8101           :8102
 vllm-metal       vllm-metal
       \          /
         MLX / Metal
     同一块 Apple GPU / 统一内存
```

两个 Worker 是两个服务副本，各有模型状态和前缀缓存。它们共用 Mac 的物理
GPU 与内存，所以可以验证实例间路由、缓存局部性、过载和故障处理，但不能据此
声称获得了两张独立 GPU 的吞吐扩展。

## 一次请求怎样走

1. Gateway 验证支持的聊天字段，拒绝未实现的 API 参数。
2. Prefix Affinity 模式使用实际模型 tokenizer 和 chat template，计算完整
   输入的前 64 个 token。模型 revision、tokenizer 指纹与模板指纹一同进入标识。
3. 请求进入有界准入过程；健康 Worker 都满时进入有限 FIFO 队列，超限返回 429。
   Worker 选择和请求计数预留在同一把锁下完成，避免并发请求读到相同的旧负载。
4. Router 优先选择可能保有该前缀缓存的实例；该实例更忙或容量不足时，转向
   负载较低的实例。亲和表是路由提示，不代表缓存块一定仍然存在。
5. vLLM 接收请求，负责内部 batching、token 调度、KV cache 管理和生成。
   vllm-metal 通过 MLX/Metal 执行模型；Gateway 不参与模型数学计算。
6. Gateway 持续转发 SSE。正常结束、超时、客户端断连和上游失败均释放连接与
   预留名额。中途失败产生错误事件，已经输出的生成不会无损迁移到其他实例。

## 哪些是我们实现的

| 部分 | 本项目负责 | 使用的现成实现 |
|---|---|---|
| 平台入口 | 请求子集、流式代理、超时与取消 | FastAPI / HTTPX |
| 跨实例路由 | 三种策略、原子负载预留、缓存亲和提示 | Python asyncio |
| 准入控制 | 有限队列、等待截止时间、明确拒绝 | Python asyncio |
| 实例生命周期 | 启动与清理、健康检查、失败隔离与重新接流量 | 本地进程 / vLLM HTTP API |
| 模型运行 | 配置、版本/模型固定、兼容性检查 | vLLM / vllm-metal / MLX |
| KV cache | 前缀身份与路由决策 | 实际 KV 分配、存储和复用由运行时负责 |
| 实验工具 | workload、客户端观测、原始报告、故障场景 | 引擎原生 Prometheus 指标 |

这里有两层调度：我们决定请求去哪个实例，vLLM 决定实例内部如何调度 token。
项目没有重写 Transformer、PagedAttention 或 Metal kernel。

## 可复现的本地配置

- 配置：`configs/macos-metal.json`。
- vLLM core：`0.31.0+cpu`；vllm-metal：`0.31.0`；MLX：`0.32.1`。
- 模型：`mlx-community/Qwen3-4B-Instruct-2507-4bit`，固定完整 commit revision。
- 运行时依赖：`requirements/macos-metal.lock`；平台依赖：`uv.lock`。
- 每个实例：2,048-token 上下文，最多 2 个引擎内请求，内存预算比例 0.15。
- Gateway：每实例最多 2 个在途请求，最多 4 个排队请求。

`+cpu` 是 macOS vLLM core wheel 的包名后缀。实际模型执行方式要看插件与设备，
不能仅凭这个后缀判断。验证时应看到 `MLX device set to: Device(gpu, 0)`，并检查
原生引擎的生成、缓存与队列指标。Metal 的内存预算作用于统一内存，不等同于
NVIDIA 显卡上独立显存的相同百分比。

## 检查和证据

```bash
uv sync --locked --extra dev --extra tokenizer
uv run python scripts/prepare_metal.py
uv run --extra tokenizer python scripts/smoke_vllm.py
```

完整检查使用临时端口启动两个真实 Worker 和同一套 Gateway，并在结束时清理。
结果位于 `results/macos-metal/verification.json`，三个 workload 的逐请求数据分别
位于同目录的 `prefix.json`、`mixed.json` 和 `overload.json`。

检查内容包括：每个实例的 tokenizer 精确匹配、普通与流式响应、usage 和结束
事件、真实前缀缓存命中 token、客户端取消、正常/混合/过载 workload、生成中
强制结束一个 Worker、存活实例继续服务、重启后两个实例重新接收请求。

取消检查先观察请求处于引擎运行状态，再断开客户端，随后要求 Gateway 和
引擎队列归零，正常完成计数未增加。原生 `abort` 标签的计数会保留在报告中，
但不会假设每次 HTTP 取消都令该计数增加。缓存计数只在同一个引擎生命周期内
计算增量，避免把重启后的计数重置错误地当作性能变化。

本轮实际结果：

| 检查 | 结果 |
|---|---|
| 机器 | Apple M1 Pro，32GB，macOS 26.7.1 |
| 两实例 tokenizer | 每个实例的长前缀与多轮中文样例都逐 token 一致 |
| 原生前缀缓存 | 冷/暖两次请求共查询 420 tokens，其中 192 tokens 命中 |
| 前缀 workload | 12/12 成功 |
| 混合 workload | 9/9 成功 |
| 过载 workload | 4 成功、8 明确拒绝，保留逐请求错误码 |
| 客户端取消 | 已观察到运行状态，取消后队列/名额归零，正常完成计数未增加 |
| Worker 失败/重启 | 断流有错误、另一实例接流量、重启后两实例均接流量 |
| 结束清理 | 服务停止，测试端口释放 |
| 自动测试 | 68 项通过；代码与格式检查通过 |

420 是两次请求的查询 token 总数，包含首次冷请求；不能将 192/420 当成
暖请求独立命中率，更不能将其解释为延迟优化百分比。
Router 使用的 64-token 分组长度也不是缓存复用上限；引擎会按实际 token 序列
与缓存块进一步匹配，所以本例可以复用更长的 192-token 前缀。

少量请求的结果用于证明系统行为。它们不能构成稳定的 P99、性能最优或生产容量
结论，也不能直接与不同精度、不同模型文件的 CUDA 结果比较。

## 与最初计划的差距

| 目标 | 当前范围 | 仍需完成 |
|---|---|---|
| 两个真实推理实例 | Mac 上两个 vLLM Metal 实例 | 两张独立 NVIDIA GPU 的验证 |
| 跨实例调度 | 三种基础策略 | token-aware 成本模型与策略对照实验 |
| 前缀缓存 | 实际 token 前缀、亲和路由、原生缓存计数 | 多前缀热点与容量压力下的系统对照 |
| 过载保护 | 固定容量、有界队列和拒绝 | 根据 token/KV 压力自适应控制 |
| 可靠性 | 超时、取消、实例失败和恢复 | 更长时间的稳定性与反复故障实验 |
| 性能分析 | 客户端延迟/吞吐、原生 TTFT/TPOT 指标 | 重复实验、正式图表和 GPU profiling |
| 部署复现 | Mac 原生准备与启动 | CUDA/RunPod 的实际部署和回读验收 |

## 两个 flavor 共用到哪一层

| 层 | macOS Metal | 未来 Linux CUDA |
|---|---|---|
| Gateway / 路由 / 队列 / 健康检查 | 同一份代码 | 同一份代码 |
| 请求格式 / SSE / benchmark | 同一套接口与 workload | 同一套接口与 workload |
| vLLM API 与原生指标检查 | 本地实际验证 | 复用检查，仍需实际验证 |
| 运行时环境 | macOS ARM Python + vllm-metal + MLX | Linux Python + CUDA vLLM |
| 模型文件 | MLX 4-bit 转换 | 选择适配 CUDA 的格式并单独固定 |
| 内存与设备配置 | 统一内存、同一块 Apple GPU | 独立显存、每实例选择 NVIDIA GPU |

将来整套平台可以部署到 GPU 服务器上，Mac 只作为操作端。当前停在本地 flavor
的审核阶段，远程 GPU 实例和云端费用不属于本轮执行范围。

参考：[vLLM Metal](https://github.com/vllm-project/vllm-metal/tree/v0.31.0)、
[支持模型](https://github.com/vllm-project/vllm-metal/blob/v0.31.0/docs/supported_models.md)、
[运行时配置](https://github.com/vllm-project/vllm-metal/blob/v0.31.0/docs/configuration.md)。
