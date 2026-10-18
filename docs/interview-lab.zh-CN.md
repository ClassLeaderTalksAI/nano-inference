# nano-inference：从基线到证据的面试实验路线

这个项目的主线是：**我搭了一个围绕 vLLM 的小型推理服务平台，用受控实验识别
调度、准入、缓存和协议边界的问题，再用数据检验改动是否值得保留。**

它是 LLM inference gateway；当前没有视觉输入，面试时不要称为 VLM 服务。
项目实际开发顺序见 Git 历史：协议/路由 → 有界准入与测量 → 前缀亲和 → Mac Metal
运行时 → 多租户 → 本批受控实验。下面的 baseline 是对现有实现关闭部分优化后的
参考配置，不等同于逐个 checkout 历史提交，更不是过去线上事故的记录。

## 1. 一个能讲清楚的迭代过程

| 阶段 | 从哪里开始 | 测试要暴露什么 | 改进与开关 | 看什么结果 |
|---|---|---|---|---|
| 0 正确性 | 一层 OpenAI-compatible HTTP/SSE 代理 | 把空 role delta 当首 token；流中断误报成功；客户端离开后占位不释放 | 正确 SSE 状态机、绝对超时、取消和幂等清理 | 增量输出；错误没有 DONE；active/queue 回到零 |
| 1 负载分布 | round_robin，两个健康实例 | 一个实例慢，平均分请求未必平均分负载 | `--policy least_inflight` | Worker 分布、均值/尾部 TTFT、失败率和 dispatch lag |
| 2 过载保护 | 较深的等待队列 | 接受更多请求，却让它们过了交互目标才开始生成 | `--max-queue 128 → 4`，其他参数固定 | 成功 TTFT + 成功数量 + 所有到达的 SLO 达标比例 + goodput |
| 3 多租户 | 同一准入规则下按可运行队首 FIFO 派发 | 大批长输出请求先到，交互请求被挡在后面 | 租户配置 `scheduler: fifo → wdrr` | 每租户 TTFT/queue、批处理完成量、实际 token 成本 |
| 4 缓存局部性 | 引擎 APC 开着，网关 round_robin | 某前缀在 A 热了，下一次却去了 B | `--policy prefix_affinity` | 原生 cache token 命中、引擎 prefill、网关首内容时间 |
| 4b 负对照 | 两个实例上相关前缀都已热 | 优化是否只在某种缓存条件下有效 | 同样两种策略，增加相同的预热次数 | 全热后是否仍有收益；不要强行解释微小差异 |
| 5 输入预算 | 只校验 JSON 字段范围 | 合法 JSON 仍可能超过模型上下文，最终成为上游错误 | `--context-guard / --no-context-guard` | 是否在派发前明确返回 400；Worker/租户额度是否受影响 |

阶段 0 的故障、取消与恢复仍由原有测试和 `scripts/smoke_vllm.py` 覆盖。本批不为了
实验而开放“绕过身份验证/缓存隔离”的开关，也不会把所有保护一次关闭。

## 2. 三个测试批次

先安装平台依赖并准备模型（已有 `.local/metal-model` 时不必重复下载）：

```bash
uv sync --locked --extra dev --extra tokenizer
uv run python scripts/prepare_metal.py
```

### Batch A：协议、计数与有效性检查

```bash
uv run --extra dev --extra tokenizer pytest -q
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
```

这里检查“实现是否正确”，不拿单元测试运行时间当推理性能。
新检查包括：相同 seed 的到达序列完全一致、派发延迟/客户端丢弃/意外拒绝使对照
失效、退化结果保留、上下文预算边界、原生直方图 delta，以及实验报告的 HTML 转义。

### Batch B：在模拟器中隔离控制逻辑

```bash
uv run nano-inference experiment --batch control
```

- routing：60 次开放到达，60 req/s；固定注入一快一慢的实例行为，使用相同 seed
  的短输出混合。比较 round_robin 与 least_inflight。
- overload：48 次到达、80 req/s、两个相同模拟实例；仅改变队列上限 128 / 4。
  每实例在途上限都是 1，TTFT 目标都是 250 ms。
- tenants：一个实例、16 个 batch 请求先进入，随后 6 个 interactive 请求到达；
  batch 输出上限 48，interactive 上限 4。仅改变 FIFO / WDRR，权重保持 5:1，
  每租户并发都为 1、队列都为 64、总队列 128、等待超时 20 s。

模拟器的延时和 token 数是合成值。这个批次说明控制逻辑在指定扰动下的作用，
不说明真实 GPU 的吞吐或 TTFT 改善。

### Batch C：真实本机 vLLM Metal

```bash
uv run --extra tokenizer nano-inference experiment --batch metal
```

启动自己的两个 Worker 和临时网关，使用空闲端口；结束后清理所属进程。不改动你
正在运行的服务。为了结果可解释，运行此批次时不要另开模型负载或并行跑 Batch B。

- cache：三个相同长度的长前缀组，每组预热一次，随后各测第一次复用。相同的
  tokenized prompt 分别经过 round_robin / prefix_affinity。每组只预热一次，
  round_robin 的第一次复用会落到另一副本；这正是该实验要观察的边界。
- cache_steady：相同三组，每组预热两次，然后测复用。round_robin 已能在两个
  实例命中，是负对照，不应把 cache 实验的收益推广到此场景。
- tenants：只使用一个真实 Worker，先到 8 个 batch 请求，再到 4 个交互请求。
  两组同样只改 FIFO / WDRR，输出上限 24 / 4，TTFT 目标 2 s。
- context：同一个真实 Worker；使用超过“输入 + 输出上限”预算的请求比较保护
  关闭/开启，再发一个正常请求，验证实例仍可服务。

缓存实验的两个配置使用同一份提示词，而每个网关的服务端 salt 不同，避免前一组
试验的缓存污染后一组。全程保持租户隔离，未开启引擎的管理/RPC 接口来清缓存。
原生 counter 必须推进到本阶段全部请求完成，才开始下一阶段或计算差值。

两个 Mac Worker 共用同一块 Apple GPU。cache 实验串行发请求，主要观察缓存局部性；
租户实验把在途上限设为 1，主要观察跨请求队列。它们不能证明两张 CUDA GPU 的扩展性，
也不能证明共享 GPU 上 token 级的硬时延隔离。

## 3. 实验纪律：如何避免“为了证明优化有效而做优化”

所有正式实验默认三次配对重复，顺序为 AB / BA / AB。`configs/experiments.local.json`
记录重复次数、固定 seed、客户端 outstanding 上限与最大派发延迟。每个原始报告保存：

- 每条请求的计划到达时间、内容、输出上限和租户；两组重放同一份 schedule。
- 网关参数、唯一改变的 knob、模拟器注入参数、真实运行时/模型 revision。
- 所有成功、拒绝、失败、客户端丢弃，以及逐请求 TTFT、queue、usage、Worker。
- 冷/热状态、排除在测量外的预热、原生 cache/TTFT/prefill/decode/TPOT delta。
- Git HEAD、工作区是否有修改、Python 源码 SHA256、运行环境。

**有效性检查先于性能结论**：客户端丢到达、派发滞后超过默认 50 ms、成功缺 usage、
非过载实验意外拒绝或失败，都会标为不可比较。过载实验有意允许 429；context 实验
有意测试错误。不能删掉慢样本、意外拒绝或失败后再宣称 p95 改善。

“连续三个方向一致、幅度超过 5%”只是报告中的描述标签，不是统计显著性检验。
反向结果会标为 regression，混合或小效应会保留为 mixed，小样本不报 P99。正式面试
若要引用百分比，应同时说明负载、缓存条件、请求数和重复次数。

### 本批实际发现的一个干扰项

最初的模拟器多租户实验中，22 个突发请求遇到默认 4 个分词准备名额，六轮各出现
2–4 个 `prefix_preparation_full`。它们根本没有进入公平队列；仅比较成功 TTFT 会把
前置拒绝的影响混到调度收益里。

原始记录保留在 `results/experiments/exploratory-preparation/`，不纳入最终性能结论。
之后两个调度配置统一使用有界的 32 个准备名额，保持其余参数一致再测。生产默认仍
是 4；这不是“越大越好”的调优结论，而是隔离实验变量，也提示未来要研究每租户
CPU/tokenization 准入，而不是把所有 noisy neighbor 都归因于 GPU。

## 4. 看指标时从哪一层开始

| 观察到的现象 | 优先查看 | 可以提出的假设 | 不能直接推出 |
|---|---|---|---|
| 客户端首内容慢 | dispatch lag → gateway queue → native TTFT/prefill | 客户端没按时发、网关拥塞、prefill 很重 | “GPU 一定慢” |
| batch 下交互慢 | 每租户 queue、在途上限、派发顺序、每租户 TTFT | 队首阻塞或租户占满容量 | WDRR 能保证每个 token 的延时 |
| cache affinity 很高但 TTFT 不降 | 原生 cached/queried tokens、prefill、输出长度、排队 | 提示已过期、引擎没命中、排队/解码占主导 | 路由 affinity hit 就是 GPU cache hit |
| p95 降了很多 | 成功数、429/错误、所有到达的 SLO 达标、goodput | 拒绝保护已接受工作，也可能只是样本被筛掉 | 服务了更多用户或总吞吐提高 |
| TPOT 仍高 | 原生 decode/TPOT、并发、KV 压力、GPU 带宽 | 解码瓶颈或 GPU 干扰 | 开 APC 必然提升解码 |
| 延迟极低但指标少 | usage 完整性、阶段请求完成计数、采样时间 | 采集滞后或漏算 | 没有指标的部分就是 0 |

TTFT 在此指首个非空内容；queue 来自网关原子预留的等待时间。原生引擎指标的边界
不同，不能机械地把这些时间相加视为精确分解。
APC 主要减少 prefill，见 [vLLM 官方说明](https://docs.vllm.ai/en/v0.13.0/features/automatic_prefix_caching/)；
当前版本的指标定义见 [vLLM 0.31.0 metrics](https://docs.vllm.ai/en/v0.31.0/design/metrics/)。

## 5. 如何复盘结果

`results/experiments/control/summary.json` 与 `metal/summary.json` 是索引，各 trial
有单独的 JSON 原始记录。生成的 `results/experiments/report.html` 是汇总页面。
重跑默认覆盖同名 trial 文件；要保留一轮新研究，请指定新的 `--output` 目录。只使用
完整结束、且 summary 索引与 trial 相符的批次，不把中断运行的残留文件当作正式结果。

正常启动任意本地网关后，在监控页点 **Experiments**，或打开
**http://127.0.0.1:8000/experiments**。实时 `/monitor` 回答“现在系统怎么样”，实验页
回答“同一负载下两个配置各表现怎样”。二者都是本机运维入口。

填充并复核实测数字后再使用 [本次结果与面试讲法](interview-results.zh-CN.md)。

## 6. 面试准备：先讲主线，再展开一个瓶颈

30 秒主线可先这样说，不填未经核实的性能百分比：

> 我做了一个围绕 vLLM 的 mini inference gateway，重点研究共享推理实例上的准入、
> 多租户公平调度和缓存局部性。我先建立可运行的简单基线，再用固定到达序列逐项
> 比较。模拟器用于隔离控制逻辑，真实 Mac Metal 用于验证引擎集成和缓存指标。
> 我把拒绝和失败一起报告，也做了缓存全热的负对照，避免只展示有利条件。

进一步只展开一个故事：**场景 → 观测 → 假设 → 一处改动 → 对照与代价 → 下一步**。
例如 noisy neighbor 不要同时讲“我还用了 Docker、Kubernetes、Redis……”；应画清
“认证 → 准入 → per-tenant FIFO → WDRR → Worker placement → vLLM local scheduler”。

最容易被追问的点：

1. **为什么不是只用 vLLM scheduler？** 它管理一个实例内的 token/批次；网关控制
   租户额度、跨租户等待顺序和跨实例放置。进入引擎的请求有界，网关才保有控制力。
2. **为什么不按请求数公平？** 长 prompt/长输出成本不同。用近似 token 成本并按
   最终 usage 纠偏；系数尚未校准，不说它等于 GPU 时间。
3. **为什么 FIFO/WDRR 对照仍有相同并发限制？** 才能把差别归因于调度顺序；一次
   改权重、并发、队列、cache 开关会产生混杂。
4. **如何防止 batch 饿死？** 有持续排队的正权重 DRR；另有长期成本份额单元测试。
   短突发的交互收益不等于已验证长期 GPU 资源份额。
5. **为什么需要 salt？** tenant_id 只隔离网关提示表；引擎必须实际按租户 salt
   分开 KV block 哈希。已有同一 Worker 的跨租户冷/热 counter 验证。
6. **为什么不重试中断的流？** 已输出内容无法透明拼接为同一生成；避免隐藏部分
   失败和放大过载。本项目明确报错，未做跨 Worker 的无损续写。
7. **这些结果能推广到 RunPod 吗？** 协议/队列正确性可迁移，性能结论必须重测。
   两个 Mac 进程共享 GPU，没有独立显存、跨节点网络和 CUDA kernel 的同等条件。

## 7. 现在值得扩展什么

| 优先级 | 下一步 | 要回答的问题与验收 |
|---|---|---|
| 已加入 | 上下文 token 预算 guard | 原先在下游报错的越界请求能否在派发/额度预留之前清楚拒绝 |
| 已加入 | 实验有效性检查与可浏览报告 | 能否重放、保留反例，避免客户端瓶颈/丢请求制造漂亮曲线 |
| 下一批 P0 | Linux/CUDA 复测相同 matrix | 固定驱动、vLLM、模型 revision；一张 GPU 再两张，记录显存/内核/网络差异 |
| P1 | 成本系数与准入容量校准 | 用 prompt/output 长度 sweep 拟合 prefill/decode 趋势，与简单系数比较 |
| P1 | 分词资源的租户隔离 | 批量长输入是否能挤掉交互租户；比较共享并发、租户并发和输入预算 |
| P1 | dedicated/shared 物理 GPU 对照 | 更稳定的交互延时换来多少空闲容量；Mac 无法验证物理隔离 |
| P2 | 基于队龄与服务估算的提前拒绝 | 在相同到达负载下提高所有到达的 SLO goodput，且预测误差可见 |
| 暂缓 | K8s、autoscaling、分离 prefill/decode、speculative decoding | 先把当前实验和瓶颈讲透，避免引入尚无验收数据的新技术栈 |

下一批扩展应由本批数据暴露的问题驱动。不要为了简历多写一个组件而失去可解释性。
