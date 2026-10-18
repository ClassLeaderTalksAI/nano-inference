# 本机实验结果与可使用的面试叙事

2026-10-18，本机 Apple M1 Pro / 32 GB；真实模型为固定 revision 的
Qwen3-4B-Instruct-2507 MLX 4-bit，vLLM 0.31.0+cpu / vllm-metal 0.31.0。
`+cpu` 是 core wheel 的包名后缀，实际执行由 MLX/Metal 的 Apple GPU 完成。
两个服务副本共享这块 GPU，不能当作两张 GPU 的扩容实验。

设计、命令、控制变量与后续计划见 [实验路线](interview-lab.zh-CN.md)。
原始数据位于 `results/experiments/`；每项三个配对重复，顺序 AB / BA / AB。
以下延迟数字是各轮相应统计量的中位数，改善百分比是逐对变化的中位数，两者
不必精确等于“两个中位数的比值”。这些是描述性小样本结果，不是统计显著性结论。

## 1. 结果总览

| 实验 / 数据来源 | 基线 → 改进 | 观察 | 必须同时说的边界 |
|---|---|---|---|
| 慢实例路由 / 模拟器 | RR → least-inflight | TTFT p95 48.3 → 47.1 ms，变化小且不稳定；均值 28.3 → 19.6 ms | 不能说已经解决尾延迟；这里只是注入延迟，非 GPU 性能 |
| 过载队列 / 模拟器 | queue 128 → 4 | 成功 TTFT p95 2.346 → 0.271 s，约降低 88.6% | 每轮 48 成功 → 14 成功、34 拒绝；所有到达的 SLO 达标数中位数仍为 6/48 |
| 租户调度 / 模拟器 | FIFO → WDRR | 交互 p95 2.807 → 0.319 s，约降低 88.6% | 每组每轮都完成 22 次请求；不把模拟器数字当 GPU 结果 |
| 首次前缀复用 / Metal | RR → affinity | TTFT p50 2.110 → 0.154 s，约降低 92.7%；原生 cache token 命中 0 → 99.0% | 三组前缀各预热一次后的第一次复用；每配置每轮仅 3 个测量请求 |
| 全热负对照 / Metal | RR → affinity | TTFT p50 142 → 138 ms；两边 cache token 命中均约 99.0% | 三轮一轮反向，整体属于小/混合效应，不能声称稳定收益 |
| 租户调度 / Metal | FIFO → WDRR | 交互 p95 5.226 → 1.155 s，约降低 78.0% | 两组每轮都完成 12 次请求；batch p95 4.03 → 4.60 s |
| 上下文预算 / Metal | guard off → on | 同一越界输入从通用上游 502 改为派发前的明确 400 `context_length_exceeded` | 六轮后的正常请求全部成功；这是正确分类与提前校验，不是推理加速 |

每个控制实验都保留全部到达，正式记录没有客户端丢弃、过度派发滞后或意外拒绝。
有意的过载拒绝、context 无效输入不会被当成测量丢失。
第一次被分词并发上限干扰的探索记录仍保留在 `exploratory-preparation/`，已排除。

## 2. 面试主故事：多租户队首阻塞

**场景**：一个 Worker 的总派发上限为 1。先到 8 个长输出 batch 请求，再到 4 个
interactive 请求。两个租户都未触发 RPM/TPM，上限、prompt、输出预算和到达顺序
相同；FIFO 与 WDRR 仅改变谁从租户队列获得下一次派发机会。

**观测**：FIFO 下 interactive 的网关 queue p95 约 5.165 s，TTFT p95 约 5.226 s。
这指向等待顺序，而不是客户端没按时发请求或单次生成变慢。

**改进**：把等待请求保留在独立租户 FIFO 中，用 token 成本 WDRR 选租户，再做
Worker placement。interactive:batch 权重 5:1，每租户并发仍是 1；按最终 usage
纠正估计成本，避免只按请求数公平。

**结果**：interactive 的 queue p95 降到约 1.094 s，TTFT p95 约 1.155 s。
每轮两组都完成 12 个请求，因此这次交互改善不是靠丢弃交互请求制造的。
但 batch p95 增至约 4.60 s；总输出吞吐仍约 34.4 token/s，没有明显增长。

**结论**：改变的是等待时间分配，并没有创造更多 GPU 算力。短突发实验支持
“在该负载下改善交互租户”，不支持长期 GPU 份额保证或 token 级硬时延隔离。
长期 5:3:1 成本份额另有饱和调度单元测试，它证明调度实现，不证明物理 GPU 时间公平。

## 3. 第二个故事：缓存是放置问题，也是测量问题

**场景**：同一模型有两个实例，各自缓存独立。三组不同的 598-token prompt
分别只预热一次；随后每组再请求一次。两种策略使用完全相同的提示词、采样和
输出上限，且每个试验网关使用新 salt，隔开前一试验的缓存。

**观测与机制证据**：

- RR 复用请求去了没有热该前缀的另一实例：测量阶段为 0 / 1,794 cached/queried tokens。
- affinity 回到各组的热实例：1,776 / 1,794，约 99.0%。affinity hit 只是路由判断，
  这组数字来自引擎真实 cache counter，不是把路由命中当作缓存命中。
- 原生 prefill 均值的跨轮中位数约从 2.073 s 降到 0.159 s。
- 原生 TPOT 仍约 34 ms；客户端 TTFT p50 从约 2.110 s 降到 0.154 s。

这说明改进主要来自减少 prefill 工作，不能宣传为解码加速。

**负对照**：每组都预热两次，使 RR 在两个实例也已缓存。两组命中都约 99%，
客户端 TTFT 约 142 / 138 ms，三轮中一轮反向。应说“未见稳定额外收益”，不能
挑一轮或换统计量，硬说 affinity 到处都更快。

**接下来的问题**：更大的 cache working set、cache 淘汰、负载倾斜、长 decode、
跨节点网络分别会怎样改变收益？本批没有证明这些场景，适合 CUDA 后续实验。

## 4. 一个反例：成功 p95 很漂亮，不代表服务更多有用请求

深队列变短队列之后，成功请求 p95 降低约 88.6%，但每轮 48 个到达中有 34 个被拒绝，
所有到达的 250 ms TTFT SLO 达标数中位数仍是 6/48 = 12.5%。
逐轮看，baseline 都是 6，candidate 是 7、6、6；不能把一轮多达标一个请求说成稳定改善。

报告中的 goodput/s 会变好，因为它的分母包含整个试验到队列清空的时间，短队列
结束得更快。这不是“同一个长期测量窗口里服务了更多达标请求”的证明。
必须同时看相同到达集合的达标数量/比例；下一步应做统一长时间窗口、到达率 sweep
及队列预算校准，而不是只挑成功 p95 或每秒指标来包装。

可讲成：

> 我确认了有界队列能够保护已接受请求的等待时间，但这次队列配置没有稳定增加所有
> 到达中的达标请求数。所以我把它视为可控的过载行为，而没有宣称总服务质量已经
> 提高。下一步需要基于服务时间和 SLO 校准可接受的排队预算。

## 5. 一个真实的小迭代：先修正实验，再讨论调度

最早的 burst 试验出现 `prefix_preparation_full`，原因是同时来的请求先争用
4 个分词准备名额。这里如果直接展示成功请求延迟，会把前置拒绝混入 WDRR 收益。

我保留原始记录，增加实验有效性检查；在两组中统一使用 32 个有界准备名额，
重新执行相同 schedule，确认无意外拒绝后再比较调度。生产默认仍为 4，没有把
所有保护都调大。这也引出了未来要研究的另一个 noisy neighbor：CPU 分词隔离。

这是本次真实测量中出现的发现，可以如实说“实验早期发现”；不要把预设的注入
慢实例或 FIFO baseline 说成真实客户事故。

## 6. 一段英文讲法（约 90 秒，按实际语速调整）

> I built a small LLM serving gateway around vLLM, focusing on admission control,
> multi-tenant scheduling, and cache-aware placement. I used a simulator to isolate
> control-plane behavior, then repeated selected experiments on a pinned vLLM Metal
> setup on my Mac.
>
> One experiment queued batch requests ahead of interactive requests. I kept the
> arrivals, concurrency limits, and token quotas fixed, and compared FIFO with
> token-cost weighted deficit round robin. On this local workload, interactive p95
> time to first content fell from about 5.23 seconds to 1.15 seconds, with every
> request completing. Batch latency increased and aggregate token throughput stayed
> roughly flat, so I describe this as better allocation of waiting time, not more GPU
> capacity or a hard latency guarantee.
>
> I also tested prefix locality. For the first reuse of three primed prefixes,
> cache-aware routing raised the native cached-token ratio from zero to about
> 99 percent and reduced median first-content latency from about 2.11 seconds to
> 154 milliseconds. Native prefill time fell while decode time stayed similar.
> Once both replicas were warm, the extra routing benefit was not consistent.
>
> I preserved raw requests, failures, warmups, and three counterbalanced repetitions.
> These are small local experiments on a shared Apple GPU. My next step is to repeat
> the same controls on independent CUDA GPUs before making broader performance claims.

## 7. 三分钟展开时可以画的边界

```text
Client arrival clock
   → auth / strict schema
   → actual tokenizer + model context budget
   → RPM / TPM admission
   → per-tenant FIFO + WDRR       [who runs next?]
   → health / capacity / allowed workers
   → prefix-affinity placement    [where does it run?]
   → HTTP/SSE → vLLM scheduler    [which tokens/batches execute?]
   → final usage / reservation cleanup / tenant settlement
```

自己的实现是队列、准入、跨实例路由和生命周期处理；KV 存储、batching 和模型运算
由 vLLM/Metal 提供。单 Gateway 拥有原子状态，水平扩展需要新的配额/队列所有权设计。

## 8. 面试前检查

- 能指出每个数字对应哪个 raw trial、哪个统计量与多少请求，而不是只背一个百分比。
- 能解释为什么同时看首内容、queue、原生 prefill/TPOT、失败和所有到达的达标比例。
- 能讲清 batch 的代价和全热负对照，而不把中性结果藏起来。
- 英文里的 “I” 指项目中设计、实现、测量和解释的责任；不冒充线上客户或大规模集群经历。
- 简历若使用数值，应标明 controlled local benchmark；更稳妥的概述可写：

> Built a vLLM-backed inference gateway with tenant quotas, cost-weighted fair
> scheduling, prefix-aware routing, and reproducible paired benchmarks; validated
> request isolation, cache locality, and overload trade-offs on a local Metal runtime.
