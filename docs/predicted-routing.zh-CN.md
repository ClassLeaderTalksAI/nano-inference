# 从前缀亲和到预测 TTFT 路由

## 本批交付与职责

请求路径：认证 → tokenization/上下文检查 → 配额 → 租户 FIFO → WDRR →
健康/容量/allowlist 过滤 → `Placement.select_worker()` → vLLM → SSE/usage 结算。

新增 `--policy predicted_ttft`，保留 RR、least-inflight、prefix-affinity 三种对照。
WDRR 分配租户的派发机会；Router 对获选请求选择预计首内容延迟最小的 Worker。
vLLM 继续负责 continuous batching、实际 KV 分配/淘汰和 token 执行顺序。

本批的 SLO 能力是 **deadline-aware placement + 可选 admission**，不是 EDF、
租户优先级提升或严格 GPU 时延隔离。没有改变 WDRR 的权重和成本结算规则。

## 预测模型

对每个候选 Worker，单位都转换为秒，再比较：

```text
predicted_remaining_TTFT = base + queue + uncached_prefill + kv_pressure_penalty

queue = (本代 Worker 的在途预估工作秒数之和
         + 原生指标中额外请求数 × external_request_seconds) / parallelism

uncached_prefill = max(1, input_tokens - cache_reuse_discount × matched_tokens)
                  / prefill_tokens_per_second

kv_pressure_penalty = max(0, kv_fraction - kv_soft_limit)
                      / (1 - kv_soft_limit) × kv_penalty_seconds

remaining_budget = API 到达时刻 + tenant.ttft_slo_seconds - 当前时刻
feasible = predicted_remaining_TTFT × slo_safety_factor <= remaining_budget
```

没有 tenant config 时使用 `default_slo_seconds`。budget 包括网关分词和等待消耗；
`X-Predicted-TTFT-Seconds` 则是**从 Worker 派发开始**的剩余预测，不含已经消耗的等待。
客户端基准的 TTFT 从预定到达计时，两者不能直接相减；客户端延迟还含 dispatch lag。
误差 Histogram 在网关内用派发到首次非空内容的时间进行比较。

每个在途请求的工作量预留包括自身 base/prefill/KV 估计和预计输出的 decode 时间。
单租户使用输出上限，多租户使用 `min(max_tokens, expected_output_tokens)`。
选择和预留共享 condition lock，并发请求立即看到新预留；正常结束、异常、取消都归还。
工作估计在完成前不衰减，可能高估已经生成大部分输出的请求；`parallelism` 是有效并发
近似，不等于 vLLM `max_num_seqs`，更不是 continuous batching 的精确仿真。

原生 running/waiting 已包含网关流量，不能和本地计数直接相加。实现按 scrape 时的
本地预留数量扣除重叠；scrape 与请求生命周期仍可能存在竞态，所以额外工作估计也是
近似值。超过 `metrics_max_age_seconds`、失败或 generation 不匹配的指标不参与预测。
缺少指标时使用本地预留与配置先验，decision 为 `predicted_ttft_local`；KV 未知显示
`kv_cache_fraction: null`，不会伪装成原生观测的零压力。此时 KV 惩罚项省略。

## 当前缓存索引：压缩 token radix tree

- 原来四个固定长度哈希检查点已替换为真实 token 路径的压缩前缀树，支持压缩边内部
  的任意长度 LPM。默认单路径最多 4096 tokens，全局同时限制记录、节点和 token 数。
- 按可信租户、模型、revision、tokenizer/template 身份隔离；同一前缀可以存在于多个
  Worker。树中的 token IDs 属于私有数据，不出现在日志、Prometheus 或 monitor JSON。
- 只有成功请求完成后记录；读取不延长 TTL，按成功使用顺序淘汰；健康状态变化清除
  提示与原生样本，旧 generation 的完成不能重新预热新实例。
- `cache_reuse_discount` 折减近似命中的价值。设为 0 可以消融缓存因素；禁用引擎 APC
  时也应设为 0。`kv_penalty_seconds=0` 可以消融 KV 惩罚。

这仍是 **success-history cache hints**，没有接收引擎 BlockStored/BlockRemoved 事件。
它不知道真实淘汰；LPM 覆盖保留路径范围，`cached_tokens` 最多为 input_tokens - 1。
`cached_tokens` 是路由估计，
真实 cached-token 比例必须看引擎原生 counters，不能互相替代。
另见 [cache-aware 策略与树实现](cache-aware-routing.zh-CN.md)。

## SLO 开关和资源账本

`slo_admission=false` 为默认：选择最小预测 TTFT，记录 feasibility，但不据此拒绝。
同一个 SLO 对所有候选 Worker 给出同一个剩余预算；最小 TTFT 的 Worker 同时是最有
机会达标的 Worker。因此 SLO 不需要另加一个任意分数，或故意选择较慢但达标的副本。

启用 `slo_admission=true` 后，派发前预测不满足预算的请求返回 429
`slo_budget_exceeded`；排队也不会越过其首内容 deadline。未派发的拒绝归还 TPM
和 WDRR 成本，不消耗 Worker/tenant inflight，RPM 仍消耗一次。
其他租户可继续获得派发机会。已经进入引擎的工作不会因为预测变差而被中途抢占。

**示例参数是未校准先验，不应直接用于有承诺的拒绝策略。** 本机冒烟中冷请求实际
首内容曾达到数秒，而示例模型预测不足一秒；热请求也可能被高估。安全系数不是保证。
保持拒绝关闭，先在目标引擎上校准并观察误差，再做 admission A/B。

## 本地运行和观测

```bash
# CPU 模拟器，无需下载模型
uv run nano-inference local --policy predicted_ttft \
  --placement-config configs/placement.example.json

# Mac 真实引擎，沿用现有环境与租户身份
source .local/tenants.env
uv run --extra tokenizer nano-inference serve --profile configs/macos-metal.json \
  --tenant-config configs/tenants.example.json --max-queue 24 \
  --policy predicted_ttft --placement-config configs/placement.example.json
```

访问 `/monitor`，Worker 卡片可看到最后一次候选的 TTFT、queue/prefill/KV 分量、
近似 cached tokens 和当时的原生数据可用性。`/monitor/api`、`/workers` 提供完整分解
及所用参数；这些分数是最后一次请求的历史快照，不是当前待派发请求的实时预测。

Prometheus 新增：

- `nano_route_predicted_ttft_seconds{worker}`：已派发请求的预测分布。
- `nano_route_prediction_absolute_error_seconds{worker}`：首次内容到达后的绝对预测误差。
- `nano_route_slo_predictions_total{tenant,feasible}`：派发时可行性计数。
- `nano_route_last_{ttft_seconds,queue_seconds,prefill_seconds,kv_seconds,cached_tokens}`：
  最后候选分数，未参与最后一次选择的 Worker 不代表没有容量或没有负载。
- `nano_rejections_total{reason="slo_budget_exceeded"}`：真实提前拒绝。

始终同时看所有到达的 SLO 达标比例、成功/拒绝/失败数和 batch 进展；不能只看成功 p95。

## 实验、校准与面试边界

```bash
# 三轮 AB / BA / AB；原始请求、配置、客户端调度滞后、失败均保留
uv run nano-inference experiment --batch placement

# 已准备的本机 Metal；两个自有临时副本，运行后清理
uv run --extra tokenizer python scripts/smoke_placement.py
```

CPU 配对批次保存在 `results/experiments/placement/`，报告可从 `/experiments` 打开：
每组先用两个短请求预热 HTTP/分词线程路径，不建立前缀缓存；预热单独记录，不计入测量。

1. **异构副本**：prefix-affinity → predicted-TTFT；先验服务率按注入的固定延迟设置，
   用于验证控制面选择。低负载时旧策略也可能停留在快副本，因此没有收益也是有效结果。
   模拟器不模拟 KV 缓存，不把这项结果称为缓存优化或 GPU 加速。
2. **过载 SLO**：两组都使用 predicted-TTFT，只改变拒绝开关。每组相同 48 个到达，
   比较达标数量/比例，并以相同的“最后到达 + SLO”窗口报告 goodput。
   baseline 的 drain 时间不用于这个固定窗口分母；拒绝必须同时展示。

Metal 冒烟记录在 `results/placement/metal-smoke.json`：真实 tokenization、租户 cold/warm
边界、原生缓存 counters、预测分解及误差；它是四次顺序请求的功能检查，不是新旧路由
的性能对照。两个 Metal 副本共享同一块 Apple GPU。

以下是 commit `5bc7303` 的检查点索引版历史结果，不能作为后续 radix tree 的性能证明。
2026-10-19 正式 CPU 批次每项三轮 AB/BA/AB，12 个 trial 均通过有效性检查：

| 对照 | 观测 | 可得出的结论 |
|---|---|---|
| 低负载异构 Worker：prefix-affinity → predicted-TTFT | 两边每轮 60 次全部成功；三轮 p95 一轮反向，属于混合/小效应 | 该负载下未见稳定额外收益，不能挑最好一轮宣传 |
| 过载：SLO admission off → on | 三轮达标数分别 3→4、2→6、3→5，分母均为 48 个到达 | 小型注入场景中达标数增加；没有证明生产或 GPU SLO 改善 |
| 同一过载实验的代价 | baseline 每轮 48 成功；candidate 每轮 7 成功、41 拒绝 | admission 以拒绝大部分工作为代价，不能只展示成功延迟或相对百分比 |

均等预热之前的探索性汇总保留在 `results/placement/exploratory-before-warmup-summary.json`，
不用于正式对比；它只保留早期汇总，不包含所有早期原始请求。

校准步骤：先关闭拒绝和缓存折扣，在低并发下测不同输入长度的 cold prefill，用
引擎 prefill 时间拟合输入 token 成本；用 decode 时间和输出 token 数估计 decode rate，
把 HTTP/调度固定开销单独保留。之后加入 warm-prefix、长 decode、负载 sweep，检验
预测误差和达标误判，再调整折扣、有效 parallelism 与安全系数。每 Worker 可通过配置
`workers: {"worker-1": {...}, "worker-2": {...}}` 使用独立 service model。
本批没有自动拟合或在线学习；目标 CUDA 硬件需要重新测量。

可以如实讲：

> I kept token-cost WDRR for tenant fairness and added an analytical TTFT placement
> policy that trades approximate prefix reuse against outstanding work and fresh
> engine KV pressure. I exposed prediction error and made deadline admission an
> opt-in experiment. I verified native cache behavior locally, while keeping
> uncalibrated predictions and independent-GPU performance claims separate.

## 技术参考

- [SGLang cache_aware.rs](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs)：
  参考其历史前缀近似与负载不平衡时覆盖亲和的思路；这里使用 token 摘要和秒单位模型，
  没有照搬 Rust 的 radix tree 或声称公式来自 SGLang。
- [llm-d Precise Prefix Cache Routing](https://llm-d.ai/docs/0.8/well-lit-paths/foundations/precise-prefix-cache-routing)：
  了解真实 KV Events 如何更新缓存索引。这是下一阶段可替换的缓存观测来源，本批没有
  实现事件订阅、事件丢失恢复、全局索引一致性或 KV 迁移。
