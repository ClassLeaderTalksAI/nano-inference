# SGLang 风格的近似前缀树路由

本批新增 `--policy cache_aware`，实现 Approximate Prefix Index、Longest Prefix Matching、
Cache Affinity 和 Load Imbalance Fallback。原有 `predicted_ttft` 也改用这棵树估计缓存复用；
`prefix_affinity` 保留为固定长度哈希提示的简单对照。

## 请求路径和索引

租户身份 → 实际模型 tokenizer/chat template → 配额和独立队列 → WDRR →
健康/容量/租户 Worker allowlist 过滤 → cache-aware placement → vLLM。

`prefix_index.py` 实现压缩 token radix tree，按 `(namespace, worker, generation)` 分树。
namespace 由 Gateway 根据可信租户身份、模型 revision、tokenizer 和模板指纹生成。
Worker ID 相同但 generation 不同，不会复用旧树。

一次成功请求的完整输入 token 路径进入树，默认最多保留前 4096 tokens。新输入沿树
逐 token 比较，匹配可以结束在压缩边中间。例如存储 `[0..149]` 后，查询前 93 tokens
相同、剩余不同的输入，返回 **93**，不需要预先设置 64/256/1024 检查点。
同一输入可以在多个 Worker 的树中存在。生成出来的 output tokens 没有加入索引。

全局限制同时覆盖：

- 完成请求的路径记录数：`cache_capacity=4096`。
- 压缩边上保留的 token 数：`cache_max_tokens=65536`。
- 节点数（含各树 root）：`cache_max_nodes=8192`。
- 单条路径长度：`cache_max_prefix_tokens=4096`。
- 路径 TTL：`cache_ttl_seconds=30`。

仅成功完成刷新路径和淘汰顺序；读取不刷新 TTL。到达任一预算时淘汰最早成功使用的
路径，删除失去有效后代的分支，合并没有终点标记的单子节点。过期的长路径不会残留
虚假的深层命中。只有较短路径过期时，共享它的有效长路径仍然保留。
匹配遍历与保留的输入长度成正比；TTL 清理另外扫描有界的路径记录集合。

树内部保存 token IDs，它们属于敏感输入的表示，不是匿名化数据。原始文本和 token
路径不写入日志、Prometheus、`/workers` 或 `/monitor/api`；这些接口仅输出统计和分数。

## Worker 选择

```python
if max_load - min_load > balance_abs_threshold and max_load > min_load * balance_rel_threshold:
    return least_loaded_worker  # cache_load_fallback
if longest_match / input_tokens > cache_threshold:
    return worker_with_longest_match  # cache_affinity
return least_loaded_worker  # cache_weak_match / cache_miss
```

默认 `balance_abs_threshold=2`、`balance_rel_threshold=1.5`、`cache_threshold=0.5`。
绝对和相对条件必须**同时严格超过**阈值。决策只比较通过硬过滤的候选 Worker。
load 使用网关原子预留的 inflight，加上最近有效原生 running/waiting 样本中的额外请求；
扣除与网关预留的重叠。指标失败、过期或属于旧 generation 时，只使用本地 inflight。

冷输入和弱匹配走最空闲 Worker。相同匹配长度时选择负载较低的 Worker；完全相同时
使用已有旋转顺序。选中后在同一把锁内预留，再发 HTTP，避免并发请求同时看见空闲。
fallback 的请求成功后同样更新树，逐渐允许热门前缀分布在多个副本。

`cache_aware` 不使用未校准的 TTFT 模型进行拒绝；其预测分数只作观测。
SLO 提前拒绝开关仍只对 `predicted_ttft` 生效，WDRR 公平性与账本保持原语义。
禁用引擎 APC 时应切换 `least_inflight`，或把 `cache_threshold` 设为 1 禁用此策略的
亲和路径；单独设置 `cache_reuse_discount=0` 只影响 TTFT 模型，不关闭 cache-aware 亲和。

## 本地运行和观测

```bash
# 无模型开发环境
uv run nano-inference local --policy cache_aware \
  --placement-config configs/placement.example.json

# 已准备的 Mac Metal 与租户身份
source .local/tenants.env
uv run --extra tokenizer nano-inference serve \
  --tenant-config configs/tenants.example.json --max-queue 24 \
  --policy cache_aware --placement-config configs/placement.example.json
```

打开 `/monitor` 查看最后一次决策、各 Worker 的 LPM token 数、选路负载和预测分量。
`/monitor/api` 的 `pool.placement.prefix_index` 显示节点、路径、token 数和预算，
`last_reason` 与 `last_loads` 显示决策依据。`X-Route-Decision` 和结构化日志保留逐请求原因；
Prometheus 的 `nano_route_decisions_total` 按原因累计，`nano_route_last_matched_tokens`
展示最近候选的最长匹配。所有 last 字段都是最后一次选择的历史快照。

## 验证和边界

- 前缀树测试覆盖边内匹配、边分裂、短路径作为另一条路径前缀、TTL 删除与重压缩、
  记录/token/node 预算，以及租户、模型和 generation 隔离。
- 500 步固定 seed 的随机操作，与独立的完整历史遍历 oracle 比较匹配结果，并核对
  树结构和内存计数，避免只用实现自身作为预期答案。
- 策略测试覆盖任意长度匹配、弱匹配、双阈值边界、负载覆盖亲和、新副本成功预热、
  原生队列样本过期、容量、取消清理、租户 allowlist 和旧 generation 完成。
- WDRR 成本份额和 HTTP 租户隔离回归也运行在 `cache_aware` 下。

真实模型检查：

```bash
uv run --extra tokenizer python scripts/smoke_placement.py \
  --policy cache_aware --output results/cache-aware/metal-smoke.json
```

两个租户各发 cold 输入与共享前缀但后缀不同的后续请求，分别核对 Router 的部分 LPM
与引擎原生 cache counters。两个 Metal 副本共享一块 Apple GPU；此检查证明本地功能，
不证明负载回退带来的 GPU 性能提升。既有 placement 对照记录属于上一版检查点索引的
历史实验，不能作为本批 radix tree 的性能结论。

2026-10-19 验证：146 项测试通过，Ruff lint/format 通过。本机四次请求全部成功：
两个租户的 cold 请求均为 `cache_miss`，原生命中为 0；后续请求均为 `cache_affinity`，
LPM 为 426/433 tokens，实际 cached tokens 为 416。路由匹配长度和引擎按块可复用长度
无需相等。报告确认临时端口已关闭、所有工作量预留已释放。

参考 [SGLang cache_aware.rs](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs)
和 [tree.rs](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/tree.rs)。
其当前实现使用字符前缀、调度时记录和随机打破平局；这里选择真实 token、成功后记录
和旋转平局。这是独立的 Python 实现，保留了“负载平衡时亲和，失衡时覆盖”的核心思路。

没有实现 llm-d 的 KV Events 索引。近似树仍不知道引擎缓存是否被淘汰；Router 的匹配
不是实际 KV hit。真实命中必须看引擎 counters，TTL 和内存预算只约束本地历史提示。
