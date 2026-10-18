# 多租户与本地监控

这批实现采用一个 Gateway、内存租户队列、共享 Worker 池，沿用 Mac Metal 和未来
CUDA flavor 的同一套 HTTP/SSE 接口。没有引入 Redis、Kubernetes 或额外的调度服务。
原有不带租户配置的启动方式仍可运行，保持单租户 FIFO 行为。

## 本地运行

首次生成三个租户的随机测试密钥，写入被 Git 忽略的 `.local/tenants.env`，权限 0600。
再次执行会保留已有文件，不会更换正在使用的密钥。密钥不写入代码、报告或监控标签。

```bash
uv sync --locked --extra dev --extra tokenizer
uv run python scripts/prepare_tenants.py
source .local/tenants.env

# 已准备好 Metal runtime / 模型时，运行两个真实服务副本。
uv run --extra tokenizer nano-inference serve \
  --profile configs/macos-metal.json \
  --tenant-config configs/tenants.example.json --max-queue 24
```

如果尚未准备模型，先运行 `uv run python scripts/prepare_metal.py`。
希望先观察控制逻辑、无需 GPU，可以用以下命令代替 `serve`：

```bash
uv run nano-inference local --tenant-config configs/tenants.example.json \
  --policy prefix_affinity --capacity 1 --max-queue 24
```

打开 **http://127.0.0.1:8000/monitor**。另开终端加载相同密钥并制造三租户流量：

```bash
source .local/tenants.env
uv run nano-inference tenant-load --model nano-qwen \
  --requests 30 --rate 5 --max-tokens 32 --output results/tenants-latest.json
```

模拟器请使用 `--model nano-model`。`--rate` 是每个普通租户每秒到达数，batch 使用
4 倍速率、较长输出。客户端有独立的 outstanding 上限，丢弃的到达也计入报告。
增加 `--requests` 可以延长观测时间；请求可能被拒绝，这是过载演示的一部分。

单独发送交互请求：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $NANO_INTERACTIVE_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nano-qwen","messages":[{"role":"user","content":"Explain a queue."}],"stream":true,"max_tokens":32}'
```

Ctrl-C 关闭启动器及其所属服务。所有服务仅绑定本机回环地址。

## 一次请求的路径

```text
Bearer API key → 服务端租户身份 → 实际 chat-template tokenization
      → RPM / TPM 准入 → 每租户 FIFO → WDRR → 健康/容量/允许 Worker 筛选
      → 租户范围内的 Prefix Affinity → vLLM（服务端 cache_salt）
      → SSE / 最终 usage → 并发释放、配额结算、成本误差校正、指标
```

- **身份**：配置中的 `api_key_env` 指向环境变量。缺失、过短或重复密钥使启动失败。
  `X-Tenant-ID` 不参与身份判断；请求体中的 `tenant_id`、`cache_salt` 都不允许。
- **独立队列与并发**：租户各自有有界 FIFO、active 上限；共享池仍有总队列与
  每 Worker 的在途上限。等待租户并发配额的请求不会提前占 Worker 名额。
- **RPM/TPM**：每租户两个连续补充的 token bucket，初始满桶、容量分别为一分钟
  RPM/TPM，允许初始突发，并非滚动一分钟的硬窗口。TPM 先扣输入 token + 完整
  `max_tokens`；完成后按最终 usage 多退少补。队列中未派发就取消/超时退还 token
  预留，已准入的 RPM 仍消耗；派发后缺失 usage，保守保留全部 token 预扣，并按完整输出上限补扣调度成本。
- **加权公平**：WDRR 按 `input_tokens + decode_cost × min(max_tokens,
  expected_output_tokens)` 收取预计成本；完成后修正 deficit。默认 decode_cost=2
  只是可配置的代理成本，尚未通过硬件校准。空闲租户不会无限累积 credit。
  配置中的 `scheduler: fifo` 可用作按可运行租户队首到达时间选择的对照；仍保留配额和
  每租户并发限制。权重不意味着每个短窗口都严格按比例服务，且只在持续排队、争抢
  同一资源、并发限制未额外压低份额时适合讨论长期比例。
- **SLO**：每租户 TTFT 目标由服务端配置，只做观测与 goodput 统计。本批没有添加
  deadline priority boost，避免在尚未校准执行时间时破坏公平性。
- **Worker 范围**：可选 `workers: ["worker-1"]` 在所有路由策略之前限制可用实例。
  这可模拟 dedicated/shared placement；Mac 上两个进程仍共用同一物理 GPU。

正式模型使用同一固定 tokenizer 和 chat template 计算输入长度；模拟器的输入计数
是空白分词，仅用于控制逻辑测试。分词并发有界，但尚无每租户 CPU 分词资源保障。
总队列满时，新请求仍会被拒绝；每租户队列上限不能替代容量预留。示例总队列 24
大于三个租户队列上限之和 20，方便观察租户队列限制自身的效果。

## 前缀缓存隔离的两层实现

路由提示使用 `hash(tenant_id, model/tokenizer/template/prefix identity)`，不同租户
不能直接借用对方的 affinity 记录。网关还生成进程内秘密 HMAC，按租户派生
64 字符 `cache_salt` 并发送给 vLLM；客户端不能覆盖。Gateway 重启后 salt 更换，
旧引擎缓存可能不再复用。运行时的 cache salt 才负责隔离实际的 KV block 哈希；
只给路由表加 tenant_id 不足以证明底层缓存隔离。

[vLLM 的前缀缓存设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)说明 salt
进入首块的哈希，后续块通过父哈希传播隔离。
[官方安全指南](https://docs.vllm.ai/en/latest/usage/security/)建议由可信网关按租户设置。
本项目还在固定的 vLLM 0.31.0 / vllm-metal 0.31.0 上进行了真实验证，见下文。

## 监控页面与 Prometheus

内置页面不依赖 Docker、Grafana 或外部 CDN，每两秒刷新。适合 Mac 本地学习：

| 页面内容 | 数据与含义 |
|---|---|
| 成功与其他结果、请求速率 | Gateway 生命周期计数；拒绝/失败/取消都展示 |
| 输出 token/s | 最近刷新间隔收到的最终 usage；长流结束前不会提前计数 |
| 每租户 active/queue、剩余 RPM/TPM | 实时准入状态，可观察 noisy neighbor |
| 每租户 TTFT mean / p95 上界 | Gateway 到首个非空内容的时间，含分词、排队和上游等待 |
| SLO met / 可观测成功流数 | 满足服务端 TTFT 目标的成功流数；非流式请求不计算 TTFT |
| Settled cost | 已收到 usage 的成本代理；缺失 usage 单独计入保守结算数 |
| Worker running/waiting、KV cache、缓存 token | 原生 vLLM Prometheus 数据，每次成功采样的时间可见 |
| Worker TTFT / TPOT mean | 原生引擎直方图 sum/count，含该实例所有租户 |

页面的 p95 是直方图桶的累计分位**上界**，不是精确的样本分位。更精确的客户端
样本分位在负载报告里。Worker 指标丢失显示 unavailable，Prometheus 的对应 gauge
置 NaN；不会把采集失败伪装成 0。页面连接失败会标记数据过期。模拟器没有原生 GPU
指标，页面明确标注；没有杜撰 Apple GPU 利用率。图表历史在浏览器内保留约四分钟。

`/monitor`、`/monitor/api`、`/workers`、`/metrics` 是本机运维入口，不做租户授权，
只暴露有界租户 ID、状态、用量和配置额度，不暴露密钥、salt、提示词或请求明细。
当前部署必须保持回环绑定；以后远程部署要给管理入口加单独访问控制，且 Worker
不能绕过 Gateway 对终端租户公开。

已有原生 Prometheus 时，可以直接使用 [monitoring/prometheus.yml](../monitoring/prometheus.yml)：

```bash
prometheus --config.file=monitoring/prometheus.yml --web.listen-address=127.0.0.1:9090
```

浏览 **http://127.0.0.1:9090**。这里提供抓取配置和查询示例，未安装或启动额外监控
服务。模拟器没有 `/metrics`，因此原生 Worker target 显示 DOWN 属于预期；Gateway
指标正常。Docker 内运行的 Prometheus 需按其网络设置修改目标地址，不能照抄回环地址。

常用查询（短暂试跑可将 `[1m]` 改为 `[15s]`）：

```promql
sum by (tenant, outcome) (rate(nano_tenant_requests_total[1m]))
sum by (tenant, reason) (increase(nano_tenant_rejections_total[1m]))
nano_tenant_inflight
nano_tenant_queued
nano_tenant_tpm_remaining
sum by (tenant) (rate(nano_tenant_tokens_total{kind="output"}[1m]))
histogram_quantile(0.95, sum by (tenant, le) (rate(nano_tenant_first_content_seconds_bucket[1m])))
sum by (tenant) (rate(nano_tenant_slo_met_total[1m]))
nano_engine_metrics_up
nano_engine_kv_cache_usage_perc
```

原生 Worker TTFT/TPOT 直方图、prefix_cache_hits/queries 等仍从 Worker `/metrics`
直接抓取；Gateway 只镜像选定的负载 gauge，避免重复实现全部 vLLM 指标。

## 验证与结果边界

```bash
uv run --extra dev --extra tokenizer pytest -q
uv run --extra dev ruff check src tests scripts
uv run --extra tokenizer python scripts/smoke_tenants.py
```

测试覆盖实际 TCP/SSE 取消、认证/参数伪造、租户队列和并发、RPM/TPM、usage 多退少补、
缺失 usage、派发与取消竞争、Worker allowlist、成本公平和监控采集失败。
持续饱和的调度测试用不同请求长度检查接近 5:3:1 的**token 成本**份额，而非请求数。

真实 Metal 缓存实验故意只用同一个 Worker，排除“去了另一副本所以未命中”的解释：

| 同一条 308-token prompt 的访问 | 原生 cached / queried tokens |
|---|---:|
| interactive 首次 | 0 / 308 |
| interactive 再次 | 304 / 308 |
| batch 首次 | 0 / 308 |
| batch 再次 | 304 / 308 |

证据在 `results/macos-metal/tenants/verification.json`。同一次真实并发试跑共 18 次到达，
12 成功、6 被拒绝（包含租户队列满和等待超时）；报告完整保留失败，不能把成功请求的
TTFT 单独解读为满足全部流量的 SLO。模拟器也保留了三租户演示报告。
Prompt 中含随机实验标记，重跑时实际 token 数可能略有变化，原始报告为准。
这些小样本用于功能验收，**还不是** FIFO/WDRR 的重复、同负载性能对照实验。

仍待后续完成：校准 prefill/decode 成本、完整 noisy-neighbor 策略对照、SLO priority boost、
专属物理 GPU 的隔离/利用率对比、长期 Prometheus/Grafana 部署、跨 Gateway 配额协调。
当前提供单 Gateway 的逻辑准入/调度和 salt 缓存边界，不保证硬 GPU 时延隔离，亦不替代
进程、操作系统、网络层安全隔离。RunPod/CUDA flavor 仍等待本地审核。
