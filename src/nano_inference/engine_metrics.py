"""The common vLLM metrics contract used by Metal and CUDA verification."""

import math

from prometheus_client.parser import text_string_to_metric_families

METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:request_time_per_output_token_seconds_count",
    "vllm:request_time_per_output_token_seconds_sum",
    "vllm:request_success_total",
}


def parse_engine_metrics(text: str) -> dict:
    values: dict[str, float] = {}
    finished: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in METRICS and math.isfinite(sample.value):
                values[sample.name] = values.get(sample.name, 0) + sample.value
                if sample.name == "vllm:request_success_total":
                    reason = sample.labels.get("finished_reason", "unknown")
                    finished[reason] = finished.get(reason, 0) + sample.value
    return {"values": values, "finished_by_reason": finished}


async def engine_snapshot(client, url: str) -> dict:
    response = await client.get(url + "/metrics")
    response.raise_for_status()
    parsed = parse_engine_metrics(response.text)
    required = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
    }
    missing = required - parsed["values"].keys()
    if missing:
        raise ValueError(f"Missing vLLM metrics: {sorted(missing)}")
    return parsed


def cache_delta(before: dict, after: dict) -> dict:
    hits = (
        after["values"]["vllm:prefix_cache_hits_total"]
        - before["values"]["vllm:prefix_cache_hits_total"]
    )
    queries = (
        after["values"]["vllm:prefix_cache_queries_total"]
        - before["values"]["vllm:prefix_cache_queries_total"]
    )
    if hits < 0 or queries < 0:
        raise ValueError("Engine counters reset; do not compare across a worker restart")
    return {
        "hit_tokens": hits,
        "queried_tokens": queries,
        "token_hit_ratio": hits / queries if queries else None,
    }
