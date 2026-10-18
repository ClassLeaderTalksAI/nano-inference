import pytest

from nano_inference.engine_metrics import cache_delta, parse_engine_metrics


def test_engine_metrics_keep_token_counts_and_finish_reasons_distinct():
    parsed = parse_engine_metrics("""
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0"} 100
vllm:prefix_cache_queries_total{engine="1"} 200
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0"} 80
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop"} 2
vllm:request_success_total{finished_reason="abort"} 1
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="1"} 2
vllm:time_to_first_token_seconds_count 2
vllm:time_to_first_token_seconds_sum 0.3
""")
    assert parsed["values"]["vllm:prefix_cache_queries_total"] == 300
    assert parsed["finished_by_reason"] == {"stop": 2, "abort": 1}
    assert "vllm:time_to_first_token_seconds_bucket" not in parsed["values"]
    before = {"values": {"vllm:prefix_cache_queries_total": 100, "vllm:prefix_cache_hits_total": 0}}
    assert cache_delta(before, parsed) == {
        "hit_tokens": 80,
        "queried_tokens": 200,
        "token_hit_ratio": 0.4,
    }


def test_cache_counters_cannot_be_compared_across_engine_restart():
    before = {
        "values": {"vllm:prefix_cache_queries_total": 100, "vllm:prefix_cache_hits_total": 80}
    }
    after = {"values": {"vllm:prefix_cache_queries_total": 20, "vllm:prefix_cache_hits_total": 0}}
    with pytest.raises(ValueError, match="reset"):
        cache_delta(before, after)
