from nano_inference.benchmark import Result, summarize


def test_missing_usage_does_not_turn_chunks_into_tokens():
    results = [Result(0, "success", first_content_seconds=0.1, content_events=5)]
    summary = summarize(results, elapsed=1, slo=0.2)
    assert summary["output_tokens_per_second"] is None
    assert summary["successful_requests_meeting_ttft_slo_per_second"] == 1


def test_errors_and_rejections_remain_visible_next_to_success_latency():
    results = [
        Result(0, "success", first_content_seconds=0.1, completion_tokens=10),
        Result(1, "rejected"),
        Result(2, "stream_error"),
    ]
    summary = summarize(results, elapsed=2, slo=0.2)
    assert summary["requests"] == 3
    assert summary["outcomes"] == {"rejected": 1, "stream_error": 1, "success": 1}
    assert summary["output_tokens_per_second"] == 5
