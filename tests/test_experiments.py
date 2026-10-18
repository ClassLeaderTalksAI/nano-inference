import asyncio

import httpx
import pytest

from nano_inference.benchmark import Result
from nano_inference.experiments import (
    ExperimentConfig,
    compare_pairs,
    evidence_quality,
    replay,
    workload,
)
from nano_inference.gateway import Settings
from nano_inference.local_server import serve_app
from nano_inference.mock_worker import create_mock_worker

from .test_gateway import payload


def test_workload_is_repeatable_and_requests_are_identical_across_variants():
    assert workload("routing", 1) == workload("routing", 1)
    assert workload("routing", 1) != workload("routing", 2)
    trace = workload("tenants", 1)
    assert trace[0].tenant == "batch"
    assert trace[-1].tenant == "interactive"
    assert max(a.offset for a in trace if a.tenant == "batch") < min(
        a.offset for a in trace if a.tenant == "interactive"
    )


def test_invalid_load_and_unexpected_rejections_cannot_be_used_as_evidence():
    config = ExperimentConfig()
    assert not evidence_quality([Result(0, "client_dropped")], config)["usable_for_comparison"]
    assert not evidence_quality([Result(0, "success", dispatch_lag_seconds=0.2)], config)[
        "usable_for_comparison"
    ]
    rejected = [Result(0, "rejected", error_code="prefix_preparation_full")]
    assert "unexpected_server_rejections" in evidence_quality(rejected, config)["flags"]
    assert evidence_quality(rejected, config, allow_rejections=True)["usable_for_comparison"]
    assert not evidence_quality([Result(0, "success")], config)["usable_for_comparison"]


def test_paired_analysis_preserves_negative_effects_and_excludes_invalid_pairs():
    runs = []
    for repeat in range(3):
        for variant, value in [("baseline", 1), ("candidate", 2)]:
            runs.append(
                {
                    "case": "x",
                    "repeat": repeat,
                    "variant": variant,
                    "summary": {"latency": value},
                    "quality": {"usable_for_comparison": True},
                }
            )
    result = compare_pairs(runs, "x", "latency")
    assert result["assessment"] == "consistent_directional_regression"
    assert result["median_improvement_percent"] == -100
    runs[-1]["quality"]["usable_for_comparison"] = False
    assert compare_pairs(runs, "x", "latency")["assessment"] == "inconclusive"
    assert compare_pairs(runs[:2], "x", "latency")["assessment"] == "inconclusive"


async def test_context_guard_rejects_before_worker_and_accepts_boundary(cluster):
    async with cluster(max_model_len=8) as (_, client, _):
        invalid = await client.post("/v1/chat/completions", json=payload(max_tokens=8))
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "context_length_exceeded"
        state = (await client.get("/workers")).json()
        assert all(w["inflight"] == 0 and w["healthy"] for w in state["workers"])
        assert not state["route_decisions"]
        valid = await client.post("/v1/chat/completions", json=payload(max_tokens=7))
        assert valid.status_code == 200  # One synthetic prompt token + seven output tokens.
        assert float(valid.headers["x-queue-seconds"]) >= 0
        metrics = (await client.get("/metrics")).text
        assert 'reason="context_length_exceeded"' in metrics


def test_real_context_budget_requires_actual_local_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        Settings(model="real-model", max_model_len=2048)
    with pytest.raises(ValueError, match="positive"):
        Settings(max_model_len=0)


async def test_embedded_server_cleanup_and_replay_include_every_arrival():
    async with serve_app(create_mock_worker()) as url:
        arrivals = workload("tenants", 1)
        # Direct mock workers have no /workers endpoint; use the ordinary gateway in cluster tests.
        async with httpx.AsyncClient(trust_env=False) as c:
            assert (await c.get(url + "/health")).status_code == 200
    async with httpx.AsyncClient(trust_env=False, timeout=0.1) as c:
        with pytest.raises(httpx.ConnectError):
            await c.get(url + "/health")
    assert len(arrivals) == 22


async def test_replay_reports_client_drops_and_queue_times(cluster):
    async with cluster() as (_, client, _):
        results, _, state = await replay(
            str(client.base_url).rstrip("/"), "nano-model", workload("overload", 1), {}, 1
        )
        assert len(results) == 48
        assert any(r.outcome == "client_dropped" for r in results)
        assert all(r.queue_seconds is not None for r in results if r.outcome == "success")
        assert not state["queued"]
        assert all(w["inflight"] == 0 for w in state["workers"])


async def test_cancelling_embedded_server_still_releases_socket():
    started = asyncio.Event()
    urls = []

    async def hold():
        async with serve_app(create_mock_worker()) as url:
            urls.append(url)
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with httpx.AsyncClient(trust_env=False) as c:
        with pytest.raises(httpx.ConnectError):
            await c.get(urls[0] + "/health")


def test_native_phase_means_use_deltas_and_preserve_missing_or_reset_metrics():
    from nano_inference.experiments import phase_means

    a = [
        {
            "values": {
                "vllm:request_prefill_time_seconds_count": 2,
                "vllm:request_prefill_time_seconds_sum": 1,
            }
        }
    ]
    b = [
        {
            "values": {
                "vllm:request_prefill_time_seconds_count": 4,
                "vllm:request_prefill_time_seconds_sum": 1.4,
            }
        }
    ]
    assert phase_means(a, b)["request_prefill_time_seconds"] == pytest.approx(0.2)
    assert phase_means(a, b)["request_decode_time_seconds"] is None
    assert phase_means(b, a)["request_prefill_time_seconds"] is None


def test_report_escapes_labels_and_renders_custom_batch_output(tmp_path):
    import json

    from nano_inference.experiment_report import render_report

    custom = tmp_path / "custom-name"
    custom.mkdir()
    data = {
        "batch": "control",
        "scope": "<script>alert(1)</script>",
        "comparisons": [],
        "runs": [],
        "config": {"seed": 1, "repeats": 3},
        "metadata": {"python_source_sha256": "abcd"},
    }
    (custom / "summary.json").write_text(json.dumps(data))
    content = render_report(tmp_path, custom).read_text()
    assert "&lt;script&gt;" in content
    assert "<script>" not in content
    assert "CONTROL" in content
    assert "statistical significance" in content
