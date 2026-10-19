import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from nano_inference.gateway import Settings, create_gateway
from nano_inference.tenant_benchmark import run_tenant_benchmark

from .conftest import serve
from .test_gateway import payload
from .test_tenants import config, config_file


@pytest.mark.parametrize("policy", ["least_inflight", "predicted_ttft"])
async def test_native_polling_exposes_values_then_marks_failed_scrapes_missing(policy):
    engine = FastAPI()
    engine.state.fail = False

    @engine.get("/health")
    async def health():
        return {"ok": True}

    @engine.get("/metrics")
    async def metrics():
        if engine.state.fail:
            return PlainTextResponse("failed", status_code=503)
        return PlainTextResponse("""
vllm:num_requests_running 2
vllm:num_requests_waiting 1
vllm:kv_cache_usage_perc 0.25
vllm:prefix_cache_queries_total 100
vllm:prefix_cache_hits_total 50
""")

    async with serve(engine) as worker:
        app = create_gateway(Settings(worker_urls=[worker], policy=policy))
        async with serve(app) as url, httpx.AsyncClient(base_url=url, trust_env=False) as c:
            await app.state.dashboard.poll()
            data = (await c.get("/monitor/api")).json()
            assert data["engines"]["worker-1"]["values"]["vllm:kv_cache_usage_perc"] == 0.25
            assert data["engines"]["worker-1"]["available"]
            assert app.state.pool.placement.native["worker-1"][2]["kv_cache_usage_perc"] == 0.25
            assert (
                'nano_engine_kv_cache_usage_perc{worker="worker-1"} 0.25'
                in (await c.get("/metrics")).text
            )
            engine.state.fail = True
            await app.state.dashboard.poll()
            data = (await c.get("/monitor/api")).json()
            assert not data["engines"]["worker-1"]["available"]
            assert data["engines"]["worker-1"]["values"] is None
            assert "worker-1" not in app.state.pool.placement.native
            assert (
                'nano_engine_kv_cache_usage_perc{worker="worker-1"} NaN'
                in (await c.get("/metrics")).text
            )


async def test_dashboard_has_no_external_assets_and_exposes_live_state(cluster):
    async with cluster() as (_, c, _):
        page = await c.get("/monitor")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert "/monitor/api" in page.text
        assert "<script src=" not in page.text
        assert "https://" not in page.text
        assert "Disconnected · displayed values are stale" in page.text
        response = await c.post("/v1/chat/completions", json=payload())
        assert response.status_code == 200
        data = (await c.get("/monitor/api")).json()
        assert data["simulated"]
        assert data["metrics"]["overall"]["outcomes"]["success"] == 1
        assert data["metrics"]["tenants"]["default"]["output_tokens"] == 8
        assert data["metrics"]["tenants"]["default"]["ttft"]["mean"] > 0


async def test_multi_tenant_workload_reports_all_arrivals_without_credentials(
    cluster, tmp_path, monkeypatch
):
    cfg = config_file(tmp_path, config(monkeypatch))
    output = tmp_path / "report.json"
    async with cluster(tenant_config=cfg) as (_, c, _):
        report = await run_tenant_benchmark(
            str(c.base_url).rstrip("/"),
            tenant_config=cfg,
            requests=4,
            rate=100,
            max_tokens=8,
            max_outstanding=2,
            output=str(output),
        )
    assert len(report["results"]) == 12
    assert sum(s["requests"] for s in report["summary"].values()) == 12
    assert any(r["outcome"] == "client_dropped" for r in report["results"])
    assert set(report["summary"]) == {"interactive", "standard", "batch"}
    assert "secret-key" not in output.read_text()
    assert "Authorization" not in output.read_text()
