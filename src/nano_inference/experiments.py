"""Paired, counterbalanced experiments: raw evidence precedes interview claims.

Control tests use injected simulator delays. Metal tests use the pinned runtime.
Neither is evidence of independent NVIDIA GPU scaling or statistical significance.
"""

import asyncio
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import time
import uuid
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .benchmark import Result, send_request, summarize
from .engine_metrics import cache_delta, engine_snapshot
from .experiment_report import render_report
from .flavors import FlavorCluster, FlavorProfile, free_port, inspect_runtime
from .gateway import Settings, create_gateway
from .local_server import serve_app
from .mock_worker import Behavior, create_mock_worker
from .placement import PlacementConfig, ServiceModel
from .tenants import Tenant, TenantConfig


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repeats: int = Field(default=3, ge=1, le=10)
    seed: int = 20261018
    max_outstanding: int = Field(default=128, ge=1, le=1024)
    max_dispatch_lag_seconds: float = Field(default=0.05, gt=0, allow_inf_nan=False)


@dataclass(frozen=True)
class Arrival:
    index: int
    offset: float
    messages: list[dict]
    max_tokens: int
    tenant: str | None = None


def short_messages():
    return [{"role": "user", "content": "List integers starting at 1, separated by spaces."}]


def workload(case, seed, *, native=False):
    rng = random.Random(seed)
    if case == "routing":
        return [Arrival(i, i / 60, short_messages(), rng.choice([2, 4, 8])) for i in range(60)]
    if case == "overload":
        return [Arrival(i, i / 80, short_messages(), 32) for i in range(48)]
    if case == "tenants":
        batch_count, interactive_count = (8, 4) if native else (16, 6)
        return [
            Arrival(i, i * 0.001, short_messages(), 24 if native else 48, "batch")
            for i in range(batch_count)
        ] + [
            Arrival(batch_count + i, 0.025 + i * 0.001, short_messages(), 4, "interactive")
            for i in range(interactive_count)
        ]
    raise ValueError("Unknown experiment workload")


async def replay(url, model, arrivals, keys, limit):
    """Open-loop schedule is identical across variants; client drops remain visible."""
    results = []
    pending = set()
    started = time.perf_counter()
    async with httpx.AsyncClient(
        trust_env=False, timeout=60, limits=httpx.Limits(max_connections=None)
    ) as client:

        async def one(a):
            h = {"Authorization": "Bearer " + keys[a.tenant]} if a.tenant else None
            r = await send_request(
                client, url, model, a.max_tokens, a.index, started + a.offset, a.messages, h
            )
            r.tenant = a.tenant
            results.append(r)

        try:
            for a in sorted(arrivals, key=lambda a: (a.offset, a.index)):
                await asyncio.sleep(max(0, started + a.offset - time.perf_counter()))
                if len(pending) >= limit:
                    results.append(
                        Result(
                            a.index,
                            "client_dropped",
                            tenant=a.tenant,
                            dispatch_lag_seconds=time.perf_counter() - started - a.offset,
                        )
                    )
                    continue
                task = asyncio.create_task(one(a))
                pending.add(task)
                task.add_done_callback(pending.discard)
            if pending:
                await asyncio.gather(*pending)
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        elapsed = time.perf_counter() - started
        # HTTP response completion can precede the gateway's resource finalizer.
        async with asyncio.timeout(5):
            while True:
                state = (await client.get(url + "/workers")).json()
                if not state["queued"] and not any(w["inflight"] for w in state["workers"]):
                    break
                await asyncio.sleep(0.01)
    return results, elapsed, state


def detailed_summary(results, elapsed, slo):
    s = summarize(results, elapsed, slo)
    firsts = [
        r.first_content_seconds
        for r in results
        if r.outcome == "success" and r.first_content_seconds is not None
    ]
    s["first_content_mean_seconds"] = statistics.mean(firsts) if firsts else None
    s["workers"] = dict(Counter(r.worker for r in results if r.worker))
    s["error_codes"] = dict(Counter(r.error_code for r in results if r.error_code))
    # P99 of a few dozen observations invites false precision; retain raw samples instead.
    return s


def evidence_quality(results, config, *, allow_rejections=False, allow_errors=False):
    flags = []
    if not allow_errors and any(
        r.outcome not in ("success", "rejected", "client_dropped") for r in results
    ):
        flags.append("unexpected_failures")
    if not allow_rejections and any(r.outcome == "rejected" for r in results):
        flags.append("unexpected_server_rejections")
    if any(r.outcome == "client_dropped" for r in results):
        flags.append("client_dropped_arrivals")
    if max((r.dispatch_lag_seconds for r in results), default=0) > config.max_dispatch_lag_seconds:
        flags.append("client_dispatch_lag")
    if any(r.outcome == "success" and r.completion_tokens is None for r in results):
        flags.append("missing_success_usage")
    return {
        "usable_for_comparison": not flags,
        "flags": flags,
        "max_allowed_dispatch_lag_seconds": config.max_dispatch_lag_seconds,
        "statistical_significance": "not_established",
    }


def compare_pairs(runs, case, field, *, higher_is_better=False, tenant=None):
    pairs = []
    for repeat in sorted({r["repeat"] for r in runs if r["case"] == case}):
        pair = {r["variant"]: r for r in runs if r["case"] == case and r["repeat"] == repeat}
        if len(pair) != 2:
            continue
        a, b = pair["baseline"], pair["candidate"]

        def value(r):
            summary = r["tenants"][tenant] if tenant else r["summary"]
            for key in field.split("."):
                summary = summary[key]
            return summary

        av, bv = value(a), value(b)
        valid = a["quality"]["usable_for_comparison"] and b["quality"]["usable_for_comparison"]
        delta = (
            None
            if av is None or bv is None or av == 0
            else ((bv - av) / abs(av) * 100 * (1 if higher_is_better else -1))
        )
        pairs.append(
            {
                "repeat": repeat,
                "baseline": av,
                "candidate": bv,
                "improvement_percent": delta,
                "usable_for_comparison": valid,
            }
        )
    values = [
        p["improvement_percent"]
        for p in pairs
        if p["improvement_percent"] is not None and p["usable_for_comparison"]
    ]
    median = statistics.median(values) if values else None
    direction = "inconclusive"
    if len(values) >= 3 and all(v > 5 for v in values):
        direction = "consistent_directional_improvement"
    elif len(values) >= 3 and all(v < -5 for v in values):
        direction = "consistent_directional_regression"
    elif len(values) >= 3:
        direction = "mixed_or_small_effect"
    return {
        "case": case,
        "metric": field,
        "tenant": tenant,
        "pairs": pairs,
        "median_improvement_percent": median,
        "assessment": direction,
        "note": "Descriptive paired differences only. Check failures, offered-load SLO "
        "attainment and throughput before making a claim; no causal GPU scaling claim.",
    }


def source_metadata(root):
    files = sorted((root / "src/nano_inference").glob("*.py"))
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.name.encode())
        digest.update(p.read_bytes())

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {
        "git_head": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "python_source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def phase_means(before, after):
    means = {}
    for name in (
        "time_to_first_token_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "request_time_per_output_token_seconds",
    ):
        count_key, sum_key = "vllm:" + name + "_count", "vllm:" + name + "_sum"
        if not all(count_key in x["values"] and sum_key in x["values"] for x in [*before, *after]):
            means[name] = None
            continue
        count = sum(
            a["values"][count_key] - b["values"][count_key]
            for b, a in zip(before, after, strict=True)
        )
        total = sum(
            a["values"][sum_key] - b["values"][sum_key] for b, a in zip(before, after, strict=True)
        )
        means[name] = total / count if count > 0 and total >= 0 else None
    return means


class ExperimentRunner:
    def __init__(self, root, config, output):
        self.root, self.config, self.output = Path(root), config, Path(output)
        self.runs = []
        self.output.mkdir(parents=True, exist_ok=True)
        self.metadata = source_metadata(self.root)
        self.started = datetime.now(UTC).isoformat()
        self.runtime = None

    async def save(self, run):
        run["artifact"] = f"{run['case']}-{run['repeat']}-{run['variant']}.json"
        self.runs.append(run)
        await asyncio.to_thread(
            (self.output / run["artifact"]).write_text, json.dumps(run, indent=2) + "\n"
        )
        print(
            f"{run['case']} repeat={run['repeat']} {run['variant']}: {run['summary']['outcomes']}",
            flush=True,
        )

    async def case(self, case, repeat, variant, urls, *, native=False):
        candidate = variant == "candidate"
        model = "nano-qwen" if native else "nano-model"
        settings = Settings(
            worker_urls=urls,
            model=model,
            capacity=1 if case != "routing" else 8,
            max_queue=4 if case == "overload" and candidate else 128,
            queue_timeout=20,
            request_timeout=45,
            health_interval=0.1,
        )
        knobs = {"max_queue": settings.max_queue}
        if case == "routing":
            settings.policy = "least_inflight" if candidate else "round_robin"
            knobs = {"policy": settings.policy}
        keys = {}
        if native:
            profile = self.profile
            settings.prefix_tokenizer = str(self.root / profile.model_path)
            settings.model_revision = profile.model_revision
            settings.max_model_len = profile.max_model_len
        if case == "tenants":
            tenants = []
            for name, weight in [("interactive", 5), ("batch", 1)]:
                env = "NANO_EXPERIMENT_" + name.upper()
                keys[name] = os.environ[env] = uuid.uuid4().hex
                tenants.append(
                    Tenant(
                        id=name,
                        api_key_env=env,
                        weight=weight,
                        rpm=10000,
                        tpm=10000000,
                        max_inflight=1,
                        max_queue=64,
                        expected_output_tokens=48,
                        ttft_slo_seconds=0.5 if not native else 2,
                    )
                )
            cfg = TenantConfig(
                tenants=tenants, scheduler="wdrr" if candidate else "fifo", quantum=32
            )
            path = self.root / ".local/experiment-tenants.json"
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_text, cfg.model_dump_json())
            settings.tenant_config = str(path)
            knobs = {"scheduler": cfg.scheduler}
        if case == "tenants":
            settings.prefix_preparations = 32  # Same bounded CPU stage in both variants.
        arrivals = workload(case, self.config.seed + repeat, native=native)
        async with serve_app(create_gateway(settings)) as url:
            results, elapsed, state = await replay(
                url, model, arrivals, keys, self.config.max_outstanding
            )
        slo = 0.25 if case == "overload" else (0.5 if not native else 2)
        return {
            "case": case,
            "repeat": repeat,
            "variant": variant,
            "backend": "vllm-metal" if native else "simulator",
            "changed_knob": knobs,
            "gateway_settings": asdict(settings),
            "schedule": [asdict(a) for a in arrivals],
            "summary": detailed_summary(results, elapsed, slo),
            "tenants": {
                t: detailed_summary([r for r in results if r.tenant == t], elapsed, slo)
                for t in keys
            },
            "quality": evidence_quality(results, self.config, allow_rejections=case == "overload"),
            "final_pool": state,
            "results": [asdict(r) for r in sorted(results, key=lambda r: r.index)],
        }

    async def control(self):
        for case in ["routing", "overload", "tenants"]:
            for repeat in range(self.config.repeats):
                for variant in (
                    ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
                ):
                    # Fresh simulator instances remove load state between runs.
                    a = Behavior(first_content_delay=0.003, chunk_delay=0.0005)
                    b = Behavior(first_content_delay=0.04, chunk_delay=0.004)
                    if case != "routing":
                        a = b = Behavior(first_content_delay=0.01, chunk_delay=0.003)
                    async with AsyncExitStack() as stack:
                        urls = [
                            await stack.enter_async_context(serve_app(create_mock_worker("a", a)))
                        ]
                        if case != "tenants":
                            urls.append(
                                await stack.enter_async_context(
                                    serve_app(create_mock_worker("b", b))
                                )
                            )
                        run = await self.case(case, repeat, variant, urls)
                        run["injected_worker_behavior"] = [a.model_dump()] + (
                            [b.model_dump()] if len(urls) == 2 else []
                        )
                        await self.save(run)
        await self.finish("control")

    async def finish(self, batch):
        comparisons = []
        if batch == "placement":
            comparisons = [
                compare_pairs(self.runs, "placement", "successful_first_content_seconds.p95"),
                compare_pairs(
                    self.runs,
                    "placement_slo",
                    "ttft_slo_fraction_of_arrivals",
                    higher_is_better=True,
                ),
            ]
        if batch == "control":
            comparisons.extend(
                [
                    compare_pairs(self.runs, "routing", "successful_first_content_seconds.p95"),
                    compare_pairs(self.runs, "overload", "successful_first_content_seconds.p95"),
                    compare_pairs(
                        self.runs,
                        "overload",
                        "ttft_slo_fraction_of_arrivals",
                        higher_is_better=True,
                    ),
                ]
            )
        if batch != "placement":
            comparisons.append(
                compare_pairs(
                    self.runs,
                    "tenants",
                    "successful_first_content_seconds.p95",
                    tenant="interactive",
                )
            )
        if batch == "metal":
            for case in ("cache", "cache_steady"):
                comparisons.append(
                    compare_pairs(self.runs, case, "successful_first_content_seconds.p50")
                )
        report = {
            "schema_version": 1,
            "batch": batch,
            "started_at_utc": self.started,
            "metadata": self.metadata,
            "runtime": self.runtime,
            "config": self.config.model_dump(),
            "comparisons": comparisons,
            "runs": [
                {
                    "case": r["case"],
                    "repeat": r["repeat"],
                    "variant": r["variant"],
                    "artifact": r["artifact"],
                    "summary": r["summary"],
                    "tenants": r["tenants"],
                    "quality": r["quality"],
                    **(
                        {
                            "native_cache": r["native_cache"],
                            "native_phase_means_seconds": r["native_phase_means_seconds"],
                        }
                        if "native_cache" in r
                        else {}
                    ),
                }
                for r in self.runs
            ],
            "scope": "Simulator isolates control-plane logic. Metal replicas share one physical "
            "Apple GPU. Small local batches do not establish production SLOs or NVIDIA scaling.",
        }
        await asyncio.to_thread(
            (self.output / "summary.json").write_text, json.dumps(report, indent=2) + "\n"
        )
        await asyncio.to_thread(render_report, self.output.parent, self.output)
        return report

    async def placement(self):
        """Injected heterogeneous workers and deadline admission; not GPU evidence."""
        for case in ("placement", "placement_slo"):
            for repeat in range(self.config.repeats):
                variants = (
                    ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
                )
                for variant in variants:
                    candidate = variant == "candidate"
                    behavior = [
                        Behavior(first_content_delay=0.003, chunk_delay=0.0005),
                        Behavior(first_content_delay=0.04, chunk_delay=0.004),
                    ]
                    if case == "placement_slo":
                        behavior = [Behavior(first_content_delay=0.01, chunk_delay=0.003)]
                    model = PlacementConfig(
                        service=ServiceModel(
                            base_seconds=0.01,
                            prefill_tokens_per_second=1e6,
                            decode_tokens_per_second=1 / 0.003,
                        ),
                        cache_reuse_discount=0,
                        default_slo_seconds=0.25,
                        slo_admission=case == "placement_slo" and candidate,
                    )
                    if case == "placement":
                        model.workers = {
                            f"worker-{i + 1}": ServiceModel(
                                base_seconds=b.first_content_delay,
                                prefill_tokens_per_second=1e6,
                                decode_tokens_per_second=1 / b.chunk_delay,
                            )
                            for i, b in enumerate(behavior)
                        }
                    async with AsyncExitStack() as stack:
                        urls = [
                            await stack.enter_async_context(
                                serve_app(create_mock_worker(str(i), b))
                            )
                            for i, b in enumerate(behavior)
                        ]
                        settings = Settings(
                            worker_urls=urls,
                            capacity=1 if case == "placement_slo" else 8,
                            max_queue=128,
                            queue_timeout=20,
                            prefix_preparations=32,
                            policy="prefix_affinity"
                            if case == "placement" and not candidate
                            else "predicted_ttft",
                        )
                        path = self.output / f"{case}-{repeat}-{variant}-model.json"
                        path.write_text(model.model_dump_json(indent=2) + "\n")
                        settings.placement_config = str(path)
                        arrivals = workload(
                            "overload" if case == "placement_slo" else "routing",
                            self.config.seed + repeat,
                        )
                        async with serve_app(create_gateway(settings)) as url:
                            # Warm the HTTP/tokenizer thread path equally, with
                            # a prompt too short to seed either prefix index.
                            warm, _, _ = await replay(
                                url,
                                "nano-model",
                                [
                                    Arrival(-1, 0, [{"role": "user", "content": "Warmup"}], 1),
                                    Arrival(-2, 0.02, [{"role": "user", "content": "Warmup"}], 1),
                                ],
                                {},
                                self.config.max_outstanding,
                            )
                            results, elapsed, state = await replay(
                                url, "nano-model", arrivals, {}, self.config.max_outstanding
                            )
                    slo = 0.25 if case == "placement_slo" else 0.5
                    summary = detailed_summary(results, elapsed, slo)
                    window = max(a.offset for a in arrivals) + slo
                    summary["fixed_window_seconds"] = window
                    summary["slo_goodput_fixed_window"] = (
                        sum(
                            r.outcome == "success"
                            and r.first_content_seconds is not None
                            and r.first_content_seconds <= slo
                            for r in results
                        )
                        / window
                    )
                    await self.save(
                        {
                            "case": case,
                            "repeat": repeat,
                            "variant": variant,
                            "backend": "simulator",
                            "gateway_settings": asdict(settings),
                            "placement_model": model.model_dump(),
                            "changed_knob": {"policy": settings.policy}
                            if case == "placement"
                            else {"slo_admission": model.slo_admission},
                            "injected_worker_behavior": [b.model_dump() for b in behavior],
                            "schedule": [asdict(a) for a in arrivals],
                            "summary": summary,
                            "tenants": {},
                            "quality": evidence_quality(
                                warm + results,
                                self.config,
                                allow_rejections=case == "placement_slo",
                            ),
                            "warmup_excluded": [asdict(r) for r in warm],
                            "final_pool": state,
                            "results": [asdict(r) for r in results],
                            "scope": "Rates calibrated to injected simulator delays; no real "
                            "prefix cache, KV pressure or GPU speedup is measured.",
                        }
                    )
        return await self.finish("placement")

    async def native_samples(self, client, urls):
        return await asyncio.gather(*(engine_snapshot(client, u) for u in urls))

    async def wait_native_usage(self, client, urls, before, results):
        expected = sum(r.prompt_tokens or 0 for r in results if r.outcome == "success")
        async with asyncio.timeout(10):
            while True:
                after = await self.native_samples(client, urls)
                queried = sum(
                    cache_delta(b, a)["queried_tokens"] for b, a in zip(before, after, strict=True)
                )
                completed = sum(
                    a["values"].get("vllm:request_success_total", 0)
                    - b["values"].get("vllm:request_success_total", 0)
                    for b, a in zip(before, after, strict=True)
                )
                if queried >= expected and completed >= sum(
                    r.outcome == "success" for r in results
                ):
                    return after
                await asyncio.sleep(0.1)

    async def cache_case(self, repeat, variant, urls, profile, *, steady=False):
        key = os.environ["NANO_EXPERIMENT_CACHE"] = uuid.uuid4().hex
        cfg = TenantConfig(
            tenants=[
                Tenant(
                    id="cache",
                    api_key_env="NANO_EXPERIMENT_CACHE",
                    max_inflight=1,
                    tpm=1000000,
                    rpm=10000,
                )
            ]
        )
        path = self.root / ".local/experiment-cache.json"
        await asyncio.to_thread(path.write_text, cfg.model_dump_json())
        settings = Settings(
            worker_urls=urls,
            model=profile.model_alias,
            capacity=1,
            policy="prefix_affinity" if variant == "candidate" else "round_robin",
            tenant_config=str(path),
            prefix_tokenizer=str(self.root / profile.model_path),
            model_revision=profile.model_revision,
            max_model_len=profile.max_model_len,
            request_timeout=60,
        )
        # Salt changes for every gateway, while both variants see exactly the same prompt tokens.
        # No reset/dev/RPC endpoint is needed. Do not disable cache isolation for an experiment.
        prompts = [
            [
                {
                    "role": "system",
                    "content": f"Group {group}. "
                    + "A distributed queue holds pending work until a worker is available. " * 48,
                },
                {"role": "user", "content": "Name two colors."},
            ]
            for group in range(3)
        ]
        warm = []
        measured = []
        started = time.perf_counter()
        async with (
            serve_app(create_gateway(settings)) as url,
            httpx.AsyncClient(trust_env=False) as client,
        ):
            cold = await self.native_samples(client, urls)
            for i, messages in enumerate(prompts * (2 if steady else 1)):
                r, _, _ = await replay(
                    url,
                    profile.model_alias,
                    [Arrival(i, 0, messages, 8, "cache")],
                    {"cache": key},
                    self.config.max_outstanding,
                )
                warm.extend(r)
            before = await self.wait_native_usage(client, urls, cold, warm)
            measurement_start = time.perf_counter()
            for i, messages in enumerate(prompts):
                r, _, state = await replay(
                    url,
                    profile.model_alias,
                    [Arrival(i, 0, messages, 8, "cache")],
                    {"cache": key},
                    self.config.max_outstanding,
                )
                measured.extend(r)
            elapsed = time.perf_counter() - measurement_start
            after = await self.wait_native_usage(client, urls, before, measured)
        deltas = [cache_delta(b, a) for b, a in zip(before, after, strict=True)]
        hits, queries = (
            sum(d["hit_tokens"] for d in deltas),
            sum(d["queried_tokens"] for d in deltas),
        )
        return {
            "case": "cache_steady" if steady else "cache",
            "repeat": repeat,
            "variant": variant,
            "backend": "vllm-metal",
            "changed_knob": {"policy": settings.policy},
            "gateway_settings": asdict(settings),
            "cache_condition": (
                "Three groups, two primes per group, then warm recurrence. "
                if steady
                else "Three groups, one prime per group, then first recurrence. "
            )
            + "New gateway salt each variant; identical prompt tokens. "
            "This is not steady-state cache-pressure performance.",
            "warmup_excluded": [asdict(r) for r in warm],
            "phase_wall_seconds": time.perf_counter() - started,
            "summary": detailed_summary(measured, elapsed, 2),
            "tenants": {},
            "quality": evidence_quality(warm + measured, self.config),
            "final_pool": state,
            "native_phase_means_seconds": phase_means(before, after),
            "native_before": before,
            "native_after": after,
            "native_cache": {
                "hit_tokens": hits,
                "queried_tokens": queries,
                "token_hit_ratio": hits / queries if queries else None,
            },
            "results": [asdict(r) for r in measured],
        }

    async def context_case(self, repeat, variant, urls, profile):
        settings = Settings(
            worker_urls=urls[:1],
            model=profile.model_alias,
            policy="prefix_affinity",
            prefix_tokenizer=str(self.root / profile.model_path),
            model_revision=profile.model_revision,
            request_timeout=30,
            max_model_len=profile.max_model_len if variant == "candidate" else None,
        )
        async with serve_app(create_gateway(settings)) as url:
            invalid, elapsed, _ = await replay(
                url,
                profile.model_alias,
                [Arrival(0, 0, short_messages(), profile.max_model_len)],
                {},
                self.config.max_outstanding,
            )
            valid, _, state = await replay(
                url,
                profile.model_alias,
                [Arrival(1, 0, short_messages(), 4)],
                {},
                self.config.max_outstanding,
            )
        return {
            "case": "context",
            "repeat": repeat,
            "variant": variant,
            "backend": "vllm-metal",
            "changed_knob": {"max_model_len": settings.max_model_len},
            "gateway_settings": asdict(settings),
            "summary": detailed_summary(invalid, elapsed, 2),
            "tenants": {},
            "quality": evidence_quality(invalid, self.config, allow_errors=True),
            "followup_valid_request": asdict(valid[0]),
            "final_pool": state,
            "results": [asdict(r) for r in invalid],
        }

    async def metal(self, profile_path):
        profile = FlavorProfile.load(profile_path)
        # Current native experiment matrix is validated for this pinned Mac model only.
        if profile.flavor != "macos-metal" or profile.model_alias != "nano-qwen":
            raise ValueError("Validate this matrix on the pinned Mac profile before porting it")
        self.profile = profile
        self.runtime = {
            "environment": await inspect_runtime(profile, self.root),
            "profile": profile.model_dump(),
            "worker_count": 2,
            "physical_gpu_shared": True,
        }
        ports = [free_port() for _ in range(3)]
        cluster = FlavorCluster(profile, self.root, port=ports[0], worker_ports=ports[1:])
        try:
            await cluster.start()
            for case in ["cache", "cache_steady", "tenants", "context"]:
                for repeat in range(self.config.repeats):
                    for variant in (
                        ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
                    ):
                        if case in ("cache", "cache_steady"):
                            run = await self.cache_case(
                                repeat,
                                variant,
                                cluster.worker_urls,
                                profile,
                                steady=case == "cache_steady",
                            )
                        elif case == "context":
                            run = await self.context_case(
                                repeat, variant, cluster.worker_urls, profile
                            )
                        else:
                            run = await self.case(
                                case, repeat, variant, cluster.worker_urls[:1], native=True
                            )
                        await self.save(run)
        finally:
            await cluster.close()
        return await self.finish("metal")


async def run_experiments(
    batch="control",
    config="configs/experiments.local.json",
    output=None,
    profile="configs/macos-metal.json",
):
    root = Path.cwd()
    settings = ExperimentConfig.model_validate_json(await asyncio.to_thread(Path(config).read_text))
    runner = ExperimentRunner(root, settings, output or f"results/experiments/{batch}")
    if batch == "control":
        await runner.control()
    elif batch == "metal":
        await runner.metal(profile)
    elif batch == "placement":
        await runner.placement()
    else:
        raise ValueError("Unknown experiment batch")
    return runner
