"""Render checked-in experiment evidence as one self-contained operator report."""

import json
import statistics
from html import escape
from pathlib import Path


def number(value, *, milliseconds=False):
    if value is None:
        return "—"
    return f"{value * 1000:.1f} ms" if milliseconds else f"{value:.2f}"


def render_report(root, source=None):
    root = Path(root)
    sections = []
    override = None if source is None else json.loads((Path(source) / "summary.json").read_text())
    for batch in ("control", "metal"):
        path = root / batch / "summary.json"
        if override is not None and override["batch"] == batch:
            data = override
        elif path.exists():
            data = json.loads(path.read_text())
        else:
            continue
        comparisons = []
        for c in data["comparisons"]:
            a = [
                p["baseline"]
                for p in c["pairs"]
                if p["usable_for_comparison"] and p["baseline"] is not None
            ]
            b = [
                p["candidate"]
                for p in c["pairs"]
                if p["usable_for_comparison"] and p["candidate"] is not None
            ]

            def med(values):
                return statistics.median(values) if values else None

            latency = "seconds" in c["metric"]
            comparisons.append(
                "<tr>"
                + "".join(
                    f"<td>{escape(str(v))}</td>"
                    for v in (
                        c["case"] + (" / " + c["tenant"] if c["tenant"] else ""),
                        c["metric"],
                        number(med(a), milliseconds=latency),
                        number(med(b), milliseconds=latency),
                        number(c["median_improvement_percent"]) + "%",
                        c["assessment"],
                    )
                )
                + "</tr>"
            )
        runs = []
        for r in data["runs"]:
            s = r["summary"]
            cache = r.get("native_cache", {}).get("token_hit_ratio")
            runs.append(
                "<tr>"
                + "".join(
                    f"<td>{escape(str(v))}</td>"
                    for v in (
                        r["case"],
                        r["repeat"],
                        r["variant"],
                        json.dumps(s["outcomes"]),
                        number(s["successful_first_content_seconds"]["p95"], milliseconds=True),
                        number(s["ttft_slo_fraction_of_arrivals"]),
                        number(s["output_tokens_per_second"]),
                        number(cache),
                        ", ".join(r["quality"]["flags"]) or "valid",
                    )
                )
                + "</tr>"
            )
        sections.append(f"""<section><h2>{escape(batch.upper())}</h2>
<p>{escape(data["scope"])}</p>
<p>Seed {data["config"]["seed"]} · {data["config"]["repeats"]} paired repeats ·
AB / BA order · source {escape(data["metadata"]["python_source_sha256"][:12])}</p>
<h3>Paired comparisons</h3><div class="table"><table><thead><tr>
<th>Case / tenant</th><th>Metric</th><th>Baseline median</th><th>Candidate median</th>
<th>Improvement</th><th>Assessment</th></tr></thead><tbody>{"".join(comparisons)}</tbody></table></div>
<h3>Every trial, including failures</h3><div class="table"><table><thead><tr>
<th>Case</th><th>Repeat</th><th>Variant</th><th>Outcomes</th>
<th>TTFT p95</th><th>SLO / all arrivals</th>
<th>Output tokens/s</th><th>Native cache token ratio</th><th>Quality</th>
</tr></thead><tbody>{"".join(runs)}</tbody></table></div></section>""")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nano-inference experiments</title>
<style>body{font:14px system-ui;color:#203b36;background:#f2f6f4;margin:0;padding:32px}
main{max-width:1400px;margin:auto}h1{font-size:26px}h2{font-size:20px}h3{font-size:15px}
a{color:#18745c}p{color:#5f736c;line-height:1.7}section{background:white;border:1px solid #dbe6df;
border-radius:12px;padding:24px;margin:24px 0}.table{overflow-x:auto}
table{border-collapse:collapse;width:100%;
white-space:nowrap}th,td{text-align:left;padding:11px 9px;
border-bottom:1px solid #e3ebe6;font-size:12px}
th{color:#6a7f76}aside{padding:16px;background:#dfede6;line-height:1.7;border-radius:8px}</style><main>
<h1>nano-inference / controlled experiments</h1><p><a href="/monitor">Live monitor ↗</a></p>
<aside>Positive improvement means the candidate moved in the intended direction.
Three repeats establish a descriptive local observation, not statistical significance.
Read success counts and SLO attainment across <strong>all arrivals</strong>
beside successful latency.
The first-recurrence cache case and the fully warm negative control are separate experiments.
Exploratory preparation-limited trials are preserved on disk and excluded here.</aside>"""
    page += "".join(sections) or "<section>No completed batches yet.</section>"
    page += (
        "<p>Raw schedules, knobs, request results, engine counters and source metadata "
        "are in results/experiments/.</p></main></html>"
    )
    path = root / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path
