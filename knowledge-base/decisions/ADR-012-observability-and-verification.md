---
id: ADR-012
title: Observability — make the claims measurable, not asserted
status: v1 accepted
date: 2026-09-01
sources: teams/T4-A5, T3-A4, T1-A4, T5-A4
---

# ADR-012 — Instrument the claims, or they are just slides

## Context

Every headline in this corpus is a number someone must be able to re-derive live. Two defects block that
today, both found by reading source rather than benchmarking:

- **`prometheus/prometheus.yml` has `scrape_interval: 15s`.** Across a 3-minute stage demo that is **12
  samples**. At 1 s it is **180**. Every panel is a flat line until this changes; it is the cheapest and most
  load-bearing item in this ADR.
- **`node.py:87` closes its timer before the logits/base64 branch.** node2's 607,744 B encode is therefore
  **uncounted** — the bottleneck stage is under-reported by exactly the amount that makes it the bottleneck.

Both `coordinator.py:180` and `node.py:113` emit hand-rolled f-string counters. Two counters can only yield a
*mean*, and a mean hides the p99 the SLO in ADR-005 is written against.

The single scalar worth putting on screen: `dllm:imbalance = max(stage p95) / min(stage p95) =
0.30897 / 0.19776 = **1.562** (measured)` — ADR-007's finding, live, going to ~1.0 after the re-split.

## Options considered

| option | verdict | why |
|---|---|---|
| Hand-rolled counters (today) | rejected | Cannot express a percentile. |
| **`prometheus_client` Histograms with a landmark bucket set** (.04 and .32 for stage service; 2 for TTFT; 10 for queue wait; 1.1 for compression ratio, to catch *expansion*) | **ACCEPTED v1** | `prometheus_client` is already a gateway dependency; Prometheus + Grafana are already in `docker-compose.yml`. Names mirror vLLM's so Grafana dashboards port. |
| Prometheus **native histograms** | v2 proposed | Stops bucket layout being a design decision frozen at authoring time. |
| **OpenTelemetry SDK 1.44.0 spans + a 32-byte `F_TRACE` DLP header extension** (`<16s8sB7x>`, `calcsize == 32`; 40+32 = 72, `72 % 8 == 0` so ADR-002's payload alignment survives — 40+25 = 65 would break it) | **ACCEPTED v1** | 0.88% of a decode frame when sampled, **0 bytes when not**. Inside ADR-002's own 1.12% header budget. |
| Untamed per-token tracing | rejected | 4 spans/token × 512 tokens = **2,049 spans in one trace**. Fix: always-sample `token_idx < 8`, then `ParentBased(TraceIdRatioBased(0.01))`. |
| `session_id` / `request_id` / `trace_id` / `token_idx` as metric **labels** | **rejected — unbounded cardinality** | Kept off labels, whole-mesh cardinality is ~420 series. Put them in exemplars instead. |
| Prometheus exemplars (`--enable-feature=exemplar-storage` + OpenMetrics scrape) + a Grafana data link into Tempo | **ACCEPTED v1** | Two flags and one datasource field; turns a p99 spike into a click-through to the offending trace. |
| Closed-loop VU load test only (locust) | insufficient alone | Closed-loop tests self-throttle — **coordinated omission** — and structurally cannot observe 429 behaviour. |
| **Open-loop probe (vegeta) at 0.1 → 1 req/s** | **ACCEPTED v1** | Against v1's `µ_req = 0.252 req/s` (modelled), rates above 0.3/s **must** produce 429 + `Retry-After`. This is the only test that proves ADR-005's admission control exists rather than merely emitting a metric. |
| `vllm bench serve` as the harness | **v2 proposed** | It computes TTFT/TPOT/ITL percentiles natively and puts our tok/s on the same axis as published vLLM numbers, which no bespoke harness can do. Requires the coordinator to emit OpenAI-shaped SSE chunks (~hours). |
| OTel Collector with tail-based sampling; k6 with arrival-rate executors | v2 proposed | Replaces head sampling and the locust+vegeta pair. k6's current version could not be pinned from sources — cite executor names only. |

## Decision

1. **`scrape_interval: 1s`**, plus `"refresh": "1s"` and per-panel `"interval": "1s"` in the dashboard.
2. Replace the f-string counters with Histograms; **fix the `node.py:87` timer scope**.
3. Ship Grafana panels: node-state timeline, tok/s against the **1.2734 tok/s v0 threshold**, per-stage p95
   bargauge (renders 0.206 / 0.198 / 0.309 as three bars — ADR-007's imbalance read off the screen), queue
   depth against N\*=3, admission outcomes, circuit state. Add Prometheus **annotations** on
   `dllm_pipeline_epoch` changes and `dllm_node_state == 0`, so ADR-009's kill-node event paints a
   time-aligned red line on every panel simultaneously.
4. OTel spans behind `F_TRACE`, head-sampled at `token_idx < 8`, exported to a Grafana Tempo container.
5. `bench/chaos_failover.sh` — SIGKILL node1 after token 8; assert `node_down` + `reshard_done` +
   `finish_reason == stop` + ≥63 tokens. **Run the `DLLM_FAILOVER=off` negative control first.** Against the
   PoC as it stands the script fails on `gateway/app.py:44` (ADR-009's breaker bug) — that is a real found
   bug, not a harness defect.
6. `bench/locustfile.py` (SSE-aware, TTFT/TPOT via `events.request.fire`) driving the 7-point closed-loop
   sweep R ∈ {1,2,3,4,6,8,12} at 3 min each, overlaid on ADR-005's `min(N/D, 1/D_max)` bound. **A knee at
   R=3 validates the semaphore; an earlier knee proves the anyio 40-thread thrash is unfixed. Either outcome
   is a result.**
7. Emit OpenAI-shaped SSE from the coordinator so a standard harness can drive us at all.

## Consequences

**Good.** Every claim in ADR-005/006/007/009 becomes continuously verifiable; N\* self-corrects at runtime
from live stage times; the strongest visual in the deck (a one-token waterfall with node2's span 1.56x the
others) costs 32 optional bytes.

**Bad.**
- **The `F_TRACE` extension is a request to ADR-002, not an agreed change.** It consumes `flags` bit 4 and
  moves the payload offset from 40 to 72 when set. Coordinate before either side implements.
- **Panel 5 is a single-use demo asset.** Once ADR-007's re-split lands it becomes three equal bars. Sequence
  the deck so the imbalance is shown *before* the fix, or the strongest visual is spent.
- Every latency number here is **inherited** from ADR-002/004/005, not independently re-measured; only the
  arithmetic (imbalance 1.5623, `calcsize` 32, alignment 72 % 8 == 0) was verified locally.
- The chaos script's expected SSE output assumes ADR-009 is implemented. Today it is a **spec for a demo, not
  a recording of one** — which is exactly why the negative control runs first.
- 1 s scraping across 3 nodes is fine at demo scale and will need revisiting at fleet scale.

## Status

**v1 accepted.** **v2 proposed:** Alertmanager with multiwindow burn-rate rules on the 10 s queue-wait bucket
edge and `dllm:imbalance > 1.25`; `vllm bench serve` adoption; OTel Collector tail sampling with metrics on
OTLP; k6 arrival-rate executors; native histograms.
