---
id: ADR-005
title: Queueing, admission control and backpressure
status: v1 accepted
date: 2026-09-01
sources: teams/T3-A4, T3-A2, T3-A3, T3-A1, T1-A4, T4-A5
---

# ADR-005 — Bounded queue, admission at N*, credit backpressure

## Context

v0 defect #8: no admission control, no queue, no backpressure, no priority. The system is a **3-station
tandem queue with c=1 per station**, and classical closed-network analysis gives the answer in one line:

```
X(N) <= min(N/D, 1/D_max)          D = sum of stage times, D_max = bottleneck stage
N*   = D / D_max                   the knee: admitting past it buys ZERO throughput
```

Measured stage times at seq=512: node0 **205.81 ms**, node1 **197.76 ms**, node2 **308.97 ms**.
So `D = 0.71254 s`, `D_max = 0.30897 s`, **N\* = 2.31 → admit 3**. After the ADR-007 rebalance
`D_max = D/3 = 0.23751 s` and **N\* = 3.00 exactly — the node count**.

| N in flight | X (tok/s) | note |
|---:|---:|---|
| 1 | 1.273 (measured, incl. transport) | v0 today; `demo.sh` sends one curl |
| 3, v0 split | 3.237 | 2.54x, concurrency only |
| 3, rebalanced | 4.210 | 3.31x |
| 8 | 3.237 | same throughput, **+167% latency** |

There is a second, larger defect hiding underneath: `node.py:82` is a **sync `def forward`**, so Starlette
runs it in the anyio threadpool whose default `total_tokens` is **40** (verified, anyio 4.12.1). Forty
concurrent torch forwards inside a `cpus: "2"` container is processor sharing, not a queue.

Where to sit on `W/S = 1/(1−ρ)`: `dW/dρ = S/(1−ρ)²`, so a +10% arrival bump costs +11% latency at ρ=0.50,
**+43% at ρ=0.75**, **+900% at ρ=0.90**. Hence ρ\* = 0.75, where p95 = 12.0·S and p99 = 18.4·S are bounded.
Queue depth then comes from Little's law used *backwards*, `K = λ_admit · W_SLO`: v1 balanced, `S_req =
32 × 41.31 ms = 1.322 s`, `µ_req = 0.756 req/s`, at ρ=0.75 `λ = 0.567 req/s`; a 10 s wait SLO gives **K = 6**
and `Retry-After = ceil(K/µ_req) = 8 s`. Total in system K + N\* = 9.

## Options considered

| option | verdict | why |
|---|---|---|
| No queue (today) | rejected | ρ drifts to 0.9+ where a 10% traffic bump is a 900% latency increase, and 20 bloated in-flight requests at 2048 ctx hold **160 MB of KV per shard doing nothing** inside a 2 GB container. |
| Unbounded `asyncio.Queue`, or per-stage buffers | rejected | Bufferbloat: accepting work you cannot serve converts a fast rejection into a slow timeout, and with ADR-001 it pins KV memory too. Per-stage buffers additionally hide the queue where nobody can see it — one admission point plus per-stage **credits** keeps the decision in one place. |
| **`asyncio.Semaphore(3)` + `asyncio.Queue(maxsize=6)` per class, `put_nowait` → HTTP 429 + `Retry-After`** | **ACCEPTED v1** | The entire scheduler is two integers, both derived rather than guessed. Never silent-drop, never accept-then-drop. |
| Credit-based flow control in the DLP header (`credit` field, ADR-002) | **ACCEPTED v1** | TCP backpressure is invisible to the application — the coordinator would keep accepting and queue in kernel buffers with no signal. Explicit credit lets it **shed at admission time**. |
| Deficit Round Robin over 2 classes (interactive w=4/q=32 tok, batch w=1/q=8 tok) | **v1** | Shreedhar & Varghese, SIGCOMM 1995. Quantum in **tokens, not requests**, because request sizes vary 100:1. Guarantees each class `w_i/Σw` of the bottleneck, so cross-class starvation is structurally impossible and general ageing is unnecessary. |
| Chunked prefill C=64 | **v1** | One 2048-token prefill blocks **28.7 decode steps** (3,551 ms measured vs 123.94 ms measured). C=64 costs 64 × 1.734 ms = 111 ms ≈ 0.90 of one decode step → **32x less head-of-line blocking**, for 32 extra traversals ≈ 1.3 MB ≈ 0.011 s on 1 GbE. |
| NATS JetStream 2.10+ as the queue; EDF + feasibility test (`W_pred + G_max·D_max > deadline` ⇒ reject) | **v2 proposed** | JetStream gives per-message ack with `AckWait` redelivery, WorkQueue retention, class subject hierarchies and request-reply cancellation — surviving coordinator restart and enabling replicated schedulers. EDF turns the SLO from an aspiration into a contract, using `max_tokens` as `G_max`. |
| **Apache Kafka on the inference path** | **rejected** | A partition-ordered log with no per-message ack ⇒ head-of-line blocking is *architectural*; `acks=all` + `min.insync.replicas=2` adds ms-to-tens-of-ms p99 against a 41 ms per-token budget; durability of a deadline-bound message is a contradiction. Telemetry/audit/billing plane only. |

## Decision

1. **Set anyio's default thread limiter to 1 in `node.py` startup** (one line) *before* anything else. Without
   it, "concurrency" is a 40-way CPU thrash and every number in this ADR is fiction.
2. `asyncio.Semaphore(3)` at the coordinator entry + `Semaphore(1)` per node URL. **N\* = 3 is the whole
   scheduler**, and after ADR-007 it equals the node count exactly.
3. `asyncio.Queue(maxsize=6)` per class; `QueueFull` → gateway returns **HTTP 429 with a computed
   `Retry-After: 8`**. Bounded, always. "Full" means rejected fast, not buffered slowly.
4. DRR across `interactive` and `batch`, quantum in tokens.
5. Chunked prefill at **C=64**, interleaved with decode at every dispatch point. The same chunks double as the
   microbatch units in ADR-006.
6. **Cancellation propagation** across all three stages: `Request.is_disconnected()` at the gateway,
   `task.cancel()` in the coordinator loop, `POST /cancel {request_id}`, `DELETE /kv/{request_id}` + a TTL
   sweeper. A running torch forward cannot be interrupted, so cancellation is cooperative at chunk boundaries;
   C=64 caps the uncancellable unit at 111 ms. Recovers 24.3 s of 3-node compute per abandoned v0 request
   (3.84 s in v1). Recompute N\* at runtime from live `dllm_stage_service_seconds`, not a hard-coded 3.

## Consequences

**Good.** 2.54x throughput from admission alone (3.31x with ADR-007); p95 wait bounded at 12·S instead of
undefined; KV memory (ADR-001) becomes an admission input; a scheduler that is two integers and ~60 LOC.

**Bad.**
- **The v1 per-stage KV-cached times (node0 35.80 / node1 34.40 / node2 53.74 ms) are apportioned** from a
  measured 123.94 ms total using measured v0 stage shares — `(modelled)`, not measured. If framework overhead
  does not scale with the shares, N\* shifts. Mitigated by recomputing it at runtime.
- **M/M/1 assumes Poisson arrivals and exponential service.** LLM service is closer to deterministic-per-token
  times a highly variable output length G, so real variance is dominated by the G distribution. The
  `1/(1−ρ)` curve is a **conservative upper bound**; the ρ\*=0.75 choice holds, absolute p99 predictions do not.
- **Output length G is unknown at admission**, so the feasibility test and K sizing both lean on the client's
  `max_tokens`; clients that pass a large value and stop early make the controller reject conservatively.
  And **the coordinator becomes the single scheduler of record** — the queue is lost on its restart until v2.
  Not a *new* SPOF (compose already runs a singleton), but now load-bearing. See ADR-009.
- **None of it shows unless concurrent load actually arrives.** `demo.sh` sends one curl; drive the demo at
  ≥3 concurrent requests or the queueing work is invisible. And closed-loop load tests structurally cannot
  observe 429 behaviour (coordinated omission) — an **open-loop** probe above `µ_req = 0.252 req/s` is the
  only test that proves admission control exists.

## Status

**v1 accepted.** NATS JetStream, EDF + feasibility admission, HTTP/2 `WINDOW_UPDATE`-carried credits, and a
replicated stateless coordinator are **v2 proposed**.
