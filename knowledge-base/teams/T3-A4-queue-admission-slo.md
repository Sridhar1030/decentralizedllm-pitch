---
team: T3 — Scheduling, Queueing & Compute Utilisation
agent: T3-A4
topic: Queueing network model, admission control, backpressure & SLO
headline: "The pipeline is a 3-station tandem queue bottlenecked on node2 at 308.97 ms (measured); the optimal in-flight depth is N* = D/D_max = 2.31 → 3, and balancing the shards makes N* land on exactly 3 = the node count — so one bounded queue (K=6, Little) plus a semaphore of 3 is the entire scheduler."
---

# T3-A4 — Queueing system: model, sizing, backpressure, SLO

Stage times are **(measured)**, from T1-A1 §5 (seq=512, `node.py` as written). Everything derived is **(modelled)** with arithmetic shown. Consistent with T3-A2, which reaches R=3 empirically; §2 is the formal derivation of why 3.

---

## 1. The system *is* a tandem queueing network — 3 stations in series, c=1 each

`coordinator.py:78-95` sends every token node0 → node1 → node2 strictly in order. Open tandem network (Jackson 1957): visit ratio V_i = 1, so service **demand** D_i = S_i.

| station | contents | S_i (measured, s) | μ_i = 1/S_i (tok/s) | share of D | ρ_i today |
|---|---|---:|---:|---:|---:|
| node0 | embed + layers 0-7 | 0.20581 | 4.859 | 28.9% | 26.2% |
| node1 | layers 8-15 | 0.19776 | 5.057 | 27.7% | 25.2% |
| node2 | layers 16-23 + norm + **lm_head** | **0.30897** | **3.237** | **43.4%** | **39.3%** |
| — | Σ = **D** | **0.71254** | — | 100% | mean 30.2% |

**D = 0.71254 s, D_max = 0.30897 s (node2).** Measured v0 wall clock incl. transport 0.7853 s/token → X_v0 = **1.2734 tok/s (measured)**. Node2 is the bottleneck for the reason in VERIFIED-FACTS FINDING 1: `lm_head` is 9.13 layer-equivalents bolted onto an 8-layer shard.

> The "2 of 3 idle → 1/3 ceiling" framing is imprecise. Per station it is **S_i/D**: 43.4% for node2,
> 25–29% for the other two. Imbalance makes it worse than 1/3 for two nodes and better for one.

---

## 2. Capacity limit and the concurrency knob: **X(N) ≤ min(N/D, 1/D_max)**

Asymptotic bounds for a closed network, zero think time (Lazowska/Zahorjan/Graham/Sevcik, *Quantitative System Performance*, 1984, §5). N is the admission semaphore. This inequality is the whole scheduler. Saturation **N\* = D/D_max = 0.71254/0.30897 = 2.306 → ⌈2.31⌉ = 3.**

> **AUDIT CORRECTION (90-AUDIT F06):** the `vs measured 1.2734` column divides a **compute-only** ceiling
> (X(N) excludes transport) by a **transport-inclusive** measured baseline, so it credits concurrency with
> ~1.10x of transport saving that belongs to the connection-pool/codec levers. The concurrency-only figures
> are **2.31x** (3.237/1.403) and **3.00x** (4.210/1.403) — which is exactly what T3-A2 reports. Use those.

| N in flight | X(N) tok/s | vs measured 1.2734 | R(N)=N/X s/token | verdict |
|---:|---:|---:|---:|---|
| 1 (today) | 1.403 compute-only / **1.273 measured** | 1.00× | 0.785 | 2 of 3 nodes idle |
| 2 | 2.807 | 2.20× | 0.713 | bubble-free |
| **3** | **3.237** | **2.54×** | 0.927 | **saturated — node2 100% busy** |
| 4 | 3.237 | 2.54× | 1.236 | +0% throughput, **+33% latency** |
| 8 | 3.237 | 2.54× | 2.472 | +0% throughput, **+167% latency** |

**Past N\*, admission buys exactly zero throughput and buys latency linearly.** That is the numerical case for a bounded queue, and why the semaphore is 3 and not 40 (§12).

With the FINDING-1 balanced split (D_max = D/3 = 0.23751 s): X = **4.210 tok/s = 3.31× v0 (modelled)** and **N\* = 3.00 exactly — the optimal in-flight depth equals the node count.** Deck line:

> **Balance the shards and the scheduler's only tuning parameter becomes "3" — the number of nodes.**

v1 (KV cache, measured Σ = 123.94 ms/token decode, apportioned by the measured shares above): node0 35.80 / node1 34.40 / node2 53.74 ms → X = 18.61 tok/s; balanced D_max = **41.31 ms**, X = **24.21 tok/s (modelled)**, N\* = 3.00.

---

## 3. The 1/(1−ρ) blow-up, and where to sit on it

M/M/1 at the bottleneck: W = S/(1−ρ), W_q = ρS/(1−ρ), L = λW (Little). Sojourn is exponential(μ−λ), so p95 = W·ln 20 = 3.00 W and p99 = W·ln 100 = 4.61 W.

| ρ | W/S | L | L_q | p95/S | p99/S |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 2.00 | 1.00 | 0.50 | 6.0 | 9.2 |
| 0.70 | 3.33 | 2.33 | 1.63 | 10.0 | 15.4 |
| **0.75** | **4.00** | **3.00** | **2.25** | **12.0** | **18.4** |
| 0.80 | 5.00 | 4.00 | 3.20 | 15.0 | 23.0 |
| 0.90 | 10.00 | 9.00 | 8.10 | 30.0 | 46.1 |
| 0.95 | 20.00 | 19.00 | 18.05 | 59.9 | 92.1 |
| 0.99 | 100.00 | 99.00 | 98.01 | 299.6 | 460.5 |

**Operating point ρ\* = 0.75** — not folklore, picked by surge sensitivity dW/dρ = S/(1−ρ)²:

| ρ now | +10% λ → | W/S before → after | change |
|---:|---:|---:|---:|
| 0.50 | 0.550 | 2.00 → 2.22 | **+11%** |
| **0.75** | 0.825 | 4.00 → 5.71 | **+43%** |
| 0.90 | 0.990 | 10.00 → 100.00 | **+900%** |

At ρ=0.90 a 10% traffic bump is a 10× latency incident; at 0.75 it is a shrug. Sanity check: L = 3.00 at ρ=0.75 coincides with N\* = 3 — the open and closed models agree.

---

## 4. Little's law used *backwards* to size the bounded queue

**K = λ_admit × W_queue_SLO.** The depth is derived, not guessed.

v1 balanced, G = 32 output tokens: S_req = 32 × 41.31 ms = **1.322 s**, μ_req = 0.756 req/s. At ρ\* = 0.75 → λ_admit = 0.567 req/s = **34.0 req/min (modelled)**.

| queue-wait SLO | K = λ·W | K | `Retry-After` = ⌈K/μ_req⌉ |
|---:|---:|---:|---:|
| 5 s | 2.84 | 3 | 4 s |
| **10 s** | **5.67** | **6** | **8 s** |
| 30 s | 17.02 | 18 | 24 s |

**Ship K = 6, W_SLO = 10 s**; total in system K + N\* = 9. v0 for reference: S_req = 32 × 0.30897 = 9.89 s, μ_req = 0.101 req/s → **6 req/min** is the honest v0 capacity.

---

## 5. Where the queue lives

| candidate | sees pipeline depth? | can 429 the client? | verdict |
|---|---|---|---|
| Gateway `gateway/app.py` | ✗ proxies blind | ✓ only holder of the client socket | **translator** |
| **Coordinator** `coordinator.py` | ✓ owns the token loop + all 3 node URLs | ✗ | **owner** |
| Per-node `node.py` | only its own | ✗ | **bufferbloat — §6** |

**One admission queue in the coordinator; the gateway translates its backpressure into HTTP 429.** Putting the queue in the gateway forces it to model state it cannot observe. This adds no new SPOF — `docker-compose.yml` already runs a singleton coordinator. v2 replicates it (§12).

---

## 6. Single admission point + per-stage credits > per-stage buffers (bufferbloat)

Throughput is bottleneck-bound either way: per-stage buffering **cannot** raise X above 1/D_max. Extra buffering therefore buys *only* latency and memory — the definition of bufferbloat (Gettys & Nichols, *Bufferbloat: Dark Buffers in the Internet*, CACM 2012): buffers absorb the congestion signal, so the sender never learns to slow down.

The LLM version is worse than the network version: **a request queued behind node2 still holds its KV cache on node0 and node1.** From FINDING 3, 12 KB/token whole-model → 4 KB/token per 8-layer shard. 20 bloated in-flight requests at 2048 ctx = **160 MB of KV per shard doing nothing (modelled)**, in a 2 GB container. Buffering does not merely delay, it evicts working set.

**Credit-based flow control instead** (link-level credits as in InfiniBand and PCIe — and usefully, HTTP/2's `SETTINGS_INITIAL_WINDOW_SIZE` / `WINDOW_UPDATE`): each stage advertises free slots, the coordinator dispatches only while holding a credit. The only queue left is the bounded admission queue, which is also the only component holding the client's socket. Prior art: **vLLM's scheduler is one global waiting/running queue with preemption, not per-layer queues.**

---

## 7. Bounded, always — and what "full" means

| condition | response | headers |
|---|---|---|
| room in queue | 200 / SSE | `X-Queue-Position`, `X-Estimated-Wait` |
| queue full (K=6) | **429** | `Retry-After: ⌈K/μ_req⌉` = 8 s, computed from live μ, never a constant |
| predicted finish > deadline | **429**, `X-Reject-Reason: deadline-infeasible` | `Retry-After` |
| coordinator down / circuit open | 503 (already `gateway/app.py:33`) | `Retry-After` |

Never silent-drop, never accept-then-drop: an accepted LLM request implies the client paid the prefill, and a silent drop guarantees a retry storm on a system already at ρ ≥ 1. `asyncio.Queue.put_nowait()` raising `asyncio.QueueFull` is the entire mechanism — stdlib, zero deps.

---

## 8. Classes, fair queueing, ageing

| class | example | weight | quantum (tokens/round) | SLO |
|---|---|---:|---:|---|
| `interactive` | chat, `max_tokens ≤ 128` | 4 | 32 | TTFT ≤ 2 s, wait ≤ 10 s |
| `batch` | eval sweeps, long generations | 1 | 8 | wait ≤ 300 s |

**Deficit Round Robin (Shreedhar & Varghese, SIGCOMM 1995), not WFQ** — same fairness bound, O(1) per item, ~15 lines; WFQ needs a virtual-clock priority queue for no extra benefit here. Quantum is in **tokens, not requests**, because request sizes vary 100:1 (`max_tokens` 8 vs 2048).

**Ageing:** DRR already guarantees each class w_i/Σw of the bottleneck (batch keeps ≥20% of node2 under an interactive flood), so cross-class starvation is structurally impossible and general ageing is unnecessary. Keep it only as a within-class deadline escape hatch: `p_eff = p_base − ⌊wait_s / 10⌋`. Two classes, not five — more classes is a v2 problem nobody has yet.

---

## 9. Head-of-line blocking — chunked prefill is the answer

In measured numbers: one 2048-token prefill costs **Σ 3551.25 ms of node compute (measured)**; a v1 decode step costs **123.94 ms (measured)**. One prefill blocks the pipeline for **28.7 decode steps** — 20 queued chat turns each eat 3.55 s of added wait.

Chunk the prefill and interleave with decode at each dispatch point. Pick C so a chunk costs ≤ one decode step (per-position prefill cost **1.734 ms, measured**):

| C | chunk cost | × one decode step | chunks for 2048 | HOL delay |
|---:|---:|---:|---:|---:|
| 32 | 55.5 ms | 0.45 | 64 | 55 ms |
| **64** | **111.0 ms** | **0.90** | **32** | **111 ms** |
| 128 | 222.0 ms | 1.79 | 16 | 222 ms |
| 256 | 443.9 ms | 3.58 | 8 | 444 ms |

**Ship C = 64: HOL blocking 3551 ms → 111 ms, 32× (modelled from the measured per-position cost).** Cost: 32 extra traversals × 3 hops × ~14 KB post-KV = 1.3 MB ≈ 0.011 s on 1 GbE. Free.

Exact prior art: **vLLM `--enable-chunked-prefill` / `enable_chunked_prefill=True`, chunk size = `max_num_batched_tokens` (default 2048 in v0.8.2; on by default wherever possible in vLLM V1)**; the technique is *stall-free batching* from **Sarathi-Serve (Agrawal et al., OSDI '24)**; SGLang ships the same. Bonus: **chunks are also the microbatch units that fill T3-A2's pipeline bubbles** — §2's N\*=3 and §9's C=64 are one mechanism seen twice.

---

## 10. Deadlines and cancellation

**Deadline-aware admission (EDF + feasibility test).** Each request carries `d = arrival + SLO(class)`; schedule EDF within a DRR class. Reject at admission if `W_pred + G_max × D_max > d`, where `W_pred = queue_depth × S_req` (Little again) and **`G_max` is the client's `max_tokens`** (`coordinator.py:107`, default 32). Output length is unknown at admission; `max_tokens` is the only honest upper bound — which is precisely why it is a scheduling input, not a formality.

**Cancellation does not exist today, and it is expensive.** `chat_completions_stream_generator` (`coordinator.py:139`) loops `for token_idx in range(req.max_tokens - 1)` with no disconnect check. A client closing the tab at token 1 burns **31 × 785.3 ms = 24.3 s of 3-node compute (modelled from measured)**; 3.84 s even in v1.

| hop | mechanism | granularity |
|---|---|---|
| client → gateway | TCP FIN → Starlette `await request.is_disconnected()` between SSE chunks | immediate |
| gateway → coordinator | exiting `httpx` `client.stream()` closes the conn → uvicorn raises the disconnect | immediate |
| coordinator loop | token loop is an `asyncio.Task`; the SSE generator's `finally:` calls `task.cancel()` | immediate |
| coordinator → nodes | explicit `POST /cancel {request_id}`. **A running `torch` forward cannot be interrupted** (`node.py:82` is a sync `def`), so cancellation is cooperative at chunk boundaries | **one chunk = 111 ms at C=64** |
| KV reclaim | `DELETE /kv/{request_id}` fire-and-forget to all 3 nodes + a TTL sweeper (TTL = 2 × p99 duration) so a dead coordinator cannot leak KV | ≤ TTL |

C therefore sets **three** things: HOL granularity, microbatch granularity, cancellation granularity.

---

## 11. Backpressure, end to end

| layer | signal | effect | why not sufficient alone |
|---|---|---|---|
| TCP | receive window shrinks as the node's socket buffer fills | slows coordinator writes | default buffers are 100s of KB; with an unbounded app queue behind them TCP does not push back until MBs are buffered — **bufferbloat at L4** |
| credit frames | node advertises `X-Credits: k` per response (or HTTP/2 `WINDOW_UPDATE`) | no dispatch without a credit | the real mechanism; makes buffering explicit |
| coordinator queue | `asyncio.Queue(maxsize=6)` → `QueueFull` | the admission decision point | — |
| gateway | admission RPC returns Full | **HTTP 429 + `Retry-After`** | — |
| client | honours `Retry-After` with jitter | closes the control loop | only works if the header is *computed* (§4) |

---

## 12. Concrete tech

### v1 — hackathon, days, CPU, docker-compose, 3 nodes: **no broker at all**

| need | v1 choice | lines |
|---|---|---:|
| admission queue, per class | `asyncio.Queue(maxsize=6)` × 2 + a DRR picker loop | ~40 |
| in-flight cap | `asyncio.Semaphore(3)` (= N\*) | 1 |
| per-stage credits | `asyncio.Semaphore(1)` per node URL in the coordinator | 3 |
| **node-side concurrency cap** | `anyio.to_thread.current_default_thread_limiter().total_tokens = 1` at `node.py` startup | **1** |
| cancellation | `asyncio.Task` + `Request.is_disconnected()` | ~10 |
| metrics | `prometheus_client` — **already a gateway dependency**; Prometheus + Grafana are already in `docker-compose.yml` | ~15 |

**That node-side one-liner is the best value/effort item in this document.** `node.py:82` is `def forward` (sync), so Starlette runs it in the anyio threadpool, whose default `total_tokens` is **40 (verified: anyio 4.12.1, run locally)**. Forty concurrent `torch` forwards, each spawning intra-op threads, in a `cpus: "2"` container is processor sharing, not a queue: service time inflates roughly linearly, FIFO order is lost, and no backpressure signal exists anywhere. Setting it to 1 turns a 40-way thrash into an actual M/M/1 station with the S_i this document measured.

Do **not** add Redis/Kafka/Celery for v1: the coordinator is already the single scheduler, and a broker adds a network hop plus a serialisation to the exact path being optimised.

### v2 — production: **NATS JetStream 2.10+**

| | Redis Streams 7.x | **NATS JetStream 2.10+** | Apache Kafka 3.7+ (KRaft) |
|---|---|---|---|
| added p99 | ~0.3–1 ms | ~0.5–2 ms | **5–50 ms** (replication acks + fsync) |
| per-message ack | ✓ `XACK` | ✓ ack/nak + `AckWait` redelivery | ✗ **offsets only** |
| out-of-order completion | ✓ | ✓ | ✗ partition-ordered |
| work-queue semantics | groups + `XAUTOCLAIM` | ✓ **WorkQueue retention**, native | emulated with partitions |
| priority classes | N streams, client-side pick | subjects `llm.req.interactive.*` / `.batch.*` + per-consumer rate limits | N topics, no native priority |
| control plane (cancel) | side channel | ✓ request-reply, same connection | side channel |
| ops | 1 process, weak durability | 1 Go binary, Raft | JVM, partitions, rebalances |

**Pick NATS JetStream.** Per-message ack with `AckWait` redelivery is exactly the semantic an inference job needs (node dies mid-request → redelivered, no offset bookkeeping); WorkQueue retention gives one-consumer semantics without partition math; subject hierarchies give §8's classes free; request-reply on the same connection carries §10's cancellation.

**Why a hard real-time inference path does NOT want Kafka:**
1. Kafka is a **partition-ordered log, not a work queue**. One stuck message blocks its partition's
consumer and you cannot ack out of order — *architectural* head-of-line blocking, the exact failure mode §9 exists to eliminate.
2. No per-message ack/nak/priority. Consumer rebalances (even cooperative-sticky, KIP-429) stall
consumption for seconds, against a 41.31 ms token budget.
3. `acks=all` + `min.insync.replicas=2` adds **ms to tens of ms of p99** durability to a path whose whole
per-token budget is 41 ms — 12–100% of the budget spent on durability nobody asked for.
4. **Durability of a deadline-bound message is a contradiction.** A replayed 5-minute-old chat
completion is garbage; the client is gone (§10).
5. Where Kafka *is* right here: the **telemetry / audit / billing plane** — per-token metrics, request
logs, cost events. Durable, replayable, no deadline. Kafka there, JetStream on the inference path.

---

## 13. Metrics to expose (names mirror vLLM's, so Grafana dashboards port)

| metric | type | labels | why |
|---|---|---|---|
| `dllm_num_requests_waiting` | gauge | `class` | Little's **L**; alert > 0.8 K = 5 |
| `dllm_num_requests_running` | gauge | — | vs N\* = 3; > 3 means the semaphore leaked |
| `dllm_request_queue_time_seconds` | histogram | `class` | **W**, p50/p95/p99. L/W cross-checks achieved λ free |
| `dllm_admission_total` | counter | `class`, `outcome=admitted\|rejected_full\|rejected_deadline` | the 429 rate — the SLO's real denominator |
| `dllm_stage_service_seconds` | histogram | `node` | live S_i → μ_i → **recompute N\* at runtime** instead of hard-coding 3 |
| `dllm_stage_utilisation` | gauge | `node` | **ρ_i**; page above 0.75 (§3); names the current bottleneck |
| `dllm_credits_available` | gauge | `node` | backpressure made visible; pinned at 0 = the bottleneck |
| `dllm_time_to_first_token_seconds` | histogram | `class` | SLI (vLLM `vllm:time_to_first_token_seconds`) |
| `dllm_inter_token_latency_seconds` | histogram | `class` | SLI (vLLM `vllm:inter_token_latency_seconds`) |
| `dllm_e2e_request_latency_seconds` | histogram | `class` | vLLM `vllm:e2e_request_latency_seconds` |
| `dllm_cancelled_total`, `dllm_wasted_compute_seconds` | counter | `stage` | turns §10's 24.3 s/abandon into a measured saving |
| `dllm_prefill_chunks_total` | counter | — | confirms chunked prefill actually engages |

Extend, don't replace: `coordinator.py:180` and `node.py:118` already emit hand-rolled counters and `gateway/app.py:20` already imports `prometheus_client`.

---

## 14. Ranked recommendations

| # | change | impact | effort | tag |
|---:|---|---|---|---|
| 1 | anyio thread limiter → 1 in `node.py`; `asyncio.Semaphore(3)` in-flight cap in the coordinator | 40-way CPU thrash → a real queue; **1.27 → 3.24 tok/s, 2.54× (modelled from measured S_i)** | hours | **v1** |
| 2 | Rebalance shards (FINDING 1) so N\* = 3.00 exactly | 2.54× → **3.31× vs v0**; removes the residual bubble | hours | **v1** |
| 3 | Bounded `asyncio.Queue(maxsize=6)` + 429 + computed `Retry-After` | ρ pinned ≤ 0.75; p99 wait bounded at 18.4 × S instead of unbounded | hours | **v1** |
| 4 | Chunked prefill, C = 64 | HOL blocking **3551 → 111 ms, 32×**; also supplies the microbatch units | days | **v1** |
| 5 | Cancellation propagation + KV `DELETE` + TTL sweeper | recovers **24.3 s of 3-node compute per abandoned v0 request** | days | **v1** |
| 6 | DRR, 2 classes, weights 4:1, quantum in tokens | neither class can starve the other | days | **v1** |
| 7 | The 12 metrics in §13 | every number above becomes continuously verifiable; N\* self-tunes | hours | **v1** |
| 8 | Credit frames over HTTP/2 `WINDOW_UPDATE` / gRPC, replacing `X-Credits` | credits free from the transport; pairs with T1-A4 | weeks | v2 |
| 9 | NATS JetStream for the inference queue; Kafka for telemetry only | survives coordinator restart; enables replicated schedulers | weeks | v2 |
| 10 | EDF + admission feasibility test on `max_tokens` | turns the SLO from aspiration into contract | weeks | v2 |

**Sources:** [vLLM metrics](https://docs.vllm.ai/en/latest/usage/metrics.html) ·
[vLLM optimization / chunked prefill](https://docs.vllm.ai/en/stable/configuration/optimization/)

