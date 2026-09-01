---
id: ADR-006
title: Concurrency model — R concurrent requests to fill an S-stage pipeline
status: v1 accepted (R=S concurrency + chunked prefill); continuous batching v2 proposed
date: 2026-09-01
sources: teams/T3-A2, T3-A3, T3-A5, T3-A4, T3-A1, T4-A1
---

# ADR-006 — Filling the pipeline: R concurrent requests, then batching

## Notation (normalised — three teams used three symbols)

`S` = pipeline stages (3). `R` = in-flight requests per pipeline. `P` = pipeline replicas. `B` = intra-stage
batch size. **`U = min(1, R / (P·S))`.** T3-A2's `R ≥ S` and T3-A5's `U = min(C/(P·S), 1)` are the same law;
this ADR uses `R`, `S`, `P`, `B` throughout and other files' `C`/`N` map onto `R`.

## Context

v0 defect #6: 2 of 3 nodes idle at any instant, utilisation ceiling 1/3. The important discovery is that this
is a **load problem, not a code problem**: `coordinator.py` already `await`s throughout and `node.py`'s
`/forward` is a sync `def` that FastAPI runs in the anyio threadpool. **Pipeline overlap works today with
zero code changes.** `demo.sh` sends one curl — that is the entire reason utilisation is 33%.

The reason a *single* request cannot be microbatched is structural and worth saying on stage: inference
pipeline parallelism differs from training PP in that there is **no backward pass**, and autoregressive decode
makes token *t+1* depend on token *t* completing all S hops. `Σt` cancels out of `U`, so **load balancing does
not change R=1 utilisation at all**. The only ways to fill the pipe are concurrent requests, chunked prefill,
and speculation.

| configuration | U | tok/s |
|---|---:|---:|
| R=1 (today) | 33.3% | 1.40 compute-only / **1.273 incl. transport (measured)** |
| R=2 | 66.7% | 2.81 (modelled) |
| R=3, v0 8/8/8 | **76.9%** = Σt/(S·t_max) = 712.54/(3×308.97) | 3.24 (modelled) |
| R=3, rebalanced (ADR-007) | **100%** | 4.21 = **3.00x** (modelled) |
| R=8 | 100% | 4.21 unchanged, at +167% latency (modelled) |

`TPOT(R) = max(Σt, R·t_max)`: flat at 712.5 ms for R=1..3, then linear (R=16 → 3,800 ms) for zero extra
throughput — **R = S = 3 is the exact knee**. Concurrency alone can never beat 76.9% on today's split; the
residual 23.1% is node0/node1 waiting on node2, which no scheduler can fill (ADR-007's job).

The second lever, intra-stage batching, is nearly free here: a decode step on one CPU shard costs **23.63 ms
at B=1 and 37.33 ms at B=32** (measured, ctx=128, 2 threads), flat from B=2 to B=32, because a linear fit of
measured prefill gives `t_shard(N) = 22.77 + 0.2931·N` ms — **98.7% of a B=1 step is fixed framework
overhead**. Per-sequence cost 23.634 → 1.167 ms = **20.2x more sequences/sec for 1.58x more time**.

## Options considered

| option | verdict | why |
|---|---|---|
| Microbatch a single request across stages | **impossible** | Token t+1 depends on token t finishing all S hops. Not a limitation of our implementation — a property of autoregressive decode. |
| **Drive the demo at R=3 concurrent requests** | **ACCEPTED v1, zero code** | 33.3% → 76.9%, 1.40 → 3.24 tok/s, **at unchanged per-request latency**. A request queued behind nothing still takes exactly one traversal. |
| **Chunked prefill, M=16 (C=128) / C=64 for SLO** | **ACCEPTED v1** | The one schedule that pipelines *within* one request: chunk k+1's attention at layer L needs chunk k's K/V at layer L, and chunk k has already left that node, so the dependency runs backwards along the pipeline and can never be violated. Bubble `(S−1)/(M+S−1)`: M=16 → 11.1%; P=2048 prefill 3,551 → 1,785 ms (1.99x), or 1,332 ms rebalanced (2.67x). Same as Sarathi-Serve (OSDI 2024) / vLLM V1, where it is always on and cannot be disabled. |
| **Continuous / iteration-level batching (Orca, OSDI 2022)** with a coordinator-minted immutable `BatchDescriptor` | **v2 proposed** | 8.07 → 265.4 tok/s at B=16 (**32.9x** aggregate) for TPOT 123.9 → 180.8 ms. Decomposes as 2.76x pipelining × 11.9x batching. |
| Flattened varlen `[n_tok, H]` instead of padded `[B, S_max]` | **v2, with batching** | **3.76x** on a ragged prefill batch (1,033.60 ms padded `[8,512]` vs 274.70 ms flat `[1,900]`, measured) — 4.55x slot inflation from padding. |
| Speculative decoding on this model | **rejected — negative result** | `lm_head` is 9.13 of 33.13 layer-eq, so *every* draft pays it: layers 0-7 early-exit = 51.7% of target cost ⇒ **0.75x, slower**. Only n-gram/prompt-lookup has c=0 (2.31x at α=0.6, k=4). `tie_word_embeddings` putting the lm_head matrix on node0 is a nice line and a dead end at 0.5B; revisit at 7B+. |
| Interleaved 1F1B, virtual pipeline stages, ZB-H1/ZB-H2 | **rejected — an active trap** | 1F1B doubles hops 3→6 with **zero** bubble reduction at R=1; zero-bubble schedules work by splitting the *backward* pass, which inference does not have. Any 1F1B implementation degenerates to plain round-robin here. |
| More pipeline replicas to fix utilisation | **rejected as a fix** | `U = min(R/(P·S), 1)`: at fixed load **more replicas make utilisation worse** (P=3, R=3 → 33%). Replication buys capacity and fault tolerance, never utilisation. |

## Decision

1. **Drive every demo and benchmark at R ≥ 3.** Change `demo.sh`, not the coordinator.
2. **Ship the semaphores from ADR-005 in the same commit as the concurrency change**, not after. Three
   concurrent forwards × torch's 2 threads inside a `cpus: "2"` container thrashes; recommendation 1 is only
   safe because of the anyio limiter and the semaphores.
3. **Chunked prefill** at C=64–128, gated on ADR-002's connection pooling: at M=32 the 5.6 ms/hop client
   construction tax is **537 ms of pure overhead and erases the entire win**.
4. Continuous batching is v2; its `BatchDescriptor` must be **minted once by the coordinator and executed
   verbatim by all three stages** — row order is the contract.
5. **Land ADR-001's KV cache and `session_id` in the same commit as any concurrency work.** `node.py` is
   stateless today, which is why R=3 works out of the box; once a cache lands, concurrent request B reads
   request A's cache unless every `/forward` carries a `session_id`.

## Consequences

**Good.** 3.00x aggregate throughput at unchanged per-request latency, from a `demo.sh` edit and one env var.
Chunked prefill cuts TTFT 518 → 393 ms **and** the TPOT spike other users see, 3.86x → 1.46x.

**Bad.**
- **KV cache × concurrency is a CORRECTNESS hazard, not a performance one** (decision 5) — the single
  highest-severity interaction in the roadmap. And **32.9x is a throughput number, not a latency number**:
  single-user TTFT is unchanged or slightly worse, and a sequence sharing a step with a 128-token prefill
  chunk pays 1.46x for it. Report p99 TPOT, never the mean, and say so before the audience does.
- Every figure past the measured stage times is **modelled and assumes stage time is independent of
  concurrency**. It is not — contention will shave R=3 below 4.21 tok/s. Say "≈3x (modelled)", then measure:
  three background curls in `demo.sh` is a 10-minute experiment that converts it to (measured). The
  chunked-prefill table also assumes prefill is linear in tokens; attention is O(P²), so M=16 sits inside
  where that model is defensible and M=32+ does not.
- **Do not multiply 32.9x into the 19x headline** — separate measurement runs (ctx=128 vs seq=512); mixing
  them double-counts. See ADR-013. Descriptor/tensor split-brain is v2's top bug class: if they travel as
  separate messages a stage can pair descriptor S with tensor S−1 and corrupt silently — one framed message,
  or `step_id` echoed in both and asserted (a coordinator restart replays `step_id`, hence the `epoch` field).

## Status

**v1 accepted** (R≥3, semaphores, chunked prefill). **v2 proposed:** continuous batching with an immutable
`BatchDescriptor`, flattened varlen, PagedAttention block tables, n-gram speculative decoding, prefill/decode
disaggregation (DistServe/Splitwise).
