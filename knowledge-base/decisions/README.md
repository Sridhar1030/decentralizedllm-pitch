# Architecture Decision Records — DecentralizedLLM

Thirteen ADRs synthesised from 25 research agents across 5 teams (`../teams/`), anchored on
`../00-SHARED-CONTEXT.md` and `../01-VERIFIED-FACTS.md`. Every number carries a `(measured)` / `(derived)` /
`(modelled)` tag. Where teams disagreed, the disagreement is resolved in writing — never averaged —
in the ADR that owns the decision, and the numeric contradictions are collected in **ADR-013**.

| # | Decision | One-liner | Status |
|---|---|---|---|
| [001](ADR-001-kv-cache-stateful-shards.md) | KV cache & session-pinned shards | Per-node `DynamicCache` keyed by `session_id` with a `position` fence — 271x fewer position-forwards, and we accept that sessions become stateful and pinned to one node triple. | v1 accepted |
| [002](ADR-002-dlp-binary-wire-protocol.md) | DLP binary wire protocol | A 40-byte header on a persistent `TCP_NODELAY` socket replaces HTTP+JSON+base64; node2 returns a 4-byte token id instead of 607,744 B of logits; routing goes star → chain. | v1 accepted |
| [003](ADR-003-activation-compression.md) | Activation compression | The winning compressor is a **dtype cast**: bf16 by default, int8+outliers gated to WAN links, and **no byte codec, no low-rank, no sparsity, no error feedback** — each rejected by measurement. | v1 accepted |
| [004](ADR-004-transport-tiering.md) | Transport tiering | Tuned TCP everywhere, UDS for co-located nodes; **RDMA deferred with its number** — 27 µs/hop saved versus 10,376 µs saved by deleting the Python stack. | v1 accepted |
| [005](ADR-005-queueing-admission-backpressure.md) | Queueing & admission | The whole scheduler is two integers derived, not guessed: `Semaphore(N*=3)` + `Queue(K=6)` → HTTP 429 with `Retry-After: 8`. | v1 accepted |
| [006](ADR-006-concurrency-and-batching.md) | Concurrency model | `U = min(1, R/(P·S))` — the 33% ceiling is a **load** problem, so drive the demo at R≥3 and add chunked prefill; continuous batching is v2. | v1 accepted |
| [007](ADR-007-layer-placement-dp.md) | Layer placement | Placement is a bottleneck shortest path, not a heuristic: `lm_head` = 9.13 layer-equivalents, so re-split to 11/11/2 for 1.539x — three env-var edits. | v1 accepted |
| [008](ADR-008-buy-vs-build.md) | Buy vs build | Buy the per-node engine (vLLM/SGLang) in v2, **build the inter-node fabric**; run vLLM PP=3 and llama.cpp `--rpc` as baselines, and never claim the layer split itself is novel. | v1 accepted |
| [009](ADR-009-failure-model-and-degradation.md) | Failure & degradation | Self-registering control plane, 1.5 s detection, re-shard onto survivors and resume the same completion at HTTP 200 `degraded:true` — ~63 net LOC. | v1 accepted |
| [010](ADR-010-trust-and-privacy-model.md) | Trust & privacy | The claim is **governance, not cryptography**: lead with data sovereignty, state the 2-of-3 collusion limit ourselves, and never claim prompt privacy — hidden states are exactly invertible. | v1 accepted |
| [011](ADR-011-weight-distribution-and-loader.md) | Weight distribution & loader | Ship pre-sliced `shard{n}.safetensors` and fix the double-allocating loader — until then "no node holds the full model" is **disproved by 40 lines of our own source**. | v1 accepted |
| [012](ADR-012-observability-and-verification.md) | Observability & verification | Make ADR-007's imbalance a rectangle judges measure with their eyes: 1 s scrapes, real histograms, OTel spans behind a 32-byte `F_TRACE` flag, and an **open-loop** probe to prove admission control exists. | v1 accepted |
| [013](ADR-013-published-claims-ledger.md) | Published claims ledger | **19.0x is the house headline** (6.8x single-stream × 2.8x concurrency, modelled from measured); 28.6x and 32.9x measure different things and must never be multiplied into it. | v1 accepted |

## Reading order

**Build order is not ADR order.** The dependency chain that matters:

```
ADR-011 (loader + pre-sliced weights)   ← do first; everything else's memory budget is fiction until it lands
   └─ ADR-007 (re-split 11/11/2)        ← finalise BEFORE fitting any codec basis or outlier index (ADR-003)
        └─ ADR-001 (KV cache + session_id) ─┬─ ADR-006 (R≥3 concurrency)  ← same commit, or silent corruption
                                            └─ ADR-009 (failover)  ← the cache CREATES the failover problem
   ADR-002 (DLP + argmax on node2 + pooling) ← argmax fix is hours and is also ADR-010's security fix
        ├─ ADR-003 (bf16 dtype)   ├─ ADR-004 (transport tier)   └─ ADR-010 (mTLS — AFTER pooling, never before)
   ADR-005 (admission) ships with ADR-006, never after it
   ADR-012 instruments all of the above;  ADR-008 and ADR-013 govern what we say about it
```

## The five decisions that carry the pitch

1. **ADR-011 first, or the product name is false.** `node.py:36` loads the whole checkpoint on every node.
   A judge reading 40 lines of source invalidates the deck. Highest-risk item in the project.
2. **ADR-001's KV cache is the largest single win** (271x redundant compute, 28.6x wall clock modelled) —
   and its price, session affinity, is the one genuine architectural regression we accept knowingly.
3. **ADR-002's argmax move is 2 lines** and is simultaneously the biggest bandwidth win (151,936x), a latency
   win, and the deletion of the strongest known prompt-inversion oracle.
4. **ADR-008: never claim the layer split is novel.** vLLM, llama.cpp, mlx-lm, Petals and exo all ship it.
   The product is the trust / heterogeneity / churn / fault model, and llama.cpp's own benchmark
   (decode 0.574x, prefill 4.17x) is why the pitch must be capacity and TTFT, never decode speed.
5. **ADR-003 and ADR-004 are both decisions to do less work.** No byte codec, no low-rank, no RDMA — each
   rejected with a measurement rather than an opinion. That is what makes the four levers we *did* pull
   believable.

## Standing caveats on every number here

- Absolute milliseconds are Apple M1 Pro / macOS bare metal with `torch.set_num_threads(2)`; the demo runs in
  Docker containers with cgroup CPU limits on a bridge network. **Quote ratios, not milliseconds.**
- Every transport figure is **loopback**. No NIC, no switch, no contention, no loss has been measured.
- Docker was not running on the bench machine, so every container-networking figure is `(modelled)`.
- No v1 component has been run as an **integrated system**. ADR-013's headline is composed arithmetic.
- All quality results are one model (Qwen2.5-0.5B-Instruct), greedy decoding, small evaluation sets.
