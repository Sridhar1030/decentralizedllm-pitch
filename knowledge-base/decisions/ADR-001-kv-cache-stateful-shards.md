---
id: ADR-001
title: Per-node KV cache and session-pinned stateful shards
status: v1 accepted
date: 2026-09-01
sources: teams/T3-A1, T1-A1, T3-A2, T3-A3, T3-A4, T1-A5, 01-VERIFIED-FACTS F3
---

# ADR-001 — KV cache and stateful, session-pinned shards

## Context

`coordinator.py:123` runs `for _ in range(max_tokens-1): logits = await forward_chain(gen_ids)` — the whole
growing token list, every step. `node.py:90/99` call `model(...)` with no `past_key_values`, no `use_cache`.
Every node recomputes every prefix position on every token, on three machines simultaneously.

| P=32, G=512 | v0 | v1 | factor | tag |
|---|---:|---:|---:|---|
| position-forwards per node | 147,200 | 543 | **271x** | derived |
| pipeline FLOPs | 145.4 TFLOP | 536.4 GFLOP | 271x | derived |
| whole-generation wall clock, 3x 2-vCPU | 2,908 s | 101.6 s | **28.6x** | modelled |
| hidden-state wire bytes | 1,406.8 MB | 5.19 MB | 271x | derived |

Per-token compute at fixed seq is measured directly: 712.54 → 123.94 ms at seq=512, and 3551.25 → 120.21 ms
at seq=2048 (measured, 3 nodes summed) — KV-cached decode is **flat** in seq, so the win grows without bound.

Cost of the cache is negligible because Qwen2.5 uses GQA (14 attn heads / 2 KV heads = **7x discount**):
`2 kv_heads x 64 head_dim x 2 (K,V) = 1,024 B per layer per token fp32` → 8,192 B/token per 8-layer shard,
16.78 MB for a full 2048-token session = **3.5% of the 477.1 MB of weights that shard already holds**.
Plain MHA would have cost 352.3 MB. There is no memory argument against caching.

## Options considered

| option | verdict | why |
|---|---|---|
| Keep v0 stateless | **rejected** | 271x redundant compute is the single largest defect in the system. |
| Cache at the coordinator | **rejected — impossible** | K/V are per-layer tensors produced inside each shard's attention. They can only live where the layers live. |
| `DynamicCache` per (session_id, node), `OrderedDict` LRU + 300 s TTL | **ACCEPTED v1** | ~35 added lines, no new dependency, verified against transformers 5.5.0. |
| PagedAttention block pool (16-token blocks, 128 KiB/block per shard) | **v2 proposed** | Buys ~10.4x concurrent sessions per GB (modelled) via <=1-block waste vs 90% reservation waste, plus prefix sharing. Does **not** buy realloc cost: `torch.cat` regrowth is 0.1% of generation time here, 6.8% even at 32k context. Build it for concurrency, not for memcpy. |
| Adopt vLLM instead of reimplementing | **v2 proposed** | See ADR-008. If we ever need paging, adopt vLLM rather than rewriting PagedAttention. |

## Decision

1. Per-node, per-session `DynamicCache` keyed by `session_id`, with a **`position` fence**: `have == position`
   → append; `have > position` → `Cache.crop(position)`; `have < position` → HTTP 409 `{cache_miss, have}` and
   the coordinator replays. LRU eviction on `MAX_SESSIONS` / `MAX_CACHE_TOKENS`, 300 s idle TTL.
2. After prefill the coordinator sends **only the new token**: payload `[seq,896]` → `[1,896]`.
3. **Hard prerequisite, ship in the same commit:** renumber sliced layers to local dense indices in
   `load_model()` — `for i, l in enumerate(model.model.layers): l.self_attn.layer_idx = i`.
4. Ship `logits_to_keep=1` and delete `output_hidden_states=True` *first*, as a standalone commit — worth
   **1.38x on the whole pipeline with no cache at all** (39.9 TFLOP of 145.4 = 27.5% is `lm_head` running on
   resent positions). On non-final nodes `lm_head` is `Identity`, so `out.logits` already IS the hidden state.
5. Recovery on 409 or node loss: reset-all-and-replay, surfaced as SSE `event: replaying`.
6. Export `node_kv_bytes`, `node_kv_sessions`, `node_kv_evictions_total` — ADR-005 needs them as backpressure.

## Consequences

**Good.** 271x less redundant compute; 77% of v0's wire bytes gone (the 414.9 MB residual is the logit vector,
fixed in ADR-002); makes continuous batching possible at all (ADR-006 — without a cache every request carries a
different seq_len and there is nothing to batch).

**Bad, and each one is real:**
- **Sessions become stateful and pinned to one node triple for life.** Round-robin load balancing breaks;
  `docker-compose scale` no longer balances; scale-out only helps *new* sessions; drains must wait or migrate;
  a node crash now kills every session pinned to it, not one in-flight token.
- **The protocol becomes order-sensitive.** One retried or duplicated `/forward` double-appends a position and
  the model emits plausible wrong text with no exception. The `position` fence is a trust boundary, not polish.
- **A stale-cache bug is silent.** The sliced-`ModuleList` `layer_idx` trap breaks RoPE positions *and* the
  causal mask at once (`get_seq_length()` reads slot 0 → 0 → position_ids restart; `get_mask_sizes` →
  kv_length=1 → the mask hides the whole past). Mitigation: one `torch.allclose` check comparing stateless vs
  token-at-a-time output, in CI. **Written and run** — `bench/parity_check.py` (real weights, measured):
  stateless 3-shard chain matches the monolithic forward at **max abs diff 0.000e+00**; cached decode with the
  renumber is token-identical; **without** it the same prompt yields `" Paris. Paris is the capital of France"`
  instead of `" Paris. It is the largest city in"` — fluent, plausible, wrong, and silent. That is why the gate
  is a test and not a code review.
- **It creates the failover problem.** v0's O(n²) resend was accidentally a free checkpoint — there was no
  distributed state to lose. See ADR-009.
- **Memory becomes a scheduling dimension.** Headroom: 53 sessions @2048 ctx on node0/node2, 83 on node1 — but
  only after `load_model()`'s ~4 GB peak (two full model copies resident) is fixed. See ADR-011.
- **Decode flips from compute-bound GEMM to DRAM-bandwidth-bound GEMV.** 271x is a FLOP ratio; wall clock is
  28.6x. Quote them separately or under-deliver on stage.

## Status

**v1 accepted.** Paging, bf16 KV, cache migration and prefix caching are **v2 proposed**.
