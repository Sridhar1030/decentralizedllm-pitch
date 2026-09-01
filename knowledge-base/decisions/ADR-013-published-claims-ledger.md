---
id: ADR-013
title: Published performance claims — which numbers we quote, with which tags
status: v1 accepted
date: 2026-09-01
sources: teams/T5-A4, T5-A3, T5-A5, T1-A1, T3-A1, T3-A3, T4-A1, 01-VERIFIED-FACTS
---

# ADR-013 — One ledger of quotable numbers, and the contradictions it resolves

## Context

Twenty-five agents produced overlapping speedup figures with different denominators; left unreconciled, the
deck contradicts itself on stage. This ADR fixes one house number per claim and records why the alternatives
were not averaged away.

## Options considered — the competing numbers, resolved explicitly rather than averaged

**C1 — How many wire crossings does v0 make per token?** FINDING 4 assumes 2 activation transfers per logical
step; T1-A1 read `coordinator.py:76-97` and found v0 is a **star** — 3 POSTs = **6 wire crossings per token,
four activation-sized**, plus one 810 KB logits blob. Both are right about different things. **Resolution:**
the star is a structural fact and goes in the diagram, so FINDING 4's total is a *lower bound* on v0's traffic;
**keep 935x as published**, because understating our own baseline is the honest direction to err and ADR-002's
chain routing deletes the extra crossings anyway. Any slide drawing v0 as a chain must be corrected.

**C2 — Is transport 9–18% of the clock, or 114 ms of it? And is a hop 38.7x, 95x or 346x cheaper?** T1-A1
decomposes seq=512 as **transport CPU 72.8 ms of 785.3 ms = 9.3%** (18.1% counting 1 GbE link separately),
cross-checked against a measured symmetric echo to within 0.3 ms; T1-A2 measures 38.06 ms for one full HTTP
hop and ×3 = 114.2 ms. **Resolution:** T1-A1 is authoritative for *share of wall clock* — the per-hop harness
includes server-side parse, does not overlap with compute, and ×3 double-counts the star — while T1-A2/A3/A4's
per-hop figures are authoritative for *protocol choice*, which is what ADR-002 and ADR-004 needed. The per-hop
spread is likewise explained, not averaged: 346x (real httpx+FastAPI, seq=1), 95.4x (stdlib `http.server`
baseline — *faster* than v0's real stack, hence a conservative lower bound), 38.7x (seq=512, where the fixed
per-call tax amortises). **Quote: "89% of v0's per-token wall clock is compute, not network," and quote the
per-hop range with its reason, never one number bare.**

**C3 — bf16 or int8 on the wire?** T2-A4 says bf16 and stop; T2-A1 recommends int8. They measured **different
codecs** — plain int8-per-token (7.32% flips) vs int8-per-token **plus 8 fp16 outlier channels** (20/20 exact
greedy) — and T2-A4's "stop at bf16" is a *deployment* claim about LAN economics, not a quality claim. No
contradiction once the codec is named (ADR-003).

**C4 — 6.7x, 19x, 28.6x or 32.9x?** Four different denominators:

| figure | what it actually measures | may be composed with |
|---|---|---|
| **6.8x** | single-stream per-token at seq=512, all v1 fixes (785.3 → 116.0 ms) | ×2.8 concurrency |
| **19.0x** | the above × R=3 concurrency on a balanced split → 24.21 tok/s | — |
| ~~**28.6x**~~ | whole 512-token generation from the KV cache alone (2,908 → 101.6 s) — **retired by 90-AUDIT F02**; on the measured basis ~227 → ~64 s = **3.6x** | nothing — do not quote |
| **32.9x** | aggregate throughput of continuous batching at B=16, off a *post-KV* 8.07 tok/s baseline, at ctx=128 | **nothing — separate measurement run** |

**Resolution: 19.0x is the house headline;** ~~28.6x is quoted separately as the whole-generation KV figure~~
(**28.6x retired — 90-AUDIT F02**; the measured-basis whole-generation figure is ~3.6x); and
**32.9x must never be multiplied into 19x** — T3-A3's run is ctx=128, T1-A1's stage times are seq=512, and
mixing them double-counts.

## The ledger

| claim | value | tag | source |
|---|---|---|---|
| redundant position-forwards removed | 147,200 → 543 = **271x** | derived | F3 |
| return path | 607,744 B → 4 B = **151,936x** | derived | F2 |
| shard imbalance | **1.55x** layer-equivalents (derived, F1); **1.30x** wall clock (measured, T3-A2) | both | F1, T3-A2 |
| wire bytes, one 512-token generation | 1,821.7 → 1.948 MB = **935x** | derived | F4 |
| v0 baseline | **1.273 tok/s** / 785.3 ms per token at seq=512 | measured | T1-A1 |
| v1 single stream / at R=3 balanced | 116.0 ms = 8.62 tok/s = **6.8x** / 24.21 tok/s = **19.0x** | modelled from measured | T5-A4 |
| KV cache, whole 512-token generation | ~~2,908 → 101.6 s = 28.6x~~ **RETIRED, 90-AUDIT F02** — modelled at 50 GFLOP/s and 198 ms/decode-step, both contradicted by T1-A1's measured stage times (which imply ~227 s v0 and 123.94 ms/decode ⇒ **~3.6x**). TTFT **1.81x** stands. | modelled, retired | T3-A1, T5-A4 |
| bf16 wire | 3,584 → 1,792 B, top-1 0.9974, KL 5.7e-5 | measured | T2-A4 |
| per-hop protocol | 8.483 → 0.089 ms (95x) … 10,406 → 30.1 µs (346x), loopback | measured | T1-A4 / T1-A3 |
| distribution's honest cost | llama.cpp RPC decode **0.574x**, prefill **4.17x** | measured, 3rd-party | T4-A2 |

## Decision

1. **19.0x is the headline**, and it ships with this sentence spoken, not just printed: *"Modelled by
   composing per-stage times measured on one laptop; 6.8x is single-stream latency, 2.8x is pipelining that
   needs three concurrent requests. v1 has never been run as an integrated system, so this is a design
   target, not a result."*
2. **The 935x caveat is attached at every occurrence** — wire bytes, not wall clock; on a fast LAN the
   pipeline is compute-bound — and it goes on the slide **in the same size type**, so the presenter says it.
3. **Ratios, not milliseconds** (absolute times are M1 Pro bare metal; the demo is Docker with cgroup CPU
   limits, so absolutes could move 20–50%), and **never mix measurement runs**: T3-A3 (ctx=128), T1-A1
   (seq=512) and T2-A2 (72.9 ms cached-decode anchor) are separate — label the run beside any composed
   figure, and name the corpus beside any compression ratio (ADR-003's 2.5x tiled-prose trap).
4. **Lead slide 3 with the self-audit** — no KV cache, argmax on the coordinator, 8/8/17 not 8/8/8. Delivered
   as confession before any claim, it buys belief in everything after it.

## Consequences

**Good.** One number per claim; a written answer to "did you run it?"; the deck cannot contradict itself.

**Bad.**
- **The headline has zero measured v1 components.** It composes independently measured pieces and assumes the
  fixes do not interact; shared allocators, caches and GIL contention make such sums optimistic more often
  than not. And **the 2.8x half requires three concurrent requests** — `demo.sh` sends one curl, so at
  concurrency 1 the screen shows 8.62 tok/s and visibly disagrees with the slide.
- Cold start is 60–90 s per node: a demo that boots on stage dies on stage. We also claim a vLLM baseline we
  have **not run** (ADR-008); until it is, that row of the ledger is empty.
- Model weights in some benches were random-init from `config.json` (the HF blob cache held an incomplete
  safetensors). Shapes and FLOPs are identical so latencies are valid, but **generated tokens would be
  garbage** — any correctness demo needs the real weights.

## Status

**v1 accepted.** **v2 proposed:** replace every modelled row with a measured one from an integrated v1 run on
the target hardware, over a real LAN, against the vLLM and llama.cpp baselines.
