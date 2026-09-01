---
team: T3 — Scheduling, Queueing & Compute Utilisation
agent: T3-A2
topic: Pipeline-parallel scheduling theory applied to the 3-stage node0→node1→node2 chain; bubbles, microbatching, chunked prefill, speculative decode
headline: >
  Utilisation = min(1, R/S). With S=3 stages and R=1 in-flight request the ceiling is 1/3 and no
  scheduler on earth beats it — the bubble is a LOAD problem, not a code problem. Autoregressive decode
  forbids intra-request pipelining (token t+1 needs token t), so the only fills are concurrent requests,
  chunked prefill, and speculation. v0 already supports R>1 concurrency in code; the demo just never
  sends it. R=3 on today's imbalanced split: 33.3% → 76.9% util, 1.40 → 3.24 tok/s (modelled from
  measured stage times). Rebalance first and it is 100% / 4.21 tok/s — 3.00x aggregate throughput at
  UNCHANGED per-request latency, for ~40 lines of coordinator code.
---

# T3-A2 — Bubbles, microbatching, and how to actually fill a 3-stage inference pipeline

Stage times are **(measured)** from `T1-A1` §5 at seq=512, `torch.set_num_threads(2)`, exactly as
`layer-nodes/node.py` builds each shard. Everything derived from them is **(modelled)**.

| symbol | meaning | value |
|---|---|---|
| S | pipeline stages | **3** (node0, node1, node2) |
| R | concurrent in-flight requests | **1** in v0 (`demo.sh` sends one curl) |
| t₀,t₁,t₂ | per-stage compute, seq=512 | **205.81 / 197.76 / 308.97 ms** (measured) |
| Σt | one full traversal | **712.54 ms** (measured) |
| t_max | bottleneck stage | **308.97 ms** = node2 (measured) |
| Σt/S | balanced stage | **237.51 ms** |

## 1. The bubble, derived

Strict sequential chain, one request in flight. Stage *s* is busy t_s and idle Σt − t_s.

```
U_stage(s) = t_s / Σt          U_aggregate = (1/S)·Σ_s (t_s/Σt) = Σt/(S·Σt) = 1/S
Bubble fraction  B = 1 − 1/S = (S−1)/S
```

**S=3 ⇒ U = 33.3%, B = 66.7%.** Note what cancels: **Σt drops out.** Load balancing does *not* change
R=1 utilisation — a perfectly balanced 3-stage chain is still 33.3% idle. Imbalance costs throughput
only once you pipeline (§5). Two independent defects, two independent fixes.

### GANTT A — v0 as written (R=1). 1 char ≈ 23.75 ms. `A0` = request A, token 0.

```
       0        200       400       600       800      1000      1200      1400 ms
       |---------|---------|---------|---------|---------|---------|---------|
node0  [###A0##].....................[###A1##].....................[A2]
node1  .........[##A0##]......................[##A1##].................
node2  .................[#####A0####].................[#####A1####]....
       ^ node2 is 13 chars wide vs node0's 9 — that is FINDING 1's imbalance, visible.
```

66.7% of that chart is dots. At 3 nodes × 2 vCPU that is **4 vCPU-seconds burnt per second of decode.**

## 2. The training schedules, and what survives contact with inference

Training bubble with M microbatches (Huang et al., *GPipe*, NeurIPS 2019): `B = (S−1)/(M+S−1)`.

| schedule | origin | bubble | carries over to **inference**? |
|---|---|---|---|
| Naive / sequential | v0 today | (S−1)/S = **66.7%** | this is what we have |
| **GPipe**, all-F-then-all-B | Huang, NeurIPS'19 | (S−1)/(M+S−1), O(M) act. mem | **Forward half only** — the F-schedule *is* chunked prefill (§6); the B-schedule is dead code |
| **1F1B** | Narayanan, *PipeDream*, SOSP'19; Megatron-LM | same bubble as GPipe, O(S) act. mem | **Degenerates to "1F".** Its entire contribution is bounding activation memory by interleaving backwards. No backward ⇒ 1F1B ≡ GPipe ≡ round-robin. Zero benefit — do not implement it. |
| **Interleaved 1F1B / virtual pipeline stages** | Narayanan et al., SC'21, `--num-layers-per-virtual-pipeline-stage` | (S−1)/(v·(M+S−1)) | **Actively harmful — a trap.** v=2 ⇒ each node holds 2 non-contiguous chunks, request visits node0→1→2→0→1→2. At R=1 there is still ONE item in flight, so U is still 1/S, but hops/token go 3 → 6. Pure loss. |
| **Zero Bubble** ZB-H1/ZB-H2 | Qi et al., ICLR 2024 | ≈0% | **Nothing.** Works by splitting backward into ∂input and ∂weight. Forward-only ⇒ nothing to split. |
| **Continuous / iteration-level batching** | Yu et al., *Orca*, OSDI'22; vLLM | n/a | **This is the one.** The inference answer to microbatching: refill the in-flight set every *iteration*, not every request. §4. |
| **Chunked prefill** | Agrawal et al., *Sarathi-Serve*, OSDI'24; vLLM | (S−1)/(M+S−1) | **Yes — exactly GPipe-forward.** The only schedule that pipelines *within one request*. §6. |
| **Speculative decoding** | Leviathan et al., ICML'23 | n/a | **Yes** — the only technique that attacks the token dependency itself. §7. |

Pipeline-parallel *inference* already shipping, for the "this shape is real" slide: **Petals** (Borzunov
et al. 2022 — closest analogue: distributed transformer blocks over volunteer nodes),
**DeepSpeed-Inference**, **TensorRT-LLM** `--pp_size`, **vLLM** `--pipeline-parallel-size`, **llama.cpp** RPC.

## 3. THE KEY INSIGHT — inference bubbles are not a scheduling problem

Training microbatching works because a minibatch of M samples is **M independent forward passes** — the
scheduler may reorder them, and that freedom is what fills the pipe. Decode has no such freedom **inside
one request**:

```
token[t+1] = argmax( f_node2( f_node1( f_node0( tokens[0..t] ) ) ) )
                                                        ^^^^^^^ contains token[t]
```

`coordinator.py:120` — `gen_ids.append(next_id)`, then loop. Token t+1's input does not exist until token
t has traversed all 3 nodes. **A single decoding request cannot be microbatched.** M is pinned at 1, so
GPipe's formula returns (S−1)/(1+S−1) = (S−1)/S — the naive bubble. The theory folds back onto itself.

> **The ONLY sources of pipeline fill during decode are (a) other requests, (b) speculatively-generated
> future tokens of the same request.** Prefill is exempt — §6.

```
U(R, S) = min(1, R/S)                      X(R) = min( R/Σt , 1/t_max )   tokens/s
TPOT(R) = R / X(R) = max( Σt , R·t_max )   ms per output token, per request
```

## 4. Utilisation and the latency-throughput tradeoff, S=3

(modelled from the measured stage times; "balanced" = the FINDING-1 rebalance, 237.51 ms/stage)

| R | U = min(1,R/3) | **balanced** TPOT | **balanced** X | **v0 split** U | **v0 split** TPOT | **v0 split** X |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **33.3%** | 712.5 ms | 1.40 tok/s | 33.3% | 712.5 ms | 1.40 tok/s |
| 2 | **66.7%** | 712.5 ms | 2.81 tok/s | 66.7% | 712.5 ms | 2.81 tok/s |
| 3 | **100.0%** | 712.5 ms | **4.21 tok/s** | **76.9%** | 926.9 ms | **3.24 tok/s** |
| 4 | 100.0% (capped) | 950.1 ms | 4.21 tok/s | 76.9% | 1235.9 ms | 3.24 tok/s |
| 8 | 100.0% | 1900.1 ms | 4.21 tok/s | 76.9% | 2471.8 ms | 3.24 tok/s |
| 16 | 100.0% | 3800.2 ms | 4.21 tok/s | 76.9% | 4943.5 ms | 3.24 tok/s |

```
X (tok/s)                                  TPOT (ms)
4.21 |          o---o---o---o---o          3800 |                        o
     |         /                                |               o
2.81 |      o                                   |        o
1.40 |  o                                  712  |  o--o--o
     +--1--2--3--4--8--16--> R                  +--1--2--3--4--8--16--> R
        RAMP    | SATURATED                        FLAT  | LINEAR PENALTY
```

> **R = S = 3 is the exact knee.** Below it, throughput is free — latency is *unchanged*, because a
> request queued behind nothing still takes exactly Σt. Above it you pay latency linearly (Little's
> Law, L = X·W with X pinned) and gain **nothing**. Admission control for this system is one number:
> **cap in-flight requests at 3.** (Holds only while each stage handles one item — §4a moves the knee.)

### 4a. Where R > S does pay: intra-stage batching

T1-A1 §7 measures KV-cached decode at ~110–124 ms/token for ~988 MFLOP ⇒ **~9 GFLOP/s, framework-
overhead-bound, not FLOP-bound**, so stage time is mostly fixed cost: `t(B) = t(1)·(f + (1−f)·B)`. At the
f=0.7 that measurement implies, **B=8 costs 3.10× the time for 8× the tokens = 2.58× per-stage
throughput** (modelled; f=0.5 → 1.78×, f=0.85 → 3.90×). R=24 with B=8 per stage ⇒ U=100% *and* 2.58× the
stage rate. Needs `node.py` to take `[B,1,896]` not `[1,seq,896]`, plus a batching queue — **v2**.

## 5. Imbalance × concurrency multiply, and the order matters

FINDING 1: node2 = 17.13 layer-equivalents vs 11.04 balanced ⇒ **1.55×** theoretical. Measured wall-clock
at seq=512: 308.97/237.51 = **1.30×** — lower, because node0/node1 also carry per-call framework overhead
that dilutes the ratio. **Use 1.30× for wall-clock claims.**

```
U_ceiling(R→∞) = Σt / (S · t_max) = 712.54 / (3 × 308.97) = 76.9%
```

**Concurrency alone cannot exceed 76.9% on the current split.** The residual 23.1% is not a bubble any
scheduler can fill — it is node0 and node1 waiting on node2, forever. Rebalancing is a *prerequisite* for
the concurrency work, not an alternative: 4.21/3.24 = **1.30× on top of concurrency's 2.31×.** Combined
**1.40 → 4.21 tok/s = 3.00×.**

## 6. Chunked prefill — the one schedule that pipelines a single request

Prefill of a P-token prompt is **not** autoregressive: all P positions compute in one causal matmul, no
token→token serialisation. Split into M chunks of C tokens and it is GPipe-forward exactly.

**Why the dependency is free:** chunk k+1's attention at layer ℓ needs chunk k's K/V *at layer ℓ* — and
in pipeline order chunk k has **already left** the node holding layer ℓ before k+1 arrives. The
dependency runs *backwards along the pipeline* relative to the flow, so it can never be violated.

### GANTT C — chunked prefill, M=4, one request
```
node0  [c0][c1][c2][c3]........................
node1  ....[c0][c1][c2][c3]....................
node2  ........[c0][c1][c2][c3]................
       |--- fill --|-- 100% busy --|- drain -|
```

`B = (S−1)/(M+S−1)`, ideal speedup `S·M/(M+S−1)`. Makespan `Σt/M + (M−1)·t_max/M` on the measured
per-stage prefill times at seq=2048 (853.97 / 1030.07 / 1667.21 ms, Σ = 3551.25 ms):

| M (chunks) | bubble | ideal | **P=2048, v0 split** | sp. | **P=2048, balanced** | sp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 66.7% | 1.00× | 3551.2 ms (measured) | 1.00× | 3551.2 ms | 1.00× |
| 2 | 50.0% | 1.50× | 2609.2 ms | 1.36× | 2367.5 ms | 1.50× |
| 4 | 33.3% | 2.00× | 2138.2 ms | 1.66× | 1775.6 ms | 2.00× |
| 8 | 20.0% | 2.40× | 1902.7 ms | 1.87× | 1479.7 ms | 2.40× |
| **16** (C=128) | **11.1%** | **2.67×** | **1785.0 ms** | **1.99×** | **1331.7 ms** | **2.67×** |
| 32 (C=64) | 5.9% | 2.82× | 1726.1 ms | 2.06× | 1257.7 ms | 2.82× |

**Stop at M=16.** Past it the return is <6% and every chunk pays the full per-hop tax — T1-A1 §4
measures **5.6 ms/hop just constructing `httpx.AsyncClient()`**, so M=32 burns 32×3×5.6 = **537 ms** of
pure overhead and erases the gain. **Chunked prefill is gated on the connection-pool fix.**

Exact names: `Sarathi` (Agrawal et al., 2023) → **Sarathi-Serve** (OSDI 2024). In **vLLM** chunked prefill
is **always on in the V1 engine and cannot be disabled** (`--no-enable-chunked-prefill` is not a valid V1
arg). Knobs: `--max-num-batched-tokens` (per-step token budget, default 8192 online in recent builds),
`SchedulerConfig.long_prefill_token_threshold` (default 0) and `--max-num-partial-prefills`, which let
short prompts jump the queue ahead of long ones.

## 7. Speculative decoding — the only attack on the token dependency itself

A draft proposes k tokens; the target **verifies all k in ONE forward pass**, because verification is
prefill-shaped (k independent positions) — k sequential traversals collapse into 1. Expected accepted
tokens per traversal at acceptance α: `E = (1−α^(k+1))/(1−α)`; `speedup = E/(1 + k·c)`, c = draft cost as
a fraction of a target traversal.

| α \ k | 2 | 4 | 8 |
|---|---:|---:|---:|
| 0.25 (n-gram, free-form chat) | 1.31 | 1.33 | 1.33 |
| 0.60 (n-gram on code / JSON / quoting) | 1.96 | **2.31** | 2.48 |
| 0.80 (well-trained draft) | 2.44 | 3.36 | 4.33 |

### The 0.5B trap: `lm_head` poisons every draft on this model
FINDING 1 — `lm_head` (896 × 151936) = **9.13 layer-equivalents**; the full model is 24 + 9.13 = **33.13
eq**. Any draft emitting a distribution over V=151936 pays that 9.13 whether it runs 1 layer or 8.

| draft hosted on node0 | layer-eq | c | speedup, k=4, α=0.6 | verdict |
|---|---:|---:|---:|---|
| layers 0–7 early-exit + tied lm_head | 17.13 | **51.7%** | 0.75× — **slower** | dead |
| layers 0–3 + full lm_head | 13.13 | 39.6% | 0.90× — slower | dead |
| EAGLE-style 1-layer head + target lm_head | 10.13 | 30.6% | 1.04× | not worth it |
| layers 0–3 + **top-32k truncated head** | 5.92 | 17.9% | 1.34× | v2, marginal |
| **n-gram / prompt-lookup**, no weights at all | **0.00** | **0.0%** | **2.31×** on quoting-heavy output | **the v1 candidate** |

**node0 can host a draft for free — but "free" is the wrong axis.** `tie_word_embeddings: true` (VERIFIED
FACTS) means node0's `embed_tokens` **is** the `lm_head` matrix, the same 136M params, so an early-exit
draft on node0 needs **zero extra weights and zero extra download**, on the least-loaded stage (28.9% of
Σt, measured). The blocker is not memory but *compute ratio*: at 51.7% of target cost the draft is only
1.9× cheaper than what it drafts for, and speculative decoding needs ≈10×. Put the free-weights fact on
the slide, then be honest that the ratio kills it at 0.5B.

vLLM API for when this reaches a real runtime: `speculative_config` = `{"method": "ngram",
"num_speculative_tokens": …, "prompt_lookup_max": …}` (zero-weight) or `{"method": "eagle3", "model": …,
"num_speculative_tokens": …}`, `"parallel_drafting": true` selecting P-EAGLE. Adjacent: **Medusa** (Cai
et al., ICML 2024), **LayerSkip** (Elhoushi et al., ACL 2024), **Draft & Verify** (Zhang et al., 2023).

## 8. GANTT B — the proposed schedule (R=3, rebalanced). Lift straight into the deck.

```
       0       238      475      713      950     1188     1425 ms
       |--------|--------|--------|--------|--------|--------|
node0  [###A0###][###B0###][###C0###][###A1###][###B1###][###C1###][A2]
node1  ..........[###A0###][###B0###][###C0###][###A1###][###B1###][C1]
node2  ....................[###A0###][###B0###][###C0###][###A1###][B1]
       |-- fill --|--------- 100% utilisation, steady state ---------|
```

| | v0 (Gantt A) | v1 (Gantt B) | factor |
|---|---:|---:|---:|
| utilisation | **33.3%** | **100%** | 3.00× |
| aggregate throughput | 1.40 tok/s | **4.21 tok/s** | 3.00× |
| per-request TPOT | 712.5 ms | **712.5 ms** | **1.00× — free** |
| idle vCPU-s per wall-second | 4.00 | 0.00 | — |
| prefill, P=2048 | 3551 ms | 1332 ms (M=16) | 2.67× |

All (modelled) from measured stage times. **The latency row is the pitch: this throughput costs nothing.**

## 9. Recommendations

| # | change | tag | effort | impact (S=3) |
|---|---|---|---|---|
| 1 | **Drive the demo with 3 concurrent requests.** `coordinator.py` is already fully `async` (`await client.post`), and `node.py`'s `/forward` is a sync `def` that FastAPI runs in the anyio threadpool — concurrency **works today**. The 33% ceiling is a *load* limitation, not a code limitation. `demo.sh` sends one curl. | **v1** | **~0 h** (3 curls + `wait`) | 33.3% → 76.9% util, 1.40 → 3.24 tok/s (modelled) |
| 2 | **Rebalance:** `NODE_LAYERS` = `0-11` / `11-22` / `22-24`+lm_head in `docker-compose.yml`. Pure env-var edit. | **v1** | ~1 h | lifts the 76.9% ceiling to 100%; 3.24 → 4.21 tok/s |
| 3 | **Admission control: cap in-flight at R=3.** One `asyncio.Semaphore(3)` at the coordinator entry, one `Semaphore(1)` per node URL. Past R=3, latency grows linearly for zero throughput (§4). | **v1** | ~2 h | prevents thread oversubscription on 2-vCPU containers; bounds tail latency |
| 4 | **Chunked prefill, M=16 (C=128).** Loop the prompt through in chunks instead of one 2048-token POST. **Do the connection-pool fix first** or the per-hop tax eats it. | **v1** | ~1 d | prefill 3551 → 1785 ms (v0 split) / 1332 ms (balanced), 2.67× |
| 5 | Continuous / iteration-level batching à la **Orca** (OSDI'22): refill the in-flight set every token. | **v2** | weeks | holds U=100% under bursty arrivals, not just steady load |
| 6 | Intra-stage batching `[B,1,896]`, B=8 | **v2** | weeks | +2.58× per-stage throughput at f=0.7 (modelled) |
| 7 | n-gram / prompt-lookup speculation, `speculative_config {"method":"ngram"}` | **v2** | weeks | 1.33× free-form, up to 2.31× on quoting/code (modelled) |
| 8 | Trained EAGLE-3 draft with truncated-vocab head | **v2** | months | ~1.34× at k=4, α=0.6 — poor ROI on a 0.5B target; revisit at 7B+ |

**Do NOT build:** interleaved 1F1B / virtual pipeline stages (§2 — doubles hops, zero bubble reduction
at R=1); zero-bubble schedules (need a backward pass); any 1F1B implementation (degenerates to
round-robin without a backward).

## Risks

1. **KV cache × concurrency is a CORRECTNESS hazard, not a performance one.** `node.py` is stateless
   today — exactly why R=3 works out of the box. Once a cache lands each node holds per-request state,
   and concurrent request B reads A's cache unless every `/forward` carries T3-A1's `session_id`.
   **Land the cache and `session_id` in the same commit.**
2. **Thread oversubscription.** FastAPI runs sync `def forward` in a 40-slot threadpool; 3 concurrent
   forwards × torch's 2 threads on a `cpus: "2"` container thrashes. Rec 3 is what makes rec 1 safe —
   ship them together.
3. Everything past the measured stage times is **modelled**, assuming stage time is independent of
   concurrency. It is not; contention will shave R=3 below 4.21 tok/s. Say "≈3× (modelled)", then
   measure — `demo.sh` with three background curls converts the headline to (measured) in 10 minutes.
4. §6 assumes prefill cost is linear in tokens. Attention is O(P²), so small chunks are cheaper than
   modelled and large ones dearer; M=16 sits inside where the linear model is defensible.
