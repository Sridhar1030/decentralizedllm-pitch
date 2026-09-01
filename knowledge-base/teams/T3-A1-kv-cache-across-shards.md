---
team: T3 — Scheduling, Queueing & Compute Utilisation
agent: T3-A1
topic: Per-shard KV cache across pipeline nodes — arithmetic, protocol, failure model, v1 patch
headline: A per-node per-session KV cache cuts position-forwards 271x (147,200 -> 543) and wire bytes 77% (1,821.7 MB -> 420.1 MB) for one 512-token generation; realised wall-clock is 28.6x (modelled), not 271x, because decode flips from compute-bound GEMM to bandwidth-bound GEMV. Price: the session becomes STATEFUL and pinned to one node triple.
---

# T3-A1 — KV cache across shards

## 0. TL;DR

| | v0 today | v1 cached | factor |
|---|---|---|---|
| position-forwards per node (P=32, G=512) | 147,200 | 543 | **271x** (derived) |
| pipeline FLOPs, same generation | 145.4 TFLOP | 536.4 GFLOP | 271x (derived) |
| wall-clock, 3x 2-vCPU chain | 2,908 s | 101.6 s | **28.6x** (modelled) |
| wire bytes, cache change only | 1,821.7 MB | 420.1 MB | 4.34x (derived) |
| hidden-state bytes alone | 1,406.8 MB | 5.19 MB | 271x (derived) |
| KV memory/node, 2048-tok session | 0 | 16.78 MB fp32 | — |
| node stateless | yes | **no** | the whole cost |

The cache alone removes 77% of v0 wire and 99.6% of v0 compute. The residual 414.9 MB of wire is the fp32 logit
vector (FINDING 2) — someone else's fix.

**v0, exactly.** `coordinator.py:123` `for _ in range(max_tokens-1): logits = await forward_chain(gen_ids)` — that
is the whole growing sequence. `node.py:90/99` call `model(...)` with no `past_key_values`, no `use_cache`. Every
node recomputes every prefix position on every token, on three machines.

## 1. Cache sizing — GQA makes it nearly free

Per layer per token: K and V are each `[batch=1, num_key_value_heads=2, seq, head_dim=64]`.
`2 heads x 64 head_dim x 2 (K,V) = 256 elements` -> **1,024 B fp32 / 512 B bf16**.

| scope | fp32 | bf16 |
|---|---|---|
| token, 8-layer shard | 8,192 B | 4,096 B |
| token, whole 24-layer model | 24,576 B | 12,288 B |
| 2048-tok session, 8-layer shard | **16.78 MB** | 8.39 MB |
| 2048-tok session, whole model | 50.33 MB | 25.17 MB |
| 32,768-tok (max_position_embeddings), whole model | 805.3 MB | 402.7 MB |

**GQA is a 7x discount** (`num_attention_heads 14 / num_key_value_heads 2`): plain MHA would make the 2048-token
whole-model cache 352.3 MB fp32 instead of 50.33 MB. Reference point — one 8-layer shard's fp32 weights are
477.1 MB (8 x 14,909,440 x 4 B), so a 2048-token session's cache is **3.5% of that**; 28 full-context sessions fit
inside one copy of the weights the node already loaded. There is no memory argument against caching.

## 2. Compute — 271x position-forwards, 28.6x wall-clock

P=32 prompt, G=512 generated (the assignment's P≈0 framing gives sum(1..512)=131,328 -> 512 = 256.5x; the house
convention below matches FINDING 3 and supersedes it):

```
v0 positions/node = G*P + G(G-1)/2 = 512*32 + 512*511/2 = 16,384 + 130,816 = 147,200
v1 positions/node = P + (G-1)      = 32 + 511            =             543
ratio = 271.1x
```

Per-position FLOPs (2 x params): 1 layer 29.82 M · 8-layer shard 238.6 M · lm_head 272.3 M · **whole model 987.9 M**
(493,961,216 params). The quadratic attention term is `4 x 14 heads x 64 head_dim x seq` = 3,584·seq FLOP/layer:
at seq=543 that is 1.95 M vs the layer's 29.82 M = **6.5%**. So the saving is essentially linear in
position-forwards — redundant *dense matmuls* are the cost, not attention.

**Wall clock (modelled).** Assumptions: 2 vCPU/node sustaining ~50 GFLOP/s fp32 GEMM, ~10 GB/s effective DRAM
read, strict sequential chain (the 33% utilisation ceiling is not fixed here).

| phase | work | time |
|---|---|---|
| v0 whole generation | 147,200 x 987.9 MFLOP = 145.4 TFLOP, all GEMM, compute-bound | 2,908 s |
| v1 prefill (32 tok) | 31.6 GFLOP GEMM | 0.63 s |
| v1 decode, per token | GEMV — bound by streaming 1.976 GB of fp32 weights once | 0.198 s |
| v1 decode, 511 tokens | | 100.9 s |
| **v1 total** | | **101.6 s -> 28.6x** |

Say this precisely on the slide: **271x is a FLOP-count ratio.** Cached decode has arithmetic intensity ~2
FLOP/byte-of-weight and is DRAM-bandwidth bound; v0's steps were fat GEMMs at intensity ~n running near peak. The
cache converts a compute problem into a bandwidth problem. bf16 weights halve the bandwidth term -> 50.8 s ->
**57.2x (modelled)**. Batching decode across sessions restores intensity — **and is only possible once the cache
exists**, because without it every request carries a different seq_len and there is nothing to batch.

## 3. Wire — what the cache alone buys

base64(B) = 4·ceil(B/3); hidden per position = 896 x 4 B = 3,584 B -> 4,780 B. Convention follows FINDING 4 (two
hidden transfers per logical step). The PoC is actually a *star* — the coordinator posts to each node — so every
logical hop is two TCP transfers; double these if you count sockets.

| | positions on wire | hidden | logits return | total |
|---|---|---|---|---|
| v0 | 147,200 | 1,406.8 MB | 414.9 MB (512 x 810,325 B) | **1,821.7 MB** |
| v1 cache only (still fp32+b64+full logits) | 543 | 5.19 MB | 414.9 MB | **420.1 MB** |
| v1 full stack (+bf16, binary, argmax on node2 — FINDING 4) | 543 | 1.946 MB | 0.002 MB | 1.948 MB |

Cache alone removes **1,401.6 MB = 76.9%** of v0's wire. All that is left is the logit vector.

## 4. Protocol change

`ForwardRequest` gains 3 fields; payload shrinks from `[seq, 896]` to `[1, 896]` after prefill.

| field | meaning |
|---|---|
| `session_id: str?` | cache key. Absent => stateless v0 path — keep it as the A/B control for the demo. |
| `position: int` | tokens this shard must ALREADY hold. Fence token. |
| `reset: bool` | drop the cache first (new session, or coordinator-driven recovery). |

With `have = cache.get_seq_length()`: `have == position` -> append; `have > position` -> `cache.crop(position)`
(rewind — retries, rejected speculative tokens, cheap); `have < position` -> **409 `{"cache_miss", have}`**,
coordinator replays.

The fence is not optional. The cache is *positional state*: one duplicated or reordered delivery double-appends a
position and the model silently emits wrong tokens with no error. `position` is simultaneously the idempotency
guard and the RoPE-correctness guard (§5). Trust boundary — do not simplify it away.

## 5. Two real bugs the patch must fix (read out of transformers 5.5.0 source)

**(a) Stale `layer_idx` -> silently wrong output.** `node.py:53` does `model.model.layers =
full.model.layers[8:16]`. Slicing an `nn.ModuleList` does not renumber: those modules keep
`self_attn.layer_idx = 8..15` (`Qwen2Attention.__init__` sets it; forward calls
`past_key_values.update(k, v, self.layer_idx)`). Cache slots 0..7 stay empty on node1/node2, and
`Qwen2Model.forward` then computes **both** of these wrong:

- `past_seen_tokens = past_key_values.get_seq_length()` — defaults to `layer_idx=0` -> **0** -> `position_ids`
  restart at 0 every decode step -> wrong RoPE;
- `masking_utils.py:850  kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)` -> kv_length
  = 1 -> the causal mask hides the entire past.

No exception, just garbage text. **Fix: renumber to local dense indices 0..k-1 in `load_model()`** — one loop.

**(b) `lm_head` runs on every position — 27.5% of v0's total FLOPs, free to fix.** Node2 computes `out.logits`
for the whole resent sequence then keeps `[:, -1, :]` (`node.py:104`). Over the generation that is 147,200
lm_head applications where 512 are needed: 146,688 x 272.3 MFLOP = **39.9 TFLOP of the 145.4 TFLOP total**.
`logits_to_keep=1` (present in `Qwen2ForCausalLM.forward`, transformers 5.5.0) fixes it: **1.38x on the whole
pipeline with no cache at all, one kwarg**. Also delete `output_hidden_states=True` (defect 7) — on non-final
nodes `lm_head` is `Identity`, so **`out.logits` already IS the final hidden state**.

## 6. Session affinity — the price of statefulness

A session is now pinned to a `(node0_i, node1_j, node2_k)` triple for life. This is the one genuine
architectural regression the cache introduces; state it in the deck rather than hiding it.

| property | v0 stateless | v1 stateful | consequence |
|---|---|---|---|
| routing | any node with the layer range | fixed triple, chosen at prefill | need a `session_id -> triple` map: v1 a dict in the coordinator, v2 etcd/Redis, replicated |
| load balancing | per token, free | per session, at admission only | a long session on a slow node stays there — head-of-line blocking at *session* granularity |
| L4/L7 LB | round-robin works | breaks | `docker-compose scale` no longer balances; needs consistent routing |
| scale-out | instant | new nodes get only NEW sessions | warm-up lag = length of live sessions |
| drain / rolling deploy | kill anytime | wait out or migrate | v2 migration costs 4.45 MB/shard for a 543-token session — cheap, just unimplemented |
| crash blast radius | one in-flight token | **every session pinned to that node** | §8 |
| memory | constant | grows with concurrency | memory becomes a scheduling dimension -> admission control (T3 hand-off) |

## 7. Eviction, TTL, memory accounting

`bytes(session, node) = tokens x layers_on_node x 256 elements x dtype_bytes = tokens x layers x 1,024 B` (fp32).

| node | resident weights fp32 | headroom (2 GB limit) | fp32 cache tokens | 2048-tok sessions |
|---|---|---|---|---|
| node0 (embed + 8 layers) | 1,021.6 MB | ~0.9 GB | 109,860 | 53 |
| node1 (8 layers) | 477.1 MB | ~1.4 GB | 170,898 | 83 |
| node2 (8 layers + lm_head) | 1,021.6 MB | ~0.9 GB | 109,860 | 53 |

Prerequisite: `load_model()` holds `full` (1.976 GB fp32) *and* a freshly constructed `Qwen2ForCausalLM(config)`
(another 1.976 GB) simultaneously -> **~4 GB peak**, which is why the compose memory limits are commented out.
Fix that before claiming any cache budget is credible.

| knob | v1 | v2 |
|---|---|---|
| structure | `OrderedDict` LRU: session_id -> `[DynamicCache, last_used]` | paged block pool (§10) |
| cap | `MAX_SESSIONS` count + `MAX_CACHE_TOKENS` | byte budget with per-tenant quota |
| TTL | 300 s idle, checked lazily on touch | active sweeper + gauge |
| victim | LRU session | longest-idle / lowest-priority, block-granular |
| evicted session returns | 409 `cache_miss` -> coordinator replays (§8) | swap blocks to disk, or recompute |
| admission | reserve `max_tokens x layers x 1,024 B` at prefill | on-demand block allocation |

Export `node_kv_bytes`, `node_kv_sessions`, `node_kv_evictions_total` beside the existing `/metrics` counters —
T3's queue needs them as its backpressure signal.

## 8. Node failure — the cache is lost; cost the recovery

The coordinator already holds the authoritative token list (`gen_ids`), so nothing is unrecoverable. But the
naive fix does not work: if node1 dies at position n, its replacement needs `hidden[0..n-1]` from node0 — and
node0's cache is already at n, so re-running node0 over those positions would double-append into its own cache.

| strategy | mechanism | cost at n=543 | tag |
|---|---|---|---|
| **reset-all + replay** | `reset=true` to all 3, one prefill of n | n position-forwards/node = 536.4 GFLOP = **10.7 s** (modelled), 5.19 MB wire | **v1** |
| upstream retains emitted hidden states | node0 keeps its `[n,896]` output ring (n x 3,584 B = 1.95 MB) so a downstream replacement re-prefills alone | ~1/3 of above, upstream untouched | v2 |
| rewind instead of reset | `Cache.crop(k)` on nodes 0..i-1, replay from k | proportional to n-k | v2 |
| checkpoint cache to a peer | 543 x 8,192 B = 4.45 MB/shard, async | ~ms on LAN, 2x memory | v2 |

**Headline: a failure at position n costs another n position-forwards — it exactly doubles that session's
compute.** Cached-with-one-failure is still 271/2 = **135x** better than v0. Eviction and crash are the same
event; handle them with the same code path. Surface the 10.7 s stall as an SSE `event: replaying` — the demo UI
already has the event channel — rather than a silent hang.

## 9. Prefill vs decode

| | prefill | decode |
|---|---|---|
| requests per session | 1 | G-1 |
| positions per request | P (=32) | 1 |
| wire shape | `[P, 896]` | `[1, 896]` |
| FLOPs, whole model | 31.6 GFLOP | 987.9 MFLOP |
| arithmetic intensity | ~P FLOP/byte-of-weight | ~2 |
| bound | compute (GEMM) | DRAM bandwidth (GEMV) |
| modelled time, 3x 2-vCPU chain | 0.63 s = TTFT | 0.198 s = TPOT |
| goes faster with | more cores, chunking | **batching across sessions**, bf16 weights |
| fails under load by | head-of-line blocking every decode behind it | queueing |

Opposite bottlenecks, therefore separate scheduling. vLLM's chunked prefill (split a long prefill into fixed-size
chunks, interleave with decode steps) is the standard mitigation; at P=32 we do not need it — flag it for T3's
scheduler agent above P ≈ 512.

## 10. PagedAttention — what paging buys, and when

vLLM, Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023. KV lives in
fixed-size blocks (16 tokens is the long-standing default; `--block-size`, CUDA supports up to 32) addressed
through a per-sequence block table — OS virtual memory for the cache. Reported: **2-4x throughput** over
FasterTransformer and Orca at equal latency; KV waste **under 4%** vs 60-80% in reservation-based systems.

One block on an 8-layer shard = 16 x 8 x 1,024 B = **128 KiB fp32** (64 KiB bf16):

| | contiguous `DynamicCache` (v1) | paged (v2) |
|---|---|---|
| admission must reserve | worst case 2048 x 8,192 B = 16.78 MB/session | on demand, 128 KiB at a time |
| session actually ends at 200 tokens | 16.78 MB held, 1.64 MB used -> **90% wasted** | waste <= 1 block = 128 KiB = 0.76% |
| concurrent sessions in 0.9 GB headroom | 53 | ~549 at a 200-token mean -> **10.4x (modelled)** |
| shared system prompt | stored N times | stored once (prefix caching, `--enable-prefix-caching`, copy-on-write) |
| preemption | evict the whole session | evict/swap individual blocks |

Prefix sharing is real money: a 32-token chat-template prefix over 50 sessions duplicates 12.8 MB/shard —
negligible; a 1,000-token RAG system prompt duplicates **401 MB/shard = 45% of the entire headroom**, for nothing.
What paging is *not* for here: `DynamicCache` grows by `torch.cat` and reallocates each step — 543²/2 x 8,192 B =
1.21 GB of memcpy per shard, 0.12 s at 10 GB/s = **0.1%**; even at 32k context it is 6.8% (modelled). **Build
paging for concurrency, reservation waste, prefix sharing and cheap preemption — none of which bite at 3 nodes and
demo-scale concurrency.** v1 stays on `DynamicCache` + LRU; if the fleet ever needs paging, adopt vLLM rather than
reimplementing it.

## 11. The v1 patch — `layer-nodes/node.py`

Verified against transformers 5.5.0: no-arg `DynamicCache()` lazily appends `DynamicLayer`s; `Cache.get_seq_length`,
`Cache.crop`, `logits_to_keep` all present. ~35 added lines, no new dependency.

```python
from collections import OrderedDict
from fastapi import HTTPException
from transformers import DynamicCache
CACHE_TTL_S  = float(os.getenv("CACHE_TTL_S", "300"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "32"))
_sessions = OrderedDict()          # session_id -> [DynamicCache, last_used_monotonic]

# in load_model(), after model.model.layers is assigned — ALL THREE branches.
# Sliced layers keep their ORIGINAL layer_idx (8..15); Qwen2Model reads position_ids from
# past_key_values.get_seq_length() (layer 0) and the mask from get_mask_sizes(q, layer_idx),
# so stale indices => garbage output, no error. Renumber to local dense indices:
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i
    model.config.num_hidden_layers = len(model.model.layers)

class ForwardRequest(BaseModel):
    hidden_states_b64: Optional[str] = None
    input_ids: Optional[list] = None
    session_id: Optional[str] = None   # None => stateless v0 path, kept as the A/B control
    position: int = 0                  # tokens this shard must already hold; fence + RoPE guard
    reset: bool = False

def _cache_for(sid, position, reset):
    if sid is None:
        return None
    if reset:
        _sessions.pop(sid, None)
    now = time.monotonic()
    ent = _sessions.get(sid)
    if ent is None or now - ent[1] > CACHE_TTL_S:
        ent = [DynamicCache(), now]
    have = ent[0].get_seq_length()
    if have > position:
        ent[0].crop(position)                 # rewind: retry / rejected speculative token
    elif have < position:
        raise HTTPException(409, {"error": "cache_miss", "have": have, "want": position})
    ent[1] = now
    _sessions[sid] = ent
    _sessions.move_to_end(sid)
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)         # ponytail: LRU by count; move to a byte budget when
    return ent[0]                             # node_kv_bytes actually nears the container limit

# in forward(), replacing the body of the `with torch.no_grad()` block:
    cache = _cache_for(req.session_id, req.position, req.reset)
    kw = {"past_key_values": cache, "use_cache": cache is not None}
    if end_layer == 24:
        kw["logits_to_keep"] = 1              # lm_head on 1 position, not all of them (see 5b)
    if req.input_ids is not None:
        out = model(torch.tensor([req.input_ids], dtype=torch.long), **kw)
    else:
        h = np.frombuffer(base64.b64decode(req.hidden_states_b64), dtype=np.float32)
        out = model(inputs_embeds=torch.from_numpy(
            h.reshape(1, -1, model.config.hidden_size).copy()), **kw)
    _forward_seconds_total += time.monotonic() - t0
    if end_layer == 24:
        return {"logits_b64": base64.b64encode(out.logits[:, -1, :].numpy().tobytes()).decode()}
    # lm_head is Identity here, so out.logits IS the final hidden state; output_hidden_states=True
    # is deleted (it materialised 8 tensors to use 1).
    return {"hidden_states_b64": base64.b64encode(out.logits[0].numpy().tobytes()).decode()}
```

Coordinator: mint `session_id = uuid4().hex` per request; first call carries `input_ids=<full prompt>,
position=0, reset=True`, every later call `input_ids=[next_id], position=len(gen_ids)-1`. On a 409 from any node,
resend with `reset=True` and the full `gen_ids`. Change nothing else.

**The one check to leave behind** (`layer-nodes/test_kv_cache.py`, no framework): load node0's shard, run the same
12-token prompt (a) stateless in one shot and (b) one token at a time with a `session_id`, then
`assert torch.allclose(h_stateless[-1], h_cached[-1], atol=1e-4)`. That single assert fails if the layer_idx
renumbering, the position fence, or the mask is wrong — the only three ways this corrupts silently.

## 12. Recommendations

| # | change | tag | effort | impact |
|---|---|---|---|---|
| 1 | `logits_to_keep=1` on node2, drop `output_hidden_states=True` | **v1** | 1 h | 1.38x whole pipeline, no cache needed (derived) |
| 2 | renumber `self_attn.layer_idx` to local dense indices | **v1** | 1 h | prerequisite — without it the cache is silently wrong |
| 3 | `DynamicCache` per session_id + `position` fence + LRU/TTL | **v1** | 1 d | 271x position-forwards, 28.6x wall-clock (modelled), 77% wire |
| 4 | coordinator sends `[1,896]` + session_id after prefill | **v1** | 0.5 d | included in #3 |
| 5 | reset-all replay on 409 / node loss + SSE `replaying` event | **v1** | 0.5 d | bounds a failure at 2x session compute |
| 6 | `node_kv_bytes` / `_sessions` / `_evictions_total` in `/metrics` | **v1** | 1 h | the input to T3's admission control |
| 7 | bf16 KV + bf16 weights | v2 | days | halves cache bytes and the decode bandwidth bound -> 57.2x (modelled) |
| 8 | continuous batching across sessions (requires #3) | v2 | weeks | attacks the 33% utilisation ceiling — the actual T3 goal |
| 9 | paged blocks + prefix caching, or adopt vLLM outright | v2 | weeks | ~10.4x concurrent sessions per GB (modelled) |
| 10 | cache migration / peer checkpoint (4.45 MB per shard-session) | v2 | weeks | rolling deploys and drains without dropping sessions |

**Risks.** (i) A stale-cache bug produces *plausible wrong text*, not an error — #2 plus the `allclose` check are
non-negotiable. (ii) Session affinity is a real regression: no per-token load balancing, and a node crash now
kills every session pinned to it. (iii) 271x is a FLOP ratio; quote **28.6x (modelled)** for wall-clock or the
demo will under-deliver against the slide. (iv) `load_model()`'s ~4 GB peak must be fixed before any per-node
memory budget is credible.

Sources: [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) ·
[vLLM engine args](https://docs.vllm.ai/en/v0.8.3/serving/engine_args.html) ·
[Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
