---
team: T3 — Scheduling, Queueing & Compute Utilisation
agent: T3-A3
topic: Continuous batching (Orca iteration-level scheduling) adapted to a 3-stage layer-sharded pipeline
headline: >
  A decode step on one CPU shard costs the same at batch 32 as at batch 1 (37.3 ms vs 23.6 ms, measured) —
  98.7% of a B=1 step is fixed framework overhead. Batching is therefore nearly free here, worth 32.9x
  aggregate throughput for 1.46x worse TPOT (modelled from measured stage times). The distributed twist:
  no stage may decide the batch. One coordinator mints an immutable BatchDescriptor per step; all 3 stages
  execute it verbatim. Row order is the contract.
---

# T3-A3 — Continuous batching in a layer-sharded pipeline

Measured on this Mac (10 cores, 32 GB), `torch 2.10.0` / `transformers 5.3.0`, fp32,
`torch.set_num_threads(2)` to mirror docker-compose `cpus: "2"`. Shard = 8 Qwen2 decoder layers
(node1-style: `inputs_embeds` in, hidden out), `DynamicCache` pre-filled to ctx=128, min-of-6 runs.

## 1. The two mechanisms, and the one that breaks when you shard by layer

| | Orca (OSDI '22, Yu et al.) | vLLM V1 (`SchedulerOutput`) | Breaks in our 3-node pipeline? |
|---|---|---|---|
| **Iteration-level scheduling** | Scheduler returns after every *iteration* (one forward step), not every request. Finished seqs leave, new ones join at the iteration boundary. Kills head-of-line blocking. | `Scheduler.schedule()` → `SchedulerOutput` per step; `ModelRunner.execute_model(scheduler_output)` | **Yes.** The boundary is now 3 machines wide. A stage cannot admit or evict on its own. |
| **Selective batching** | Batch the token-wise ops (LayerNorm, QKV proj, MLP) over a flattened `[ΣL, H]` 2-D tensor; run Attention *per sequence* (split → attend → merge), since attention is not token-wise. | `flash_attn_varlen_func(q,k,v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)`; `cu_seqlens_*` are `int32[B+1]` | No — it is local to a stage, but **every stage must derive the same `cu_seqlens`**, so it must come from the same descriptor. |
| **Paged KV** | (not in Orca) | PagedAttention (SOSP '23), `block_size=16`, per-request block table | No — but it is what makes leave cheap: a seq exiting mid-batch frees blocks without compacting anyone else's KV **on three machines simultaneously**. |
| Reported | 36.9x throughput at equal latency vs FasterTransformer, GPT-3 175B | production default since the V1 engine | |

vLLM V1 collapsed the V0 "prefill batch vs decode batch" split into one dict,
`num_scheduled_tokens: dict[req_id, int]` — a step freely mixes a chunked prefill of A with decodes of
B, C, D. That dict *is* the batch descriptor this design needs. Its siblings on the same struct —
`scheduled_new_reqs`, `scheduled_cached_reqs`, `finished_req_ids`, `num_common_prefix_blocks`,
`preempted_req_ids` — are exactly the join/leave/evict channels enumerated in §4.

## 2. Measured: on a CPU shard, batching is nearly free

**Decode step** (B sequences × 1 new position, ctx=128):

| B | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| shard step, ms | 23.63 | 42.39 | 43.06 | 44.81 | 44.81 | **37.33** |
| ms per sequence | 23.634 | 21.197 | 10.765 | 5.601 | 2.801 | **1.167** |
| `lm_head` (896×151936, last pos only), ms | 21.23 | — | 15.37 | 16.75 | 15.47 | 16.23 |

Flat from B=2 to B=32 within noise (37–45 ms); B=32 < B=16 is measurement noise, not a real inversion.
**20.2x more sequences per second, for 1.58x more time.**

**Prefill** (1 seq, P positions, no cache) — this is where the fixed cost shows up:

| P | 1 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| ms | 21.29 | 32.15 | 65.53 | 172.83 |
| ms/position | 21.29 | 1.005 | 0.512 | 0.338 |

Linear fit from P=32 and P=512: **`t_shard(N) = 22.77 ms + 0.2931 ms × N`** (N = positions in the step;
predicts P=128 at 60.3 vs 65.5 measured, −8%). Two consequences that drive everything below:

- A B=1 decode step is `22.77 + 0.29` → **98.7% fixed overhead**. Batching amortises it.
- `lm_head` is a 545 MB fp32 weight read; it is memory-bandwidth-bound, so **B=32 costs the same as B=1**.
  It is 51.7% of node2 (VERIFIED FINDING 1) and it batches for free.

## 3. Pipeline arithmetic (stage times measured, composition modelled)

Current split: node0/node1 = 8 layers; node2 = 8 layers + `lm_head`. Pipeline depth D=3 → 3 microbatches
in flight, each producing a token every `3 × t_bottleneck`.

| B/microbatch | node0/1 ms | node2 ms | bottleneck | TPOT = 3× ms | seqs in flight = 3B | tok/s = 3B/TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.63 | 44.86 | 44.86 | 134.6 | 3 | 22.3 |
| 4 | 43.06 | 58.43 | 58.43 | 175.3 | 12 | 68.5 |
| 8 | 44.81 | 61.56 | 61.56 | 184.7 | 24 | 129.9 |
| **16** | 44.81 | **60.28** | 60.28 | **180.8** | **48** | **265.4** |
| 32 | 37.33 | 53.56 | 53.56 | 160.7 | 96 | 597.4 |

Baseline = v0 **+ KV cache**, strictly serial, 1 seq: **123.94 ms/token = 8.07 tok/s** (T1-A1, measured).

> **AUDIT CORRECTION (90-AUDIT F03) — 32.9x mixes two measurement runs.** T1-A1's 123.94 ms is a seq=512
> run; every stage time in this file is ctx=128. On *this file's own* measured stages the serial baseline is
> `23.63 + 23.63 + (23.63 + 21.23 lm_head) = 92.1 ms = 10.86 tok/s`, so the honest same-run figures are
> **pipelining 2.05x** (not 2.76x) and **combined 24.4x at B=16** (not 32.9x). Quote 24.4x, or quote 32.9x
> only with "against T1-A1's seq=512 serial baseline" attached.
*(Reconciles with T3-A2, which anchors on the **pre**-KV baseline 712.5 ms → 1.40 tok/s; both are T1-A1 §5/§7.
Batching only means something after the KV cache exists — without it every step recomputes the whole sequence
and there is no fixed cost to amortise. A2's 3.00x pipelining rung is row 2 below: the files multiply, not conflict.)*

| step | tok/s | × over previous | why |
|---|---:|---:|---|
| v0 + KV cache, serial, B=1 | 8.07 | — | 2 of 3 nodes idle (ceiling 1/3) |
| + microbatch pipelining, B=1 | 22.3 | **2.76x** | fills the idle stages; ceiling is 3.00x, lost to node2 being 1.34x heavier |
| + **continuous batching, B=16** | 265.4 | **11.9x** | amortises the 22.77 ms fixed cost over 16 seqs |
| **= combined** | **265.4** | **32.9x** | TPOT 123.9 → 180.8 ms (**1.46x worse per user**) |
| + rebalanced split (FINDING 1) | 320.2 | 1.21x | equal stages: (44.81×2+60.28)/3 = 49.97 ms → TPOT 149.9 ms (modelled) |

At B=32: **74.0x** throughput for 1.30x worse TPOT. The 2.76x and the 11.9x multiply because they are
orthogonal: pipelining fills idle *stages*, batching fills an idle *step*.

## 4. The twist: one decision, three executors

> **Invariant.** Row *i* of the `[n_tok, H]` activation belongs to `desc.slots[i].req_id` at every stage.
> No stage may reorder, insert or drop a row. Only the coordinator changes the row set, and only by
> minting a new `step_id`.

| transition | mechanism | what breaks if a stage decides alone |
|---|---|---|
| **JOIN** — request arrives while a batch is at stage 2 | It never joins mid-flight. The coordinator dispatches a microbatch every stage-step, so the newcomer enters the *next* descriptor (`kind=PREFILL_CHUNK`, `ctx_len=0`, `slot` from the coordinator's free-list). Admission delay ≤ 1 stage-step ≈ 45–60 ms (measured), not 1 TPOT. | stage 1 admits, stage 2 does not → row counts disagree; best case a shape error, worst case a silent row shift and every sequence after row *i* attends to another user's KV |
| **LEAVE** — EOS | EOS is only visible at node2 (the sampler must live there — FINDING 2). Node2 **cannot** drop the row: it returns `finished=[req_id]` on the return path; the coordinator omits it from step S+1 and puts it in `desc.evict[]`, so **all three stages free the same blocks on the same step**. The dead row occupies its slot for one more step at stages 0/1 — they are downstream-blind by construction. | node2 drops row 5; stages 0/1 still ship 16 rows → off-by-one for rows 6..15 |
| **RAGGED** — mixed chunked-prefill + decodes | Flatten to `[Σ n_new, H]`; `cu_seqlens_q`/`cu_seqlens_k` carry the segment boundaries; attention runs per segment. Never pad. | padding to `max(n_new)` — **measured 3.76x waste**, §5 |
| **RECLAIM latency** | A freed slot is only reusable once the coordinator has *seen* the EOS, i.e. one full traversal (D=3 stage-steps ≈ 1 TPOT) after the step that produced it. Over-provision the slot table by one microbatch (`MAX_SLOTS = D × B + B`). | slot reuse race: two requests written into the same KV rows on different stages |

## 5. Tensor layout: padded vs flattened varlen (measured)

Ragged prefill batch, lengths `[16, 200, 40, 8, 512, 24, 96, 4]` — Σ=900 real positions, max=512:

| layout | shape | position-slots | measured ms | |
|---|---|---:|---:|---|
| padded | `[8, 512]` | 4096 | **1033.60** | 4.55x slot inflation |
| flattened varlen | `[1, 900]` | 900 | **274.70** | |
| | | | **3.76x** | measured saving from not padding |

`cu_seqlens_q = cumsum([0] + n_new)`, `cu_seqlens_k = cumsum([0] + ctx_len + n_new)`, both `int32[n_seq+1]`.
**Do not ship them** — every stage derives them from `n_new`/`ctx_len`, which are already in the descriptor.
That drops 136 B of a 479 B descriptor at n_seq=16.

## 6. The batch descriptor

```c
struct BatchDescriptor {          // minted ONCE by the coordinator, immutable, travels with the microbatch
  u64  step_id;                   // monotonic fencing token; stage rejects step_id <= last_seen
  u32  epoch;                     // bumped on any shard/topology change; stage rejects mismatch
  u8   microbatch_id;             // 0..D-1
  u16  n_seq;
  u32  n_tok;                     // == sum(slots[].n_new) == rows in the activation tensor
  struct Slot {
    u64 req_id;
    u16 slot;                     // row index into THIS stage's KV block table  <- "at which position"
    u8  kind;                     // DECODE | PREFILL_CHUNK
    u8  emit;                     // 1 => node2 runs lm_head + sampler on this row. Non-final
                                  //      prefill chunks set 0: saves 16.5 ms/step (measured)
    u16 n_new;                    // positions this seq contributes to THIS step
    u32 ctx_len;                  // KV positions already committed for this seq at this stage
    u16 block_ids[];              // paged-KV block-table DELTA (new blocks only)
  } slots[n_seq];
  u64  evict[];                   // req_ids every stage frees AFTER this step (EOS / preemption)
  u32  crc32;
}                                 // wire size @ n_seq=16: 19 B hdr + 16x17 B + 32 B blocks + 16 B evict
                                  // + 4 B crc = 343 B, vs 28,672 B of bf16 activation => 1.20% overhead
```

The descriptor and the activation **must be one framed message** (T1-A4's wire protocol), or carry
`step_id` in both — otherwise a stage can pair descriptor S with tensor S−1 and corrupt silently.

## 7. Batch size under an SLO: TTFT vs TPOT

One knob, not two — vLLM's `max_num_batched_tokens` (`T`). Decodes contribute 1 token each, a prefill
chunk contributes `c`. From the measured fit `t_shard(N) = 22.77 + 0.2931 N` and `TPOT = 3 · t_shard(T)`:

> **`T_max = (TPOT_slo / 3 − 22.77) / 0.2931`**

| TPOT SLO | 150 ms | 200 ms | 300 ms | 500 ms |
|---|---:|---:|---:|---:|
| token budget `T` | 93 | **150** | 264 | 491 |

**Prefill and decode want opposite batch sizes**, and here is why, measured:

| | marginal cost of one more unit | verdict |
|---|---|---|
| **decode** (+1 sequence) | ~0 ms from B=4→32 (43.06 → 37.33 ms) | batch as wide as KV memory allows |
| **prefill** (+1 position) | 0.2931 ms — real work, already saturating | batch narrow; a big prefill monopolises the pipe |

A one-shot P=512 prefill costs 172.83 ms/stage and **stalls every decoding sequence for 3.86x a normal
step**. Chunked prefill at c=128 (measured 65.53 ms) fixes both ends:

| | TPOT spike for other users | TTFT for the P=512 prompt |
|---|---:|---:|
| one-shot prefill | 172.83 / 44.81 = **3.86x** | 3 stage-steps × 172.83 = **518 ms** (a monolithic prefill cannot pipeline) |
| chunked, c=128 | 65.53 / 44.81 = **1.46x** | 4 chunks pipelined over 3 stages = 6 steps × 65.53 = **393 ms** |
| | **2.64x smaller spike** | **1.32x better TTFT** |

Chunking improves TTFT *and* TPOT here — non-obvious, and it falls out of the pipeline: chunks flow
across stages concurrently, a single fat prefill cannot. (Components measured, composition modelled.)

**KV memory bound on `B`.** GQA gives 512 B/token/layer fp16 (FINDING 3) → 4,096 B/token per 8-layer
shard. At ctx=2048: 48 seqs = 402 MB/shard, 96 seqs = 805 MB/shard. node0 already holds ~1.02 GB of fp32
weights in a 2 GB container. **Admission ceiling ≈ B=32/stage at ctx=2048** (modelled). Below that, the
binding constraint is the TPOT SLO, not memory.

## 8. Prefill/decode disaggregation IS layer sharding (the pitch point)

| | DistServe (OSDI '24) / Splitwise (ISCA '24) | DecentralizedLLM |
|---|---|---|
| cut along | **phase**: prefill machines \| decode machines | **layer**: 0-7 \| 8-15 \| 16-23 |
| crosses the wire | the prompt's whole KV cache, **once per request** | activation `[n_tok, H]`, **twice per step** |
| bytes, P=512 G=512 B=16 | 512 × 12 KB = 6.1 MB, once | 16×1792 B = 28.7 KB × 2 hops × 512 steps = 29 MB |
| stages | 2 | 3 |
| motivation | prefill (compute-bound) and decode (memory-bound) interfere in one engine | no node holds the whole model |
| **shared requirement** | **one scheduler decides the batch; every stage honours it verbatim** | identical |
| reported | DistServe: 7.4x more requests, or 12.6x tighter SLO, >90% within latency constraints. Splitwise: 1.4x throughput at 20% lower cost, or 2.35x under the same power+cost budget | — |

Same shape: break a monolithic engine into stages, connect them with a network hop carrying tensor state,
add a control plane. **This is why v2 is not "throw away the shards and adopt vLLM" — it is "adopt vLLM V1's
`SchedulerOutput` as our descriptor and run `--pipeline-parallel-size 3"`.** vLLM already broadcasts one
scheduler decision to all PP ranks; that is precisely §4 and §6, already written and tested.

## 9. Scheduler pseudocode

```python
# coordinator.py — replaces the `for _ in range(req.max_tokens - 1)` loop at coordinator.py:122-127
D          = 3                                          # stages == microbatches in flight
T          = int((TPOT_SLO_MS / D - 22.77) / 0.2931)     # token budget/step; 150 @ 200 ms (measured fit)
CHUNK      = 128                                         # measured sweet spot, §7
MAX_SLOTS  = D * B + B                                   # +1 microbatch: EOS reclaim lags 1 traversal

waiting, running, slots, step_id = deque(), {}, FreeList(MAX_SLOTS), 0
just_finished = []                                       # seen at node2 last traversal, evict next step

def schedule() -> BatchDescriptor:                       # ONLY the coordinator ever calls this
    d, budget = BatchDescriptor(step_id, EPOCH), T
    for r in running.values():                           # decodes first — never starve a live sequence
        if budget < 1: break
        d.add(r.id, r.slot, DECODE, n_new=1, ctx_len=r.ctx_len, emit=1); budget -= 1
    while waiting and budget > 0:                        # fill the remainder with prefill chunks
        r = waiting[0]
        n = min(CHUNK, budget, r.n_prompt - r.ctx_len)
        if r.slot is None:
            if not slots.can_alloc(): break              # admission control: KV headroom, §7
            r.slot = slots.alloc()
        last = (r.ctx_len + n == r.n_prompt)
        d.add(r.id, r.slot, PREFILL_CHUNK, n_new=n, ctx_len=r.ctx_len, emit=int(last))
        budget -= n; r.ctx_len += n
        if last: running[r.id] = waiting.popleft()
    d.evict = just_finished; just_finished = []          # all 3 stages free the same ids, same step
    step_id += 1
    return d

async def pipeline():
    for m in range(D): dispatch(schedule(), microbatch=m)          # fill the pipe
    while True:
        done = await stage2_return()                               # {tokens, finished, microbatch_id}
        for rid, tok in done.tokens: running[rid].emit(tok); running[rid].ctx_len += 1
        for rid in done.finished:                                  # EOS decided at node2 only
            just_finished.append(rid); slots.free(running.pop(rid).slot)
        dispatch(schedule(), microbatch=done.microbatch_id)         # refill the freed stage
```

```python
# node.py — /step replaces /forward. The stage EXECUTES the descriptor; it never edits it.
def step(desc: BatchDescriptor, act: bytes):             # act = [desc.n_tok, H] bf16, flat, unpadded
    assert desc.epoch == MY_EPOCH and desc.step_id > last_step_id     # fence: reject stale / forked
    x    = np.frombuffer(act, bf16).reshape(desc.n_tok, H)
    cu_q = cumsum([0] + [s.n_new            for s in desc.slots])     # DERIVED, not shipped (§5)
    cu_k = cumsum([0] + [s.ctx_len + s.n_new for s in desc.slots])
    for s in desc.slots: kv.bind(s.slot, s.block_ids)                 # paged: join/leave never compacts
    y = layers(x, cu_q, cu_k, slots=[s.slot for s in desc.slots])     # varlen attention, no padding
    if IS_LAST:                                                       # node2 only
        rows = [i for i, s in enumerate(desc.slots) if s.emit]        # skip non-final prefill chunks
        return sample(lm_head(y[rows]))                               # 4 B token ids, not 607 KB logits
    for rid in desc.evict: kv.free(rid)
    last_step_id = desc.step_id
    return y                                                          # [desc.n_tok, H], SAME row order
```

## 10. Recommendations

| # | change | impact | effort | v1/v2 |
|---:|---|---|---|---|
| 1 | `/step` + `BatchDescriptor` + per-slot KV cache (a `dict[req_id, DynamicCache]` suffices at B≤32) | 8.07 → 265.4 tok/s at B=16, **32.9x** (modelled from measured stage times) | days | **v1** |
| 2 | Sampler + EOS detection on node2, return token ids | prerequisite for `evict[]`; also 607,744 B → 4 B on that hop (FINDING 2) | hours | **v1** |
| 3 | Flatten to `[n_tok, H]`, never pad; derive `cu_seqlens` per stage | **3.76x** on a ragged prefill batch (measured) | days | **v1** |
| 4 | Chunked prefill, c=128 | TPOT spike 3.86x → **1.46x**; TTFT 518 → 393 ms (measured components) | hours | **v1** |
| 5 | Token budget `T` from the TPOT SLO; decode-first, prefill-fills | turns the SLO into one number; ~60 lines in `coordinator.py` | hours | **v1** |
| 6 | `emit` bit — `lm_head` only on rows that need logits | 16.5 ms saved per non-final chunk step (measured) | hours | **v1** |
| 7 | Rebalance the split (FINDING 1) so stages are equal | 265.4 → 320.2 tok/s, **1.21x** (modelled); one env var | hours | **v1** |
| 8 | Replace the descriptor with vLLM V1 `SchedulerOutput`; run `--pipeline-parallel-size 3` | deletes the hand-rolled scheduler; inherits preemption, priority, prefix caching | months | v2 |
| 9 | PagedAttention block tables (`block_size=16`) per stage | join/leave never compacts KV on 3 machines at once | months | v2 |
| 10 | 2-D disaggregation: phase × layer (DistServe/Splitwise on top of layer sharding) | 7.4x more requests / 12.6x tighter SLO (DistServe, paper) | months | v2 |

## 11. Risks

| risk | mitigation |
|---|---|
| **Descriptor/tensor split-brain** — separate messages let a stage pair descriptor S with tensor S−1: silent corruption, no exception. Highest-severity bug class here. | one framed message, or `step_id` echoed in both and asserted |
| **Coordinator restart replays `step_id`** | `epoch`; a stage rejects a mismatched epoch outright |
| **B=32 < B=16 is noise; B>32 untested** | KV memory binds ≈ B=32/stage at ctx=2048 fp16 in a 2 GB container |
| **Batching worsens the unlucky tail** — a seq sharing a step with a 128-token chunk pays 1.46x | report **p99** TPOT, never the mean |
| **All coefficients are CPU/fp32/2-thread.** On GPU the 22.77 ms fixed cost is 10–100x smaller | the *argument* ports (amortise fixed cost → amortise arithmetic intensity); the *numbers* do not |
| **32.9x is throughput, not latency.** Single-user TTFT is unchanged or slightly worse | say it on the slide before the audience does |
