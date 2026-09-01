---
team: "T3 — Scheduling, Queueing & Compute Utilisation"
agent: T3-A5
topic: Layer-to-node placement, heterogeneous hardware, dynamic rebalancing, multi-pipeline routing
headline: "Layer placement is a shortest-path DP over cut points, and the PoC's 8/8/8 split is already the wrong answer: node2 carries 17.13 layer-equivalents vs 8.00, a 1.539x throughput loss fixable by three env-var edits."
---

# T3-A5 — Placement & Heterogeneity

All arithmetic below is reproducible from `Qwen/Qwen2.5-0.5B-Instruct` `config.json`; consistent with
`01-VERIFIED-FACTS.md` FINDING 1. `(derived)` = arithmetic on verified constants. `(modelled)` = assumption stated inline.

## 0. TL;DR

| # | Finding | Number |
|---|---|---|
| 1 | 8/8/8 is already imbalanced — node2 holds `lm_head` | bottleneck 17.13 eq vs optimal 11.13 eq → **1.539x** (derived) |
| 2 | 8/8/8 is optimal only if node2 is intrinsically faster | needs **f₂ ≥ 2.14x** f₀ (derived) |
| 3 | Optimal split is a 3-line DP over cut points | O(M²N) = 2,028 relaxations for M=26, N=3 (derived) |
| 4 | Re-cutting is the *only* lever for a 2x-slow node | equal split 2.00x slower; DP recovers **1.60x** of it (derived) |
| 5 | Rebalancing raises **throughput**, not single-request latency | Σ stage times is invariant at 131.7 ms (derived) |
| 6 | Node loader allocates the model **twice** | peak **3.95 GB**/container, not the 2 GB the compose comment claims (derived) |
| 7 | Work stealing cannot apply to a pipeline | stealable unit is 15 KB of activation vs **59.6 MB** of weights = 4,000x (derived) |
| 8 | Replicas do **not** fix the 33% utilisation figure | utilisation = min(C/(R·S), 1); more R at fixed C is *worse* (derived) |
| 9 | Decentralisation has a hop-count ceiling | above **~23 ms** per-hop RTT a 2-node chain beats a 3-node one (modelled) |

## 1. Cost model — the model is a sequence of M = 26 indivisible blocks

Not 24. Placement must treat `embed_tokens` and `lm_head` as first-class blocks, which is precisely what the PoC does not do.

| block idx | contents | params | FLOPs/token | **layer-eq** | fp32 MB |
|---|---|---|---|---|---|
| 0 | `embed_tokens` | 136,134,656 | ~0 (row lookup, 3.6 KB read) | **0.00** | 544.5 |
| 1..24 | one decoder layer (attn 1,835,008 + MLP 13,074,432) | 14,909,440 | 29.82 M | **1.00** | 59.6 |
| 25 | `norm` + `lm_head` (896 × 151936) | 136,134,656 | 272.27 M | **9.131** | 544.5 |
| | **total pipeline work** | | | **33.131 eq** | 2,520.4 |

Why `layer-eq` is the right unit: FLOPs = 2 × params and bytes-read = 4 × params for every block here, so the
compute-bound and memory-bandwidth-bound ratios are **identical** — `lm_head` = 9.131 layers either way, and batch-1
CPU decode (bandwidth-bound) does not break the unit. (derived) Meanwhile `embed_tokens` costs 0 FLOPs and 544.5 MB:
**work and memory are not proportional**, which is exactly why llama.cpp's memory-proportional default is the wrong
objective (§7).

Modelled clock: 15 GB/s effective DRAM bandwidth per 2-vCPU container → **1 layer-eq = 3.98 ms/token** (modelled).
Where T3-A2 has *measured* per-stage times, prefer theirs; only the ratios below are load-bearing here.

## 2. The optimisation, and why it is a shortest path

**Given** N nodes in fixed pipeline order, speed `f_i` (layer-eq/s), memory cap `mem_i`, link `(bw, lat)_{i,i+1}`.
**Choose** cuts `0 = c₀ ≤ … ≤ c_N = M`, node i holding blocks `[c_{i-1}, c_i)`, s.t. `Σ mem(b) ≤ mem_i`.
**Minimise** either objective:

| objective | meaning | combine op |
|---|---|---|
| `min max_i t_i` | steady-state **throughput** (a pipeline runs at its slowest stage) | `max` |
| `min Σ_i t_i` | single-request **end-to-end latency** | `+` |

with `t_i = W(c_{i-1}, c_i)/f_i + [i<N] · (H·dtype/bw_{i,i+1} + lat_{i,i+1})`.

**Key structural fact:** contiguity + fixed node order ⇒ the feasible set is exactly the paths in a DAG with vertices
`(cut point b, node index i)` and edge `(a,i-1) → (b,i)` of weight `t_i(a,b)`. Min-sum = ordinary shortest path;
min-max = bottleneck shortest path. Same DP, combine operator swapped.

```
B[0][0] = 0 ;  B[0][b>0] = ∞
B[i][b] = min over a ∈ [0,b] with mem(a,b) ≤ mem_i  of  combine( B[i-1][a] , t_i(a,b) )
answer  = B[N][M]      # backtrack P[i][b] for the cuts
```

O(M²N) time, O(MN) space. M=26, N=3 → **2,028 relaxations, microseconds**. There is no excuse for a static split.

```python
# ponytail: prefix sums + O(M^2 N) DP. Exact for N<=~64 nodes; swap in a greedy pass only if N grows past that.
def place(w, mem, f, cap, obj="minmax"):        # w,mem indexed by block; f,cap by node
    M, N, INF = len(w), len(f), float("inf")
    W = [0.0]*(M+1); MM = [0.0]*(M+1)
    for i in range(M): W[i+1] = W[i]+w[i]; MM[i+1] = MM[i]+mem[i]
    B = [[INF]*(M+1) for _ in range(N+1)]; P = [[None]*(M+1) for _ in range(N+1)]
    B[0][0] = 0.0
    for i in range(1, N+1):
        for b in range(M+1):
            for a in range(b+1):                       # a == b => node i holds nothing (legal, and sometimes optimal)
                if B[i-1][a] == INF or MM[b]-MM[a] > cap[i-1]: continue
                t = (W[b]-W[a])/f[i-1]                 # + comm term; constant per hop, see section 6
                v = max(B[i-1][a], t) if obj == "minmax" else B[i-1][a] + t
                if v < B[i][b]: B[i][b], P[i][b] = v, a
    if B[N][M] == INF: return None                     # memory infeasible
    cuts, b = [M], M
    for i in range(N, 0, -1): b = P[i][b]; cuts.append(b)
    return B[N][M], cuts[::-1]

if __name__ == "__main__":                             # the one check that fails if the DP breaks
    E = 136134656/14909440                             # lm_head in layer-equivalents = 9.1308
    w  = [0.0] + [1.0]*24 + [E]
    mm = [544.5] + [59.6]*24 + [544.5]
    assert place(w, mm, [1,1,1], [1e9]*3)[1] == [0,12,23,26]       # -> layers 0-10 / 11-21 / 22-23+lm_head
    assert abs(place(w, mm, [1,1,1], [1e9]*3)[0] - 11.131) < 1e-3  # vs 17.131 for the current 8/8/8
    assert place(w, mm, [1,.5,1], [1e9]*3)[1] == [0,14,21,26]      # slow middle node -> 13/7/4
    print("ok")
```

## 3. FINDING — the current 8/8/8 split is already imbalanced (homogeneous hardware, no straggler)

`docker-compose.yml` sets `NODE_LAYERS` to `0-8`, `8-16`, `16-24`; all three containers get `cpus: "2"`.
The hardware is identical; the **work** is not.

| shard | contents | layer-eq | share | fp32 MB | ms/token (modelled) |
|---|---|---|---|---|---|
| node0 | `embed_tokens` + layers 0–7 | 8.000 | 24.1% | 1,021.6 | 31.8 |
| node1 | layers 8–15 | 8.000 | 24.1% | 477.1 | 31.8 |
| node2 | layers 16–23 + `norm` + **`lm_head`** | **17.131** | **51.7%** | 1,021.6 | **68.1** |

Bottleneck 17.131 eq. DP optimum **11.131 eq** (`[0,12,23,26]` → layers **0–10 / 11–21 / 22–23 + lm_head**).
Fractional lower bound 33.131/3 = 11.044, so the integer split is within 0.8% of ideal.

> **17.131 / 11.131 = 1.539x throughput, for three env-var edits.** (derived)

Inverting it: 8/8/8 is optimal only when `17.131/f₂ ≤ 8`, i.e. **node2 must be ≥ 2.14x faster** than node0/node1.
Nothing in the compose file makes that true. (derived)

The v1 patch is `docker-compose.yml` only, no code change — `node.py` keys off `start_layer==0` / `end_layer==24`,
both preserved: `NODE_LAYERS` `"0-8"→"0-11"`, `"8-16"→"11-22"`, `"16-24"→"22-24"`. Memory moves with it: node0
1,021.6 → 1,200.6 MB, node1 477.1 → 656.0, node2 1,021.6 → 663.8. Largest shard grows 17.5%, still far under the
constraint that actually binds (§6).

Honest caveat for the deck: `tie_word_embeddings: true`, so node0's `embed_tokens` and node2's `lm_head` are the
**same 136M-param matrix** — 544.5 MB duplicated across two nodes, 21.6% of the 2,520.4 MB deployed footprint.

## 4. Equal layer counts is the wrong split when nodes differ

Pedagogical case first — 24 layers, `lm_head` set aside, node1 at half speed (`f = 1, 0.5, 1`):

| split | t₀ | t₁ | t₂ | bottleneck | throughput |
|---|---|---|---|---|---|
| 8/8/8 (equal counts) | 8.0 | **16.0** | 8.0 | 16.0 | 1.00x |
| homogeneous reference | 8.0 | 8.0 | 8.0 | 8.0 | 2.00x |
| DP optimum 9/5/10 | 9.0 | 10.0 | 10.0 | **10.0** | **1.60x** |

One node 2x slower makes the whole pipeline exactly 2x slower under an equal split — a pipeline runs at the speed of
its slowest stage. Balancing recovers 1.60x of the 2.00x; the residual 1.25x is irreducible (capacity fell 3.0 → 2.5
speed-units; 24/2.5 = 9.6 is the fractional floor).

Full model (`lm_head` included), same `f = (1, 0.5, 1)`:

| split | shards | stage eq | bottleneck | ms/token (modelled) |
|---|---|---|---|---|
| current 8/8/8 | 0–7 / 8–15 / 16–23 | 8.00 / 16.00 / 17.13 | 17.131 | 68.2 |
| DP optimum | **0–12 / 13–19 / 20–23+lm_head** | 13.00 / 14.00 / 13.13 | **14.000** | 55.7 |

**1.224x** — smaller than the homogeneous 1.539x because the straggler now constrains the achievable balance too.
Proof it is optimal: to beat 14 you need `L₀ ≤ 13`, `L₁ ≤ 6`, `L₂ ≤ 4`; that sums to 23 < 24 layers. (derived)

**Demoable heterogeneity, v1:** set `cpus: "1"` on node1 in `docker-compose.yml`. Three identical containers, one
throttled — measure, re-run the DP, re-cut, measure again. That is the whole experiment.

## 5. Two objectives, and the invariance nobody mentions

`Σᵢ tᵢ = 33.131 eq = 131.7 ms` for **every** contiguous split when `f` is uniform — total work is conserved, and the
cut location does not change it.

> **Rebalancing does not make one request faster. It makes the pipeline 1.539x higher-throughput.**
> With one request in flight and no microbatching, the user feels the *sum*, and sees zero improvement.

This is the hard dependency on T3-A2: **rebalancing only pays once ≥ 2 requests (or microbatches) are in flight.**
Ship them together or the demo shows nothing.

Under heterogeneity min-sum becomes non-degenerate and instructive. With `f = (1, 0.5, 1)` and no memory cap it
returns `[0,0,0,26]` — **every block on one fast node, zero hops**: the correct answer to the latency question and the
wrong answer to the project. Only the memory cap forces distribution: at `mem_i ≤ 1400 MB` it becomes
`0–9 / — / 10–23+lm_head`, two nodes, one hop. (derived) The DP will *always* collapse the pipeline unless memory
forces it apart — decentralisation buys memory capacity and trust, and costs latency. Say so on the deck.

## 6. FINDING — the real memory constraint is a loader bug, not the shard

`node.py` calls `AutoModelForCausalLM.from_pretrained(...)` (1,975.8 MB fp32) and *then* `Qwen2ForCausalLM(config)`,
allocating a **second complete randomly-initialised model** (another 1,975.8 MB) before overwriting three attributes
and `del full`.

> **Peak RSS = 3,951.7 MB per container** (derived). The compose comment claims "model load needs ~2GB per node" and
> `memory: 4G` is commented out on all three services. The comment understates by 2x — that is why the limit is off.

So the cap binding the DP is ~3.95 GB per node regardless of shard size; `mem_i` is meaningless until this is fixed.
v1, ~5-line diff: mutate `full` in place rather than build a second model (`full.model.layers = full.model.layers[a:b]`;
`full.model.embed_tokens = torch.nn.Identity()`; …; `model = full`) → peak 3,951.7 → 1,975.8 MB. v2:
`safetensors.safe_open` + `get_slice`, materialising only the shard's tensors → peak = shard = 663.8–1,200.6 MB.
**This is the gate on the multi-pipeline demo (§9).**

**Comm term.** A cut always ships `H × dtype` bytes wherever it falls — **cut *location* does not affect comm, only
cut *count* does.** At bf16: 1,792 B/token/hop = 14.3 µs on 1 GbE + ~0.3 ms RTT ≈ 0.31 ms, i.e. 0.7% of a 44.3 ms
stage. On LAN the comm term is noise and min-max reduces to pure work balancing. Crossover: a 3-node chain
(44.3 ms + 2R) beats a 2-node chain (67.7 ms + R) only while **R < 23.4 ms** per hop (modelled). Above that — real WAN —
the DP drops nodes. Decentralisation over the open internet has a hard node-count ceiling.

## 7. Prior art — this is the standard formulation

| system | what it does | relation to the DP |
|---|---|---|
| **Petals v2.0.0** (bigscience-workshop) | release notes: "shortest-path routing and direct server-to-server communication"; client picks a chain of servers each covering a block range | the min-sum DP, run client-side over a live server DAG, re-run per request |
| **llama.cpp RPC backend** (`tools/rpc`) | default splits weights + KV cache across local and RPC devices **in proportion to available memory**; `--tensor-split 4,3,3` overrides | `--tensor-split` **is** the cut vector, exposed as a manual knob. The memory-proportional default is the wrong objective — §1 shows work ⊥ memory (`embed_tokens`: 544.5 MB, 0 FLOPs) |
| **GPipe / `torch.distributed.pipelining`** | balanced partitioning by profiled per-layer time | same DP, profiled `w(b)` instead of analytic |
| **DeepSpeed-Inference / Megatron-LM** | uniform layers-per-stage, homogeneous-GPU assumption | the assumption the PoC inherited and should not have |
| **AlpaServe / Alpa** | ILP over stage assignment + replication | the v2 generalisation: joint placement **and** replication |

Sources: [Petals v2.0.0 release](https://github.com/bigscience-workshop/petals/releases/tag/v2.0.0.post1) ·
[llama.cpp RPC README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) ·
[llama.cpp multi-GPU docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)

## 8. Dynamic rebalancing, stragglers, and why work stealing does not apply

**Detection.** `node.py` already exports `node_forward_seconds_total` / `node_forward_total`. EWMA of the ratio gives
measured `f_i` for free. Straggler when `t_i > 1.3 × median(t)` over 30 s. Re-run the DP (µs), re-cut only if predicted
gain > 15% — hysteresis, or it oscillates.

**Migration cost, 1 GbE (125 MB/s):**

| moved | fp32 | bf16 | + KV @2048 ctx (512 B/tok/layer) | wall-clock bf16 |
|---|---|---|---|---|
| 1 layer | 59.6 MB | 29.8 MB | +1.05 MB | 0.25 s |
| 8/8/8 → 11/11/2 (9 layers move) | 536.7 MB | 268.4 MB | +9.4 MB | **2.2 s** |

**…except in this PoC it is free.** Every container already `from_pretrained`s the *whole* model at boot and discards
the rest. Keep `full` resident and a re-shard is a Python attribute reassignment — **~50 ms, zero network** (costs
1,975.8 MB RSS, which §6's v2 loader is the alternative to). v1 re-shard = env var + `docker compose up -d`.

**Warm handover** (v2): target loads the range while the incumbent serves → target replays the migrating layers' KV
(1.05 MB/layer @2048 ctx) → cut over **at a token boundary**, since decode is stateless across tokens once the KV
moves → drain, release. Zero downtime: the two cuts coexist for exactly one token.

**Why work stealing does not apply.** Stealing requires relocatable tasks — any idle worker runs any queued task. Here
a queued item is bound to the *weights resident on that node*; an idle peer cannot run node1's backlog because it does
not hold layers 11–21. Stealing one queued token means shipping 59.6 MB × 11 of weights to move 15 KB of activation —
**4,000x the payload of the work itself, per layer.** A pipeline is a fixed *chain*, not a bag of tasks.

**What does apply, in order of effect:**

| mechanism | unit relocated | applies to the PoC? |
|---|---|---|
| **Re-cut** (this DP) | layer ranges, minutes-scale | **yes, v1** — the primary lever |
| **Replica-level load balancing** | a whole *request*, across k nodes holding the **same** shard | v2, needs k>1 per shard (Petals' model) |
| **Microbatch interleaving (1F1B)** | fills the 2-of-3 idle nodes | T3-A2's lever, orthogonal, multiplicative with this one |
| Work stealing | — | **no** |

## 9. Multi-pipeline: P replicas, and the router

Notation, to avoid a collision: T3-A2's `R` is **in-flight concurrency**; I write that **C**. **P** = parallel pipeline
replicas, **S** = 3 stages, so P·S nodes. Work-conserving router:

> **U = min( C / (P·S), 1 )** — and T3-A2's rule `R ≥ S` generalises to **`C ≥ P·S`**.

| P | nodes | C=1 | C=3 | C=9 |
|---|---|---|---|---|
| 1 (today) | 3 | 33% | **100%** | 100% (queued) |
| 2 | 6 | 17% | 50% | 100% |
| 3 | 9 | 11% | 33% | **100%** |

**Replicas do not fix the 33% ceiling — at fixed load they make it worse.** Concurrency, not replication, fills a
pipeline; at P=1 the rule reduces to T3-A2's C ≥ 3. Replicas buy *capacity* and *fault tolerance* (v0 defect #9:
node1 down = total outage), not utilisation. Corollary the deck must not get wrong: routing cannot change U — work
conservation makes breadth-first and depth-first both 3-busy-of-9. Routing changes **tail latency, straggler
avoidance, and fairness**.

**Router policy (v1, ~20 lines in `coordinator.py`):** track per-replica in-flight `C_r` and bottleneck `maxₛ t_{r,s}`
(from `/metrics`, already exported); route to `argmin_r ( C_r × maxₛ t_{r,s} )` — least-outstanding-**time**, which
avoids a straggler-containing replica with no explicit straggler logic; admit only while
`Σ_r C_r < R·S + queue_depth` (v0 defect #8); keep a request's decode loop sticky to its replica, where its KV lives.

**Feasibility gate.** P=2 needs 6 containers. At today's 3,951.7 MB peak that is **23.7 GB** — infeasible on a laptop.
With §6's v1 loader fix, 11.9 GB; with the v2 loader, **5.04 GB** (2 × 2,520.4 MB). Fix the loader, then P=2 demos.

## 10. Recommendations

| # | tag | change | measured/modelled impact | effort |
|---|---|---|---|---|
| 1 | **v1** | `NODE_LAYERS` → `0-11` / `11-22` / `22-24` in `docker-compose.yml` | bottleneck 17.131 → 11.131 eq = **1.539x throughput** (derived); zero single-request latency change | **minutes** |
| 2 | **v1** | Loader: mutate `full` in place, drop `Qwen2ForCausalLM(config)` | peak RSS 3,951.7 → 1,975.8 MB/node, unblocks `memory:` limits (derived) | hours |
| 3 | **v1** | 30-line `place()` DP + `/metrics`-derived `f_i`, printed at coordinator boot | makes the split explainable on stage; runs in µs | hours |
| 4 | **v1** | Demo: `cpus: "1"` on node1, show 68.2 → 55.7 ms/token after re-cut | **1.224x** under an induced straggler (modelled) | hours |
| 5 | **v1** | Router: least-outstanding-*time* across P replicas, sticky KV, admission cap | fixes v0 defects #8/#9; needs rec. 2 first | days |
| 6 | **v2** | Warm handover: shadow-load + KV replay + token-boundary cutover | re-shard with zero downtime; 2.2 s of transfer at bf16/1 GbE (derived) | weeks |
| 7 | **v2** | `safetensors.safe_open` shard-only loading | peak RSS = shard = 663.8–1,200.6 MB; R=2 fits in 5.04 GB (derived) | weeks |
| 8 | **v2** | Joint placement + replication ILP (AlpaServe-style), continuous re-solve | handles churn/heterogeneity at N ≫ 3 | months |

**Risks.** (a) Rec. 1 shows nothing without T3-A2's concurrency — pair them or the demo flatlines. (b) `f_i` from
`/metrics` is noisy at low request rates; EWMA + 15% hysteresis or the split will thrash. (c) `tie_word_embeddings`
means 544.5 MB is duplicated on node0 and node2 — do not claim "no node holds a duplicated weight". (d) The 15 GB/s
bandwidth figure is modelled; every ms/token number scales linearly with it, so re-measure before it reaches a slide.
