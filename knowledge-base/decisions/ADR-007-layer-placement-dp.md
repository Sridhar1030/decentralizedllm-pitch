---
id: ADR-007
title: Layer placement — cost-aware DP split, not equal layer counts
status: v1 accepted
date: 2026-09-01
sources: teams/T3-A5, T3-A2, T3-A4, T4-A1, T4-A3, 01-VERIFIED-FACTS F1
---

# ADR-007 — Placement is a shortest path, not a heuristic

## Context

The model is **26 indivisible blocks, not 24**: `embed_tokens` (0 FLOPs, 544.5 MB), 24 layers
(14,909,440 params each), and `norm + lm_head` (136,134,656 params). Because FLOPs = 2×params *and*
bytes-read = 4×params for every block, the compute-bound and bandwidth-bound cost ratios are identical, so a
single unit — the **layer-equivalent** — survives batch-1 CPU decode.

> **`lm_head` = 9.131 layer-equivalents.** Total pipeline work = 33.131 eq.

| shard | contents | layer-eq | share |
|---|---|---:|---:|
| node0 | embed + layers 0-7 | 8.000 | 24.1% |
| node1 | layers 8-15 | 8.000 | 24.1% |
| node2 | layers 16-23 + norm + **lm_head** | **17.131** | **51.7%** |

The 8/8/8 split is therefore already wrong **on identical hardware**. DP optimum is 11.131 eq at cuts
`[0,12,23,26]` → **1.539x throughput**, and the fractional lower bound is 11.044, so the integer split is
within 0.8% of ideal. 8/8/8 would only be optimal if node2 were intrinsically ≥ **2.141x faster** than the
others; nothing in `docker-compose.yml` makes that true.

Two figures circulate for the same fact and must not be conflated: **1.539x is the layer-equivalent
(analytic) ratio**; **1.30x is the measured wall-clock imbalance** (308.97 / 237.51 ms). Use 1.30x for any
wall-clock claim, 1.55x/1.539x for the FLOP/placement claim.

The critical property: **the sum of stage times is invariant across every contiguous split at uniform speed**
(33.131 eq = 131.7 ms modelled). Rebalancing raises throughput 1.539x and changes single-request latency by
**exactly zero**.

## Options considered

| option | verdict | why |
|---|---|---|
| Equal layer counts (8/8/8, today) | **rejected** | Ignores that one block is worth 9.13 layers. |
| Memory-proportional split (llama.cpp `--tensor-split` default) | **rejected** | Wrong objective: `embed_tokens` is 544.5 MB of **zero FLOPs**. Optimising bytes places 545 MB of lookup table as if it were work. |
| **Min-max DP over cut points, cost = layer-equivalents, weighted by measured per-node speed `f_i`** | **ACCEPTED v1** | Contiguous shards over a fixed node order form a DAG whose vertices are cut points; min-max is bottleneck shortest path. O(M²N) = **2,028 relaxations** for M=26, N=3 — microseconds. 30 lines. Petals v2.0.0 ships shortest-path routing; llama.cpp exposes the same cut vector as a manual knob. |
| Min-**sum** DP | **rejected as the objective** | It collapses the pipeline onto one node whenever memory permits. Distribution is never latency-optimal — own that in the pitch rather than hiding it. |
| Work stealing between stages | **impossible** | Relocating one queued token means shipping **59.6 MB of weights per layer to move 15 KB of activation state — 4,000x the payload of the work itself.** The stealable unit is a whole request across *replicas* of the same shard (Petals' model), never a queued token across different shards. |
| Static split forever | rejected | `f_i` is available for free from the `node_forward_seconds_total` / `node_forward_total` counters `node.py` already exports. |
| More nodes | **bounded** | A 3-node chain (44.3 ms + 2R) beats a 2-node chain (67.7 ms + R) only while per-hop RTT R < **23.4 ms** (modelled). Above that the optimiser actively drops nodes. |

## Decision

1. **Re-split to `NODE_LAYERS = "0-11" / "11-22" / "22-24"`** in `docker-compose.yml`. Pure env-var edit — no
   code change, since `node.py` already keys off `start_layer == 0` and `end_layer == 24`, both preserved.
2. **Add the 30-line `place()` min-max DP to the coordinator**, fed by `f_i` derived from the exported
   timing counters, and print the chosen cuts at boot. Placement becomes explainable live rather than asserted.
3. **Ship it together with ADR-006's concurrency change.** Alone it is invisible: the sum of stage times is
   invariant, so at R=1 a rebalance shows nothing and the 1.539x claim looks fabricated on stage.
4. Heterogeneity demo: set `cpus: "1"` on node1, measure, re-run the DP, re-cut to `0-13 / 13-20 / 20-24`,
   measure again. Bottleneck 17.131 → **14.000 eq = 1.224x** — and it is provably optimal (beating 14 needs
   L0≤13, L1≤6, L2≤4, summing to 23 < 24 layers). The pedagogical 24-layer-only version is cleaner: an equal
   split is exactly **2.00x** slower and the DP recovers **1.60x**, with 1.25x irreducible.
5. A re-cut needs **15% hysteresis over a 30 s EWMA window** or the split oscillates on noise.

## Consequences

**Good.** 1.539x throughput for three env-var edits; N\* (ADR-005) becomes exactly 3.00 = the node count,
which makes the whole scheduler one integer; and the same DP reproduces vLLM's own imbalance — naive
`--pipeline-parallel-size 3` gives 8/8/17.13 too, fixed by `VLLM_PP_LAYER_PARTITION="11,11,2"` (ADR-008).

**Bad.**
- **It shows nothing at concurrency 1.** Pair with ADR-006 or the live demo flatlines.
- **Moving the cut invalidates every fitted artifact** — PCA bases, outlier channel indices, learned codecs
  (ADR-003). Finalise the split *before* fitting anything.
- **Compute-balancing makes parameter spread worse**, not better: node0 goes to 300,138,496 params = **60.8%**
  under the 0-10/11-21/22-23 variant. The fix is to move embeddings and `lm_head` off the pipeline (ADR-011),
  which simultaneously equalises parameters at 24.1% each *and* makes 8/8/8 compute-balanced again.
- The `15 GB/s` DRAM bandwidth used for every ms/token figure here is **modelled**, and those figures scale
  linearly with it. The **ratios** (1.539x, 1.224x, 2.141x) are derived and safe; the absolute milliseconds
  are not. Where T3-A2/T3-A4 have measured stage times, those supersede.
- `f_i` estimated from `/metrics` is noisy at low request rates; without the hysteresis band the DP will
  re-cut on noise and thrash weights across the network.
- Rebalancing a *live* fleet costs 9 layers = 536.7 MB fp32 / 268.4 MB bf16 + 9.4 MB KV @2048 ctx ≈ **2.2 s
  on 1 GbE**. In this PoC it is ~50 ms and zero network only because every container already loads the whole
  model — an artifact of the defect ADR-011 fixes.

## Status

**v1 accepted.** **v2 proposed:** warm handover for zero-downtime re-sharding (shadow-load the range, replay
the migrating layers' KV at 1.05 MB/layer @2048 ctx, cut over at a token boundary since decode is stateless
across tokens once the KV moves); shard-only `safetensors.safe_open` loading so the DP's `mem_i` constraint
means something; joint placement-and-replication ILP (AlpaServe/Alpa) once node order becomes a decision
variable; a continuous straggler control loop at `t_i > 1.3x` median.
