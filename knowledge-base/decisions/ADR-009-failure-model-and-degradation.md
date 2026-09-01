---
id: ADR-009
title: Failure model, replication and graceful degradation
status: v1 accepted
date: 2026-09-01
sources: teams/T1-A5, T3-A1, T4-A1, T4-A5, T5-A2, T3-A5
---

# ADR-009 — Failure model: detect in 1.5 s, re-shard onto survivors, resume the same completion

## Context

v0 has **no control plane at all**: three hardcoded env URLs (`coordinator.py` r0/r1/r2 blocks), zero
heartbeats, `timeout=60` so a *hung* node hangs the request for 60 s, and node1 down = HTTP 500. The shipped
`demo.sh` step 5 is literally titled "Failover Demo" and demonstrates the outage.

Two live bugs found by reading source:
- **`gateway/app.py` B1:** the circuit breaker sets `circuit_state['failures'] = 0` unconditionally after any
  successful HTTP exchange with the coordinator, so **a coordinator 500 caused by a dead node RESETS the
  breaker instead of tripping it.** It never trips on the exact failure it was built for. 2-line fix.
- Default `CIRCUIT_FAILURE_THRESHOLD=3` / `COOLDOWN=30 s` means one slow cold-CPU first request kills the
  demo for 30 seconds. Set 10 / 5 for the demo.

**The free lunch, and its expiry date.** Because v0 re-sends the entire `gen_ids` every token (the O(n²)
defect), **there is no distributed state to lose** — failover costs exactly one extra forward pass. ADR-001's
KV cache is what *creates* the failover problem. Sequence the roadmap accordingly, and say it on stage: our
worst performance bug is currently our best failover feature.

Recovery arithmetic once a cache exists (modelled): a node dying at position n costs another n
position-forwards — it **exactly doubles that session's compute**. At n=543 that is 536.4 GFLOP ≈ 10.7 s and
5.19 MB of wire. Cached-with-one-failure is still **135x** better than v0. A 3.67 MB boundary-activation
journal (n × 896 × 4 B × 2 boundaries at n=512, ~5 LOC) buys that back; without it, recomputing a dead 8-layer
shard is 122 GFLOP = **6.1 s** on 2 CPU cores.

## Options considered

| rung | mechanism | extra cost | recovery | verdict |
|---|---|---|---|---|
| 0 | today: 500 to the client | 0 | never | rejected |
| 1 | **Self-registering control plane + dynamic re-shard onto survivors.** `POST /register` with the layer range each node already exposes on `/health`, plus a state field `{LOADING, READY, DRAINING, DEAD}`; coordinator polls every 500 ms and computes the chain by greedy cover of [0,24); `POST /load_layers {"range":[8,12]}` parameterises the existing `load_model()` | +4 layers × 59.65 MB = **+238.6 MB per survivor**; 3–8 s reload from warm HF cache | in-flight request **survives**, replaying `gen_ids` | **ACCEPTED v1**, ~63 net LOC (+81 added, −20 deleted from the hardcoded r0/r1/r2 blocks) |
| 2 | **Boundary-activation journal** — retain the shard-boundary hidden states already in flight | 3.67 MB per in-flight request at n=512, 14.7 MB at n=2048, ~5 LOC | ~0 beyond rung 1 | **v1.5**, mandatory once ADR-001 lands |
| 3 | Hot standby replica of the highest-risk shard | +1 node (477.2 MB) | ≤1.5 s, no reload | v2 |
| 4 | N+1 chained shard replication (4 nodes × 6 own + 6 shadow = 50% of the model each) | 2× layer weights = 2,863 MB fleet-wide | ≤1.5 s | v2 |
| — | Phi-accrual failure detection (Hayashibara SRDS 2004; φ=8 is the default in both Cassandra and Akka) | | | **v2 only.** At 3 nodes on a wired LAN a `LOADING` state field solves the same problem for one field. |
| — | etcd 3.6.6 / Consul 1.22.7 registry | | | **v2 proposed** — not because 3 nodes need consensus, but because it makes the coordinator stateless so two can run behind the gateway. |
| — | SWIM gossip (`hashicorp/memberlist` v0.5.x) | | | **rejected below ~50 nodes.** Gossip beats O(N²) polling only above ~n=50; we have n=3, where 6 probes/s of direct polling is strictly better. Also needs a Go sidecar — no maintained Python port. |

## Decision

1. **Self-registering control plane** replacing the three hardcoded `NODE*_URL` env vars: 500 ms poll,
   3 misses = down. Detection **60 s → 1.5 s** on a hung node. Overhead 3 × 2 Hz × ~200 B = **1.2 KB/s =
   0.001% of 1 GbE**. Net **−20 LOC in `coordinator.py`** despite adding the feature.
2. **A `LOADING` state is not optional polish.** A naive 1.5 s fixed timeout will declare a *re-sharding* node
   dead while it loads weights, cascading into a second reshard and looping the demo.
3. **Dynamic re-shard on failure**, resuming the *same* `gen_ids`. Requires `HF_HOME` on a **named docker
   volume** so the reload is from warm cache.
4. **Fix `gateway/app.py` B1** — treat coordinator 5xx as a breaker failure. Set
   `CIRCUIT_FAILURE_THRESHOLD=10`, `CIRCUIT_COOLDOWN_SEC=5` for the demo.
5. **The journal lives on the sending node, not the coordinator.** ADR-002 moves the data plane to chain
   routing, so `node_i` retains the `h` it sent. Same 3.67 MB, same ~5 LOC, different owner. This resolves the
   tension T1-A5 flagged — decide it here, before the wire format freezes.
6. **Replace `demo.sh` step 5** with the scripted kill-node1-mid-generation run, emitting SSE
   `node_down` / `resharding` / `reshard_done`, resuming the same completion, finishing HTTP **200** with
   `degraded: true`, then `docker compose start node1` and healing back to three nodes. Ship a
   `DLLM_FAILOVER=off` negative control and run it **first**.
7. Recovery from a KV `409 cache_miss` and recovery from a node crash are the **same event**; use one code path.

## Consequences

**Good.** A judge-verifiable claim that needs no trust: the completion is coherent end to end, the status is
200 not 503, `docker ps` shows two containers mid-demo. Detection 60 s → 1.5 s. ~63 net LOC.

**Bad.**
- **The reshard pause (3–8 s) is dead air on stage unless the SSE stream narrates it.** If the HF cache is
  cold (no `HF_HOME` volume), the node re-downloads and the demo stalls indefinitely. **This is the single
  highest-probability demo failure in the project.**
- **The coordinator is a SPOF exactly as total as any node, and no rung of this ladder addresses it.** If a
  judge asks "what if I kill the coordinator", the honest answer is "total outage; v2 fixes it with a
  stateless coordinator behind etcd".
- **Session affinity (ADR-001) makes a crash blast radius much larger**: every session pinned to that node
  dies, not one in-flight token. Rung 1 recovers the *chain*; the sessions still replay.
- Rung 1 costs +238.6 MB per survivor, which is only affordable after ADR-011 fixes the ~4 GB loader peak.
- The 1.5 s detection budget assumes ADR-002's `PING`/`PONG` heartbeat; without it, liveness is
  indistinguishable from idleness and a hung node still hangs for 60 s.
- Verification of *correctness* (did node1 compute honestly?) is out of scope here and belongs to ADR-010.
  Naive redundant execution false-positives across heterogeneous BLAS kernels — compare under tolerance.

## Status

**v1 accepted** (rungs 1 and 2, breaker fix, demo script). **v2 proposed:** hot standby and N+1 chained
replication, phi-accrual detection, etcd-backed stateless coordinator, latency-weighted Dijkstra chain
selection over the layer-boundary DAG (Petals' approach, V ≤ 25, edge weight `EWMA(latency) + α·queue_depth`),
and cache migration / peer checkpointing (4.45 MB per shard per 543-token session).
