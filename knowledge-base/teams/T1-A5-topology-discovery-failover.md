---
team: T1 — Transport & Protocol
agent: T1-A5
topic: Control plane — membership, chain routing, failure detection, failover ladder, trust model
headline: >
  v0 has no control plane — three hardcoded env URLs, zero heartbeats, and a circuit breaker that is
  *reset* by node death. ~63 net LOC buys a self-registering chain that survives `docker compose stop node1`
  with HTTP 200 and one token of added latency. Separately: the pitch's headline claim is false today —
  every node calls `AutoModelForCausalLM.from_pretrained(MODEL_NAME)` and holds the entire checkpoint at boot.
---

# T1-A5 — Control plane: discovery, routing, failover, trust

## 0. What exists today (read from source, not assumed)

| Concern | v0 reality | File |
|---|---|---|
| Membership | 3 env vars `NODE0_URL/NODE1_URL/NODE2_URL`, baked at compose time | `layer-nodes/coordinator.py:19-21` |
| Layer→node map | Implicit in variable *names*. Coordinator never learns a node's range | `coordinator.py:44-70` |
| Node self-knowledge | `LAYER_RANGE` env; `/health` already returns `{"layers":"8-16"}` | `layer-nodes/node.py:16,110` |
| Heartbeat | None. Only Docker `healthcheck` (5 s interval) — compose uses it for *startup ordering*, nobody reads it at runtime | `docker-compose.yml` |
| Failure detection | The failing request itself. `timeout=60` per hop | `coordinator.py:46` |
| Failover | None. Node down ⇒ model incomplete ⇒ outage | shared-context defect 9 |
| Auth node↔node | None. Plain HTTP on the `decentralized-net` bridge, no token, no TLS | all |
| Auth client→gateway | One shared static key `"sridhar-intern-2026"` in source | `gateway/app.py:15` |

### Two live bugs found while reading

**B1 — the circuit breaker is disarmed by exactly the failure it exists for.** `gateway/app.py` does
`resp = await client.post(...)` then unconditionally `circuit_state["failures"] = 0` before returning
`JSONResponse(status_code=resp.status_code, ...)`. A dead node makes the *coordinator* return 500 — a
successful HTTP exchange — so the breaker counter is **reset**, not incremented. It only ever trips when
the coordinator process itself is unreachable. 2-line fix.

**B2 — `demo.sh` step 5 is titled "Failover Demo" and demonstrates the opposite.** It stops node1 and
prints the error; today's demo ends on the outage. §6 replaces it.

---

## 1. Weight arithmetic (needed by every later section) — (modelled from `config.json`)

Qwen2.5-0.5B-Instruct: H=896, L=24, V=151936, intermediate=4864, 14 q-heads / 2 kv-heads / head_dim 64,
`tie_word_embeddings: true`.

| Tensor group | Params | fp32 bytes |
|---|---|---|
| `embed_tokens` = 151936×896 | 136,134,656 | 544.5 MB |
| attn/layer (q+b, k+b, v+b, o) + mlp/layer (3 × 896×4864) + 2 RMSNorm | 1,836,160 + 13,074,432 + 1,792 = **14,912,384** | **59.65 MB** |
| 24 layers | 357,897,216 | 1431.6 MB |
| **total** | **494,031,872** (matches the published 0.49 B) | **1976.1 MB** |

Shard footprints (fp32, tied `lm_head` ⇒ node2 also carries the 544.5 MB embedding matrix):

| Node | Holds | MB | % of model |
|---|---|---|---|
| node0 | embed + L0-7 | 1021.7 | 51.7% |
| node1 | L8-15 | 477.2 | 24.1% |
| node2 | L16-23 + norm + lm_head(tied) | 1021.7 | 51.7% |
| **fleet total** | | **2520.6** | **127.6%** — 27.6% replication overhead, all of it the embedding matrix |

**Consequence for the pitch:** two of three nodes hold 51.7% of the model each — the single largest tensor
(27.6% of all params) sits on both ends of the chain. "No node holds the full model" is true at 51.7%, not
33%. Say the real number on stage; a judge who reads `config.json` will find it.

---

## 2. Membership / registry — pick one

Crossover first: all-to-all polling is O(n²) messages; SWIM is O(n)/round with O(log n) rounds to
dissemination. At n=3 that is **6 probes/s** vs the operational cost of a gossip library. Gossip starts
winning ≈ n > 50 (modelled). We have 3.

| Option | Version | Integration cost | New infra | Verdict |
|---|---|---|---|---|
| Coordinator-held table + `POST /register` + async poller | — | ~55 LOC in `coordinator.py` | none | **v1 — ship this** |
| etcd | 3.6.6 (Nov 2025) | ~120 LOC + lease/TTL + watch | 1 container (3 for real quorum) | **v2** |
| HashiCorp Consul | 1.22.7 (Apr 2026) | ~100 LOC + agent per node | agent sidecars | v2 alt — only if you also want Consul Connect for mTLS |
| SWIM via `hashicorp/memberlist` | v0.5.x, Go | no maintained Python port; needs a Go sidecar | sidecar per node | v2 only, and only above ~50 nodes |

**The v2 case for etcd is not membership — it is that the coordinator is an undiscussed SPOF.** Killing
node1 is the demo; killing the coordinator is an equally total outage and no rung of §5's ladder fixes it.
Moving the registry + routing table into etcd 3.6.6 makes the coordinator stateless, so you run two behind
the gateway. That, not "3 nodes need consensus", is the argument.

v1 registration payload — nodes already compute every field:

```json
{"node_id":"node1","url":"http://node1:8002","layers":[8,16],
 "state":"READY","dtype":"fp32","epoch":3}
```

`state ∈ {LOADING, READY, DRAINING, DEAD}`. `epoch` bumps on every re-shard so stale routes are rejected.

---

## 3. Routing: how the coordinator computes the chain

Model the fleet as a DAG on layer boundaries `0…24`. A node advertising `[s,e)` is an edge `s → e`.
A valid pipeline is any path `0 → 24`. Cover of `[0,24)` = path existence.

| | Algorithm | Cost | When |
|---|---|---|---|
| v1 | sort READY nodes by `s`, greedy walk from 0, take the node with largest `e` at each frontier | O(n log n), n=3 ⇒ µs | ship this |
| v2 | Dijkstra with edge weight = EWMA(node latency) + α·queue_depth | O(E log V), V≤L+1=25 | multi-replica, heterogeneous hardware |

v2 weighting is Petals' approach (Borzunov et al., ACL 2023 demo, arXiv:2209.01188) — servers chosen to
minimise total chain latency, not statically. Recompute on registry change, not per token; cache under
`epoch`. If no path `0→24` exists ⇒ structured `503 {"error":"layers 8-16 uncovered"}` — **naming the
missing range** — instead of today's opaque `Internal Server Error`.

---

## 4. Failure detection: phi-accrual vs fixed timeout

| | Fixed timeout | Phi-accrual (Hayashibara et al., SRDS 2004) |
|---|---|---|
| Config | interval 500 ms, dead after 3 misses ⇒ **1.5 s** | φ threshold 8 ⇒ ~10⁻⁸ chance the beat is merely late (the default in both Apache Cassandra `phi_convict_threshold` and Akka Cluster) |
| Fits | wired LAN, σ(RTT) ≈ 0.1–0.2 ms, homogeneous | WAN / heterogeneous / GC-pausing peers |
| Failure mode here | declares a *re-sharding* node dead while it loads weights (seconds) | absorbs it via `acceptable-heartbeat-pause` |
| Verdict | **v1** | **v2** (needed once nodes are third-party / WAN, Petals-style) |

**The pause problem is not solved by a smarter detector — it is solved by a state field.** A node loading
layers answers `/health` with `state: "LOADING"`: not DEAD, not routable. One registry field removes the
only reason we would need phi-accrual at 3 nodes. Don't import a failure detector to paper over a missing
state machine. Heartbeat overhead at v1 settings: 3 × 2 Hz × ~200 B = **1.2 KB/s = 0.001% of 1 GbE** (modelled).

Detection latency today, for contrast (modelled from source):

| Failure mode | v0 detection time | v1 |
|---|---|---|
| `docker compose stop node1` (container removed from embedded DNS) | ms — but surfaces as an unhandled `httpx.ConnectError` ⇒ 500 | ≤1.5 s, routed around |
| Process hung / `docker compose pause` | **60 s** (`timeout=60`, `coordinator.py:46`) | ≤1.5 s |
| Node slow but alive | never detected | EWMA in the routing weight (v2) |

---

## 5. The failover ladder — what it costs, in order

**The free lunch: v0's worst performance defect is currently its recovery mechanism.** Because
`coordinator.py` re-sends the entire `gen_ids` list through all three nodes every token (defect 1, O(n²)),
**there is no distributed state to lose** — failover costs one extra full forward = **one token of added
latency**. The moment T2 lands a KV cache that stops being true: the cache *creates* the failover problem.
Sequence the roadmap accordingly.

| Rung | Mechanism | Extra RAM / nodes | Recovery | In-flight request | Tag |
|---|---|---|---|---|---|
| 0 | Today: 500 to the client | 0 | never | lost | — |
| 1 | **Dynamic re-shard onto survivors.** Coordinator recomputes cover, tells node0 `0-12` and node2 `12-24`; each loads 4 layers from its local HF cache | +4 layers × 59.65 MB = **+238.6 MB per survivor** | 3–8 s (modelled: safetensors read + fp32 upcast of 4 layers) | **survives** — replay `gen_ids` | **v1** |
| 2 | **Boundary-activation journal.** Coordinator already relays every boundary `h`; just keep it. `n × 896 × 4 B × 2 boundaries` = 3.67 MB at n=512, 14.7 MB at n=2048. On failure, recompute only the dead shard's 8 layers from the stored input | +3.7 MB per in-flight request, ~5 LOC | ~0 beyond rung-1 reshard | survives *with* a KV cache | **v1.5** |
| 3 | **Hot standby replica** of the highest-risk shard | +1 node (477.2 MB) | ≤1.5 s, no reload | survives | v2 |
| 4 | **N+1 chained shard replication** — 4 nodes × 6 own + 6 shadow layers = 12 layers = 50% of the model per node | 2× layer weights = 2863 MB fleet-wide | ≤1.5 s | survives | v2 |

**Tension to flag:** rung 2 assumes the coordinator stays in the data path; T1's thesis is getting it *out*.
If nodes talk peer-to-peer the journal moves to the sender (`node_i` keeps the `h` it sent) — same 3.67 MB,
same ~5 LOC, different owner. Decide before T1-A1/A2 freeze the wire format.

Recompute cost without the journal, with a KV cache (modelled): 8 layers × 2 × 14.91 M = 238.6 MFLOP/token;
n=512 ⇒ 122 GFLOP; 2 cores at ~20 GFLOP/s fp32 ⇒ **6.1 s**. The 3.67 MB journal buys that back.

---

## 6. v1 failover demo script — what the judge watches

Replaces `demo.sh` step 5, which currently demonstrates the outage.

```
=== 5. Failover: node1 (layers 8-16) is killed MID-GENERATION ===
[ 0.0s] POST /v1/chat/completions/stream  max_tokens=64
[ 0.0s] chain: node0[0-8] -> node1[8-16] -> node2[16-24]
[ 2.1s] tokens 0-7 stream normally, UI lights all three nodes
[ 2.2s] $ docker compose stop node1        <-- run live, on stage
[ 3.6s] SSE {"event":"node_down","node":"node1","layers":[8,16],"detected_ms":1400}
[ 3.6s] SSE {"event":"resharding","plan":{"node0":[0,12],"node2":[12,24]},"epoch":4}
[ 9.2s] SSE {"event":"reshard_done","seconds":5.6,"chain":"node0[0-12] -> node2[12-24]"}
[ 9.4s] SSE {"event":"token","idx":8,...}   <-- generation RESUMES, same completion
[..   ] finishes. HTTP 200. {"finish_reason":"stop","degraded":true,"nodes":2}
[ 40s ] $ docker compose start node1 ; node1 re-registers ; chain shrinks back to 3 nodes
```

Judge-verifiable without trusting us: the completion is coherent end to end, the status is 200 not 503,
and `docker ps` shows two containers through the middle. Line for the reshard pause: *"the pipeline just
got 1.5× slower and stayed correct — that is graceful degradation. Watch it heal."*

Implementation, honest LOC against current source:

| Change | File | LOC |
|---|---|---|
| `POST /load_layers {"range":[8,12]}` — parameterise the existing `load_model()` | `node.py` | ~15 |
| `state` field + `LOADING` guard on `/forward` | `node.py` | ~6 |
| `POST /register` + `_registry` dict + 500 ms asyncio poller | `coordinator.py` | ~40 |
| `build_chain(alive)` greedy cover + `epoch` | `coordinator.py` | ~12 |
| `forward_chain` iterates the chain — **deletes** the hardcoded `r0/r1/r2` blocks | `coordinator.py` | −20 |
| New SSE events `node_down` / `resharding` / `reshard_done` | `coordinator.py` | ~8 |
| **B1**: count coordinator 5xx as a breaker failure | `gateway/app.py` | 2 |
| **Net** | | **~63** |

Pre-req: keep the HF cache warm (`HF_HOME` on a named volume) or the reshard re-downloads and the demo
stalls. One volume mount in `docker-compose.yml`.

---

## 7. Security — because "decentralized" invites the question

### 7.1 Hidden states are not plaintext, and are not private either

| Result | Source | Number |
|---|---|---|
| Text recovered from a *sentence embedding* | Morris et al., *Text Embeddings Reveal (Almost) As Much As Text*, EMNLP 2023, arXiv:2310.06816 (vec2text) | **92% of 32-token inputs recovered exactly**, BLEU 97.3 |
| Prompt recovered from *next-token probabilities alone* | Morris et al., *Language Model Inversion*, ICLR 2024, arXiv:2311.13647 | Llama-2-7b: BLEU 59, token-F1 78, **27% of prompts exact** |
| Tokens recovered from a *split-inference activation*, by split depth | Cunningham, *Privacy-Aware Split Inference with Speculative Decoding…*, arXiv:2602.16760 (2026) | **~59% at a 2-layer split, ~35% at an 8-layer split** |
| Prior art acknowledging exactly this risk | Borzunov et al., *Petals*, ACL 2023 demo, arXiv:2209.01188 | "peers serving the first layers can use their inputs to recover input tokens"; recommends trusted servers or a private swarm |

### 7.2 Who sees what, in *our* topology

| Party | Receives | Leakage |
|---|---|---|
| coordinator | tokenizes the prompt itself + full logits every token | **total.** It has the plaintext |
| node0 | `{"input_ids": [...]}` — **raw token ids, not hidden states** (`coordinator.py:46`) | **total.** Hop 0 is plaintext |
| node1 | `h` after 8 layers | ~35% of tokens by analogy to arXiv:2602.16760's 8-layer split (modelled — different model, treat as order-of-magnitude) |
| node2 | `h` after 16 layers; emits logits; holds the tied embedding matrix | lower on input; owns the output distribution |

**The single worst object on the wire is the logits tensor.** `node.py:104` returns `out.logits[:, -1, :]`
— 151936 × 4 B = **607,744 B raw, ~810 KB after base64, per token**. At seq_len 50 the hidden state is
179,200 B (239 KB b64), so **the logits message is 3.4× larger than the activation message** and the
largest payload in the system: 25.9 MB per 32-token completion, 0.21 s of wire time on 1 GbE. It is also
precisely the oracle arXiv:2311.13647 inverts.

`coordinator.py` uses it only for `np.argmax`. **Fix (v1, ~2 lines): node2 returns the token id.**
607,744 B → 4 B = **151,936× reduction**, and the inversion oracle disappears. For sampling, top-k=50 as
(id, logit) pairs = 400 B — still 1,520×. Biggest bandwidth win, a latency win, and a security fix in one
diff. Perf framing belongs to T1-A1/A2; it is here because the security case makes it non-negotiable.

### 7.3 Transport & identity

| | Mechanism | Cost | Tag |
|---|---|---|---|
| v1 | uvicorn native mTLS: `--ssl-certfile --ssl-keyfile --ssl-ca-certs --ssl-cert-reqs 2`, certs from a 15-line `openssl` script, CN = node id | ~free **once connection reuse is fixed** (see below) | **v1** |
| v1 | `X-Node-Id` + HMAC over the body with a per-node key, if TLS termination is a hassle | ~10 LOC | v1 fallback |
| v2 | SPIFFE/SPIRE X.509 SVIDs, ~1 h rotation; identity drives *authorization* ("node1 may serve layers 8-16 and nothing else") | SPIRE server + agent per node | v2 |
| v2 | WireGuard mesh — kernel-level, app code untouched, ~0.03 ms added | 1 peer stanza per node | v2 alt |

**mTLS is free only after shared-context defect 5 is fixed.** Today `async with httpx.AsyncClient()` is
built inside every forward call ⇒ a fresh TCP connection per hop per token. TLS there costs a 2-RTT
handshake + asymmetric crypto ≈ 2 ms × 3 hops × 32 tokens = **~0.2 s per completion** (modelled). With one
module-level `AsyncClient` + keepalive: 3 handshakes total, then AES-GCM at GB/s on 3.5 KB payloads —
unmeasurable. Sequence: connection pooling first, then mTLS.

### 7.4 The claim that does not survive contact with the source

`node.py:36` — **every node calls `AutoModelForCausalLM.from_pretrained(MODEL_NAME)` and materialises the
complete checkpoint**, then slices and `del full`. Worse, `Qwen2ForCausalLM(config)` first allocates a
randomly-initialised *full* 24-layer model, so peak RSS ≈ 2 × 1976 MB ≈ **3.95 GB per node** — which is
what the commented-out `memory: 4G` in `docker-compose.yml` is quietly telling you. At boot each node
holds two copies of the entire model in RAM. Where nodes belong to different parties, that is the whole
ballgame: shard isolation is a runtime convention, not an enforced property.

Fix (**v1**, ~40 LOC offline): pre-slice into `shard{0,1,2}.safetensors` via `safetensors.torch.save_file`,
ship only the shard, load with `load_file` into a `config` whose `num_hidden_layers` is that shard's count.
Kills three problems — the trust claim becomes enforceable, peak RSS ~3.95 GB → ~1.1 GB, and cold start
stops re-parsing 1.98 GB per node. Honest residual: with `tie_word_embeddings: true`, node0 and node2 both
legitimately need the 544.5 MB embedding matrix (§1); sharding cannot remove that.

Client-side embedding (client holds `embed_tokens`, sends `h` not `input_ids`) fixes the plaintext hop-0
problem and is exactly what Petals recommends. **v2** — it moves 544.5 MB to the client.
