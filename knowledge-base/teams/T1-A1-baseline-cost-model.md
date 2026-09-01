---
team: T1 — Transport & Protocol
agent: T1-A1
topic: Definitive cost model of the v0 transport; where per-token wall clock actually goes
headline: >
  Transport is only 9–16% of v0's per-token wall clock — the O(n^2) recompute is the real cost.
  But 100% of transport's fixed cost is software, not wire: 17.6 ms/token building and tearing down
  httpx clients (12.1 ms of it an unused TLS context on plain http://) versus 0.3 ms of real TCP
  connect and 0.85 ms of real 1 GbE transfer. Fix compute first; then transport is the whole ceiling.
---

# T1-A1 — Baseline cost model of the v0 transport

## 0. Measurement environment

| Item | Value |
|---|---|
| Host | Apple M1 Pro, 10 core, 32 GB, darwin 25.6.0 |
| Compute cap | `torch.set_num_threads(2)` — mimics `docker-compose.yml` `limits.cpus: "2"` |
| Stack | Python 3.12.12, torch 2.10.0 (CPU), transformers eager attn, httpx 0.28.1, uvicorn 0.52.4, fastapi 0.141.1, numpy 2.4.3 |
| Model | Shapes from `Qwen/Qwen2.5-0.5B-Instruct` `config.json`: H=896, ffn=4864, 24 layers, 14 Q / 2 KV heads, V=151936, fp32, `tie_word_embeddings=true` |
| Weights | **Random-init from config** (the HF blob cache holds an `.incomplete` safetensors). Timing is shape-determined, so this is FLOP-identical to the real model. Outputs are meaningless; latencies are not. |
| Estimator | `min` of 3–40 reps after warm-up (cleanest under OS/thermal noise) |

Absolute ms will differ on the Linux/docker target. **The ratios and the byte counts carry.**
Source read: `layer-nodes/node.py`, `layer-nodes/coordinator.py`, `gateway/app.py`, `docker-compose.yml`.

## 1. Correction to the shared context: v0 is a STAR, not a chain

`coordinator.py:76-97` posts to each node and receives each node's output back before posting it
onward. Hidden states cross the wire **twice per logical hop**. Per generated token there are
**3 POSTs = 6 wire crossings**, four of them full hidden-state-sized:

① coord→node0 `{"input_ids":[…]}` ~7 B/token · ② node0→coord, ③ coord→node1, ④ node1→coord,
⑤ coord→node2 all carry `hidden_states_b64` **[seq,896] fp32** · ⑥ node2→coord carries
`logits_b64` [1,151936] fp32 = **810 KB, fixed**.

The house number "3 hops/token" is the *logical* count. The *wire* count is 4 activation crossings
plus one 810 KB logits blob. Every byte figure below uses the real count.

## 2. Byte accounting

`b64(n) = 4·ceil(n/3)`. JSON wrapper `{"hidden_states_b64": "…"}` = +25 B, `{"logits_b64": "…"}` = +18 B
(both verified against `json.dumps` output). HTTP request+response framing ≈ 350 B per crossing.

### (a) v0 as written — bytes on the wire per **generated token**

| seq | ①ids | ②H | ③H | ④H | ⑤H | ⑥logits | **TOTAL B** | MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 127 | 76,485 | 76,485 | 76,485 | 76,485 | 810,346 | **1,118,513** | 1.12 |
| 128 | 911 | 611,697 | 611,697 | 611,697 | 611,697 | 810,346 | **3,260,145** | 3.26 |
| 512 | 3,599 | 2,446,705 | 2,446,705 | 2,446,705 | 2,446,705 | 810,346 | **10,602,865** | 10.60 |
| 2048 | 14,351 | 9,786,737 | 9,786,737 | 9,786,737 | 9,786,737 | 810,346 | **39,973,745** | 39.97 |

(measured `raw`/`b64`/`json` lengths; TOTAL includes 6×350 B framing)

Asymptote: **19,220 B per sequence position per token**, i.e. 4 × b64(3584)+25. Note at seq=16 the
**810 KB logits blob is 72% of all traffic** and is 100% waste — the coordinator does `np.argmax` on it
(`coordinator.py:96`) and throws away 151,935 of 151,936 floats.

### (b) v0 + KV cache — only the last position crosses. **seq-independent.**

| Variant | B/token | vs (a) @seq=512 | vs (a) @seq=2048 |
|---|---:|---:|---:|
| KV cache only (still b64+JSON, logits blob, star) | 829,588 | 12.8× | 48.2× |
| + argmax at node2 (return token id, not logits) | 19,262 | 550× | 2,075× |
| + raw fp32 body (drop base64 + JSON) | 14,344 | 739× | 2,787× |
| + chain routing (node0→node1 direct; 2 crossings not 4) | 7,176 | 1,478× | 5,571× |
| + bf16 on the wire | 3,592 | 2,952× | 11,128× |
| + int8 on the wire (needs calibration — see T1 compression) | 1,800 | **5,890×** | **22,208×** |

## 3. The base64 tax and the JSON string-parse tax (measured)

Expansion is exactly **4/3 = +33.3%**. The CPU cost is the bigger half.

| seq | raw B | b64 B | tobytes | b64enc | json.dumps | json.loads | b64dec | frombuf | **Σ ms** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,584 | 4,780 | 0.0002 | 0.0039 | 0.0094 | 0.0054 | 0.0050 | 0.0008 | **0.025** |
| 16 | 57,344 | 76,460 | 0.0011 | 0.0579 | 0.1389 | 0.0673 | 0.0756 | 0.0015 | **0.342** |
| 128 | 458,752 | 611,672 | 0.0079 | 0.4573 | 1.1261 | 0.5205 | 0.6036 | 0.0071 | **2.72** |
| 512 | 1,835,008 | 2,446,680 | 0.0460 | 2.0724 | 4.4815 | 2.1349 | 2.4158 | 0.0405 | **11.19** |
| 2048 | 7,340,032 | 9,786,712 | 0.1829 | 8.2372 | 17.9833 | 8.4619 | 9.5885 | 0.1628 | **44.62** |

Throughputs (measured, on raw byte volume): **b64encode 936 MB/s, b64decode 753–765 MB/s,
json.loads 1,151 MB/s, json.dumps ~546 MB/s.** `json.dumps` is the single most expensive step —
it re-scans every one of the 2.4 M base64 characters for escape sequences that can never occur in
the base64 alphabet. Zero-copy alternative measured at **0.050 ms** for the same seq=512 tensor
(`tobytes` + `frombuffer`), i.e. **224× cheaper than the 11.19 ms b64+JSON round trip.**

Ops per generated token at seq=512, counted from source: **4× json.dumps** (node0 resp, coord→node1,
node1 resp, coord→node2), **4× json.loads** (coord `r0.json()`, node1 req, coord `r1.json()`, node2 req),
**2× b64encode + 2× b64decode** on hidden states, plus the same set once on the 810 KB logits blob.

End-to-end, measured against a real uvicorn+FastAPI server with the model replaced by a no-op
(so every ms is pure transport), the whole b64+JSON+pydantic+ASGI path costs
**≈5.0–5.5 ms per MB of activation, versus 0.49 ms/MB for a raw fp32 body — a 10.4× CPU tax.**

## 4. The `httpx.AsyncClient()`-per-call tax — and it is *not* the TCP handshake

`coordinator.py:44` and `:78` build `async with httpx.AsyncClient()` inside every forward call.
Because node0/node1/node2 are three different hosts, each client opens **3 fresh TCP connections
per generated token**, then closes them all.

Measured, one round trip to a local uvicorn server:

| seq | v0 fresh-client JSON+b64 | pooled JSON+b64 | pooled + raw bytes | fresh-client tax | b64+JSON tax | total tax |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.717 ms | 1.150 | 0.910 | **5.566** | 0.241 | 5.81 (7.4×) |
| 16 | 7.493 | 2.064 | 0.997 | **5.430** | 1.067 | 6.50 (7.5×) |
| 128 | 13.495 | 7.770 | 1.498 | **5.725** | 6.272 | 12.00 (9.0×) |
| 512 | 32.731 | 26.854 | 3.781 | **5.877** | 23.073 | 28.95 (8.7×) |
| 2048 | 105.308 | 97.354 | 9.678 | **7.955** | 87.676 | 95.63 (10.9×) |

The fresh-client tax is a flat **≈5.6 ms/hop = 16.8 ms/token**, independent of payload size.
Decomposing it (measured, min ms): raw loopback TCP connect+accept+close **0.099**;
`ssl.create_default_context()` **3.790**; `httpx.AsyncClient()` construct **4.027**;
`httpx.AsyncClient(verify=False)` construct **0.159**; `async with httpx.AsyncClient()`
enter+exit with no request **4.808**.

**≈4.0 of the 5.6 ms/hop is `ssl.create_default_context()` — httpx 0.28.1 builds a TLS context in the
client constructor regardless of scheme, and every v0 URL is plain `http://`.** The actual TCP
handshake is 0.099 ms, 1.8% of the tax. 12.1 ms/token is spent parsing X.509 certificates for
connections that will never use TLS.

## 5. Per-token wall-clock decomposition

Node compute, measured, exactly as `node.py` builds it (`output_hidden_states=True`, full sequence):

| seq | node0 ms | node1 ms | node2 ms | Σ compute ms |
|---:|---:|---:|---:|---:|
| 16 | 48.50 | 48.56 | 68.75 | **165.81** |
| 128 | 73.71 | 73.58 | 106.39 | **253.68** |
| 512 | 205.81 | 197.76 | 308.97 | **712.54** |
| 2048 | 853.97 | 1030.07 | 1667.21 | **3551.25** |

Full decomposition at **seq=512, v0 as written, 1 GbE** (compute & transport-CPU measured; link
transfer modelled from §2a byte counts):

| Stage | ms/token | % | tag |
|---|---:|---:|---|
| Tokenisation (`apply_chat_template` 0.0084 + `tokenizer()` 0.0566, once per *request*, ÷32 tokens) + `decode` 0.0015/token | **0.004** | 0.0% | measured |
| **Node compute, 3 nodes, full seq** | **712.54** | 89.4% | measured |
| httpx client construct + teardown × 3 | **17.63** | 2.2% | measured |
|  ‑ of which `ssl.create_default_context()` × 3 | *12.08* | *1.5%* | measured |
| TCP connect × 3 (LAN RTT 0.3 ms) | **0.90** | 0.1% | modelled (0.30 measured on loopback) |
| b64 + JSON + pydantic + ASGI + socket copies, 10.60 MB @ 5.2 ms/MB | **55.14** | 6.9% | measured coefficient |
| 1 GbE link transfer, 10.60 MB @ 125 MB/s | **84.82** | — | modelled *(overlaps the row above on a real NIC; counted separately below)* |
| **Total, transport CPU only (loopback-equivalent)** | **785.3** | 100% | |
| **Total, + 1 GbE serialisation-on-wire** | **870.1** | | |

**Transport share of wall clock: 9.3% (loopback) to 18.1% (1 GbE).** Cross-check: 3 × the measured
symmetric-echo round trip at seq=512 = 98.2 ms, scaled to v0's real 10.60 MB (vs the echo's 14.7 MB)
= **72.8 ms**, against 72.5 ms from the component sum. The model closes.

## 6. Crossover: when does network overtake compute?

### On the seq axis — **it never crosses in v0.** Both terms are O(seq).

| seq | wire MB | 1 GbE ms | 10 GbE ms | compute ms | net/comp 1 GbE | net/comp 10 GbE |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.12 | 8.95 | 0.89 | 165.8 | 5.4% | 0.54% |
| 128 | 3.26 | 26.08 | 2.61 | 253.7 | 10.3% | 1.03% |
| 512 | 10.60 | 84.82 | 8.48 | 712.5 | **11.9%** | 1.19% |
| 2048 | 39.97 | 319.79 | 31.98 | 3551.2 | 9.0% | 0.90% |

Asymptotically: 19,220 B/position → 0.154 ms/pos on 1 GbE, 0.0154 ms/pos on 10 GbE, against a
measured 1.734 ms/pos of compute. The ratio is **pinned at ~9% (1 GbE) / ~0.9% (10 GbE) for all seq**.
v0's two stupidities — resending the whole sequence, and recomputing the whole sequence — are the
same stupidity, so they scale together and never cross.

### The crossover is on the **compute-speed** axis

| seq | compute speedup needed before 1 GbE wire dominates | before 10 GbE dominates |
|---:|---:|---:|
| 512 | **8.4×** | 84.0× |
| 2048 | **11.1×** | 111.0× |

### The seq crossover *does* exist once compute is fast (modelled: 1.5 ms/token, KV-cached GPU runtime)

| Link | 1.5 ms byte budget | crossover seq, logits blob kept | crossover seq, logits blob removed |
|---|---:|---:|---:|
| 1 GbE | 187,500 B | **seq = 0** (the 810 KB blob alone blows the budget 4.3×) | **seq ≈ 10** |
| 10 GbE | 1,875,000 B | seq ≈ 55 | **seq ≈ 98** |

**Read this as the v2 argument:** on CPU nodes the LAN has 18–780× of headroom and a faster wire buys
nothing. The moment the compute is done properly (KV cache + a real runtime + GPU), the wire becomes
the binding constraint at sequence lengths as short as 10 on 1 GbE.

## 7. What KV cache alone does (measured), and the node-local waste it exposes

| seq | v0 Σ compute ms | +KV cache Σ compute ms | speedup |
|---:|---:|---:|---:|
| 16 | 165.8 | 108.92 | 1.5× |
| 128 | 253.7 | 112.54 | 2.3× |
| 512 | 712.5 | 123.94 | **5.7×** |
| 2048 | 3551.2 | 120.21 | **29.5×** |

KV-cached decode is **flat at ~110–124 ms/token** regardless of context — and at ~988 MFLOP/token that
is only ~9 GFLOP/s, so it is **framework-overhead-bound, not FLOP-bound**. That sets the compute floor
transport has to beat, and it is why §6's "8.4× compute speedup" is reachable in software alone.

Node2's two local wastes, isolated at seq=512 (measured): v0 `output_hidden_states=True` + full
`lm_head` = **419.65 ms**; `output_hidden_states=False` + full `lm_head` = 348.30 ms;
`output_hidden_states=False` + `lm_head` on the last position only = **280.80 ms (1.49×)**.

`node.py:99,104` runs the tied 896×151,936 `lm_head` over every position — 272.3 MFLOP/position, more
than the 238.6 MFLOP/position of node2's own 8 transformer layers — then slices `[:, -1, :]`.

## 8. Ranked transport defects, by expected speedup

Anchored at seq=512, v0 baseline 785.3 ms/token (loopback-equivalent). "×" = speedup of the whole
per-token wall clock unless stated.

| # | Defect | Fix | Byte / time effect | Whole-system × | v1/v2 |
|---:|---|---|---|---:|---|
| 1 | **No KV cache** (`coordinator.py:123-126` resends `gen_ids`) — O(n²), and the reason 4 full-sequence tensors cross per token | Per-node `past_key_values` keyed by request id; send last position only | compute 712.5→123.9 ms (measured); bytes 10.60 MB→0.83 MB (12.8×) | **5.4×** @512, **20×** @2048 | v1 |
| 2 | **810 KB logits blob** returned per token; coordinator only does `argmax` (`node.py:105`, `coordinator.py:96`) | Do `argmax` on node2, return the int | −810,346 B/token; −4.2 ms/token. 97.7% of post-KV bytes. 3-line diff | 1.03× alone; **43× of post-KV bytes** | v1 |
| 3 | **`httpx.AsyncClient()` per call** (`coordinator.py:44,78`) — 3 TCP conns/token, 12.1 ms/token of unused TLS context | Module-level pooled `AsyncClient(limits=…, verify=False)`, `Connection: keep-alive` | −17.1 ms/token, size-independent. **After fixes 1+2 this is ~99% of remaining transport** | 1.02× alone; **~14× of post-KV transport** | v1 |
| 4 | **base64 + JSON + pydantic** for binary | `await request.body()` → `np.frombuffer`; `Response(arr.tobytes(), media_type="application/octet-stream")`; shape in headers | 5.2 → 0.49 ms/MB (**10.4×**, measured); −33% bytes | 1.07× @512 pre-KV | v1 |
| 5 | **`output_hidden_states=True` + full-sequence `lm_head`** (`node.py:90,99,104`) | `False`; project last position only | 419.7 → 280.8 ms on node2 (1.49×, measured) | 1.21× @512 | v1 |
| 6 | **Star routing** — every activation crosses twice | Give each node a `NEXT_URL`; node0→node1→node2 direct | 4 → 2 activation crossings, **2×** bytes and serialisation | 1.03× post-KV | v1 |
| 7 | **fp32 on the wire** | bf16 cast at the boundary (activations tolerate it; keep fp32 compute) | 3,584 → 1,792 B/crossing, **2×** | small on LAN; matters at 10 GbE+ | v1 |
| 8 | **No quantised transport** | int8/fp8 + per-channel scale | **4×** vs fp32 — needs calibration, hand to T1 compression | — | v2 |
| 9 | **HTTP/1.1 framing** (h11 parse, header churn, chunked encoding) | Length-prefixed frames on one persistent TCP socket w/ `TCP_NODELAY`; or gRPC over HTTP/2, or Arrow Flight (zero-copy IPC) | removes the residual ASGI/h11 copies inside the 0.49 ms/MB floor | 1.0–1.1× post-KV | v2 |
| 10 | **RDMA / RoCEv2 / UCX / NCCL** | — | **Not justified by this analysis.** Post-KV the 1 GbE link carries 14 KB/token = 0.11 ms against ≥120 ms of compute: 1,000× headroom. Revisit only for 70B-class tensor parallelism, where per-token all-reduce volume is ~100× larger and sits on the critical path | ~1.00× | v2 (defer) |

### Compounded, seq=512, all v1 items (modelled from the measured components)

```
v0                                          785.3 ms/token
  #1 KV cache      compute 712.5 → 123.9;  bytes 10.60 → 0.83 MB   →  145.8 ms   (5.4×)
  #5 node-local    node2 419.7 → 280.8 on prefill only; decode −8   →  137.8 ms
  #2 argmax@node2  −810,346 B → transport 21.9 → 17.7 ms           →  133.6 ms
  #3 pooled client −17.1 ms                                        →  116.5 ms
  #4 raw bytes     14,344 B @0.49 ms/MB ≈ 0.007 ms                 →  116.4 ms
  #6 chain routing 4 → 2 crossings                                 →  116.4 ms
                                                                      ────────
                                            116.4 ms/token   = 6.7× end-to-end
```

Post-fix split: **compute ≈116 ms (99.6%), transport ≈0.5 ms (0.4%)**. Transport is then *solved* on
this hardware, and every further win must come from compute (T1's ceiling is real, and low). The
transport work only pays again at v2 scale — GPU nodes, larger models, or batch>1 — which is exactly
where §6's crossover table says the wire starts binding at seq≈10.

## 9. Three claims to defend in the deck

1. **"We measured it: 89% of v0's per-token time is compute, not network."** Anyone who opens with
   RDMA is optimising 0.9% of the clock (10 GbE, §6).
2. **"The transport cost that *is* real is 4.0 ms/hop of TLS certificate parsing on unencrypted
   localhost HTTP."** `ssl.create_default_context()` = 3.790 ms measured; TCP connect = 0.099 ms.
3. **"39,973,745 bytes per token at seq=2048 → 1,800. 22,208×."** (§2b, all-fixes column.)
