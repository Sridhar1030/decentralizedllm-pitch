---
status: TARGET ARCHITECTURE — supersedes the v0 PoC at /Users/srpillai/CODING/DecentralizedLLM
inputs: 25 agent reports in teams/, 01-VERIFIED-FACTS.md (authoritative on all four findings)
rule: every number carries (measured) / (modelled) / (derived). No number here was invented.
---

# DecentralizedLLM — v1 / v2 architecture

v0 is a correct, working proof that one LLM can be split across three machines. It is also
**89% compute, 11% transport** (measured, T1-A1), recomputes every position on every token
(271x redundant, FINDING 3), ships 607,744 B of logits to perform a 4-byte argmax (FINDING 2),
and runs its bottleneck stage at 1.55x the others (FINDING 1). This document is what replaces it.

**The single-stream ladder (the honest headline):**

| stage | ms/token | tok/s | vs v0 | basis |
|---|--:|--:|--:|---|
| v0 today, seq=512 | 785.3 | 1.273 | 1.00x | **measured** (T1-A1 §5) |
| v1 single stream | 116.0 | 8.62 | 6.77x | modelled from measured components |
| v1 at R=3, balanced shards | 41.31 (D_max) | 24.21 | **19.0x** | modelled (T3-A4, T5-A4) |

6.8x of that is latency we can demo with one request. The remaining 2.8x is pipelining and
**requires three concurrent requests** — `demo.sh` sends one curl, so at concurrency 1 the screen
will read 8.62 tok/s and disagree with the slide. Drive the demo at R=3 or drop the 19x.

---

## 1. Layered architecture

```
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ CLIENT             curl · browser SSE · any OpenAI-compatible SDK                        │
  │                    POST /v1/chat/completions[/stream]           ~200 B / request         │
  └───────────────────────────────────────┬──────────────────────────────────────────────────┘
                                          │ HTTP/1.1 + JSON — curl-able, NOT DLP (§6)
  ┌───────────────────────────────────────▼──────────────────────────────────────────────────┐
  │ GATEWAY :8080      api-key · circuit breaker · SSE relay · static UI                     │
  │                    the ONLY process on an untrusted boundary                             │
  └───────────────────────────────────────┬──────────────────────────────────────────────────┘
                                          │
  ╔═══════════════════════════════════════▼══════════════════════════════════════════════════╗
  ║ COORDINATOR :8081                     ── the scheduler of record ──                      ║
  ║                                                                                          ║
  ║  ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────────────┐             ║
  ║  │ ADMISSION        │   │ SCHEDULER         │   │ SESSION TABLE            │             ║
  ║  │ Semaphore(R*=3)  │──▶│ mints ONE         │──▶│ session_id → node triple │             ║
  ║  │ Queue(K=6)       │   │ BatchDescriptor   │   │ session_id → position    │             ║
  ║  │ 429 + Retry-After│   │ per step  (§4.3)  │   │ epoch                    │             ║
  ║  └──────────────────┘   └───────────────────┘   └──────────────────────────┘             ║
  ║  R* = D / D_max = 0.71254 / 0.30897 = 2.31 → 3   (measured stage times, T3-A4)           ║
  ╚═══════════════════════════════════════╤══════════════════════════════════════════════════╝
                                          │ DLP  TOKENS  40+4 B
  ┌───────────────────────────────────────▼──────────────────────────────────────────────────┐
  │ DLP TRANSPORT — persistent TCP · TCP_NODELAY · 40-byte header · credit=4  (§3)           │
  │ zero-copy:  sendmsg([hdr, memoryview(tensor.storage)]) → recv_into(tensor)               │
  └─────────┬─────────────────────────────┬─────────────────────────────┬────────────────────┘
            │ TOKENS 44 B                 │ ACTIVATION 1,832 B          │ ACTIVATION 1,832 B
            ▼                             ▼                             ▼
   ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
   │ node0           │  chain    │ node1           │  chain    │ node2           │
   │ embed + L0-10   │──────────▶│ L11-21          │──────────▶│ L22-23 + norm   │──▶ TOKENS 44 B
   │                 │  1,832 B  │                 │  1,832 B  │ + lm_head       │    to coordinator
   │ ┌─────────────┐ │           │ ┌─────────────┐ │           │ ┌─────────────┐ │
   │ │  KV cache   │ │           │ │  KV cache   │ │           │ │  KV cache   │ │    the sampled id —
   │ │ 5.5 KB/token│ │           │ │ 5.5 KB/token│ │           │ │  1 KB/token │ │    NOT 607,744 B
   │ │  bf16, LRU  │ │           │ │  bf16, LRU  │ │           │ │ + SAMPLER   │ │    of logits (F2)
   │ └─────────────┘ │           │ └─────────────┘ │           │ │ + EOS check │ │
   │                 │           │                 │           │ └─────────────┘ │
   │  11.00 layer-eq │           │  11.00 layer-eq │           │  11.13 layer-eq │
   └─────────────────┘           └─────────────────┘           └─────────────────┘

  ╔═════════════════════════════════════════════════════════╗   ╔═══════════════════════╗
  ║  CONTROL PLANE       (HTTP/JSON — never DLP)            ║   ║  OBSERVABILITY        ║
  ║  POST /register {node_id, url, layers:[11,22],          ║   ║  /metrics @ 1 s       ║
  ║                  state, dtype, epoch}                   ║   ║  Prometheus → Grafana ║
  ║  poll 500 ms · 3 misses = DEAD  (1.5 s)                 ║   ║  OTel 1.44.0 → Tempo  ║
  ║  state ∈ {LOADING, READY, DRAINING, DEAD}               ║   ║                       ║
  ║  greedy cover of [0,24) → chain · epoch++ on re-shard   ║   ║  dllm:imbalance       ║
  ║  POST /load_layers {"range":[8,12]}   on failover       ║   ║    1.562  →  ~1.00    ║
  ╚═════════════════════════════════════════════════════════╝   ╚═══════════════════════╝
```

Two things the diagram is asserting, both load-bearing:

**v0 is a star; v1 is a chain.** v0's coordinator POSTs to each node and gets the tensor back
(`coordinator.py:46-63`), so one logical hop is two TCP crossings: **6 crossings per token, 4 of
them activation-sized** (measured, T1-A1 — this corrects the "3 hops" house number). v1 gives each
node a `NEXT_URL` so activations go node0→node1→node2 directly: **4 crossings, 2 activation-sized.**
That is a 2x cut in both wire bytes and serialization CPU before any other change.

**The control plane is not on the data path.** It speaks HTTP/JSON at 2 Hz on a different socket
from the DLP fabric, so a registry stall can never stall a token (§6).

---

## 2. Components

| component | responsibility | v1 tech | v2 tech |
|---|---|---|---|
| **Client** | Prompt in, tokens out | curl / browser SSE / OpenAI SDK | unchanged — the OpenAI shape is the moat, it lets `vllm bench serve` drive us |
| **Gateway** | api-key, circuit breaker, SSE relay, static UI. Only untrusted boundary | FastAPI + `prometheus_client`, in-proc breaker | SPIFFE/SPIRE X.509 SVIDs; rate limit per tenant; WireGuard mesh if nodes leave L2 |
| **Admission** | Bound the queue. `Semaphore(3)` + `Queue(maxsize=6)` → 429 + `Retry-After: 8` | `asyncio` primitives, ~20 LOC | NATS JetStream 2.10+ durable queue; EDF within DRR classes |
| **Scheduler** | Mint ONE immutable `BatchDescriptor` per step; decode-first, prefill-fills to token budget `T` | ~60 LOC in `coordinator.py`; `T = (TPOT_slo/3 − 22.77)/0.2931` | vLLM V1 `SchedulerOutput` verbatim + `--pipeline-parallel-size 3`; DRR priority classes |
| **Coordinator** | Tokenize, route, own the session→triple map, drive the chain, emit SSE | FastAPI + in-proc dicts | stateless behind etcd 3.6.6; run 2 replicas (it is an undiscussed SPOF today) |
| **DLP transport** | Move tensors. 40-B header, persistent TCP, `TCP_NODELAY`, credit=4 | stdlib `socket` + `struct`, ~150 LOC (`layer-nodes/dlp.py`) | `google-crc32c`; `MSG_ZEROCOPY`/io_uring; Arrow Flight if frames >1 MB or nodes go polyglot; RDMA/NIXL when GPUs land |
| **Layer shard** | Hold `[s,e)`, run forward, own its KV cache. node2 also samples | FastAPI (control) + DLP listener (data), `transformers` + `DynamicCache` | vLLM as the per-node engine — never reimplement PagedAttention/prefix caching/the sampler |
| **KV cache** | Per-node, per-session, keyed by `session_id`, fenced by `position` | `OrderedDict` LRU + 300 s TTL, bf16 | PagedAttention 16-token blocks + prefix caching, or adopt vLLM |
| **Control plane** | Registry, heartbeat, layer cover, re-shard on failure | coordinator-held table + `POST /register`, 500 ms poll | etcd 3.6.6 leases/watches; Dijkstra chain selection on EWMA latency (Petals' model) |
| **Observability** | Make FINDING 1 visible, not asserted | Prometheus @ **1 s** + Grafana + OTel 1.44.0 → Tempo | Alertmanager burn-rate rules; OTel Collector tail sampling; native histograms |

---

## 3. DLP frame layout

Verbatim from **T1-A4** — this is the frozen spec, not a re-invention. Fixed **40 bytes**,
little-endian, every field naturally aligned; `40 % 8 == 0` so the payload starts 8-byte aligned and
a `torch.float32` view over it needs no realignment copy. `struct` format `"<4sBBHIIIIIIBBHI"`,
`struct.calcsize` == 40 (asserted in the reference impl).

| off | size | field | type | why it exists |
|--:|--:|---|---|---|
| 0 | 4 | `magic` | `u8[4]` = `"DLP1"` | Rejects a stray HTTP probe at byte 0 instead of mis-parsing 4 GB of "payload" |
| 4 | 1 | `version` | `u8` | Major version. Mismatch → hard close |
| 5 | 1 | `msg_type` | `u8` | Frame discriminator (§3.1) |
| 6 | 2 | `flags` | `u16` | Bitfield (§3.4). `u16` so flags can be added without a version bump |
| 8 | 4 | `request_id` | `u32` | **Pipelining key** — demultiplexes out-of-order responses on one stream |
| 12 | 4 | `session_id` | `u32` | **Binds the frame to a KV-cache slot.** The field that makes KV caching possible at all |
| 16 | 4 | `seq_len` | `u32` | `dims[0]`. 1 on decode, N on prefill |
| 20 | 4 | `dim1` | `u32` | `dims[1]` — 896 or 151936. Must be `u32`: vocab overflows `u16` |
| 24 | 4 | `payload_len` | `u32` | **Length prefix** — all the reader needs to find the next frame boundary |
| 28 | 4 | `credit` | `u32` | Piggybacked flow-control grant. Free: rides an existing frame |
| 32 | 1 | `dtype` | `u8` | 0 fp32 · 1 fp16 · **2 bf16** · 3 int8 · 4 int32 · 5 fp8_e4m3. Per-frame, no renegotiation |
| 33 | 1 | `codec` | `u8` | 0 raw · 1 ~~lz4~~ (**retired**, T2-A5 measured it *expanding* activations 1.0036–1.0056x) · 2 zstd · 3 topk-delta · 4 blosc2:zstd+bitshuffle |
| 34 | 2 | `reserved` | `u16` | Pads `crc32` to a 4-byte boundary. Room for a priority/tenant byte |
| 36 | 4 | `crc32c` | `u32` | Payload integrity. **Optional — gated by `F_CRC`** |
| 40 | N | `payload` | raw | Tensor storage verbatim. No framing, no escaping, no base64 |

**`msg_type`:** `0x01 HELLO` · `0x02 HELLO_ACK` · `0x10 ACTIVATION` · `0x11 LOGITS` ·
`0x12 TOKENS` · `0x20 CREDIT` · `0x30/0x31 PING/PONG` · `0x40 CACHE_EVICT` · `0x7F ERROR`.

**`flags`:** bit 0 `F_CRC` (default **off** on the docker bridge — Ethernet FCS + TCP checksum
already cover it, 0.16 µs/frame saved) · 1 `F_LAST` · 2 `F_MORE` · 3 `F_PREFILL` ·
**4 `F_TRACE`** (granted to T4-A5: appends a 32-byte trace extension `"<16s8sB7x"` = trace_id +
span_id + flags + pad; header becomes 72 B, `72 % 8 == 0` so alignment survives; costs
32/3624 = **0.88%** of a decode frame when sampled, 0 bytes when not).

Overhead: `40 / 3584 = 1.12%` fp32, `40 / 1792 = 2.23%` bf16 — against base64's flat +33.3%.
Serialization CPU is **O(1) in payload size** (1.08 µs at seq=1, 1.80 µs at seq=1024, measured)
because DLP hands the socket a `memoryview` of the tensor's own storage; v0 is O(n)
(25.4 µs → 23,429 µs over the same range, measured). Python's stdlib has **no CRC32C** —
v1 fills the field with `zlib.crc32` (~30 GB/s, measured); the name is for the v2 `google-crc32c` swap.

---

## 4. End-to-end request lifecycle

House case: **P = 32 prompt tokens, G = 512 generated, bf16 on the wire, chain routing, balanced
shards.** Byte counts include the 40-byte DLP header on every frame.

### 4.1 Prefill (once per request) — TTFT path

| # | hop | frame | bytes | note |
|--:|---|---|--:|---|
| 1 | client → gateway | JSON | ~200 | api-key checked, breaker checked |
| 2 | gateway → coordinator | JSON | ~200 | |
| 3 | *coordinator* | — | 0 | `apply_chat_template` 0.0084 ms + `tokenizer()` 0.0566 ms (measured, once per request — tokenisation is a non-issue) |
| 4 | *admission* | — | 0 | `Semaphore(3)`; queue full → **429 + `Retry-After: 8`**, never a silent drop |
| 5 | *scheduler* | — | 0 | allocate `session_id`, pin the `(n0,n1,n2)` triple, mint `BatchDescriptor{step_id, epoch, kind=PREFILL_CHUNK}` |
| 6 | coordinator → node0 | `TOKENS` | **40 + 128 = 168** | 32 × int32 ids. `F_PREFILL` set |
| 7 | node0: embed + L0-10, append 32 positions to KV (32 × 11 × 512 B bf16 = 180 KB) | | | |
| 8 | node0 → node1 | `ACTIVATION [32,896]` bf16 | **40 + 57,344 = 57,384** | direct, not via coordinator |
| 9 | node1 → node2 | `ACTIVATION [32,896]` bf16 | **57,384** | |
| 10 | node2: L22-23 + norm, `logits_to_keep=1`, **argmax on node2** | | | lm_head on 1 position, not 32 — saves 31 × 272.3 MFLOP |
| 11 | node2 → coordinator | `TOKENS` | **40 + 4 = 44** | the token id. **Not** 607,744 B of logits |
| 12 | coordinator → gateway → client | SSE | ~120 | first token visible ⇒ TTFT |
| | **DLP fabric total** | | **114,980 B** | vs v0's **1,422,285 B** = **12.4x** (derived) |

Long prompts (P > 512) are **chunked at c = 128** and interleaved with decode steps: TTFT
518 → 393 ms (1.32x better) *and* the TPOT spike other users pay drops 3.86x → 1.46x
(2.64x smaller) — both, because chunks pipeline across stages and a monolithic prefill cannot
(measured components, modelled composition, T3-A3 §7). Chunking is **gated on the connection-pool
fix**: at M=32 the 5.6 ms/hop `httpx.AsyncClient()` construction tax is 537 ms and erases the win.

### 4.2 Decode (511 times) — TPOT path

| # | hop | frame | bytes | note |
|--:|---|---|--:|---|
| 1 | coordinator → node0 | `TOKENS` | **44** | one int32 + `position` fence in `seq_len` |
| 2 | node0: embed 1 token, L0-10 with `past_key_values`, append 1 position | | | ~35.8 ms (modelled from measured shares) |
| 3 | node0 → node1 | `ACTIVATION [1,896]` bf16 | **1,832** | 1792 payload + 40 header |
| 4 | node1 → node2 | `ACTIVATION [1,896]` bf16 | **1,832** | |
| 5 | node2: L22-23 + norm + lm_head, **sample + EOS check here** | | | node2 is the only node that can see EOS |
| 6 | node2 → coordinator | `TOKENS` | **44** | id + optional logprob; `finished=[req_id]` on EOS |
| 7 | coordinator → client | SSE | ~80 | |
| | **DLP fabric total per token** | | **3,752 B** | 3,592 B excluding headers (matches T1-A1) |

**Against v0 at seq=512: 10,602,865 B → 3,752 B = 2,826x** (measured byte counts / derived).
Whole 512-token generation: **1,821.7 MB → 1.948 MB = 935x** (FINDING 4; +2.2% = 43.4 KB if you
count DLP headers, which FINDING 4 does not).

**The caveat that must travel with 935x:** it is a *wire-bytes* reduction, not a wall-clock one.
On a LAN the pipeline is compute-bound — post-KV the 1 GbE link carries ~14 KB/token = 0.11 ms
against ≥116 ms of compute, ~1,000x of headroom. The wall clock comes from FINDING 3's 271x
recompute elimination and FINDING 1's 1.55x rebalance. Bytes dominate on WAN, 1 GbE, and long context.

### 4.3 What travels with the activation

The `BatchDescriptor` and the activation tensor **must be one framed message**, or `step_id` must be
echoed in both and asserted. Otherwise a stage pairs descriptor S with tensor S−1 and corrupts
silently — no exception, just every row after the mismatch attending to another user's KV cache.
This is the highest-severity bug class in the design.

Descriptor wire size at n_seq=16 is **343 B against 28,672 B of bf16 activation = 1.20% overhead**
(derived, T3-A3 §6). `cu_seqlens` is *derived per stage* from `n_new`/`ctx_len`, never shipped.

---

## 5. The five levers, and the one that does not move the clock

| # | lever | wire | clock | effort |
|--:|---|---|---|---|
| 1 | **KV cache** per node per session, keyed by `session_id`, fenced by `position` | 1,406.8 → 5.19 MB hidden (271x) | 712.5 → **123.94 ms** compute at seq=512 (**measured**) | days |
| 2 | **Sample on node2**, return 4 B not 607,744 B | 810,325 → 44 B/token | −4.285 ms/token of codec to do 0.074 ms of argmax = **58x** (measured) | hours |
| 3 | **DLP + pooled connections** replacing HTTP/JSON/base64 | +34% → +1.1% overhead | 8.483 → 0.089 ms/hop (95x, measured loopback); −17.6 ms/token from the client pool alone | days |
| 4 | **Rebalance** `NODE_LAYERS` to `0-11`/`11-22`/`22-24` | 0 | bottleneck 17.13 → 11.13 layer-eq = **1.539x throughput, 0x single-stream latency** | hours |
| 5 | **bf16 on the wire** | 3,584 → 1,792 B/hop (2.00x) | **~0 ms on a LAN.** 3.5 µs cast | hours |

**Lever 5 is the one that does not move the clock, and saying so is worth more than hiding it.**
bf16 costs 3.5 µs and is quality-free (KL 5.7e-5, top-1 0.9974, greedy output bit-identical on 4/4
prompts — measured, T2-A4). But bf16→int8 saves 1,788 B/token = **14.3 µs on 1 GbE against 0.88 ms
of compute = 1.6%**. Sub-bf16 compression only starts to matter below **~163 Mbit/s per hop**
(modelled, T2-A4). Ship bf16 as the default; ship int8 as a WAN *toggle*, not as the headline.

Two teams reached this independently and they agree: **do not ship a byte codec.** At 1 GbE exactly
2 of 60 measured codec/payload combinations beat raw bytes, and both are 2048-token prefills; at
10 GbE, zero do (measured, T2-A3). Real fp32 activations compress to r=0.843 — barely better than
white Gaussian noise at r=0.850 — because the only low-entropy byte is the IEEE-754 exponent
(2.838 of 8 bits). The compression is the dtype cast, not the codec.

**If int8 is ever enabled, per-token scaling is mandatory and per-tensor scaling is a code-review
ban.** One channel of 896 in our own residual stream is **972x** larger than the median
(ch 62, |1701.9| vs 1.75 — measured on our model, T2-A1). Per-tensor int8 erases every other token:
top-1 agreement 0.0135, perplexity 411,041 vs 18.64, output degenerates to
`" time declaration declaration declaration"` (measured). Per-token scaling costs **+0.22% wire bytes
and buys a 328x KL reduction**. With 8 fp16 outlier channels pinned it reproduces fp32's greedy
output exactly for 20/20 tokens at 906 B/token/hop.

---

## 6. Data path vs control path

| | **data path** | **control path** |
|---|---|---|
| carries | activations, token ids, batch descriptors | registration, heartbeat, layer cover, re-shard, metrics, traces |
| protocol | **DLP** — binary, persistent TCP, `TCP_NODELAY` | **HTTP/1.1 + JSON** — text, curl-able |
| socket | dedicated, one per node pair, opened at boot | separate, short-lived |
| frequency | 4 frames per token | 2 Hz heartbeat + on-change |
| latency budget | 0.089 ms/hop (measured) | seconds |
| failure mode | a stall stalls a token | a stall stalls **nothing** — last-known-good routing table stays hot |
| auth | none in v1 (trusted L2) | api-key at the gateway only |
| debuggability | `dlp.py demo()` self-check, `dllm_frame_errors_total{reason}` | `curl node1:8002/health` |

The separation is not aesthetic. Three concrete consequences:

1. **A hung registry cannot hang inference.** The coordinator polls `/register` on its own task;
   the token loop reads a cached routing table keyed by `epoch`.
2. **A dead node is detected by the control path (1.5 s), not by a 60 s data-path timeout**
   (`coordinator.py:46` today waits `timeout=60` on a paused node — measured from source).
3. **`epoch` is the fence between them.** Every DLP frame carries the epoch it was minted under;
   a stage rejects a frame whose epoch does not match its own. A re-shard bumps the epoch, so a
   frame in flight across a topology change is rejected loudly instead of being applied to the
   wrong layer range.

Keep HTTP `/forward` on every node as a debug/health endpoint. It costs nothing and it is the
A/B control that proves the DLP number on stage.

---

## 7. State model — what statefulness actually costs

v0 is **completely stateless**, and that is not an accident of design — it is a side effect of its
worst bug. Because the coordinator re-sends the whole sequence every token, *there is no distributed
state to lose*, so failover today costs exactly one extra forward pass. **The KV cache is what
creates the failover problem.** Sequence the roadmap accordingly.

### 7.1 What becomes stateful

| state | lives on | size | lost on crash? |
|---|---|--:|---|
| KV cache | each node, per session | **512 B / token / layer** bf16 (GQA: 2 kv heads × 64 head_dim × 2 (K,V) × 2 B) | yes |
| — per token, balanced split | node0 / node1 (11 layers) | 5,632 B; node2 (2 layers) 1,024 B | yes |
| — 2048-token session | node0 / node1 | 11.53 MB bf16 (node2: 2.10 MB) | yes |
| — 2048-token session | whole model | 25.17 MB bf16 | yes |
| session → node-triple map | coordinator | ~100 B/session | yes (v1); no (v2, etcd) |
| `position` fence | node + coordinator | 4 B | recoverable from `gen_ids` |
| routing table + `epoch` | coordinator | ~1 KB | rebuilt in 500 ms from `/register` |
| `gen_ids` (authoritative token list) | coordinator | 4 B/token | **the recovery anchor** |

**GQA makes the cache nearly free.** 2 KV heads instead of 14 is a **7x discount**: a full
2048-token session costs 11.53 MB on an 11-layer node, **1.8% of the 656 MB of layer weights that
node already holds**.
There is no memory argument against caching. Headroom in a 2 GB container is ~0.9 GB on node0/node2
and ~1.4 GB on node1 → **~78 and ~121 concurrent 2048-token sessions** at the balanced split (modelled) — *after* the loader
fix in §8, because `load_model()` currently peaks at ~3.95 GB by holding two full copies of the model.

### 7.2 What it costs

| property | v0 stateless | v1 stateful | consequence |
|---|---|---|---|
| routing | per token, any node with the range | **fixed triple, chosen at prefill** | needs a `session_id → triple` map |
| load balancing | free, per token | per session, at admission only | a long session on a slow node stays there |
| L4/L7 LB | round-robin works | **breaks** | `docker-compose scale` no longer balances |
| scale-out | instant | new nodes get only NEW sessions | warm-up lag = length of live sessions |
| drain / rolling deploy | kill anytime | wait out or migrate | 4.45 MB/shard to migrate a 543-token session — cheap, just unimplemented |
| crash blast radius | one in-flight token | **every session pinned to that node** | §7.3 |
| memory | constant | grows with concurrency | memory becomes a scheduling dimension |

Session affinity is the one genuine architectural regression the cache introduces. Put it in the
deck rather than hiding it — a judge who knows serving will ask.

### 7.3 Recovery: the `position` fence is not optional

The cache is *positional* state. One duplicated or reordered delivery double-appends a position and
the model emits wrong tokens **with no error**. Every frame carries `position` = the number of tokens
this shard must already hold:

- `have == position` → append (the normal path)
- `have > position` → `Cache.crop(position)` — rewind, cheap, covers retries and rejected speculation
- `have < position` → **409 `{"cache_miss", have}`** → coordinator replays

`position` is simultaneously the idempotency guard and the RoPE-correctness guard. Do not simplify
it away.

**Failure at position n costs another n position-forwards — it exactly doubles that session's
compute.** At n=543 that is 536.4 GFLOP ≈ 10.7 s (modelled). Cached-with-one-failure is still
**135x** better than v0. Surface it as an SSE `event: replaying`; the UI already has the channel.
v1.5 adds the boundary-activation journal (the coordinator already relays every boundary hidden
state — just keep it: `n × 896 × 2 B × 2 boundaries` = 1.83 MB at n=512 bf16, ~5 LOC) so only the
dead shard's layers are recomputed.

### 7.4 The correctness trap that ships silently

`node.py:53` does `model.model.layers = full.model.layers[8:16]`. **Slicing an `nn.ModuleList` does
not renumber.** Those modules keep `self_attn.layer_idx = 8..15`, so with a cache attached:

- `past_key_values.get_seq_length()` reads slot 0 → **0** → `position_ids` restart every step → wrong RoPE
- `get_mask_sizes(q_length, layer_idx)` returns `kv_length = 1` → **the causal mask hides the whole past**

No exception. Just plausible wrong text. **Renumber to local dense indices in `load_model()` —
one loop — and gate the merge on a `torch.allclose` check comparing stateless vs token-at-a-time
output.** This is non-negotiable, and it is derived from reading `transformers` 5.5.0 source.

**This is now run, not asserted** — `bench/parity_check.py`, output in `bench/parity_check.out`,
real Qwen2.5-0.5B-Instruct weights, torch 2.10.0 / transformers 5.3.0, prompt `"The capital of
France is"` (all **measured**):

| # | what | result |
|---|---|---|
| **A** | 3 chained shards, stateless (v0 semantics) vs 1 monolithic forward | **max abs diff = 0.000e+00** — bit-identical, so the split itself is exact |
| **B** | 3 shards KV-cached token-at-a-time, `layer_idx` renumbered | `" Paris. It is the largest city in"` — **token-identical to the reference** |
| **C** | same, renumber removed | `" Paris. Paris is the capital of France"` — **fluent, plausible, and wrong** |

Row C is the whole argument in one line: the bug does not crash, it does not warn, and it does not
even produce nonsense. It produces a *different, perfectly reasonable sentence*. No eyeball review
catches that; only test B does. Run it before merging the cache, and keep it in CI after.

`parity_check.py` also pins two requirements the prose above leaves implicit: non-final shards must
run with `norm = Identity` (the final node owns `norm` + `lm_head`), and each shard owns its **own**
cache object — the three caches are independent and are handed back per hop, never shared.

Same class of trap on the wire: a dtype mismatch between sender and receiver **reshapes silently**
when the sizes divide evenly (a 16×896 bf16 payload reshapes cleanly into a half-height fp32
tensor). The mandatory length assert in `decode()` must not be optimised away.

---

## 8. File-by-file change list

Against `/Users/srpillai/CODING/DecentralizedLLM`. Ordered by dependency, not by size.

### New files (2)

| file | ~LOC | why |
|---|--:|---|
| `layer-nodes/dlp.py` | ~150 | The frame codec + persistent-socket client/server. Reference impl already written and self-checked in T1-A4 §5 — `struct.Struct("<4sBBHIIIIIIBBHI")`, `sendmsg([hdr, mv])`, `recv_into(tensor)`, `demo()` asserts version negotiation, PING/PONG, out-of-order demux `[103,102,101]`, CRC, bit-exact fp32 round trip, bf16 halving, and rejection of a stray HTTP request. Copy it, do not rewrite it. Stdlib only. |
| `tools/slice_weights.py` | ~40 | `safetensors.torch.save_file` → `shard{0,1,2}.safetensors`, run once offline. **This is the file that makes the pitch's headline claim true.** |

Everything else in v1 is an edit. Registry (~55 LOC) and scheduler (~60 LOC) live inside
`coordinator.py`; they do not need modules of their own at n=3.

### Edits

**`layer-nodes/node.py`** — the largest diff, and the one that carries most of the win.
1. `load_model()`: **renumber `layer_idx` to local dense indices** (§7.4). Hard prerequisite for the
   cache; without it the cache is silently wrong.
2. `load_model()`: mutate `full` in place instead of allocating a second `Qwen2ForCausalLM(config)`
   — ~5 lines. Peak RSS **3,951.7 → 1,975.8 MB**. Unblocks the commented-out `memory:` limits.
3. `load_model()`: load `shard{i}.safetensors` when present, fall back to `from_pretrained`.
   Peak RSS → ~1.1 GB, and "no node holds the full model" stops being false at boot.
4. `forward()`: **delete `output_hidden_states=True`** (line 90/99). On non-final nodes `lm_head` is
   `Identity`, so `out.logits` *already is* the final hidden state.
5. `forward()`: `logits_to_keep=1` on node2 (line 104). **1.38x whole-pipeline with no cache at all**
   — 39.9 TFLOP of 145.4 TFLOP is redundant lm_head applications.
6. `forward()`: **argmax + EOS check move here** (line 105). Return the token id, not the logits.
   `607,744 B → 4 B`. Also deletes the exact oracle that recovers 27% of prompts verbatim
   (arXiv:2311.13647) — a bandwidth win, a latency win, and a security fix in one commit.
7. Add `_sessions: OrderedDict[session_id → [DynamicCache, last_used]]`, `MAX_SESSIONS`,
   `CACHE_TTL_S=300`, and the `position` fence with a 409 `cache_miss` (§7.3).
8. Add a DLP listener on a second port; keep `/forward` for health/debug/A-B.
9. Add `POST /load_layers {"range":[8,12]}` — parameterise the existing `load_model()`. This is the
   failover mechanism.
10. `/health` returns `state ∈ {LOADING, READY, DRAINING, DEAD}` and POSTs `/register` on startup.
11. **Fix the timer scope at line 87**: `_forward_seconds_total +=` sits *before* the logits branch,
    so node2's base64 encode is never counted — the bottleneck stage is under-reported by exactly the
    amount that makes it the bottleneck.
12. Replace the hand-rolled f-string counters with `prometheus_client` **Histograms** on the
    landmark buckets (`.04` = the v1 per-stage target, `.32` just above node2's measured 0.30897 s).
    Two counters can only yield a mean, and a mean hides the p99 the SLO is written against.

**`layer-nodes/coordinator.py`** — net *smaller* despite gaining features (the hardcoded r0/r1/r2
blocks go away).
1. **Delete the three `async with httpx.AsyncClient()` blocks** (lines 44, 78). One module-level
   pooled `AsyncClient(verify=False, limits=…)`. **−17.6 ms/token, payload-independent** — 3.95 ms
   of every hop is `ssl.create_default_context()` parsing X.509 for a plain `http://` URL
   (4.123 ms with TLS vs 0.176 ms without, measured; the real TCP handshake is 0.099 ms).
   *Do this first even if everything else slips.*
2. Delete `NODE0_URL/NODE1_URL/NODE2_URL`. Add `POST /register`, the node table, a 500 ms poller,
   and greedy cover of `[0,24)`. **Failure detection 60 s → 1.5 s.**
3. Replace the token loop (lines 122-127) with: prefill once, then send **only the new token** with
   `session_id` + `position`. `[seq,896] → [1,896]`.
4. Add `Semaphore(3)` + `Queue(maxsize=6)` + 429 with `Retry-After: 8`
   (`K = λ·W_SLO`, Little's law solved for L — not guessed).
5. Add the `BatchDescriptor` minting loop (decode-first, prefill-fills to budget `T`), and speak DLP
   to node0 instead of HTTP.
6. Chain routing: pass each node its `NEXT_URL` so activations never return to the coordinator.
   **4 activation crossings → 2.**
7. Add SSE events `node_down` / `resharding` / `reshard_done` / `replaying`.
8. Emit OpenAI-shaped SSE chunks so `vllm bench serve` can drive us — today's bespoke event schema
   means no professional benchmark can run against this system at all.

**`gateway/app.py`** — two small fixes, one of them a real bug.
1. **Bug B1:** line 43 sets `circuit_state["failures"] = 0` unconditionally after any successful HTTP
   exchange. A dead node makes the coordinator return 500 — a *successful* exchange — so the breaker
   is **reset by exactly the failure it exists for**. It only ever trips when the coordinator process
   is unreachable. Treat coordinator 5xx as a failure. 2-line fix.
2. Rename `sridhar_gateway_*` → `dllm_*`; add TTFT/TPOT histograms.

**`docker-compose.yml`**
1. `NODE_LAYERS`: `"0-8"/"8-16"/"16-24"` → **`"0-11"/"11-22"/"22-24"`**. Pure env-var edit, no code
   change — `node.py` keys off `start_layer==0` and `end_layer==24`, both preserved. **1.539x
   throughput, zero single-stream latency change** (the sum of stage times is invariant).
2. Add `OMP_NUM_THREADS=2` / `MKL_NUM_THREADS=2`. `deploy.resources.limits.cpus` is a cgroup *quota*,
   not a core mask — torch reads the host's 10 cores and spawns 10 OpenMP threads to thrash inside
   2 CPUs of quota.
3. Shared `hf-cache` named volume + `HF_HOME` on all three nodes. Cold start ~255 s → ~65 s
   (modelled), and it is the difference between a 3-8 s re-shard pause and an indefinite stall on
   stage. **This is the single highest-probability demo failure.**
4. `mem_limit: 6G` (not the commented-out `4G` — the loader peaks at 3.95 GB today). After the
   loader fix, `2G`.
5. Add `NEXT_URL` per node for chain routing, and a `dlp` port.
6. Add Tempo. Keep the `sleep 45` / `depends_on` chain until #3 lands — the hack is load-bearing.

**`prometheus/prometheus.yml`** — `scrape_interval: 15s` → **`1s`**. This is the worst observability
bug in the repo for a demo: 15 s yields **12 samples across a 3-minute run**, and `rate(...[5m])`
over 12 samples is a flat line. Every other observability item is decoration without it.

**`grafana/provisioning/dashboards/sridhar-mesh.json`** — `"refresh": "1s"`, per-panel
`"interval": "1s"`; panels for per-stage p95 bargauge (renders 0.206 / 0.198 / 0.309 as three bars —
the imbalance read off the screen), node-state timeline, queue depth vs `N*=3`, admission outcomes;
Prometheus annotations on `dllm_pipeline_epoch` changes and `dllm_node_state == 0`.
**Sequence the deck so the imbalance panel is shown *before* the rebalance lands** — after the fix it
is three equal bars and stops being a demo asset.

**`demo.sh`** — step 5 is titled "Failover Demo" and demonstrates the *opposite*: it stops node1,
prints the error, and the shipped demo ends on the outage. Replace with the scripted
kill-node1-mid-generation run asserting `node_down` → `resharding` → `reshard_done` →
`finish_reason == "stop"` → HTTP **200** with `degraded: true`. Run `DLLM_FAILOVER=off` first as the
negative control. **And drive step 3 with three concurrent curls plus `wait`** — one curl is the
entire reason utilisation is 33%, and at R=1 the 19x slide disagrees with the screen.

**`README.md`** — lift §1's diagram and §5's lever table. Add the honesty clause: `tie_word_embeddings`
makes node0's `embed_tokens` and node2's `lm_head` the **same 136M-param matrix**, so 545 MB fp32 is
duplicated and max single-node parameter share is **51.7%, not 33%**. Saying it first is worth more
than the claim it costs.

**`layer-nodes/requirements.txt`** — add `safetensors`, `prometheus_client`,
`opentelemetry-sdk==1.44.0`. **No new transport dependency**: DLP is stdlib `socket` + `struct`.

**`ray-vllm/`** — leave as-is, bring it up as the **baseline**, not the product. `vllm serve
--pipeline-parallel-size 3 --distributed-executor-backend mp` with
`VLLM_PP_LAYER_PARTITION="11,11,2"`, plus one run *without* it as the control. vLLM PP reproduces
FINDING 1 exactly (8 / 8 / 17.13 layer-equivalents, because `lm_head` lands on the last rank) — the
control run proves the imbalance is generic, not a PoC bug. Expect to lose on tok/s and win on the
"one node killed" row. This is the comparison the pitch currently lacks.

**Untouched:** `layer-nodes/Dockerfile.*`, `gateway/Dockerfile`, `gateway/static/index.html`
(the SSE event names it consumes are additive), `grafana/provisioning/datasources/`, `temp.ipynb`.

---

## 9. Contradictions between teams, resolved

Not averaged. Each of these had two teams disagreeing; here is which one wins and why.

| # | conflict | resolution |
|--:|---|---|
| 1 | **"3 hops" (shared context) vs "6 wire crossings" (T1-A1)** | **T1-A1 is right**, verified by reading `coordinator.py:46-63`: v0 is a *star*, so 6 crossings/token, 4 activation-sized. Anyone anchoring on 3 hops undercounts v0 wire by ~2x. FINDING 4's 935x uses 2 logical hops and is therefore *conservative* about v0 — leave it as-is. |
| 2 | **KV-cache wall clock: 28.6x (T3-A1) vs 5.35–6.8x (T1-A1, T5-A4)** | **T1-A1 wins because it measured.** T3-A1's 28.6x models cached decode at 198 ms/token from a DRAM-bandwidth argument; T1-A1 *measured* 123.94 ms. Both baselines differ too. Quote **271x for redundant compute (FLOPs, derived) and 6.77x for single-stream wall clock (modelled from measured)** — never 271x for time. |
| 3 | **int8 (T2-A1: 906 B, 20/20 exact) vs bf16-and-stop (T2-A4, T2-A5)** | **Both are right about different links.** bf16 is the v1 default and the demo-day safety net; int8+8 outlier channels is a **WAN toggle**. bf16→int8 saves 14.3 µs/token on 1 GbE against 0.88 ms of compute = 1.6%. Crossover is ~163 Mbit/s per hop. |
| 4 | **Byte codecs: worth it or not** | **Not, and this is unanimous once you read the measurements.** 2 of 60 combinations beat raw at 1 GbE, 0 at 10 GbE. Retire `lz4` from the codec enum — measured *expanding* activations 1.0036–1.0056x. Keep the `codec` field; ship `codec=0`. |
| 5 | **RDMA (T1-A2/A3 both defer)** | Agreed, and worth the number: RDMA saves **27 µs/hop**; deleting Python's HTTP+JSON+base64 saves **10,376 µs/hop**. Post-KV the link has ~1,000x headroom. v2, and only at 70B-class tensor parallelism. |
| 6 | **Utilisation notation: T3-A2's `R` vs T3-A5's `C`/`P`** | Normalised here to **U = min(1, R/(P·S))**, R = in-flight requests, S = stages (3), P = pipeline replicas. T3-A5 *generalises* T3-A2's `R ≥ S`; it does not contradict it. Note the trap: at fixed load, more replicas make utilisation **worse**. |
| 7 | **Split notation: `0-10/11-21/22-23` (FINDING 1) vs `0-11/11-22/22-24` (T3-A2/A5/T4-A4)** | Same split, two notations. **Use the half-open env-var form**, `NODE_LAYERS="0-11"` = layers 0..10, because that is what `node.py:21` parses. |
| 8 | **DLP header 40 B (T1-A4) vs 32 B (prototype HTML)** | **40 B.** T1-A4 is the spec; `assets/split-model-bench.html` has a factual defect and must be corrected. |
| 9 | **T4-A5 wants `flags` bit 4 for tracing; T1-A4 defined bits 0-3** | **Granted.** Bit 4 = `F_TRACE`, 32-byte extension, header 72 B when set, `72 % 8 == 0` so payload alignment survives. 0.88% of a decode frame when sampled, 0 when not. |
| 10 | **Boundary journal owner: coordinator (T1-A5) vs sending node (T1's P2P thesis)** | **The sending node**, because §1 freezes chain routing — the coordinator is no longer in the activation path and cannot journal what it never sees. |

---

## 10. Non-goals

Explicit, because each of these is a thing a reviewer will ask for and each is the wrong answer here.

1. **RDMA / RoCEv2 / UCX / NCCL / DPDK / XDP / SR-IOV.** 27 µs/hop of remaining headroom against
   ~1,000x of link slack. Also physically unavailable on the demo hardware (Apple M1 Pro, no PCIe
   slot, no libibverbs, Docker Desktop inside a Linux VM). Any slide implying we could demo RDMA is
   false. Revisit at 70B-class tensor parallelism.
2. **Reimplementing PagedAttention, prefix caching, or the sampler.** v1 stays on `DynamicCache` +
   LRU. If the fleet ever needs paging, **adopt vLLM** — that is a shipped, tested implementation of
   exactly this.
3. **gRPC / Arrow Flight / Cap'n Proto / MessagePack / ZeroMQ / QUIC.** Arrow Flight is the only
   serious contender and it loses on payload *shape*: our hot frame is 3,584 B, which is
   control-plane-shaped, against a ~120 MB pyarrow wheel. Revisit if batched frames exceed ~1 MB or
   nodes go polyglot. (And note: the common "protobuf varints pessimise float tensors" argument is
   **factually wrong** — `float` is wire type 5, fixed-width. Do not put it on a slide.)
4. **A DHT / gossip membership layer.** hivemind's Kademlia is right at 1,000 peers and pure overhead
   at n=3. SWIM beats O(n²) polling only above ~50 nodes. We have 3.
5. **Byte compression on the decode path.** §5 and §9.4. Hard-disable below 1 MiB.
6. **A service mesh on the tensor path.** Istio's own benchmark: +0.63 ms/hop sidecar, +0.16 ms
   ambient L4 → +1.26 ms/token → +645 ms on a 512-token response, paid forever. Mesh the control
   plane only; if the tensor path needs mTLS, do it in-process or with WireGuard (~0.05 ms).
7. **Multi-region pipeline striping.** us-east-1 ↔ us-west-2 is ~60-70 ms × 2 hops = +130 ms/token =
   **66.6 s** of pure RTT on a 512-token response. Replicate whole pipelines per region; the KV cache
   never leaves its region.
8. **Speculative decoding at 0.5B.** `lm_head` is 9.13 of a 33.13 layer-equivalent model, so every
   draft pays it: layers 0-7 early-exit costs 51.7% of the target = **0.75x, i.e. slower**. Only
   n-gram/prompt-lookup (c=0) is positive. Revisit at 7B+.
9. **Interleaved 1F1B / virtual pipeline stages / zero-bubble schedules.** These split the *backward*
   pass, which inference does not have. 1F1B degenerates to round-robin without one. Active trap.
10. **Prompt privacy, or calling this encryption.** Hidden states are exactly invertible — SIPIT
    recovers prompts in provably linear time (ICLR 2026, arXiv:2510.15511); vec2text recovers 92% of
    32-token inputs (arXiv:2310.06816); ~35% of tokens are recoverable at an 8-layer split. **The
    honest claim is governance and blast radius, not confidentiality.** MPC is 3-4 orders of
    magnitude too slow (BumbleBee >13 min/token); ZK is 26,460x too slow (2,646 s/token). TEE
    (2-8% overhead) is the v2 answer. Say which one we are.
11. **Training, fine-tuning, or backward passes.** Inference only. This is why half the pipeline-
    parallelism literature does not apply.
12. **Model-agnostic support.** Every constant here — 896, 24, 151936, GQA 14/2, the 8 outlier channel
    indices `[62, 490, 570, 53, 262, 591, 450, 208]`, `lm_head` = 9.13 layer-equivalents — is
    Qwen2.5-0.5B-specific. The **method** generalises; the numbers do not. `lm_head` being 27.6% of
    parameters is a 0.5B artifact and the shape of the win changes at 7B+.
13. **Beating a single box on price or latency.** For Qwen2.5-0.5B on CPU, **one Mac mini is 3.5x
    cheaper per token than three** — the model already fits one node. A 16-node fleet produces
    $1.73/day of tokens while burning $5.76/day of electricity, 3.3x underwater. The justification is
    the **memory wall**: 70B fp16 3-way-sharded is 47.0 GB/shard and fits nothing consumer; int4
    16-way is 2.2 GB/shard and fits a laptop. **That boundary is the product** — not price, not
    novelty, not "three small beat one big".

---

## 11. Sequencing (the one ordering constraint that matters)

```
  layer_idx renumber ──┐
                       ├──▶ KV cache ──┬──▶ continuous batching  (nothing to batch without it)
  pooled httpx ────────┤               ├──▶ session affinity + admission control
                       │               └──▶ failover journal     (the cache CREATES this problem)
  argmax on node2 ─────┤
  rebalance NODE_LAYERS┤
  chunked prefill ─────┘  (gated: at M=32 the un-pooled client tax is 537 ms and erases the win)

  DLP ──▶ bf16 (dtype field already carries it) ──▶ int8 toggle (WAN only)
  registry ──▶ reshard-on-failure ──▶ chaos demo
```

Three ordering rules, each learned the hard way by a different team:

- **`layer_idx` renumbering before the cache**, or the cache is silently wrong (§7.4).
- **Pooled connections before chunked prefill**, or the client-construction tax erases the gain.
- **Rebalancing alongside concurrency**, never alone: the sum of stage times is *invariant*, so a
  rebalance is invisible at R=1 and the 1.539x claim will look fabricated on stage.
