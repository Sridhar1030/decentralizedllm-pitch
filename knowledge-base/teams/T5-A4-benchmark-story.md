---
team: T5 — Product, Narrative & Deliverables
agent: T5-A4
topic: Benchmark protocol, projected v0→v1 results with per-cell arithmetic, the one headline number, and the honesty slide
headline: >
  Same three 2-vCPU containers, 19x the tokens per second: 1.27 → 24.2 tok/s (modelled by composing
  measured per-stage times). 6.8x of that is single-stream latency we can demo end-to-end; the other
  2.8x needs three concurrent requests in flight and has never been run as an integrated system.
  Nothing in this file is an end-to-end measurement of v1, because v1 does not exist yet — say that out loud.
---

# T5-A4 — The benchmark story

**Rule of this file: no number goes on a slide unless it appears here with its arithmetic.**
`(measured)` = someone ran it. `(modelled)` = arithmetic on measured inputs, assumptions stated.
`(derived)` = pure arithmetic on `01-VERIFIED-FACTS.md` constants. **There are zero measured v1 numbers.**

## 0. Environment and how to reproduce

Bench host: Apple M1 Pro (MacBookPro18,1), 10 cores, 32 GB, Darwin 25.6.0 arm64. My micro-bench:
CPython **3.14.3**, numpy 2.4.4, **stdlib only**. Stage times borrowed from T1-A1 §5 / T3-A3 §2
(CPython 3.12.12, torch 2.10.0, `torch.set_num_threads(2)` — mirrors `docker-compose.yml` `cpus: "2"`).
Estimator: min of 15–30 reps after warm-up.

```bash
python3 knowledge-base/bench/t5a4_micro.py --selftest   # arithmetic identities: 4/3 b64, bf16 half, 3584/607744
python3 knowledge-base/bench/t5a4_micro.py              # ~15 s, writes bench/t5a4-micro-results.json
```

Absolute ms will not transfer to the demo Linux box. **Ratios and byte counts transfer. Quote ratios.**

## 1. What gets measured, and where the probe already exists

| Metric | Definition | Probe | New code? |
|---|---|---|---|
| **TTFT** | `t(first token_done SSE event) − t(request sent)` | `POST /v1/chat/completions/stream` | none |
| **TPOT / ITL** | mean and p95 of gaps between consecutive `token_done` events | same SSE stream | none |
| **E2EL** | request start → `done` event | same | none |
| **tok/s (per-stream)** | `1 / TPOT` | derived | — |
| **tok/s (aggregate)** | `Σ tokens across all streams / wall` | driver | none |
| **p95 latency** | p95 of ITL **and** of E2EL. Report both; they answer different questions | driver | none |
| **Per-stage time** | Δ between consecutive `at_node` events, n=0,1,2 | `coordinator.py:39-42,47,54` already emits `{"event":"at_node","node":N,...}` | **none — free instrumentation** |
| **Node utilisation** | `rate(node_forward_seconds_total[30s])` ∈ [0,1] | `node.py:118` counter, Prometheus already scraping | **none** |
| **Wire bytes/token** | `Δ(rx+tx) / tokens` from `docker exec <node> cat /proc/net/dev`, before/after | kernel counter | **none** |

**Nothing above needs a line of new code.** The one gap: `vllm bench serve` speaks OpenAI
`chat.completion.chunk` SSE; the coordinator emits a custom event schema, so the standard harness
**cannot drive the PoC today**. Fix is ~15 lines (accept `"stream": true` on `/v1/chat/completions`,
wrap `token_done` as `{"choices":[{"delta":{"content":...}}]}`) — **v1, hours**, and it buys the whole
professional harness for free. Until then, drive the existing SSE with a 40-line asyncio driver.

## 2. Systems under test

| # | System | Role | How to run |
|---|---|---|---|
| **B0** | `AutoModelForCausalLM`, one process, `torch.set_num_threads(2)`, `use_cache=True` | **Upper bound.** No network, no shards — everything we do is a tax against this | `.venv/bin/python` greedy generate loop, same prompts |
| **B1** | **vLLM v0.27.0** CPU, `--pipeline-parallel-size 3`, `VLLM_PP_LAYER_PARTITION="11,11,2"` | **Professional baseline** — a correct layer-split implementation | verbatim from **T4-A1 §7(A)/(B)**; do not re-derive |
| **B2** | **v0** — the PoC exactly as committed | Starting point | `docker compose up`; drive `http://localhost:8081` |
| **B3** | **v1** — ours | The claim | **does not exist yet** |

Drive **B2/B3 at the coordinator (:8081), not the gateway (:8080)** — `gateway/app.py:29` demands
`x-api-key`, which `vllm bench serve` cannot send, and excluding the proxy isolates the pipeline.
Run the gateway once separately to price the proxy hop.

```bash
vllm bench serve --backend openai-chat --base-url http://localhost:8081 \
  --endpoint /v1/chat/completions --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 48 --max-concurrency 4 --request-rate inf --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 --save-result
```

`--ignore-eos` is mandatory: without it a 0.5B model emits EOS at unpredictable lengths and TPOT
becomes a sample of prompt luck, not of the system.

## 3. Load matrix — 9 cells per system

| axis | values | why |
|---|---|---|
| concurrency | **1, 4, 16** | 1 = latency truth and the 1/3-bubble proof; 4 ≈ N\*=3 saturation (T3-A4); 16 = past saturation, where v0 has no admission control and must be shown falling over |
| shape (in→out) | **32→128** (chat), **512→128** (RAG-ish), **512→512** (long decode) | separates prefill cost from decode cost; the O(n²) defect only screams on the third |
| repeats | 3 runs, median; discard run 1 | first run pays page-cache + lazy-init |

Every run reports **8 numbers**: TTFT p50/p95, ITL p50/p95, aggregate tok/s, E2EL p95, mean node util,
wire B/token. 4 systems × 3 concurrency × 3 shapes = the table. Budget ≈ 3 h wall for B0/B2.

## 4. Measured — the cheap numbers, this box, today

### 4a. base64 + JSON is the entire v0 serialisation cost (measured, `t5a4-micro-results.json`)

Per **one** crossing of a `[seq,896]` fp32 tensor, the four v0 ops (`b64encode`→`json.dumps`→`json.loads`→`b64decode`):

| seq | raw B | b64 B | JSON B | b64enc ms | dumps ms | loads ms | b64dec ms | **Σ ms** | ms/MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,584 | 4,780 | 4,805 | 0.0048 | 0.0097 | 0.0054 | 0.0080 | **0.028** | 7.81 |
| 128 | 458,752 | 611,672 | 611,697 | 0.5691 | 1.0577 | 0.5804 | 0.9955 | **3.203** | 6.98 |
| **293 (≈1 MB)** | **1,050,112** | 1,400,152 | 1,400,177 | 1.3158 | **2.4300** | 1.3259 | 2.2822 | **7.354** | **7.00** |
| 512 | 1,835,008 | 2,446,680 | 2,446,705 | 2.4453 | 4.4005 | 2.3157 | 4.0162 | **13.178** | 7.18 |
| 2048 | 7,340,032 | 9,786,712 | 9,786,737 | 9.7727 | 17.8959 | 9.3411 | 16.7387 | **53.748** | 7.32 |

Throughputs on the raw basis: **b64enc 750–806 MB/s, b64dec 439–461 MB/s, json.dumps 410–434 MB/s,
json.loads 786–792 MB/s** (measured). The zero-copy alternative — `tobytes` + `np.frombuffer` — is
**0.00038 ms at every size** (measured; it is a pointer cast, not a copy).

**Independent replication of T1-A1** (same script, both interpreters, seq=512): CPython **3.12.12 = 5.59 ms/MB**,
CPython **3.14.3 = 7.17 ms/MB** (measured). T1-A1 reported 5.2 ms/MB end-to-end through uvicorn on 3.12.
**Agreement within 8% on the same interpreter** — the coefficient is real. (`b64decode` is 1.76x slower on
3.14 than 3.12: 4.01 vs 2.28 ms. Pin the interpreter before quoting absolute ms.)

### 4b. The logits blob: 4.28 ms of codec to do 0.07 ms of work (measured)

VERIFIED FINDING 2 — node2 ships all 151,936 fp32 logits back so the coordinator can `argmax` them.

| | B | ms |
|---|---:|---:|
| raw fp32 logits | 607,744 | — |
| base64 | 810,328 (+33.3%) | enc 0.756 + dec 1.318 |
| JSON-wrapped | 810,346 | dumps 1.445 + loads 0.766 |
| **codec total, per generated token** | | **4.285** |
| `np.argmax` — the *only* operation that touches it | | **0.0736** |

> **58x more time spent transporting the logit vector than reading it.** Fix = move `argmax` into
> `node.py` and return an int. **607,744 B → 4 B.** (v1, hours)

### 4c. Compression on the decode payload never pays (measured, stdlib codecs)

3,584 B activation, N(0,1.75) + the 972x outlier channel 62 that T2-A1 measured on the real model.
`net_us_1GbE` = wire-µs saved − CPU-µs spent; **negative means the codec loses.**

| payload | codec | out B | ratio | comp MB/s | decomp MB/s | CPU µs | net µs @1GbE | net µs @100 Mbit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3,584 fp32 | zlib-1 | 3,368 | 1.064 | 94.6 | 342.7 | 48.3 | **−46.6** | **−31.1** |
| 3,584 fp32 | lzma-0 | 3,528 | 1.016 | 15.9 | 33.5 | 333.1 | −332.7 | −328.6 |
| 3,584 fp32 | bz2-1 | 3,969 | **0.903** | 6.4 | 25.8 | 699.9 | −703.0 | −730.7 |
| 3,584 fp32 | **base64 (v0)** | 4,780 | **0.750** | 735.2 | 445.7 | 12.9 | **−22.5** | −108.6 |
| 1,792 bf16 | zlib-1 | 1,482 | 1.209 | 67.1 | 257.5 | 33.7 | −31.2 | −8.9 |
| 1.84 MB fp32 | zlib-1 | 1,709,760 | 1.073 | 39.0 | 435.0 | 51,218 | **−50,216** | −41,198 |

Every cell is negative at every link class. Consistent with T2-A3 (lz4/zstd, real weights) and T2-A5.

> **v0's only "codec" is base64 — a compressor with ratio 0.750.** The winning compression algorithm
> is *deleting base64 and casting to bf16*: 3,584 → 1,792 B, ratio 2.00, cost ~3 µs (T2-A5, measured).

### 4d. Transport: the cost is the library, not the wire (measured)

| | ms |
|---|---:|
| Empty framed round trip, persistent loopback TCP, `TCP_NODELAY` | **0.0185** |
| TCP connect + close, loopback | **0.0626** |
| HTTP/1.1 POST round trip, pooled connection, 3,584 B raw body | 0.2225 |
| HTTP/1.1 POST round trip, **fresh connection per call**, 4,805 B b64+JSON | 0.3747 |
| `ssl.create_default_context()` | **5.415** |
| `httpx.AsyncClient()` constructor (httpx 0.28.1) | **4.123** |
| `httpx.AsyncClient(verify=False)` constructor | **0.176** |

`coordinator.py:44,78` builds `async with httpx.AsyncClient()` inside every forward call, three times
per generated token, against three plain `http://` URLs.

> **3.95 ms of the 4.12 ms is X.509 certificate parsing for connections that will never use TLS —
> 23.4x the cost of the same constructor with `verify=False`, and 63x the cost of the loopback TCP handshake
> it is nominally standing in for.** Independently confirms T1-A1 §4 (4.027 / 0.159 ms on 3.12).
> The true network floor for a 3-hop token is **3 × 0.0185 = 0.056 ms**.

Honest note: my raw-TCP echo at 7 MB (5.11 ms) is *slower* than pooled HTTP (2.83 ms) — a naive Python
`recv` loop, not a protocol result. **HTTP framing is not the cost at any size we care about; the client
library and the payload are.**

## 5. Projected v0 → v1 — every cell (modelled), single stream, 512-token context

Compute inputs are **(measured)**: v0 stages 205.81 / 197.76 / 308.97 = Σ 712.54 ms (T1-A1 §5);
KV-cached decode Σ = 123.94 ms (T1-A1 §7); v0 wall = 785.3 ms/token → **1.273 tok/s (measured)**.
Transport rows use the 5.2 ms/MB coefficient (T1-A1, py3.12) and the byte counts in T1-A1 §2.
Each row is cumulative. **All totals (modelled).**

| # | Change | Arithmetic | wire B/tok | ms/tok | tok/s |
|---:|---|---|---:|---:|---:|
| — | **v0 as committed** | 712.54 compute + 17.63 httpx + 0.90 TCP + 55.14 codec (10.60 MB × 5.2) | 10,602,865 | **785.3** | **1.273** |
| 1 | **KV cache** per node per session | compute 712.54→123.94; codec 0.83 MB × 5.2 = 4.31 | 829,588 | **146.8** | 6.81 |
| 2 | **argmax on node2**, return int | −810,346 B ⇒ codec 4.31→0.10 | 19,262 | **142.6** | 7.01 |
| 3 | **Pooled client**, `verify=False`, keepalive | −17.63 −0.90 +0.06 (3 × 0.0185 measured) | 19,262 | **124.1** | 8.06 |
| 4 | **Raw binary body**, drop b64+JSON | 14,344 B @ 0.49 ms/MB = 0.007 | 14,344 | **124.0** | 8.06 |
| 5 | `output_hidden_states=False`, `lm_head` last position only | −8 (decode; −139 on prefill) | 14,344 | **116.0** | 8.62 |
| 6 | **Chain routing** node0→node1 direct (4→2 crossings) | −0.01 loopback; **halves link time on real LAN** | 7,176 | **116.0** | 8.62 |
| 7 | **bf16 on the wire** | 2x bytes, ~3 µs cast (T2-A5 measured) | **3,592** | **116.0** | **8.62** |
| | **v1 single stream** | | **2,952x fewer bytes** | **6.77x** | **6.77x** |
| 8 | **Rebalance 11/11/2+`lm_head`** (FINDING 1) | Σ unchanged ⇒ **no single-stream gain**; D_max 53.74→41.31 ms | 3,592 | 116.0 | 8.62 |
| 9 | **Concurrency N\*=3** + bounded queue K=6 | X = 1/D_max = 1/0.04131 | 3,592 | — | **24.21** |
| | **v1 at N=3** | 24.21 / 1.273 | | | **19.02x** |

Row 9 uses T3-A4's D=123.94 ms, which does **not** credit row 5's −8 ms. Deliberately conservative:
crediting it gives 25.9 tok/s. **We quote 24.2.**

### Derived rows for the other slides

| metric | v0 | v1 | factor | tag |
|---|---:|---:|---:|---|
| **TTFT**, P=32 | 204.3 ms | 112.7 ms | **1.81x** | (modelled) — v0: linear fit through measured Σ(16)=165.81 / Σ(128)=253.68 ⇒ 153.26 + 0.7846·32 = 178.4, + 25.9 transport. v1: T3-A3 fit `22.77 + 0.2931N` × 3 shards + 16.2 `lm_head`. **KV cache does not help prefill.** |
| **Node utilisation**, aggregate | **33.3%** (26.2 / 25.2 / 39.3) | **100%** | 3.00x | (derived) U = min(1, R/S); R=1, S=3 ⇒ 1/3 exactly, and Σt cancels |
| **p95 ITL** | **undefined** | ≤ (K+N)·D_max = 9 × 41.31 = **372 ms** | — | v0 has no admission control, so p95 is a property of the arrival process, not the system. v1 makes p95 a **design parameter**. |
| **Position-forwards/node**, P=32 G=512 | 147,200 | 543 | **271x** | (derived) FINDING 3 |
| **Wire bytes**, whole 512-token generation | 1,821.7 MB | 1.948 MB | **935x** | (derived) FINDING 4 |
| **Aggregate tok/s at B=4 continuous batching** | 1.273 | 68.5 | 53.8x | (modelled) T3-A3 §3, **different measurement run (ctx=128)** — do not add to row 9, quote separately or not at all |

## 6. Assumptions, stated loudly

1. **Stage times come from one M1 Pro laptop with 2 torch threads, not from the 3 Docker containers.**
   Container scheduling, cgroup throttling and the Docker bridge are absent from every number here.
2. Random-init weights were used for some stage timings (T1-A1) — FLOP-identical, so latency carries,
   but **no output-quality claim may rest on those runs.**
3. Rows 1–7 compose *independently measured* components. Composition is itself an assumption: fixes
   interact, and shared caches/allocators make such sums optimistic more often than not.
4. Row 9 assumes **3 requests are actually in flight**. At true concurrency 1 the answer is 8.62 tok/s
   and no scheduler on earth changes that (T3-A2: U = min(1, R/S)).
5. Wire bytes assume the star becomes a chain (row 6); if routing stays a star, halve the byte win.
6. bf16 treated as quality-free — T2-A4 measured KL 5.7e-5, 99.41% top-1. Free *enough*, not free.
7. Everything is batch=1 per stream, greedy, fp32 compute, CPU. **No GPU number appears anywhere.**

## 7. THE headline number

> # 19x more tokens per second from the same three containers
> ### 1.27 → 24.2 tok/s

**The caveat sentence that must appear on the same slide, in the speaker notes, and in the form:**

> *Modelled by composing per-stage times we measured on one laptop; 6.8x of it is single-stream latency
> and 2.8x is pipelining that requires three concurrent requests — we have not yet run v1 as an
> integrated system, so treat this as a design target, not a result.*

Two supporting numbers, in this order:
1. **271x** redundant position-forwards eliminated by the KV cache (derived, pure arithmetic — the
   most defensible number we own).
2. **935x** fewer bytes on the wire for one 512-token generation (derived) — *bytes, not seconds; on a
   fast LAN v0 is compute-bound, so this wins on WAN, 1 GbE and long context, not on the demo box.*

**Do not lead with 935x.** It is the biggest number and the weakest claim; leading with it invites the
one question that unravels the deck ("so how much faster is it?").

## 8. What we did NOT measure — put this slide in

| # | Not measured | Why it matters |
|---:|---|---|
| 1 | **v1 end-to-end.** Nothing in §5 has been run. Every v1 cell is arithmetic. | The headline is a target |
| 2 | **Anything in Docker.** All timings are bare-metal macOS; the demo runs in containers on a bridge network with cgroup CPU limits | Could move every absolute ms by 20–50% |
| 3 | **Anything on a real LAN.** Loopback only. Zero NIC, zero switch, zero contention, zero packet loss | The entire "wired fast path" thesis is untested on wire |
| 4 | **Output quality after the full v1 stack.** bf16 was checked in isolation (T2-A4); KV cache + bf16 + rebalanced shards together were not | Compounding error across 3 shards is unmeasured |
| 5 | **vLLM PP head-to-head.** T4-A1 gives the commands; nobody has run them | We claim a professional baseline we have not stood up |
| 6 | **Concurrency > 1, ever.** `demo.sh` sends one curl. The 1/3-bubble claim is arithmetic, not observation | 2.8x of the headline lives here |
| 7 | **Failover latency.** `demo.sh` step 5 stops node1 and shows an outage; time-to-recover is not measured because there is no recovery | The differentiator slide is a design, not a demo |
| 8 | **Memory under load.** KV cache is sized (16.78 MB/shard @2048) but never allocated in a container with a 2 GB limit | 2 GB minus 477 MB of fp32 weights is not a lot of headroom |
| 9 | **Models above 0.5B.** Every number is Qwen2.5-0.5B. `lm_head` being 27.6% of params is a *small-model* artifact | The shape of the win changes at 7B+ |
| 10 | **Multi-tenancy, p99, sustained soak, cold start.** Cold start is ~60–90 s per node (`docker-compose.yml` health `start_period`) | Judges may ask "how long to boot?" — the answer is 90 s |

Say it in one line on the slide: **"Measured: the defects. Modelled: the fixes. Nothing here is a v1 benchmark."**

## 9. Falsification gates — what would prove us wrong

| Gate | Pass threshold | Fails if |
|---|---|---|
| G1 KV cache lands | decode TPOT flat within ±15% from seq 128 → 2048 | it still grows with seq ⇒ cache is not being hit |
| G2 Latency claim | v1 TPOT ≤ **150 ms** on the demo box at concurrency 1, seq 512 | > 200 ms ⇒ drop the 6.8x, quote what we got |
| G3 Utilisation claim | `rate(node_forward_seconds_total[30s])` ≥ 0.85 on all 3 nodes at concurrency 4 | < 0.6 ⇒ the pipeline is not filling; the 19x is dead |
| G4 Wire claim | `/proc/net/dev` delta ÷ tokens ≤ **8 KB/token** at seq 512 | > 20 KB ⇒ logits or base64 still on the wire |
| G5 Quality gate | top-1 agreement vs B0 ≥ 99% over 200 greedy tokens | < 95% ⇒ pull bf16, ship fp32, lose 2x of bytes and nothing else |

**If G2 or G3 fails, the honest move is to re-cut the slide to the number we actually got.** A measured
3x beats a modelled 19x in front of judges who ask one follow-up question.
