---
status: SYNTHESIS — one analytic model, one ladder, one scoreboard
scripts: bench/perf_model_micro.py (measures) · bench/perf_model_ladder.py (evaluates the model)
raw: bench/perf-model-micro-results.json · bench/perf-model-micro.out · bench/perf-model-ladder.out
supersedes: nothing. Defers to 01-VERIFIED-FACTS.md on every constant.
---

# 30 — THE PERFORMANCE MODEL AND THE HONEST SCOREBOARD

> ## 19x more tokens per second from the same three containers: **1.27 → 24.2 tok/s**
> **Caveat, mandatory on any slide carrying the number:** this is arithmetic composing components we
> measured separately on one laptop — 6.3x of it is single-stream latency and 3.0x needs three
> concurrent requests plus a one-env-var layer rebalance; **v1 has never been run as an integrated
> system, so it is a design target, not a result.**

Everything below is reproducible:

```bash
python3 knowledge-base/bench/perf_model_micro.py   --selftest   # arithmetic identities
python3 knowledge-base/bench/perf_model_micro.py                # ~6 s, writes the JSON
python3 knowledge-base/bench/perf_model_ladder.py               # every ladder cell, from that JSON
```

---

## 1. The model

One generated token, one request in flight, a 3-node chain driven by a coordinator that makes
**3 HTTP POSTs per token** (`coordinator.py:78-95`).

```
T_token(R=1) = T_compute  +  Σ over the 3 POSTs of [ T_ser + T_wire + T_deser + T_fixed ]  +  T_queue

              3
T_compute  =  Σ  t_i(n_eff)          n_eff = seq_len without a KV cache, 1 with one
             i=1

T_ser  + T_deser  =  c · (B_req + B_resp) / 2^20            c in ms per MB per wire crossing
T_wire            =  (B_req + B_resp) / BW  +  RTT          BW in B/s, RTT per round trip
T_fixed           =  A                                      ms per POST, payload-independent

  ⇒  T_POST(B_req, B_resp) = A + c·(B_req + B_resp)/2^20 + (B_req+B_resp)/BW + RTT

T_queue    =  0            for R ≤ N*                       (admission semaphore holds it there)
              W_q = ρ·S/(1−ρ) at the bottleneck otherwise    (M/M/1, T3-A4 §3)

X(R)       =  min( R / T_token , 1 / D_max )    tokens/s     D_max = slowest stage
N*         =  T_token / D_max                                saturation depth
U(R,S)     =  min(1, R/S)                                    S = 3 stages
```

### Every input, with its tag

| symbol | meaning | value | tag | source |
|---|---|---|---|---|
| `t_i` | per-stage forward, v0, full seq=512 | 205.81 / 197.76 / 308.97 ms | **measured** | T1-A1 §5 |
| `Σt_i` | v0 compute | **712.54 ms** | measured | T1-A1 §5 |
| `Σt_i` | KV-cached decode, flat in seq | **123.94 ms** | measured | T1-A1 §7 |
| `t_i` | KV-cached per stage | 35.80 / 34.40 / 53.74 ms | *modelled* | apportioned by the measured v0 shares (T3-A4 §2) |
| `A` | fixed cost per POST | 4 values, table §3 | **measured** | this doc |
| `c` | serialise+deserialise, ms/MB/crossing | 4 values, table §3 | **measured** | this doc |
| `BW` | link | 125 MB/s (1 GbE) · 1.25 GB/s (10 GbE) | given | 00-SHARED-CONTEXT |
| `RTT` | per POST | 0.30 ms (1 GbE) · 0.08 ms (10 GbE) · 0 (loopback) | *modelled* | 00-SHARED-CONTEXT |
| `B` | payload bytes | table §4 | derived | b64 = 4·⌈n/3⌉, exact |
| `D_max` | bottleneck stage | 53.81 ms (8/8/8) · 41.37 ms (11/11/2) | *modelled* | FINDING 1 |

**Why this shape.** `T_ser` and `T_deser` are folded into one coefficient `c` because they are measured
together (a round trip serialises twice and deserialises twice, symmetrically) and because splitting
them adds a parameter the data cannot identify. `T_fixed` is separate because it is the term that
does not scale with payload — and on this system it turns out to be the term that matters.

---

## 2. What I measured (all tagged **measured**, this host, today)

Host: Apple M1 Pro (Darwin 25.6.0, arm64, 10 core). **CPython 3.12.12, numpy 2.4.3, httpx 0.28.1,
fastapi 0.141.1, uvicorn 0.52.4** — deliberately the same interpreter and venv T1-A1 used, so the
numbers compose. Estimator: min of 15–200 reps after warm-up. Raw JSON in `bench/`.

### 2a. base64 + JSON — the v0 codec (measured)

Activation `[seq,896]` fp32, N(0,1.75) with the 972x outlier channel 62 that T2-A1 measured on the
real model. `v0_rt` = b64encode + json.dumps + json.loads + b64decode. `v1_rt` = tobytes + 40-byte
struct header + np.frombuffer.

| seq | raw B | b64 B | JSON B | b64enc ms | dumps ms | loads ms | b64dec ms | **v0_rt ms** | **v1_rt ms** | v0 ms/MB | v1 ms/MB | **x** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,584 | 4,780 | 4,844 | 0.0035 | 0.0093 | 0.0053 | 0.0048 | **0.0228** | **0.00071** | 6.68 | 0.207 | **32.3** |
| 128 | 458,752 | 611,672 | 611,738 | 0.4242 | 1.0428 | 0.5223 | 0.5798 | **2.569** | **0.0081** | 5.87 | 0.019 | **316** |
| 293 (≈1 MB) | 1,050,112 | 1,400,152 | 1,400,218 | 0.9708 | **2.3945** | 1.2356 | 1.3319 | **5.933** | **0.0208** | 5.92 | 0.021 | **285** |
| 512 | 1,835,008 | 2,446,680 | 2,446,746 | 1.8638 | 4.3815 | 2.0905 | 2.3247 | **10.661** | **0.0382** | 6.09 | 0.022 | **279** |
| 2048 | 7,340,032 | 9,786,712 | 9,786,779 | 7.6288 | 18.1138 | 8.5147 | 9.4688 | **43.726** | **0.1498** | 6.25 | 0.021 | **292** |

Throughputs on the raw basis (measured): **b64encode 980 MB/s, b64decode 750 MB/s, json.loads 827 MB/s,
json.dumps 406 MB/s.** `json.dumps` is the most expensive of the four — it rescans every base64
character for escapes that cannot occur in the base64 alphabet.

**struct.pack vs base64 (measured):** the 40-byte DLP header (`struct '<4sBBHIIIIIIBBHI'`,
`calcsize == 40`) packs in **0.00021 ms**; base64-encoding the *same* 3,584 B activation takes
**0.0035 ms**, and JSON-wrapping it another 0.0093 ms. `np.frombuffer` is **0.00033 ms at every size**
— it is a pointer cast, not a copy, which is why `v1_rt` is flat and `v0_rt` is O(n).

### 2b. The logits return path (measured) — VERIFIED FINDING 2, priced

| | B | ms |
|---|---:|---:|
| raw fp32 logits, V=151,936 | 607,744 | — |
| base64 | 810,328 (+33.3%) | enc 0.562 + dec 0.768 |
| JSON-wrapped | 810,346 | dumps 1.379 + loads 0.691 |
| **codec total, per generated token** | | **3.400** |
| `np.argmax` — the only op that touches it | | **0.064** |

> **53x more time spent moving the logit vector than reading it** (measured). T5-A4 measured 58x on
> CPython 3.14; same conclusion, different interpreter.

### 2c. Byte codecs on activation-like float data (measured; lz4 4.4.5, zstandard 0.25.0, blosc2 4.12.0)

`ratio` > 1 shrinks. `BE MB/s` = bytes saved ÷ CPU seconds spent = the link speed below which the
codec breaks even. `net_us@1GbE` = wire µs saved − CPU µs spent; **negative means the codec loses.**

| dtype | payload | codec | in B | out B | ratio | comp MB/s | decomp MB/s | CPU µs | BE MB/s | **net µs @1GbE** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 3,584 (decode) | lz4 | 3,584 | 3,607 | **0.994** | 2103 | 5132 | 2.3 | −10.0 | **−2.5** |
| fp32 | 3,584 | zstd-1 | 3,584 | 3,370 | 1.064 | 414 | 720 | 13.0 | 16.5 | **−11.3** |
| fp32 | 3,584 | blosc2 lz4+bitshuf | 3,584 | 3,409 | 1.051 | 494 | 30 | 122.7 | 1.4 | **−121.3** |
| fp32 | 1.84 MB (prefill) | zstd-1 | 1,835,008 | 1,701,901 | 1.078 | 1123 | 1122 | 3118 | 42.7 | **−2053** |
| fp32 | 1.84 MB | blosc2 lz4+bitshuf | 1,835,008 | 1,719,753 | 1.067 | 1587 | 4811 | 1466 | 78.6 | **−544** |
| fp32 | 7.34 MB | zstd-1 | 7,340,032 | 6,807,450 | 1.078 | 1066 | 986 | 13,669 | 39.0 | **−9,408** |
| bf16 | 1,792 (decode) | zstd-1 | 1,792 | 1,461 | 1.227 | 244 | 494 | 10.5 | 31.7 | **−7.8** |
| bf16 | 0.92 MB | zstd-1 | 917,504 | 718,642 | 1.277 | 1063 | 994 | 1704 | 116.7 | **−113** |
| bf16 | 0.92 MB | **blosc2 lz4+bitshuf** | 917,504 | 794,637 | 1.155 | 1413 | 3083 | 903 | **136.0** | **+79.7** |
| int8 | 895 (decode) | zstd-1 | 895 | 864 | 1.036 | 156 | 341 | 8.0 | 3.9 | **−7.7** |
| int8 | 0.46 MB | zstd-1 | 458,240 | 420,066 | 1.091 | 1146 | 1129 | 769 | 49.7 | **−463** |

**1 of 35 measured combinations is net-positive at 1 GbE** — bf16 prefill under blosc2 LZ4+bitshuffle,
worth **+80 µs on a 0.92 MB payload**. Every decode-sized cell is negative. This independently
reproduces T2-A3's conclusion (they found 2 of 60) on different data with different codecs.

> **Trap I hit and had to fix, flagged for anyone re-running this.** My first int8 pass scaled by the
> per-row absmax *including* the 972x outlier channel, so every other channel quantised to ~0 and zstd
> read **ratio 38.96**. That is T2-A5's "corpus trap" in a different costume. Excluding the outlier
> channel from the scale — which is what T2-A1's shipped codec actually does — drops it to **1.036**.
> A 37x error, silent, in the direction you want to believe. The fix is in the script with a comment.

### 2d. Transport (measured)

| | ms |
|---|---:|
| Raw framed TCP round trip, persistent socket, `TCP_NODELAY`, 3,584 B | **0.055** |
| Raw framed TCP round trip, 1.835 MB | **0.491** |
| TCP connect + close, loopback | **0.044** |
| HTTP POST round trip via real uvicorn+FastAPI, pooled, raw binary body, 3,584 B | 0.478 |
| HTTP POST round trip, pooled, b64+JSON, 3,584 B | 0.545 |
| HTTP POST round trip, **fresh `httpx.Client()` per call**, b64+JSON, 3,584 B — *v0 as written* | **5.591** |
| `ssl.create_default_context()` | **3.704** |
| `httpx.AsyncClient()` constructor | **4.232** |
| `httpx.AsyncClient(verify=False)` constructor | **0.160** |

> **3.7 ms of the 4.2 ms httpx constructor is X.509 certificate parsing for connections that will
> never use TLS** — 26x the same constructor with `verify=False`, and **84x the 0.044 ms TCP handshake
> it is nominally standing in for.** Third independent replication (T1-A1: 3.790 / 4.027 / 0.159;
> T5-A4: 5.415 / 4.123 / 0.176). This one is not noise.

---

## 3. Fitting the model: A and c (measured, two-point fit)

Each transport is characterised by two round trips (3,584 B and 1,835,008 B, symmetric). Two points,
two parameters, exact fit — asserted in `perf_model_ladder.py --selftest`, which reproduces all 8
measured round trips from the 4 fitted pairs.

| transport | **A** ms/POST | **c** ms/MB/crossing | what changed |
|---|---:|---:|---|
| **v0**: fresh `httpx` + b64/JSON | **5.546** | **6.561** | — |
| pooled `httpx` + b64/JSON | 0.500 | 6.552 | connection reuse only → **A ÷ 11.1** |
| pooled `httpx` + raw binary body | 0.476 | 0.400 | + drop base64/JSON → **c ÷ 16.4** |
| **framed TCP + zero-copy** | **0.054** | **0.125** | + drop HTTP/ASGI → **A ÷ 103, c ÷ 52** vs v0 |

The two b64/JSON rows have **identical `c` to 0.14%** (6.561 vs 6.552) while `A` falls 11x. That is the
fit validating itself: pooling changes only the payload-independent term, exactly as the model says.

Cross-check against T1-A1, which measured the fixed tax directly by a different method: they got
**5.566 ms/hop**; this fit yields **5.546 ms/POST**. Agreement to **0.4%**.

---

## 4. The ladder — cumulative, seq=512 context, R=1

Payload column definitions (`b64json(n) = 4·⌈n/3⌉ + 25`, verified against `json.dumps`):

| row | POST 0 (req→resp) | POST 1 | POST 2 | **B/token** |
|---|---|---|---|---:|
| v0 | ids 3,599 → H 2,446,705 | H → H | H → logits 810,346 | **10,600,765** |
| +KV | id 7 → h 4,805 | 4,805 → 4,805 | 4,805 → logits 810,346 | 829,573 |
| +argmax | 7 → 4,805 | 4,805 → 4,805 | 4,805 → **4** | 19,231 |
| +binary | 44 → 3,624 | 3,624 → 3,624 | 3,624 → 44 | 14,584 |
| +bf16 | 44 → 1,832 | 1,832 → 1,832 | 1,832 → 44 | 7,416 |
| +int8 | 44 → 946 | 946 → 946 | 946 → 44 | 3,872 |

*Validation:* 10,600,765 + 6 crossings × 350 B of HTTP framing = **10,602,865 B — byte-for-byte
identical to T1-A1 §2a's measured total.* The two byte models were built independently.

### 4a. Loopback (`T_wire = 0`) — isolates software cost

| # | step | B/token | B/act-hop | T_compute | T_transport | **T_token ms** | tok/s | **cum x** |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | **v0 as written** | 10,600,765 | 2,446,705 | 712.54 | 82.97 | **795.5** | 1.26 | 1.00x |
| 1 | + KV cache (last position only) | 829,573 | 4,805 | 123.94 | 21.83 | **145.8** | 6.86 | **5.46x** |
| 2 | + argmax on node2 (return the int) | 19,231 | 4,805 | 123.94 | 16.76 | **140.7** | 7.11 | 5.65x |
| 3 | + binary frame (DLP 40 B, raw fp32) | 14,584 | 3,624 | 123.94 | 16.64 | **140.6** | 7.11 | 5.66x |
| 4 | + **bf16** on the wire | 7,416 | 1,832 | 123.94 | 16.64 | **140.6** | 7.11 | 5.66x |
| 5 | + **int8** + 8 fp16 outlier channels | 3,872 | 946 | 123.94 | 16.64 | **140.6** | 7.11 | 5.66x |
| 6 | + connection reuse → framed TCP | 3,872 | 946 | 123.94 | **0.16** | **124.1** | **8.06** | **6.41x** |

### 4b. 1 GbE (`BW = 125 MB/s`, `RTT = 0.30 ms/POST`)

| # | step | B/token | T_compute | T_transport | **T_token ms** | tok/s | **cum x** |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | v0 as written | 10,600,765 | 712.54 | 168.68 | **881.2** | 1.13 | 1.00x |
| 1 | + KV cache | 829,573 | 123.94 | 29.37 | **153.3** | 6.52 | 5.75x |
| 2 | + argmax on node2 | 19,231 | 123.94 | 17.81 | **141.8** | 7.05 | 6.22x |
| 3 | + binary frame | 14,584 | 123.94 | 17.66 | **141.6** | 7.06 | 6.22x |
| 4 | + bf16 | 7,416 | 123.94 | 17.60 | **141.5** | 7.07 | 6.23x |
| 5 | + int8 | 3,872 | 123.94 | 17.57 | **141.5** | 7.07 | 6.23x |
| 6 | + connection reuse → framed TCP | 3,872 | 123.94 | **1.09** | **125.0** | **8.00** | **7.05x** |

### 4c. One worked substitution, so the table is auditable

Row 6 (int8, framed TCP, 1 GbE). `A = 0.054`, `c = 0.125`, `BW = 125e6`, `RTT = 0.30`:

```
POST0: 0.054 + 0.125·(44+946)/2^20   + (44+946)/125e6·1e3   + 0.30 = 0.054+0.000118+0.0079+0.30 = 0.362 ms
POST1: 0.054 + 0.125·(946+946)/2^20  + (946+946)/125e6·1e3  + 0.30 = 0.054+0.000226+0.0151+0.30 = 0.369 ms
POST2: 0.054 + 0.125·(946+44)/2^20   + (946+44)/125e6·1e3   + 0.30 = 0.362 ms
T_transport = 1.09 ms ;  T_compute = 123.94 ms  ⇒  T_token = 125.03 ms  ⇒  7.998 tok/s          ✔ table
```

### 4d. Concurrency — row 6 → R = 3

`D = 124.10 ms` (loopback row 6). Per-stage, transport apportioned by the measured compute shares
(*modelled*): **35.85 / 34.44 / 53.81 ms**, `D_max = 53.81` (node2, because it carries `lm_head`).

```
N* = D / D_max = 124.10 / 53.81 = 2.306  →  admit ⌈2.31⌉ = 3
```

| R | `X = min(R/D, 1/D_max)` | tok/s, 8/8/8 split | tok/s, **rebalanced** 11/11/2 (`D_max = D/3 = 41.37`) | TPOT ms |
|---:|---|---:|---:|---:|
| 1 | min(1/124.10, 1/53.81) | 8.06 | 8.06 | 124.1 |
| 2 | min(2/124.10, 1/53.81) | 16.12 | 16.12 | 124.1 |
| **3** | min(3/124.10, 1/53.81) | **18.58** | **24.17** | 161.4 |
| 4 | — | 18.58 | 24.17 | 215.3 |
| 8 | — | 18.58 | 24.17 | 430.5 |

**Past R = N\* = 3, admission buys exactly zero throughput and buys latency linearly.** That is the
numeric case for `asyncio.Semaphore(3)` and a bounded queue of 6 (T3-A4 §4), not for a 40-slot
threadpool. The rebalance (FINDING 1, one env var) is what makes R=3 worth 24.17 rather than 18.58 —
**it is not optional for the headline.**

### 4e. Scoreboard

| | v0 | v1 @ R=1 | v1 @ R=3 + rebalance |
|---|---:|---:|---:|
| tok/s | **1.273** (measured, T1-A1) | 8.06 (modelled) | **24.17** (modelled) |
| ms/token | 785.3 (measured) | 124.1 | 124.1 (TPOT 161.4 at R=3) |
| bytes/token | 10,602,865 (measured) | 3,872 | 3,872 |
| **speedup** | 1.00x | **6.3x** | **19.0x** |
| **wire reduction** | 1.00x | **2,738x** | 2,738x |
| node utilisation | 33.3% (`U = min(1,R/S)`) | 33.3% | ~100% |

`24.17 / 1.273 = 18.99x`. Independent of T5-A4, which reached **24.21 tok/s** by a different route —
**0.2% apart.**

---

## 5. The finding the ladder makes unavoidable

**Rows 3, 4 and 5 move the clock by zero.** Binary framing, bf16 and int8 — the entire compression and
wire-format workstream — buy **0.15 ms, 0.06 ms and 0.03 ms per token** at 1 GbE, against a 141.5 ms
budget. They are 0.02% of the clock each. The reason is in `A`: while `httpx.AsyncClient()` is
constructed inside every forward call, **16.64 of row 3's 17.66 ms of transport is the constructor
alone** (`3 x A = 3 x 5.546`) and another 0.90 ms is RTT — leaving 0.12 ms of payload-dependent cost.
Shrinking the payload 4x shrinks the part that was already ~0.

Re-running the same six changes cheapest-first (`perf_model_ladder.py`, ORDER SENSITIVITY block):

| # | step | B/token | T_transport | T_token ms | tok/s | cum x |
|---:|---|---:|---:|---:|---:|---:|
| 0 | v0 as written | 10,600,765 | 168.68 | 881.2 | 1.13 | 1.00x |
| 1 | **+ connection reuse FIRST** | 10,600,765 | 153.44 | 866.0 | 1.15 | 1.02x |
| 2 | + KV cache | 829,573 | 14.22 | 138.2 | 7.24 | **6.38x** |
| 3 | + argmax on node2 | 19,231 | 2.67 | 126.6 | 7.90 | 6.96x |
| 4 | + binary frame (framed TCP) | 14,584 | 1.18 | 125.1 | 7.99 | 7.04x |
| 5 | + bf16 | 7,416 | 1.12 | 125.1 | 8.00 | 7.05x |
| 6 | + int8 + outliers | 3,872 | 1.09 | 125.0 | 8.00 | 7.05x |

Same endpoint (the model is additive, so it must be), **different intermediate story**: two changes —
KV cache and argmax-on-node2 — deliver 6.96x of the 7.05x. Everything else is 1.3%.

**When compression starts to matter** (criterion: bytes saved ÷ BW > 10% of `T_token`):

| step | bytes saved/token | pays below, at measured CPU decode (123.94 ms) | pays below, at a *modelled* GPU-class 1.5 ms/token |
|---|---:|---:|---:|
| fp32 → bf16 | 7,168 | **4.6 Mbit/s** | **382 Mbit/s** |
| bf16 → int8 | 3,544 | 2.3 Mbit/s | 189 Mbit/s |
| fp32 → int8 | 10,712 | 6.9 Mbit/s | 571 Mbit/s |

> On CPU nodes, bf16 only pays below dial-up. **Once compute is GPU-fast it pays below 382 Mbit/s —
> which is most of the internet.** T2's work is a WAN and a v2-scale feature, correctly built and
> incorrectly scheduled. Ship bf16 anyway (T2-A4: KL 5.7e-5, 99.41% top-1, ~3.5 µs) because it costs
> nothing and halves the bytes on the slide; do **not** put it on the latency critical path narrative.

---

## 6. Contradictions between teams, resolved (not averaged)

| # | Conflict | Resolution |
|---|---|---|
| 1 | **"3 hops/token"** (00-SHARED-CONTEXT, T1-A4, T2, T3) vs **"6 wire crossings"** (T1-A1, T5-A3) | Both right, different units. `coordinator.py` star-routes: **3 POSTs, 6 crossings, 4 of them activation-sized.** The model uses `T_POST(B_req, B_resp)`, which is the only formulation where both counts fall out correctly. Anyone quoting "3 hops × 3,584 B" **undercounts v0 wire bytes ~2x.** FINDING 4's v1 figure assumes chain routing (2 crossings) — a change not in this ladder, so my final 3,872 B/token is the **star** number; chain routing takes it to ~1,972 B. |
| 2 | **base64 coefficient**: 5.2 ms/MB (T1-A1, end-to-end) vs 5.59 (T5-A4, py3.12 ops) vs 7.18 (T5-A4, py3.14) vs **6.56 ms/MB/crossing** (this doc) | Not a conflict: three different quantities. This doc's `c` is per **crossing** through the full uvicorn+FastAPI+httpx stack and is the term the model needs; it subsumes the others. **Pin the interpreter before quoting absolute ms** — CPython 3.14's `b64decode` is 1.76x slower than 3.12's (T5-A4, measured). |
| 3 | **Per-hop transport speedup**: 38.7x (T1-A2) vs 346x (T1-A3) vs 95.4x (T1-A4) | All three are measured and none are comparable: different payload sizes, different baselines (fresh vs pooled), different servers (uvicorn vs stdlib `http.server`). My measured equivalents, one host, one baseline: **58.1x at seq=512** (28.51 → 0.491 ms) and **101.6x at seq=1** (5.591 → 0.055 ms). Quote a per-hop ratio only with its payload size and its baseline attached, or it is meaningless. |
| 4 | **Compression**: T2-A1 ships int8+outliers (906 B, 3.96x) vs T2-A3/A4/A5 "do not compress, bf16 and stop" | Regime, not disagreement. §5 prices it: on a LAN with CPU compute, int8 is worth **0.03 ms/token**; at 382 Mbit/s with GPU compute, bf16 is worth 10% of the clock. **T2-A1 is right about the codec, T2-A4/A5 are right about the schedule.** Both go in the doc; only T2-A4/A5's framing goes on the latency slide. |
| 5 | **v0 baseline tok/s**: 1.40 (T3-A2) vs 1.273 (T1-A1, T3-A4, T5-A4) | 1.40 = `1/0.71254` = **compute only**. 1.273 = measured wall clock including transport. **Use 1.273**; it is the measured one. |
| 6 | **Utilisation notation**: `U = min(1, R/S)` (T3-A2) vs `U = min(C/(P·S), 1)` (T3-A5) | T3-A5 generalises T3-A2 to `P` pipeline replicas. Normalised here: **R** = in-flight requests, **S** = 3 stages, **P** = replicas. At P=1 they are identical. Replicas do not raise utilisation at fixed load — they lower it. |
| 7 | **Batching**: T3-A3's 32.9x at B=16 | Measured at **ctx=128** in a separate run; T1-A1's stage times are seq=512. **Do not add it to the 19x.** Quote separately or not at all. |
| 8 | **935x wire vs 19x wall clock** | Both correct, different units, and the gap *is* the engineering story: v0 is compute-bound on a LAN. My model gives **2,738x** bytes (star routing, DLP headers, int8) and **19.0x** seconds from the same ladder. Never let 935x/2,738x appear without "bytes, not seconds". |
| 9 | **Node2's bottleneck factor**: 1.55x (FINDING 1, layer-equivalents) vs 1.30x (T3-A2, measured wall clock) | 1.55x is the FLOP/parameter ratio; 1.30x is measured wall clock (`308.97/237.51`), lower because node0/node1 also carry per-call framework overhead. **Use 1.30x for any wall-clock claim, 1.55x only when explicitly talking about layer-equivalents.** |

---

## 7. Honesty section

### 7a. Assumptions, every one of them

1. **Additivity.** `T_token` is a sum of independently measured components. Fixes interact — shared
   allocators, the GIL, page cache, torch threads contending with codec threads on 2 vCPUs. Such sums
   are optimistic more often than not. This is the single largest structural risk in the ladder.
2. **`T_compute` is unchanged by transport work.** True on loopback with one request; false at R=3,
   where three concurrent forwards compete for `cpus: "2"`. **The R=3 row will come in below 24.17.**
3. **Per-stage v1 times are apportioned, not measured.** Only the 123.94 ms *sum* is measured; the
   35.80/34.40/53.74 split assumes framework overhead scales with the measured v0 shares. `D_max` and
   therefore `N*` depend on it.
4. **`RTT` is modelled** (0.30 ms on 1 GbE, 0.08 on 10 GbE) from the house numbers. No packet has
   crossed a NIC in any measurement in this document.
5. **Bare-metal macOS, not Docker.** No cgroup CPU quota, no bridge network, no veth, no conntrack.
   T1-A2 models the container tax at 10–25 µs RTT and 5–15% throughput; unverified because Docker was
   not running.
6. **`c` is measured on an idle CPU.** In a saturated pipeline the codec competes with matmuls, so
   real `c` is higher and the codec wins in §2c get *worse*, not better.
7. **The 40-byte DLP header is charged on every crossing** (T1-A4's spec). T1-A1's byte ladder omits
   it; that is why my 3,872 B/token differs from their 3,592 B.
8. **Star routing is retained throughout.** Chain routing (`NEXT_URL`) halves activation crossings and
   is not credited anywhere in the ladder — the ladder is conservative by ~1,900 B/token.
9. **`output_hidden_states=False` + last-position `lm_head` (−8 ms decode, measured) is not credited
   either.** Crediting it gives ~25.9 tok/s at R=3; **we quote 24.17.**
10. **Greedy decoding, batch 1 per stream, fp32 compute, CPU, Qwen2.5-0.5B.** No GPU number appears
    anywhere in this document except the explicitly *modelled* 1.5 ms/token crossover row.
11. **T1-A1's stage times used random-init weights** (FLOP-identical, so latency carries; output is
    garbage). No quality claim may rest on them.
12. **`T_queue = 0`** throughout, because admission holds `R ≤ N*`. Beyond `N*` use T3-A4's M/M/1
    table; at ρ=0.75, p99 = 18.4·S.

### 7b. NOT measured — by anyone, anywhere in this knowledge base

| # | Not measured | Consequence |
|---:|---|---|
| 1 | **v1, end to end.** Not one cell of §4 has been run. | The headline is a target. Say so out loud. |
| 2 | **Anything in Docker.** All timings bare-metal. | Every absolute ms could move 20–50%. |
| 3 | **Anything on a real LAN.** Loopback only — zero NIC, zero switch, zero contention, zero loss. | The whole "wired fast path" thesis is untested on wire. T1-A2's 40 ms Nagle/delayed-ACK stall **cannot fire on loopback** and must not be promised in a live demo. |
| 4 | **Concurrency > 1, ever.** `demo.sh` sends one curl. | **3.0x of the 19x lives here.** |
| 5 | **Quality after the full v1 stack.** bf16 checked in isolation (T2-A4); bf16 + KV cache + rebalanced shards together, never. | Compounding error across 3 shards is unmeasured. |
| 6 | **vLLM PP head-to-head.** T4-A1 wrote the commands; nobody ran them. | We claim a professional baseline we have not stood up. |
| 7 | **Failover time-to-recover.** There is no recovery to time. | T1-A5's ladder is a design, not a demo. |
| 8 | **Memory under load.** KV sized (16.78 MB/shard @2048) but never allocated in a 2 GB container that already peaks at ~3.95 GB from the double model load. | The memory budget is fiction until `node.py`'s loader is fixed. |
| 9 | **Models above 0.5B.** `lm_head` = 27.6% of params is a small-model artifact. | The *shape* of the win changes at 7B+. |
| 10 | **p99, soak, multi-tenancy, cold start.** Cold start is 60–90 s/node. | "How long to boot?" → 90 seconds. |
| 11 | **Real activations.** §2c uses synthetic N(0,1.75)+outlier channel; T2-A3 used real hidden states. | Opposite ratio conventions: T2-A3 reports compressed/original (fp32 best = **0.843**, i.e. shrink to 84.3%), §2c reports original/compressed (fp32 best = **1.078**, shrink to 92.8%). Real activations are slightly *more* compressible than my synthetic. **T2-A3's absolute ratios supersede mine**; the verdict (nothing pays on the decode path) is identical on both. |

### 7c. Falsification gates — run these before the number goes on a slide

| gate | pass | fails if |
|---|---|---|
| G1 KV cache landed | decode TPOT flat ±15% from seq 128 → 2048 | it still grows ⇒ cache is not being hit |
| G2 single-stream claim | v1 TPOT ≤ 150 ms, concurrency 1, seq 512 | > 200 ms ⇒ drop 6.3x, quote what you got |
| G3 concurrency claim | `rate(node_forward_seconds_total[30s])` ≥ 0.85 on all 3 nodes at R=4 | < 0.6 ⇒ the pipeline is not filling; **the 19x is dead** |
| G4 wire claim | `/proc/net/dev` delta ÷ tokens ≤ 8 KB/token at seq 512 | > 20 KB ⇒ logits or base64 still on the wire |
| G5 quality | top-1 agreement vs single-process baseline ≥ 99% over 200 greedy tokens | < 95% ⇒ pull bf16, ship fp32, lose bytes and nothing else |

**A measured 3x beats a modelled 19x in front of anyone who asks one follow-up question.**

---

## 8. Three sentences for the deck

1. **"We modelled it and then measured the model: 89% of v0's per-token clock is compute, and the
   transport cost that *is* real is 3.7 ms per hop of TLS certificate parsing on plain-HTTP
   localhost."** (`ssl.create_default_context()` = 3.704 ms measured; TCP handshake = 0.044 ms.)
2. **"Two changes out of six deliver 6.96x of the 7.05x — the KV cache and moving `argmax` onto the
   last node. bf16 and int8 together are worth 0.09 ms per token on a 125 ms budget, in either
   ordering, and we are shipping them anyway because they cost ~3.5 µs and they are what makes WAN
   work."**
3. **"1.27 → 24.2 tok/s, same three containers, 19x. 6.3x is latency and 3.0x is filling a pipeline
   that is idle two-thirds of the time — and none of it has been run yet."**
