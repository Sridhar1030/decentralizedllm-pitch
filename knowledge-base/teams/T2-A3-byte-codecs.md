---
team: T2 — Activation Compression
agent: T2-A3
topic: General-purpose byte-level codecs (LZ4/Snappy/Zstd/zlib/Brotli/Blosc2) on real fp32/bf16 activation tensors
headline: "On the 3584 B decode payload entropy coding NEVER pays — every codec is net-negative at 1 GbE and above; it only flips for prefill payloads ≥1 MB on links ≤100 MB/s, where blosc2 LZ4+bitshuffle gives r=0.746 (bf16) and +3.5 ms/hop. And the whole thing is a rounding error next to deleting base64, which costs 36 ms/hop on a 7 MB prefill."
---

# T2-A3 — Byte-level codecs on activation tensors

**Verdict: DO NOT COMPRESS the decode-step activation. Do compress long prefill payloads on slow links only.
Both of those are worth less than deleting base64 and casting to bf16.**

## Method

| | |
|---|---|
| Data | **Real** Qwen2.5-0.5B-Instruct hidden states, captured at the PoC's actual shard boundaries (`hidden_states[8]` = out of layer 7 → node1; `hidden_states[16]` = out of layer 15 → node2), fp32, on real English prose. Not synthetic. |
| Payloads | seq ∈ {1, 128, 512, 2048} × {fp32, bf16} → 3584 B … 7,340,032 B |
| Control | i.i.d. `N(0,1)` fp32 array of identical shape, to separate "the model is compressible" from "IEEE-754 is compressible" |
| Machine | Apple M1 Pro, 10 core. Codecs pinned **single-threaded** (`nthreads=1`) — a layer node is a 2-CPU container already busy doing matmuls |
| Libs | lz4 4.4.5 (liblz4 1.9.4), zstandard 0.25.0 (libzstd 1.5.7), brotli 1.2.0, blosc2 4.12.0 (c-blosc2 3.3.3), cramjam 2.12.1 (Snappy), zlib stdlib |
| Throughput | Both compress and decompress MB/s are on the **uncompressed** byte basis (zstd/lz4 benchmark convention). MB = 10⁶ B. Median of ≥5 reps over a 0.25 s budget. Round-trip equality asserted for every cell. |
| Script / raw | `/Users/srpillai/CODING/DecentralizeLLMs/knowledge-base/bench/t2a3_byte_codecs.py` → `t2a3-byte-codecs-results.json` (210 measured cells) |

All numbers below are **(measured)** unless tagged (modelled).

---

## 1. Why bitshuffle — and why it doesn't save you

A float array is a **column-major structure stored row-major**. Every 4th byte is the sign+exponent byte, which
for activations takes ~30 distinct values; the other 3 bytes are mantissa, which is white noise. An LZ77 matcher
(LZ4, Snappy, Deflate, Zstd) scans linearly and never sees a match longer than 1–2 bytes, because a repeated
exponent byte is always separated by 3 random bytes. **Shuffle** transposes to plane-major order (all byte-3s,
then all byte-2s…), handing the matcher a 2048-byte run of near-identical exponent bytes. **Bitshuffle** does the
same at bit granularity, which additionally isolates the 7 always-similar high mantissa bits.

Shannon entropy per byte plane, `L8_s2048_fp32` (measured):

| plane (little-endian) | byte 0 (mantissa LSB) | byte 1 | byte 2 | byte 3 (sign+exp) | Σ / 32 bits |
|---|---|---|---|---|---|
| fp32, real activations | 7.842 | 8.000 | 7.970 | **2.838** | **0.833** |
| bf16 (2 planes) | 7.971 | — | — | **2.838** | **0.676** |
| fp32, Gaussian control | 8.000 | 8.000 | 7.969 | 2.620 | 0.831 |

Exponent field: **30 distinct values**, top 5 cover **87.2%**. Value range 2⁻²² … 2¹⁰, σ = 1.259, absmax 1571.6.
That single byte carries all the redundancy in the tensor. 26.65 of 32 bits are irreducible.

**The order-0 entropy floor for fp32 activations is r = 0.833.** Measured best across all 21 codecs: **r = 0.843**
(`blosc2:zstd+shuffle`). Zstd+shuffle is already within 1.2% of the information-theoretic order-0 bound. There is
nothing left for a better general-purpose codec to find.

**And the redundancy is not in the model.** Gaussian noise compresses to r = 0.850 vs. real activations at
r = 0.843 — a 0.7 pp difference. Activations are essentially *incompressible white noise wearing an IEEE-754
costume*. Any real gain must come from throwing bits away (quantization — T2-A1/A2's lane), not from entropy coding.

---

## 2. Compression ratio (lower = better). Full codec sweep

| codec | 3584 B fp32 (decode) | 458 KB fp32 | 7.34 MB fp32 (prefill 2048) | 3.67 MB bf16 (prefill 2048) |
|---|---|---|---|---|
| **base64 (PoC v0 today)** | **1.334** | **1.333** | **1.333** | **1.333** |
| lz4-block | 1.004 | 1.004 | 1.004 | 1.004 |
| lz4hc-9 | 1.004 | 1.004 | 1.004 | 0.995 |
| snappy | 1.001 | 1.000 | 1.000 | 1.000 |
| zstd-1 | 0.947 | 0.928 | 0.926 | 0.787 |
| zstd-3 | 0.947 | 0.929 | 0.928 | 0.786 |
| zstd-9 | 0.947 | 0.928 | 0.926 | 0.780 |
| zstd-19 | 0.945 | 0.929 | 0.927 | 0.758 |
| zlib-1 | 0.946 | 0.933 | 0.932 | 0.810 |
| zlib-6 | 0.946 | 0.931 | 0.929 | 0.803 |
| brotli-1 | 0.961 | 0.929 | 0.927 | 0.788 |
| brotli-5 | 0.943 | 0.927 | 0.926 | 0.736 |
| blosc2 lz4 + none / shuffle / **bitshuffle** | 1.009 / 1.009 / 1.009 | 1.000 / 0.955 / **0.880** | 1.000 / 0.934 / **0.880** | 1.000 / 0.869 / **0.746** |
| blosc2 lz4hc + bitshuffle | 0.923 | 0.865 | 0.855 | 0.666 |
| blosc2 zstd + none / **shuffle** / **bitshuffle** | 0.958 / 0.887 / 0.920 | 0.928 / 0.863 / 0.862 | 0.926 / **0.843** / 0.852 | 0.787 / 0.687 / **0.649** |

Reading it:
- **Raw LZ4/Snappy are useless on float data** (r ≈ 1.00). They find zero matches. This is the interleaving problem.
- **Shuffle/bitshuffle is worth 8–13 pp on fp32 and 12–14 pp on bf16.** blosc2 lz4: 1.000 → 0.880 (fp32),
  1.000 → 0.746 (bf16). That is the entire value of Blosc2 in one line.
- **Zstd level buys almost nothing.** zstd-1 → zstd-19 on fp32 prefill: 0.926 → 0.927. Level 19 is 95× slower
  for **zero** ratio gain, because there is no long-range redundancy to find. Never use level > 3 here.
- **bf16 compresses better than fp32** (0.649 vs 0.843) because dropping 16 noise bits raises the exponent
  plane's share of the payload. Quantization and entropy coding compound.

**Composition, vs. the fp32 baseline (measured × arithmetic):** fp32 raw = 1.000 → bf16 cast = 0.500 →
bf16 + blosc2 lz4+bitshuffle = 0.500 × 0.746 = **0.373**. The dtype cast contributes 2× and costs ~0 CPU;
the codec contributes 1.34× and costs milliseconds. **The cast is the win; the codec is the garnish.**

---

## 3. Throughput and CPU cost, 7.34 MB fp32 prefill payload

| codec | ratio | comp MB/s | decomp MB/s | CPU µs (c+d) |
|---|---|---|---|---|
| lz4-block | 1.004 | 15196.8 | 18706.7 | 875 |
| snappy | 1.000 | 13349.6 | 16474.4 | 995 |
| blosc2 lz4 + none | 1.000 | 9868.4 | 25750.7 | 1029 |
| **blosc2 lz4 + bitshuffle** | **0.880** | **1456.3** | **7164.8** | **6064** |
| blosc2 lz4 + shuffle | 0.934 | 1404.4 | 16693.0 | 5666 |
| zstd-1 | 0.926 | 1172.3 | 1204.7 | 12354 |
| blosc2 zstd + bitshuffle | 0.852 | 397.2 | 6916.7 | 19539 |
| brotli-1 | 0.927 | 377.8 | 179.3 | 60358 |
| blosc2 zstd + shuffle | 0.843 | 134.0 | 11633.1 | 55414 |
| zlib-1 / lz4hc-9 / zstd-19 | 0.932 / 1.004 / 0.927 | 38.8 / 54.8 / 12.4 | 436 / 17646 / 1158 | 205839 / 134294 / 599048 |
| base64 (PoC v0) | 1.333 | 991.8 | 789.3 | 16700 |

`blosc2:zstd+shuffle` wins on ratio (0.843) but at 134 MB/s compress it is 11× slower than
`blosc2:lz4+bitshuffle` for 4 pp of ratio. **Pareto winner is `blosc2:lz4+bitshuffle`.** zlib and Brotli are
dominated on every axis — never use them here.

---

## 4. The decisive arithmetic

Let S = payload bytes, r = compression ratio, B = link MB/s, T_c / T_d = compress / decompress MB/s on the
uncompressed basis. RTT and per-hop compute are identical either way, so they cancel.

```
t_plain = S/B                       t_comp = S/T_c + S·r/B + S/T_d
compression pays  iff  S/B > S/T_c + S·r/B + S/T_d
                  iff  (1−r)/B > 1/T_c + 1/T_d          ← S cancels
                  iff  B  <  (1−r) · T_eff ,   T_eff = 1/(1/T_c + 1/T_d)   ≜ B_cross
```

**S cancels out.** The crossover looks size-independent — but it is not, because T_c and T_d are *strongly*
size-dependent: every codec has a fixed per-call cost (frame header, context init, buffer allocation, Python
FFI) of ~1–150 µs. At 3584 B that fixed cost *is* the entire cost, so measured T_c collapses and B_cross with it.
This is exactly why small payloads lose. B_cross measured at each real size:

**B_cross (MB/s) — compression pays only if your link is SLOWER than this**

| codec | 3584 B fp32 | 458 KB fp32 | 1.84 MB fp32 | 7.34 MB fp32 | 1792 B bf16 | 3.67 MB bf16 |
|---|---|---|---|---|---|---|
| **blosc2 lz4 + bitshuffle** | **−0.2** | 94.1 | 135.6 | **145.6** | 1.8 | **233.3** |
| blosc2 lz4 + shuffle | −0.2 | 42.5 | 71.1 | 85.1 | −0.2 | 84.1 |
| blosc2 zstd + bitshuffle | 1.7 | 42.7 | 56.5 | 55.4 | 1.7 | 82.2 |
| zstd-1 | 14.5 | 43.7 | 43.4 | 43.8 | 27.8 | 105.5 |
| lz4-block | −5.9 | −31.6 | −35.8 | −32.7 | −4.1 | −24.1 |
| base64 (PoC v0) | −142.1 | −151.3 | −140.7 | −146.5 | −134.6 | −133.0 |

Reference links: 100 Mb WAN = 12.5 · 1 GbE = **125** · 10 GbE = 1250 · 25 GbE = 3125 MB/s.
Negative B_cross = compression is net-negative at *any* bandwidth.

## 5. Net milliseconds saved per hop (the money table)

`net = S·(1−r)/B − CPU_time`. Positive = faster. **Bold = the only wins.**

| payload | codec | r | 100 Mb WAN | **1 GbE** | 10 GbE | 25 GbE |
|---|---|---|---|---|---|---|
| 3584 B fp32 (decode) | zstd-1 | 0.947 | +0.002 | −0.012 | −0.013 | −0.013 |
| 3584 B fp32 | blosc2 lz4+bitshuffle | 1.009 | −0.158 | −0.156 | −0.156 | −0.156 |
| 3584 B fp32 | blosc2 zstd+shuffle | 0.887 | −0.146 | −0.175 | −0.178 | −0.179 |
| 1792 B bf16 (decode) | zstd-1 | 0.834 | +0.013 | −0.008 | −0.010 | −0.011 |
| 458 KB fp32 | blosc2 lz4+bitshuffle | 0.880 | **+3.80** | −0.14 | −0.54 | −0.57 |
| 7.34 MB fp32 (prefill 2048) | blosc2 lz4+bitshuffle | 0.880 | **+64.6** | **+1.00** | −5.36 | −5.78 |
| 7.34 MB fp32 | zstd-1 | 0.926 | **+31.0** | −8.02 | −11.92 | −12.18 |
| 7.34 MB fp32 | brotli-1 | 0.927 | −17.4 | −56.1 | −59.9 | −60.2 |
| 3.67 MB bf16 (prefill 2048) | blosc2 lz4+bitshuffle | 0.746 | **+70.5** | **+3.46** | −3.25 | −3.69 |
| 3.67 MB bf16 | blosc2 zstd+bitshuffle | 0.649 | **+87.3** | −5.36 | −14.63 | −15.25 |

**In the entire 1 GbE column, exactly two cells are positive, and both are `blosc2:lz4+bitshuffle` on a
2048-token prefill. At 10 GbE and 25 GbE, every single cell is negative.**

Sanity check on the decode row: at 1 GbE a 3584 B hop is 3584/125e6 = **28.7 µs** of wire time. The best possible
codec saves at most (1−0.843)×28.7 = **4.5 µs** — while LAN RTT alone is 200–500 µs and the shard's own forward
pass is 17.1 ms (measured, §7). You are fighting for 0.03% of the hop.

---

## 6. The finding that dwarfs all of the above: base64

The PoC ships `base64.b64encode(hidden.tobytes())` inside a JSON string. base64 is a **codec with r = 1.333** —
it is negative compression, applied unconditionally, on every hop.

| payload | extra wire bytes | extra wire ms @1 GbE | encode+decode CPU ms | **total cost/hop @1 GbE** |
|---|---|---|---|---|
| 3584 B fp32 (decode) | +1196 B | 0.0096 | 0.0084 | **−0.018 ms** |
| 7.34 MB fp32 (prefill 2048) | +2.44 MB | 19.55 | 16.70 | **−36.25 ms** |

**Deleting base64 on the 7.34 MB prefill is worth +36.25 ms/hop. The best compressor in this study is worth
+1.00 ms/hop.** base64 removal is a **36× larger win**, costs one line of code, and is lossless and risk-free.
(JSON string parsing of a ~10 MB base64 blob is on top of this and not counted.)

## 7. Compression CPU vs. the compute budget it steals

8-layer shard forward pass, fp32, `torch.set_num_threads(2)` (matches the PoC's 2-CPU container) — measured:

| seq_len | shard forward | wire @1 GbE (fp32) | wire as % of hop | best codec CPU | codec CPU as % of forward |
|---|---|---|---|---|---|
| 1 | 17.1 ms | 0.029 ms | **0.17%** | 0.156 ms | 0.9% |
| 128 | 40.7 ms | 3.67 ms | 8.3% | 0.58 ms | 1.4% |
| 512 | 123.0 ms | 14.7 ms | 10.7% | ~2.5 ms | 2.0% |
| 2048 | 501.0 ms | 58.7 ms | 10.5% | 6.06 ms | 1.2% |

At decode the network is **0.17% of the hop**. Compressing it is optimizing the wrong 0.17%. At prefill the
network is ~10% and the codec CPU is ~1% of compute — that is the regime where it is worth doing, and even then
the ceiling is ~1 pp of end-to-end latency at 1 GbE.

Second-order cost not in these tables: the codec runs on the same 2 cores as the matmuls. In a saturated
pipeline, compression CPU is *not* free — it directly extends the shard's critical path.

---

## 8. Recommended policy

**v1 (hackathon-demoable, hours):** one function, two thresholds, no negotiation protocol.

```python
# ponytail: constants from measured B_cross; recalibrate if you change the link or the dtype.
MIN_BYTES = 1 << 20            # below 1 MiB no codec ever wins at >= 1 GbE
MAX_LINK_MBPS = 120.0          # B_cross of blosc2 lz4+bitshuffle at 1 MiB, minus margin
CP = blosc2.CParams(codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.BITSHUFFLE],
                    clevel=5, nthreads=1)   # typesize set per dtype: 4 for fp32, 2 for bf16

def maybe_compress(buf, itemsize, link_mbps):
    if len(buf) < MIN_BYTES or link_mbps > MAX_LINK_MBPS:
        return b"\x00", buf                       # raw. the common case.
    return b"\x01", blosc2.compress2(buf, cparams=replace(CP, typesize=itemsize))
```

- **1-byte tag prefix**, not a header field. Receiver branches on it. Decompression is unconditional-safe.
- Send **raw little-endian bytes, not base64, not JSON.** This is worth more than the codec (§6).
- **`typesize` must match the element width** (4 for fp32, 2 for bf16). A wrong `typesize` silently degrades
  bitshuffle to noise — this is the single easiest bug to ship here. (I hit exactly this: passing blosc2's
  `filter=` instead of `filters=` is silently ignored and every filter variant returns the default SHUFFLE.)
- `link_mbps`: EWMA of `bytes_sent / wall_time` per peer, from the transport layer. No probing needed.
- **Do not compress decode-step tensors at all.** With a KV cache the payload is 3584 B; there is no bandwidth
  regime above 100 Mb WAN where compressing it wins, and the WAN win is +2 µs against a 10–50 ms WAN RTT.

**v2 (production, months):**

| # | Recommendation |
|---|---|
| 1 | Move the decision off the hot path entirely: **negotiate the wire dtype at pipeline setup** (fp32/bf16/fp8), and let quantization — not entropy coding — carry the compression. r = 0.500 for bf16 and 0.250 for fp8 at ~0 CPU beats the 0.843 ceiling of every byte codec in this study. |
| 2 | Fuse cast + bitshuffle + LZ4 into one pass over the tensor. Three separate passes over 7 MB is 3 extra round-trips through L2; blosc2 already blocks internally, so use `SChunk` with a blocksize tuned to L2 rather than `compress2` on a monolithic buffer. |
| 3 | Per-peer online calibration: measure `(r, T_c, T_d)` for the first N payloads on each link, compute B_cross live, drop the hard-coded 120 MB/s. Cheap and removes the only magic constant. |
| 4 | On RDMA/RoCEv2 or 25 GbE, hard-disable the codec path. B_cross never exceeds 233 MB/s for any codec measured; a 3125 MB/s link is 13× past every crossover. |
| 5 | If a WAN/federated deployment is real (Petals-style over the open internet, B ≈ 1–12 MB/s), the entire calculus inverts — then use `blosc2:zstd+bitshuffle` (r = 0.649 on bf16, +87 ms/hop at 100 Mb) and accept the CPU. |

## 9. Risks and caveats

| risk | note |
|---|---|
| Numbers are M1 Pro, single-threaded | Xeon/EPYC LZ4 throughput is broadly comparable; **ratios are hardware-independent** and are the load-bearing half of the analysis. Re-run `t2a3_byte_codecs.py` on the target CPU before trusting the ms columns. |
| Model is `f(single 0.5B model)` | Exponent-plane entropy (2.84 bits) is a property of activation *scale distribution*, not model size. Expect r within ±3 pp for larger models. The Gaussian control (0.850) bounds how much model-specific structure can ever be exploited: ~1 pp. |
| Outlier activations | absmax 1571.6 vs σ 1.259 — a ~1250σ outlier exists in real hidden states. It does not hurt lossless codecs, but it will wreck naive per-tensor int8 scaling. Flag for T2-A1/A2. |
| Blosc2 Python binding fixed cost | `decompress2` measured at ~143 µs on a 3584 B buffer (allocation-dominated). A C/Rust transport would cut this to ~5 µs — which raises small-payload B_cross from −0.2 to roughly +18 MB/s (modelled). Still below 1 GbE. The conclusion does not change. |
| Compression on a saturated pipeline | All net-ms figures assume idle CPU. Under load the codec competes with matmuls and the real net saving is lower. |
