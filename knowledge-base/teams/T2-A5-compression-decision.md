---
team: T2 — Activation Compression
agent: T2-A5
topic: Codec selection matrix, DLP negotiation, pipeline order, and the v1 diff that lands the win
headline: On the 3.5 KB decode frame that is DecentralizedLLM's hot path the winning compressor is a dtype cast, not a codec — bf16 halves the wire (3624→1832 B) at 99.41% top-1 agreement for 3.5 µs, while zstd-1 saves 194 B for 14.5 µs and loses at every link class from loopback to WAN 100 Mbit.
---

# T2-A5 — the decision

Bench: Apple M1 Pro, torch 2.10.0, numpy 2.4.3, `torch.set_num_threads(2)` (matches the 2-CPU container).
Payloads are **real Qwen2.5-0.5B-Instruct layer-7 outputs** — the actual node0→node1 tensor.
Scripts: `scratchpad/t2a5_{decide,verify,select}.py`, `wirecodec.py`.
Integrates T2-A1 (numeric formats), T2-A2 (structural), T2-A3 (byte codecs), T1-A4 (DLP frame).

> **Corpus warning, and it bit me.** My first run used T2-A3's seed prose tiled 60×; tiled text lets zstd
> find long-range matches, and int8@2048 read **ratio 0.268**. On non-repeating **wikitext-2-raw-v1** the
> same measurement reads **0.677** — a **2.5x overstatement**. Benchmarking with a repeated prompt produces
> a fake compression ratio. All numbers below are the wikitext ones. (measured)

## 1. The selection matrix

Objective per hop: `T = quantise + dequantise + compress + decompress + wire_bytes / B_link`
(`B_link` in MB/s = bytes per µs, so the division yields µs). Ties within 5% go to the simpler codec — CPU
is stolen from the layer forward pass on a 2-CPU box, so a codec that merely ties is a loss. Links:
loopback/UDS 5000 MB/s (modelled, memcpy-bound), 10 GbE 1250, 1 GbE 125, WAN 100 Mbit 12.5. Budgets:
**Q0** bit-exact · **Q1** safe (top-1 ≥ 99%) · **Q2** lossy-ok. `int8` means **T2-A1's int8 per-token +
8 fp16 outlier channels, 906 B/token** — not naive int8 (§2).

| payload | budget | loopback/UDS | 10 GbE | 1 GbE | WAN 100 Mbit |
|---|---|---|---|---|---|
| **decode 3.5 KB** | Q0 | `fp32/raw` | `fp32/raw` | `fp32/raw` | `fp32/raw` |
| | Q1 | `fp32/raw` | `fp32/raw` | **`bf16/raw`** 1.98x | **`bf16/zstd-1`** 2.37x |
| | Q2 | `fp32/raw` | `fp32/raw` | **`bf16/raw`** 1.98x | **`int8/zstd-1`** ~5.0x |
| **prefill 57 KB** | Q0 | `fp32/raw` | `fp32/raw` | `fp32/raw` | `fp32/zstd-1` 1.08x |
| | Q1 | `fp32/raw` | **`bf16/raw`** 2.00x | **`bf16/raw`** 2.00x | **`bf16/zstd-1`** 2.52x |
| | Q2 | `fp32/raw` | `bf16/raw` 2.00x | **`int8/zstd-1`** ~6.0x | **`int8/zstd-1`** ~6.0x |
| **prefill 7 MB** | Q0 | `fp32/raw` | `fp32/raw` | `fp32/blosc2:zstd+bitshuf` 1.18x | same 1.18x |
| | Q1 | **`bf16/raw`** 2.00x | **`bf16/raw`** 2.00x | **`bf16/blosc2:zstd+bitshuf`** 2.87x | same 2.87x |
| | Q2 | `bf16/raw` 2.00x | `bf16/raw` 2.00x | **`int8/blosc2:zstd+bitshuf`** ~5.8x | **`int8/zstd-1`** ~5.8x |

Ratios are vs DLP raw fp32 (payload + 40 B header). Q0/Q1 cells measured; Q2 ratios marked `~` are
**(modelled)** — measured on plain 898 B int8, then rescaled to T2-A1's 906 B layout (+0.9% bytes; moves no
winner). Full grid: `scratchpad/t2a5-selection-torch.json`.

**Three rules fall out, and they are the whole decision:**

1. **Loopback / the docker-compose demo: send raw.** Every Q0/Q1 cell at 5000 MB/s is `raw`. The hackathon
   demo runs here, so *the demo should ship no codec at all.*
2. **≥ 10 GbE: quantise, never compress.** A dtype cast is O(n) at ~1 B/cycle; a byte codec is O(n) with a
   ~100x worse constant.
3. **No byte codec ever wins a decode-sized frame on any LAN** — that frame is *latency*-bound, not
   bandwidth-bound: T1-A4 measured **89 µs RTT** per DLP hop vs **2.9 µs** of serialization, so
   compressing it optimises 3% of the hop. Independently reproduces T2-A3's verdict.

## 2. Why a dtype cast beats a codec (the mechanism)

Byte-plane order-0 entropy of real fp32 activations, bits/byte, LE plane 0 = low mantissa
(measured, T2-A3, seq=2048):

| dtype | plane 0 | plane 1 | plane 2 | plane 3 | total | floor ratio |
|---|--:|--:|--:|--:|--:|--:|
| fp32 | 7.842 | 8.000 | 7.970 | 2.838 | 26.65 / 32 b | 0.833 |
| bf16 | — | — | 7.971 | 2.838 | 10.81 / 16 b | 0.676 |

Planes 0–1 carry **15.84 of a possible 16.0 bits — 99.0% of maximum entropy.** They are noise, which is
exactly why every general codec bottoms out near ratio 0.93 on fp32. **bf16 deletes those two planes
outright** — the cast wins because it *removes* incompressible bytes instead of compressing them.

End-to-end quality, quantisation injected at **both** shard boundaries (layer 7→node1, layer 15→node2),
512 wikitext positions, vs fp32 baseline:

| wire dtype | B/token/hop | top-1 agreement | KL(q ‖ fp32) | source |
|---|--:|--:|--:|---|
| fp32 | 3584 | 100% | 0 | — |
| **bf16** | **1792** | **99.41%** | **2.6e-5 nats** | measured here |
| int8 per-token absmax | 898 | 93.16% | 3.04e-2 | measured here |
| **int8 per-token + 8 fp16 outliers** | **906** | **98.65%, 20/20 exact greedy** | — | **T2-A1 (measured)** |

**Adopt T2-A1's outlier variant as the only int8 we ship** — call their `pack`/`unpack` (their §6), do not
reinvent it. My plain per-token int8 flips 6.8% of argmax decisions; 8 extra bytes/token (0.9%) buys almost
all of it back, because ch 62 alone is a 972x outlier that absmax cannot survive. Cosine similarity reads
0.9992 for plain int8 and looks fine: **it is the wrong metric.** Top-1 agreement predicts divergence.

## 3. What is pointless — verified, not assumed

| combination | verdict | measured evidence |
|---|---|---|
| **LZ4 on any activation payload** | **Dead. Retire the enum value.** | ratio **1.0036–1.0056 in all 12** realistic dtype×size cases — it *expands*. Activations hold no repeated ≥4-byte strings. |
| **Bitshuffle on int8** | Pointless | int8 decode: `blosc2:zstd+bitshuf` 0.789 vs plain `zstd-1` **0.744**; at 7 MB a tie (0.6746 vs 0.6765). Bitshuffle separates exponent from mantissa planes; int8 has neither. |
| **Any codec on fp32** | Pointless | best fp32 ratio 0.848, for 2.4 ms. bf16 gets 0.50 for 3.5 µs. **Always quantise first; never compress fp32.** |
| **blosc2 on frames < ~1 MB** | Pointless | ~400 µs fixed cost regardless of size (198 µs comp + 186 µs decomp on a 1792 B payload). Crossover sits between 57 KB and 7 MB — call it **1 MB (modelled)**. |
| **Low-rank / top-k / PCA at H=896** | Pointless | T2-A2: the 896×k matmul costs 12–23 µs/hop and saves 8–20 µs of 1 GbE wire. Do not revisit at this hidden size. |
| **Compressing the logit vector** | **Do not — delete it instead.** | VERIFIED-FACTS FINDING 2: argmax on node2, 607,744 B → 4 B. No codec beats not sending it. T2 must not "optimise" this hop. |

**The int8+zstd question, settled.** "Compressing already-quantised int8 with zstd is pointless" is **false
on ratio, true on time.** int8 activations have order-0 entropy **5.11–5.31 bits/byte** (floor ratio
0.64–0.66) — per-token absmax squeezes most values into a narrow band (6.1% exact zeros at seq=2048).
zstd-1 hits 0.744 / 0.651 / 0.677 at 3.5 KB / 57 KB / 7 MB, **within 1.4–4.5 points of that floor**: real
redundancy, not spinning. But break-even `saved_bytes / (comp+decomp µs)` = **25.6 MB/s at decode size**,
**4.9x below 1 GbE**. Keep it for WAN and ≥57 KB prefill (break-even rises to 170 MB/s, so it wins on
1 GbE); drop it everywhere else.

## 4. Codec negotiation — coordination request to T1-A4

T1-A4's DLP header already carries `dtype` (u8 @32) and `codec` (u8 @33). Two changes; **neither touches
the 40-byte layout**, so this is enum-only churn.

**4.1 Revised `codec` enum** (replaces T1-A4 §2.3). The old one conflated dtype with codec:

| val | name | note |
|--:|---|---|
| 0 | `NONE` | the default, and the demo setting |
| 1 | `ZSTD_1` | small/medium frames on slow links |
| 2 | `ZSTD_3` | marginal (0.651→0.607 at 57 KB for 2.5x CPU); reserve, ship 1 |
| 3 | `BLOSC2_ZSTD_BITSHUF` | frames ≥ 1 MB only; `typesize` = itemsize(`dtype`), so no new field |
| 4 | ~~`lz4`~~ **retired** | measured to expand activations (§3). Burn the value, never emit it. |
| — | ~~`int8-blockwise`~~ **moved** | that is `dtype=3` plus a scale/outlier prefix, not a codec. Delete from this axis. |

**4.2 `HELLO` gains a 4-byte payload** — a `u32` codec bitmask (bit *i* = codec *i* supported).
`HELLO_ACK` returns the **intersection**. `payload_len` already describes it; no header change.

| mechanism | how |
|---|---|
| **agreement** | `HELLO{mask}` → `HELLO_ACK{mask_c & mask_s}`. Once per connection, never per frame. |
| **per-message override** | The `dtype` and `codec` fields *are* the override — already per-frame. The sender picks from §1 using `seq_len` (which it knows) and its configured link class. No renegotiation, no state. |
| **receiver lacks the codec** | Reply `ERROR` (0x7F) carrying the frame's `request_id` + `"unsupported codec N"`. **Do not close the connection** — that kills every in-flight pipelined request. Sender demotes the session to `codec=0` permanently and retries once. Unreachable if the mask was honoured; it exists so version skew degrades instead of hanging. |
| **decompression bound** | Receiver **must** pass `max_output_size = seq_len × dim1 × itemsize(dtype)`. The header states the uncompressed size, so an oversized frame is a bug or an attack, never a surprise. Trust boundary — do not drop it. |

## 5. Pipeline and its exact inverse

```
SEND   tensor[seq,H] fp32, contiguous
 1 quantise    -> dtype ∈ {fp32, bf16, int8}
                  int8 frame body = [fp16 scales | fp16 outliers | int8 rest]  (T2-A1 layout)
 2 serialise   -> little-endian bytes; zero-copy memoryview over the tensor storage
 3 filter      -> bitshuffle: ONLY inside blosc2, ONLY if dtype≠int8, ONLY if ≥1 MB
 4 byte codec  -> NONE | ZSTD_1 | BLOSC2_ZSTD_BITSHUF   (compress the whole body, scales included)
 5 frame       -> DLP header{dtype, codec, seq_len, dim1, payload_len = len(step 4)}

RECV   frame
 5' parse header; bound = seq_len × dim1 × itemsize(dtype) (+ int8 scale/outlier prefix)
 4' byte decodec, capped at `bound`             (skip if codec == 0)
 3' bitshuffle inverse                          (inside blosc2; self-describing)
 2' assert len == bound, then bytes -> tensor[seq_len, dim1] of `dtype`; split the int8 prefix
 1' dequantise -> fp32
```

Step 3 lives **inside** blosc2, never beside it — standalone bitshuffle plus standalone zstd is two full
memory traversals instead of blosc2's blocked, cache-resident one. Order is not negotiable: **quantise
before compress** (§3), **bitshuffle before the entropy coder**. Scales ride **inside the frame ahead of
the payload** (T2-A1), never in a JSON header. Step 2' must assert the length: a bf16 payload of 16×896
divides evenly into fp32 and reshapes silently into a half-height tensor — I hit this writing the reference
impl, and a wrong-but-plausible tensor is worse than a crash.

## 6. The headline claim

> **The best activation compressor for a sharded LLM is a dtype cast, not a codec.** On DecentralizedLLM's
> hot path — the 3.5 KB single-token decode frame — bf16 halves the wire, **3624 B → 1832 B (1.98x)**, for
> **3.5 µs** of CPU at **99.41% top-1 agreement** (KL 2.6e-5). zstd-1 on that frame saves **194 B** and
> costs **14.5 µs**: a break-even link of 13.4 MB/s, **9.3x slower than 1 GbE**. It loses everywhere.

Arithmetic: fp32 mantissa planes 0–1 carry 15.84/16.0 bits (99.0% of max entropy, measured) — noise, so
codecs bottom out at 0.93; bf16 deletes them for exactly 2.00x. `3584+40 = 3624`; `1792+40 = 1832`.
zstd-1: `3584 → 3390` = 194 B saved in `9.08+5.44 = 14.52 µs`; `194/14.52 = 13.4 MB/s` vs 1 GbE's 125.

**Caveats that must travel with it.** (a) 1.98x is *wire bytes*; the demo hop is RTT-bound (89 µs, T1-A4)
so wall-clock barely moves — bytes matter on 1 GbE, WAN, long prefill. (b) 99.41% is single-step
teacher-forced agreement, **not** autoregressive; errors compound over a real generation. (c) Two shard
boundaries; more shards means more quantisation steps. (d) M1 Pro, one model, one corpus.
**Where compression does earn its keep** (the honest second slide): 7 MB prefill over 1 GbE, `fp32/raw`
**58.7 ms/hop → int8+blosc2:zstd+bitshuf 13.3 ms/hop, 4.4x** — a visible TTFT win.

## 7. v1 — the smallest diff that lands it

Default `WIRE_DTYPE=bf16`. **`coordinator.py` needs no compression logic at all** — it already treats the
hidden state as an opaque base64 string and never decodes it.

### 7.1 `layer-nodes/node.py`

```python
# --- after LAYER_RANGE ---
WIRE_DTYPE = os.getenv("WIRE_DTYPE", "bf16")          # fp32 | bf16 | int8
ITEMSIZE = {"fp32": 4, "bf16": 2}
_wire_bytes_total = 0
# ponytail: int8 delegates to T2-A1's verified outlier codec; do not reimplement it here.
from int8_outlier import pack as _i8_pack, unpack as _i8_unpack, NBYTES as _I8_NB  # 906 B/token


def encode(h: torch.Tensor, wire: str) -> str:
    """[seq,H] fp32 tensor -> base64 str."""
    if wire == "bf16":
        b = h.to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    elif wire == "int8":
        b = _i8_pack(h.numpy())
    else:
        b = h.float().numpy().tobytes()
    return base64.b64encode(b).decode()


def decode(b64: str, wire: str, hidden: int, seq: int) -> torch.Tensor:
    """Inverse of encode; always returns fp32 [seq,H]. `seq` is mandatory and
    checked: without it a bf16 payload silently reshapes into a half-height
    fp32 tensor whenever the sizes divide evenly."""
    b = base64.b64decode(b64)
    want = seq * _I8_NB if wire == "int8" else seq * hidden * ITEMSIZE[wire]
    if len(b) != want:
        raise ValueError(f"{wire} payload is {len(b)} B, expected {want} B "
                         f"for [{seq},{hidden}] — dtype/shape disagreement")
    if wire == "bf16":
        a = np.frombuffer(b, dtype=np.uint8).copy()
        return torch.from_numpy(a).view(torch.bfloat16).reshape(seq, hidden).float()
    if wire == "int8":
        return torch.from_numpy(_i8_unpack(b, seq))
    return torch.from_numpy(np.frombuffer(b, dtype=np.float32).reshape(seq, hidden).copy())
```

`ForwardRequest` gains two optional fields: `seq_len: Optional[int] = None` (makes the length check
possible) and `wire_dtype: Optional[str] = None` (per-message override, §4).

```python
# --- in forward(), the incoming branch ---
        else:
            wire = req.wire_dtype or WIRE_DTYPE
            hidden = decode(req.hidden_states_b64, wire,
                            model.config.hidden_size, req.seq_len)
            out = model(inputs_embeds=hidden.unsqueeze(0), output_hidden_states=True)
            hidden = out.hidden_states[-1][0]          # [seq,H]; keep it a tensor

# --- the non-last-node return ---
        else:
            payload = encode(hidden, WIRE_DTYPE)
            _wire_bytes_total += len(payload) * 3 // 4
            return {"hidden_states_b64": payload,
                    "seq_len": int(hidden.shape[0]),
                    "wire_dtype": WIRE_DTYPE}
```

The `input_ids` branch also becomes `hidden = out.hidden_states[-1][0]` (drop `.numpy()`), so both branches
hand `encode` a `[seq,H]` tensor. Add `node_wire_bytes_total{layers="..."}` to `/metrics` — the counter the
animated prototype graphs when the demo flips `WIRE_DTYPE` live. Self-check
(`scratchpad/wirecodec.py demo()`, passes): round-trips fp32/bf16/int8 at seq ∈ {1,16,2048}, asserts exact
byte counts, and asserts a wrong `wire_dtype` **raises** rather than silently reshaping.

### 7.2 `layer-nodes/coordinator.py` — forward the node's response verbatim

```python
        r0.raise_for_status()
        h0 = r0.json()                        # was: r0.json()["hidden_states_b64"]
        r1 = await client.post(f"{NODE1_URL}/forward", json=h0, timeout=60)
        r1.raise_for_status()
        h1 = r1.json()                        # was: r1.json()["hidden_states_b64"]
        r2 = await client.post(f"{NODE2_URL}/forward", json=h1, timeout=60)
```

node0's response *is* node1's request — payload, `seq_len` and `wire_dtype` travel together and pydantic
ignores extra keys. Same two lines in `forward_chain_stream`. That is the entire coordinator change.

### 7.3 Ranked v1 actions (all tagged v1: days, CPU, docker-compose, 3 nodes)

| # | change | effort | impact |
|--:|---|---|---|
| 1 | `WIRE_DTYPE=bf16` (§7.1–7.2) | ~2 h | **2.00x wire**, 99.41% top-1, KL 2.6e-5 (measured) |
| 2 | **Ship no byte codec in the demo.** Delete it from the plan. | 0 | avoids 400 µs/frame of blosc2 fixed cost that buys nothing on loopback |
| 3 | `WIRE_DTYPE=int8` via T2-A1's codec, as the WAN-slide toggle | ~4 h | 3.83x wire at 20/20 exact greedy match (T2-A1, measured) |
| 4 | Send T1-A4 the §4 enum revision + `HELLO` mask | 30 min | retires LZ4 before anyone implements it |
| 5 | `ZSTD_1` behind `WIRE_CODEC`, off by default, on only for `seq_len ≥ 16` **and** a WAN link | ~3 h | 2.52–6.0x on WAN prefill (measured); zero cost when off |

## 8. v2 — production horizon

- **QuaRot** (Ashkboos et al., NeurIPS 2024, arXiv:2404.00456) — a Hadamard rotation *removes* the outlier
  channels, so plain int4 works with no outlier bookkeeping. The only route to 8x that survives §2.
- **fp8_e4m3** — DLP `dtype=5` is already reserved. 4x, no scale prefix, better dynamic range; needs torch
  `float8_e4m3fn` on a CPU that does not emulate it.
- **Error feedback** — carry the quantisation residual into the next step (Seide et al. 1-bit SGD;
  DeepSpeed). Turns per-step bias into zero-mean noise, which is what makes §6(b) compound slowly.
- **Re-measure on real 1 GbE / 10 GbE hardware.** Every §1 crossover is measured CPU against a *modelled*
  link speed. Revisit blosc2's multithreaded path once nodes exceed 2 cores.

## 9. Risks

| risk | severity | mitigation |
|---|---|---|
| Tiled/repetitive prompts inflate every ratio ~2.5x | **high** — it fooled my own first run | all numbers are wikitext-2; name the corpus on any slide |
| top-1 agreement is teacher-forced, not autoregressive | high | bf16's 2.6e-5 KL makes it safe regardless; **int8 needs a real generation test before it ships on** |
| Link speeds modelled, CPU measured | medium | the loopback and 10 GbE verdicts ("send raw") are robust — they hold for *any* faster link |
| Q2 ratios rescaled from 898 B to T2-A1's 906 B rather than re-measured; one model, one corpus, two shard boundaries | medium | +0.9% bytes moves no winner, but re-measure zstd on the outlier layout before quoting a Q2 ratio precisely; re-run `t2a5_verify.py` per model |
| blosc2's ~1 MB threshold interpolated between 57 KB and 7 MB | low | only affects prefill on slow links |
| `dtype` mismatch reshapes silently when sizes divide evenly | low, but corrupts output | mandatory length assert in `decode` (§5, §7.1) — hit in testing, now guarded |
