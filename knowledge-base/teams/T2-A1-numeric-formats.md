---
team: T2 — Activation Compression
agent: T2-A1
topic: Numeric precision reduction for activations on the wire (fp32 → bf16/fp8/int8/int4)
headline: Qwen2.5-0.5B's residual stream has a 972x outlier channel (ch 62, |1701.9| vs 1.75 median) that
  makes naive per-tensor int8 collapse to 1.4% top-1 agreement — but int8 per-token + 8 fp16 outlier
  channels hits 906 B/token/hop (3.96x) with 20/20 exact greedy-token match, measured on our own model.
---

# T2-A1 — Numeric formats for activations on the wire

## Measurement setup (everything tagged (measured) comes from this)

Real `Qwen/Qwen2.5-0.5B-Instruct` fp32 on CPU (`torch` 2.10.0 / `transformers` 5.3.0), captured at the exact
tensors `node.py` base64s today: `hidden_states[8]` (node0→node1) and `hidden_states[16]` (node1→node2).
Eval prompt S=74; outlier channels calibrated on a **different** prompt (poem/primes/compiler/Hamlet).
Metrics: `relerr` = ‖q−x‖₂/‖x‖₂ at the cut · `agree` = top-1 next-token match vs fp32 over all 74 positions ·
`KL` = KL(fp32 ‖ quant) over the 151936-way softmax · `prefix` = identical greedy tokens out of 20 with
**both** hops quantised at every decode step. Scripts: `…/scratchpad/act_quant2.py`, `amplify.py`.

## 1. Byte table for H=896 (the arithmetic)

Per token, per hop. `H·b` where `b` = bytes/element, plus scale metadata.

| Format | elem | payload | scale metadata | **B/token** | vs fp32 | vs fp32+base64 |
|---|---|---|---|---|---|---|
| fp32 (v0 today) | 4 B | 896·4 = 3584 | — | **3584** | 1.00x | 0.75x |
| fp32 + base64 (**actual v0 wire**) | — | 3584·4/3 | — | **4779** | 0.75x | 1.00x |
| bf16 | 2 B | 896·2 = 1792 | — | **1792** | 2.00x | 2.67x |
| fp8 e4m3 / e5m2, raw cast | 1 B | 896 | — | **896** | 4.00x | 5.33x |
| fp8 + per-tensor fp16 scale | 1 B | 896 | 2 B | **898** | 3.99x | 5.32x |
| int8 **per-token** (fp16 scale) | 1 B | 896 | 2 B/token | **898** | 3.99x | 5.32x |
| int8 **per-channel** (fp16 vec) | 1 B | 896 | 1792 B **per message** | **896 + 1792/S** | 4.00x @ S→∞ | — |
| **int8 per-token + 8 fp16 outliers** | 1 B | 888·1 + 8·2 = 904 | 2 B/token | **906** | 3.96x | 5.27x |
| int4 per-token | 0.5 B | 448 | 2 B | **450** | 7.96x | 10.6x |
| int4 group-128 (7 groups) | 0.5 B | 448 | 7·2 = 14 B | **462** | 7.76x | 10.3x |

**Per-channel is a trap in the KV-cache regime.** Once T1 lands the KV cache the payload is a *single* row
`[1, 896]`, so a per-channel scale vector costs 1792 B to move 896 B of data — *worse than fp32*. Per-channel
only pays during prefill; per-token is the only granularity size-stable across prefill and decode. (Same
reason the default scheme pins its outlier channels **statically** rather than deriving them per message.)

## 2. Format anatomy — where the bits go

| Format | S/E/M | max finite | min normal | relative step (eps) | dynamic range |
|---|---|---|---|---|---|
| fp32 | 1/8/23 | 3.40e38 | 1.18e-38 | 1.19e-7 | ~2^277 |
| bf16 | 1/8/7 | 3.39e38 | 1.18e-38 | **7.81e-3** | ~2^277 (= fp32) |
| fp16 | 1/5/10 | 65504 | 6.10e-5 | 9.77e-4 | ~2^40 |
| fp8 **e4m3** | 1/4/3 | **448** | 1.56e-2 | **0.125** | ~2^18 |
| fp8 **e5m2** | 1/5/2 | **57344** | 6.10e-5 | **0.25** | ~2^35 |

(all `torch.finfo` (measured))

**bf16 vs fp16:** bf16 keeps fp32's *exponent* field and spends the loss on mantissa. It cannot overflow
where fp32 doesn't, so it needs **no scaling logic at all** — one `.astype`. fp16 is 8x more precise but caps
at 65504; our max is 1701.9 so fp16 is safe *here*, and bf16 is the safer default for the bigger model later.

**e4m3 vs e5m2 — the trade:** e4m3 buys a mantissa bit (eps 0.125) and pays with a hard ceiling at **448**.
e5m2 buys exponent range (57344) and keeps only 2 mantissa bits — a **25% quantisation step**. That is exactly
why the OCP 8-bit FP spec (Rev 1.0, Sept 2023) and NVIDIA Transformer Engine's `Format.HYBRID` use **E4M3
forward (weights + activations)** and **E5M2 backward (gradients)**: gradients span many orders of magnitude
and need range; activations are range-bounded and need precision.

**Measured consequence for us:** max activation **1701.9 > 448**, so a *raw* e4m3 cast **overflows to NaN on
the very first tensor** (1 non-finite element (measured), agreement 0.0). Raw e5m2 survives but its 2-bit
mantissa costs 5.27% relerr. **Neither raw fp8 cast is usable — fp8 requires a scale.** Given one fp16
per-tensor scale, e4m3 beats e5m2 (0.439% vs 0.770% relerr (measured)) — "precision over range for
activations", confirmed on our own model.

## 3. The outlier problem, measured on our own residual stream

Per-channel max |activation| over the 74 tokens:

| | h8 (node0→node1) | h16 (node1→node2) |
|---|---|---|
| max abs | **1701.9** | **1709.8** |
| mean abs | 0.355 | 0.465 |
| median of per-channel maxima | 1.75 | 1.865 |
| **outlier factor (max / median)** | **972.5x** | **916.9x** |
| channels > 10x median | 22 / 896 (2.5%) | 20 / 896 (2.2%) |
| channels > 20x median | 13 | 12 |
| channels > 100x median | 1 | 1 |
| excess kurtosis | **62 180** | **61 587** |
| top-8 channel indices | **62, 490, 570, 53, 262, 591, 450, 208** | **62, 490, 570, 53, 262, 591, 450, 208** |
| top-8 magnitudes | 1701.9, 124.4, 107.8, 95.9, 78.4, 73.2, 66.8, 58.6 | 1709.8, 118.3, 108.3, 96.4, 77.6, 72.0, 67.3, 58.9 |

All (measured). Two facts decide the design:

1. **A per-tensor scale is set by channel 62 and rounds everything else to zero.** int8 step = 1701.9/127
   = **13.4**, while the median channel's *entire* range is ±1.75. Not a rounding error — total information
   loss on 97% of the tensor.
2. **The outlier set is a property of the model, not the input.** The calibration prompt and the eval prompt
   yield the **identical top-8 index list**, and it is the same at both cut points → negotiate the list
   **once at handshake** (16 B), never resend it.

### Prior art (this is a solved problem; we are porting the solution to a network link)

| Work | Venue / id | What it does | What we take |
|---|---|---|---|
| **LLM.int8()** Dettmers et al. | NeurIPS 2022, arXiv:2208.07339 | Mixed-precision decomposition: outlier feature dims kept fp16, rest int8 vector-wise; emergent outliers past ~6.7B | **Our default codec — verbatim, applied to the wire instead of the GEMM** |
| **SmoothQuant** Xiao et al. | ICML 2023, arXiv:2211.10438 | Migrates difficulty activations→weights via per-channel `s_j = max|X_j|^α / max|W_j|^(1−α)`, α=0.5; W8A8 | v2 only. **Cannot help us**: we dequantise on the far side, there is no weight matrix on the wire to absorb `s` |
| **AWQ** Lin et al. | MLSys 2024 (best paper), arXiv:2306.00978 | Protects ~1% activation-salient *weight* channels, group-128 | Confirms the 1% figure; weights-only, orthogonal to the wire |
| **ZeroQuant** Yao et al. | NeurIPS 2022, arXiv:2206.01861 | Group-wise weights + **token-wise activations**, layer-by-layer KD | Token-wise activation scaling is our baseline granularity |
| **Atom** Zhao et al. | MLSys 2024, arXiv:2310.19102 | W4A4: mixed-precision outlier channels + group-128 + INT4 KV cache | The int4 recipe if we ever need 8x |
| **KIVI** Liu et al. | ICML 2024, arXiv:2402.02750 | 2-bit KV cache; **keys per-channel, values per-token** | When T1 quantises the KV cache: keys have the outliers, values don't — do not use one scheme for both |
| **KVQuant** Hooper et al. | NeurIPS 2024, arXiv:2401.18079 | Per-channel pre-RoPE key quant, non-uniform NUQ, ~1% dense outliers, 3-bit | v2 KV-cache path |
| **QuaRot** Ashkboos et al. | NeurIPS 2024, arXiv:2404.00456 | Hadamard rotation *removes* outliers → clean W4A4, no outlier bookkeeping | **The v2 answer**: rotate once offline, then plain int4 works |
| **Petals** Borzunov et al. | ACL 2023 demo, arXiv:2209.01188 | Closest system to ours: dynamic blockwise 8-bit quant of hidden states before pipeline-parallel send; **halves bandwidth, no noticeable quality effect** | Direct precedent that 8-bit on the pipeline wire is production-safe |
| **QuantPipe** Wang et al. | ICASSP 2023 | Adaptive PTQ for distributed transformer pipelines on dynamic edge links | v2: adapt bit-width to measured link bandwidth |

## 4. Measured quality — all schemes, both hops quantised

`relerr` at h8; `agree`/`KL` with hop1 only, then with **both** hops quantised; `prefix` = identical greedy
tokens out of 20 with both hops quantised at every decode step.

| Scheme | B/tok | ratio | relerr | agree (1 hop) | agree (2 hops) | KL (2 hops) | **prefix /20** |
|---|---|---|---|---|---|---|---|
| fp32 baseline | 3584 | 1.00x | 0 | 1.0000 | 1.0000 | 0 | **20** |
| **bf16** | 1792 | 2.00x | 0.00126 | 1.0000 | 1.0000 | 0.00008 | **20** |
| fp8 e4m3 raw cast | 896 | 4.00x | **NaN** | 0 | 0 | NaN | — |
| fp8 e5m2 raw cast | 896 | 4.00x | 0.05269 | 0.9054 | 0.9459 | 0.0886 | — |
| **fp8 e4m3 + per-tensor scale** | 898 | 3.99x | 0.00439 | 0.9595 | 0.9459 | 0.0327 | **20** |
| fp8 e5m2 + per-tensor scale | 898 | 3.99x | 0.00770 | 0.9595 | 0.9459 | 0.0419 | — |
| **int8 per-tensor** | 898 | 3.99x | 0.08204 | **0.0135** | **0.0000** | **11.71** | **0** |
| int8 per-token | 898 | 3.99x | 0.04552 | 0.8919 | 0.8919 | 0.1509 | 12 |
| int8 per-channel (oracle¹) | 896 | 4.00x | 0.00603 | 0.9459 | 0.9324 | 0.0438 | — |
| **int8 per-token + 8 fp16 outliers** | **906** | **3.96x** | **0.00234** | 0.9865 | 0.9865 | **0.00205** | **20** |
| int8 2-level (calib chan × per-token) | 898 | 3.99x | 0.00634 | 0.9459 | 0.9459 | 0.0564 | 11 |
| int4 per-token | 450 | 7.96x | 0.16506 | 0.6216 | 0.4459 | 1.795 | — |
| int4 group-128 | 462 | 7.76x | 0.07818 | 0.8514 | 0.7838 | 0.337 | 8 |
| int4 2-level (calib chan × g128) | 462 | 7.76x | 0.03952 | 0.3243 | 0.1351 | 4.802 | 0 |

All (measured). ¹ per-channel scale computed from the message being sent — not causal for single-token
decode, listed only as an upper bound on what channel-wise scaling can buy.

**Three findings.**
- **int8 per-tensor is the failure mode, and it is spectacular.** 1.4% agreement, KL 11.7, generation
  degenerates to `" time declaration declaration declaration declaration…"`. This row is the slide —
  LLM.int8()'s thesis reproduced on our own model in an afternoon.
- **Two schemes reproduce fp32 output exactly:** bf16 (2x) and int8+8 fp16 outliers (3.96x). fp8 e4m3+scale
  also gets 20/20 despite lower per-position agreement — its disagreements land on low-confidence positions.
- **Honest negatives.** My calibrated 2-level scheme (static per-channel vector, then per-token int8)
  *underperforms* simply keeping 8 channels in fp16, and at int4 is actively harmful (0/20) — a calibration
  vector from another prompt mis-scales the tail. Keep the outliers; don't get clever. int4 is unusable here
  in every variant.

## 5. Error compounding across the remaining layers

node2 has 8 more layers plus `norm` plus a 151936-wide `lm_head` to run on whatever node1 hands it.

| Scheme | relerr at h8 | at h16 (after 8 layers) | **amp** | at logits | **amp** |
|---|---|---|---|---|---|
| bf16 | 0.00126 | 0.00125 | 0.99x | 0.00394 | 3.14x |
| int8 per-token + 8 fp16 outliers | 0.00234 | 0.00232 | **0.99x** | 0.01598 | 6.82x |
| fp8 e4m3 + per-tensor scale | 0.00439 | 0.00448 | 1.02x | 0.03835 | 8.75x |
| int8 per-token | 0.04552 | 0.04544 | 1.00x | 0.17549 | 3.86x |
| int4 group-128 | 0.07818 | 0.07806 | 1.00x | 0.26433 | 3.38x |
| **int8 per-tensor** | 0.08204 | **0.13099** | **1.60x** | **1.25509** | **15.30x** |

All (measured). Mechanism: the residual stream is a *sum* whose L2 norm is dominated by those same outlier
channels. A scheme that preserves them has its relative error **carried, not amplified**, through 8 more
layers (0.99–1.02x). Per-tensor int8, which destroys them, is the *only* scheme that amplifies layer-to-layer
— 1.60x over 8 layers, then 15.3x through `lm_head`. **Design rule: protect the outlier channels and node2's
residual stream stays numerically sane for free; break them and the error compounds multiplicatively with
every remaining layer.** The 3–9x jump at the logits is `lm_head` projecting 896 → 151936, unavoidable and
survivable because argmax only needs the ordering preserved.

## 6. Reference codec (self-checked: emits exactly 906.0 B/token, relerr 0.00234)

```python
import numpy as np
H = 896
# Calibrated ONCE offline, identical for both cut points and across unrelated prompts (measured).
# Negotiated at handshake (16 B), never resent.
OUTLIER = np.array([62, 490, 570, 53, 262, 591, 450, 208], dtype=np.uint16)
KEEP = np.ones(H, bool); KEEP[OUTLIER] = False          # the 888 int8 channels

def encode(h: np.ndarray) -> bytes:                      # h: fp32 [S, 896]
    hi   = h[:, OUTLIER].astype(np.float16)              # outliers stay fp16
    rest = h[:, KEEP]                                    # [S, 888]
    s    = (np.abs(rest).max(1, keepdims=True) / 127.0).astype(np.float16)   # PER-TOKEN scale
    q    = np.clip(np.rint(rest / s.astype(np.float32)), -128, 127).astype(np.int8)
    return s.tobytes() + hi.tobytes() + q.tobytes()      # 2S + 16S + 888S = 906 B/token

def decode(buf: bytes, S: int) -> np.ndarray:
    n, o = len(OUTLIER), 0
    s  = np.frombuffer(buf, np.float16, S,       o).astype(np.float32).reshape(S, 1); o += 2*S
    hi = np.frombuffer(buf, np.float16, S*n,     o).astype(np.float32).reshape(S, n); o += 2*S*n
    q  = np.frombuffer(buf, np.int8,   S*(H-n),  o).reshape(S, H-n).astype(np.float32)
    h = np.empty((S, H), np.float32)
    h[:, OUTLIER] = hi
    h[:, KEEP]    = q * s                                # broadcast per-token dequant
    return h
```

Notes that matter: scale is **fp16, not fp32** (2 B/token; plenty of precision for a scalar divisor).
`np.rint` is round-half-to-even — never `astype(np.int8)` on the raw quotient, it truncates toward zero and
adds a systematic bias. Clamp `[-128, 127]`, not `[-127, 127]` — the extra negative code is free. Guard
`np.maximum(s, 1e-8)` for all-zero padding rows at prefill.

### Where the scale metadata lives in the frame

Replace the base64-in-JSON body with a little-endian binary frame (`Content-Type: application/x-dllm-act`).
Scales live **inside the frame, ahead of the payload** — never in a JSON header, because a per-token scale
array is O(S) and would put us straight back into string parsing.

```
off  size            field
  0  u32             magic 0x444C4C44 'DLLD'
  4  u8              version = 1
  5  u8              codec   0=fp32 1=bf16 2=fp8e4m3+scale 3=int8_pt_outlier 4=int4_g128
  6  u16             n_outlier (0 if codec has none)   <- 8 for the default
  8  u32             seq_len S
 12  u16             hidden H = 896
 14  u16             flags (bit0 = last-position-only, i.e. KV-cache decode step)
 16  u64             request_id  (for T4's queueing / tracing)
 24  --------------- 24 B fixed header, amortised to 0.25 B/token at S=96
 24  fp16[S]         per-token scales          <- SCALE METADATA LIVES HERE
     fp16[S*n_out]   outlier channel values
     int8[S*(H-n)]   quantised payload
```

`OUTLIER` (16 B) is **not** in the frame: it rides the one-time `/handshake` response next to the model hash,
so a scale-metadata change forces a version bump instead of a silent mismatch. Both sides assert
`len(buf) == 24 + S*(2 + 2*n + (H-n))` before decoding — a truncated frame must fail loudly, not decode into
garbage activations that will look like a model bug.

## 7. What this is worth on the real path

v0 today: 74-token prompt + 32 generated tokens, no KV cache (hop payload = the *whole* growing sequence),
2 inter-node hops. Σ of sequence lengths over the 32 steps = 2864 token-rows per hop.

| Wire format | bytes moved | 1 GbE @125 MB/s | per token |
|---|---|---|---|
| fp32 + base64 (**v0 actual**) | 2864·2·3584·4/3 = **27.37 MB** | **219 ms** | 6.84 ms |
| bf16, raw binary | 2864·2·1792 = **10.26 MB** | 82 ms | 2.57 ms |
| **906 B codec, raw binary** | 2864·2·906 = **5.19 MB** | **41.5 ms** | 1.30 ms |

**5.27x fewer bytes, 177 ms saved over a 32-token completion (modelled from measured byte counts).**

Two honesty notes for the deck. (a) On a single-host docker-compose the link is a loopback bridge, so most of
that 177 ms surfaces as **base64 + JSON-parse CPU**, not bandwidth; the bandwidth win is real only across
physical machines — which is the project's whole premise. (b) Once T1 lands the KV cache the hop payload
collapses to one row: 32·2·4779 = 0.31 MB → 32·2·906 = 0.058 MB. Quantisation **multiplies** with the KV
cache rather than competing with it — but the KV cache is the larger single lever and should ship first.

## 8. Recommendation

**Default, plainly: int8 per-token with the 8 outlier channels kept in fp16.** 906 B/token/hop — **3.96x
smaller than fp32, 5.27x smaller than the base64 fp32 on the wire today** — reproducing fp32's greedy output
**exactly for 20/20 tokens with both hops quantised** (measured), at KL 0.002 and 0.99x error amplification
through node2's remaining 8 layers. ~15 lines of numpy per side, no new dependency, no GPU, no runtime
calibration (the index list is fixed and measured to be input-independent). **v1, ~1 day.**

**Safe fallback: bf16.** 1792 B, 2x, one cast per side, zero scale metadata, zero outlier bookkeeping,
bit-identical greedy output (measured), and it cannot overflow anywhere fp32 doesn't. Ship this first
(**v1, ~1 hour**), then turn the int8 codec on; if it misbehaves on demo day, flip `codec=1` in the header.

**Middle option if the outlier list feels like too much machinery: fp8 e4m3 + one fp16 per-tensor scale.**
898 B, 3.99x, 20/20 greedy match (measured), ~5 lines, and it is the format H100/Ada/Blackwell accelerate
natively, so it survives the port to GPUs. Less headroom (8.75x logit amplification), and **it must have the
scale** — a raw e4m3 cast NaNs on our very first tensor (measured). **v1, ~2 hours.**

**Do not ship: int8 per-tensor (any variant), raw fp8 of either flavour, int4 (any grouping).** Per-tensor
int8 is not "slightly worse", it is broken output. int4's best variant manages 8/20 tokens, and the 456 B/tok
it saves over the default does not buy a different completion.

**v2 (months):** (1) **QuaRot**-style offline Hadamard rotation of the two cut-point bases — *removes* the
outliers instead of special-casing them, after which plain group-wise int4 becomes viable and the 448 B tier
opens. (2) Native **fp8 e4m3** end-to-end on Hopper/Blackwell with Transformer Engine-style delayed scaling
(amax history) instead of a per-message scale. (3) Adaptive bit-width per link à la **QuantPipe**, driven by
T4's measured queue depth. (4) When T1 quantises the KV cache, follow **KIVI**: per-channel for keys,
per-token for values — the outlier structure lives in the keys, and one scheme for both repeats exactly the
per-tensor mistake above.
