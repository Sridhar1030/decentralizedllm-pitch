---
team: T2 — Activation Compression
agent: T2-A4
topic: Quality metrics, error compounding, and the adaptive rate controller
headline: Measured on the real model — bf16 is free (KL 5.7e-5, zero visible change) but int8 flips 7.3% of tokens to save 14 µs/token on 1 GbE; below bf16, activation compression costs more quality than it buys latency until link bandwidth drops under ~160 Mbit/s.
---

# T2-A4 — Quality guardrails for activation compression

All numbers **(measured)** unless tagged. Harness `bench/t2a4_quality_harness.py`, raw output
`bench/t2a4-quality-results.json`. Real `Qwen/Qwen2.5-0.5B-Instruct` fp32; the codec is installed as a forward
hook on the **output of layer 7 and layer 15** — the exact two tensors that cross the wire in the PoC — and
everything downstream runs on the reconstructed tensor, so compounding is real, not simulated.
Corpus: 3072 tokens of wikitext-2-raw-v1 **test** split (6 x 512). fp32 reference ppl = **18.6389**.
Codec contract `fn(np.float32[seq,896]) -> np.float32[seq,896]`; built-in self-check aborts unless the identity
codec yields KL 0 / top-1 1.0.

```
python t2a4_quality_harness.py --chunks 6 --trace --margin    # full sweep, ~4 min CPU
python t2a4_quality_harness.py --codecs bf16,int8-tok         # or: Harness().report({"mine": fn})
```

## 1. The metric table (measured)

`gain` = rel-L2 at final hidden state ÷ rel-L2 at injection. `exact` = fraction of 4 chat prompts whose
16-token greedy continuation is bit-identical to fp32. `div@` = mean token index of first divergence (16 = never).

| codec | B/tok/hop | relL2 inj | cos inj | relL2 fin | gain | KL mean | KL p99 | top-1 | ppl | Δppl% | exact | div@ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 3584 | 0 | 1.000000 | 0 | — | 0 | 0 | 1.0000 | 18.639 | 0 | 1.00 | 16.0 |
| fp16 | 1792 | 0.00009 | 1.000000 | 0.00027 | 3.01 | 0.000000 | 0.000002 | **1.0000** | 18.639 | −0.00 | 1.00 | 16.0 |
| bf16 | 1792 | 0.00210 | 0.999999 | 0.00429 | 2.05 | **0.000057** | 0.000497 | **0.9974** | 18.633 | **−0.03** | 1.00 | 16.0 |
| int8-tok | 898 | 0.02869 | 0.999581 | 0.09307 | 3.24 | 0.030573 | 0.17311 | 0.9268 | 19.239 | +3.22 | 0.50 | 10.3 |
| int6-tok | 674 | 0.05528 | 0.993096 | 0.11763 | 2.13 | 0.052537 | 0.27409 | 0.8897 | 19.290 | +3.49 | 0.50 | 12.5 |
| int4-g128 | 462 | 0.08123 | 0.967417 | 0.19070 | 2.35 | 0.148924 | 0.86291 | 0.7959 | 21.009 | +12.7 | 0.25 | 8.0 |
| int3-g128 | 350 | 0.11812 | 0.903058 | 0.37369 | 3.16 | 0.686231 | 3.54796 | 0.5713 | 35.612 | +91.1 | 0.25 | 4.5 |
| int4-tok | 450 | 0.15195 | 0.905483 | 0.41744 | 2.75 | 0.905344 | 4.62630 | 0.5215 | 40.729 | +119 | 0.00 | 4.5 |
| **int8-tensor** | 896 | 0.18781 | **0.038655** | 1.09687 | 5.84 | **10.0289** | 18.152 | **0.0068** | **411041** | +2.2e6 | 0.00 | 0.0 |
| gauss (control) | — | 0.01482 | 0.997641 | 0.04329 | 2.92 | 0.006989 | 0.05045 | 0.9557 | 18.826 | +1.00 | 0.00 | 4.5 |

`gauss` = isotropic noise injected at the same rel-L2, no compression. It is the control that separates
"error magnitude" from "error structure".

**Read the table this way:** only fp16 and bf16 are safe. Everything at or below int8 changes 7–48% of tokens.

## 2. Which metrics predict user-visible damage, which are vanity

| metric | verdict | the evidence that settles it |
|---|---|---|
| **cosine sim at injection** | **VANITY** | int8-tok scores 0.999581 — "99.96% identical" — while flipping **7.3%** of tokens. The entire usable range is 0.999→1.000; it has no resolution where decisions are made. Never put it on a slide. |
| **rel-L2, block-Frobenius** | **MISLEADING as reported** | `‖Â−A‖_F/‖A‖_F` over a [512,896] block is a *norm-weighted* RMS, so the one massive-activation token (§4, 100x the mean norm) carries ~95% of the weight and the metric reports mostly *its* error. Measured on identical tensors (both cuts, 6 chunks): |
| | | `block_F → per-token mean (ratio)`: bf16 0.0031→0.0033 (1.05x) · int8-tok 0.0441→0.0283 (0.64x) · int6-tok 0.0756→0.1158 (1.53x) · int4-tok 0.1901→0.4406 (2.32x) · int4-g128 0.0967→0.2583 (**2.67x**) · int3-g128 0.1342→0.4567 (3.40x) · int8-tensor 0.2161→0.9326 (**4.31x**). **The understatement grows with aggressiveness — it lies hardest exactly where you need it.** Report per-token mean **and** p99. |
| **rel-L2, per-token** | **USEFUL — as a free proxy** | Predicts KL via the law in §5. Free at the sender. No absolute meaning without that calibration. |
| **Δ perplexity** | **WEAK / actively misleading** | bf16 gives **−0.031%** — noise *improves* perplexity — while flipping 0.26% of tokens. The gauss control gives only +1.00% ppl while its greedy chat output **diverges at token 4.5**. Perplexity is an average over 151936 log-probs; it only fires once things are already obviously broken (int8-tensor: 411041). |
| **KL(P_fp32‖P_codec) mean** | **USEFUL** | Monotone across the whole ladder, 5 orders of dynamic range (5.7e-5 → 10.0). This is the controller's signal. |
| **KL p99** | **BEST single number** | Catches the tail the mean hides: int8-tok mean 0.0306 but p99 0.1731 (**5.7x**). Damage is concentrated, not spread. |
| **top-1 agreement** | **USEFUL, and the one to say out loud** | "7.3% of your tokens change" is a sentence a judge understands. Directly comparable across codecs. |
| **greedy exact-match / divergence step** | **THE end-task metric** | It is literally what the animated demo renders. It is also the noisiest (4 prompts) — quote it as a demo, not as a benchmark. |

**Vanity metric of the year:** cosine similarity. It is what every compression paper reports and it cannot
distinguish a codec that works from one that flips one token in fourteen.

## 3. Error compounding — the residual stream is a near-lossless channel with two amplifying layers

`--trace` injects at **cut L7 only** and measures rel-L2 at every subsequent layer against the fp32 run.

| layer | 9 | 12 | 16 | 21 | 22 | 23 | 24 (post-norm) |
|---|---|---|---|---|---|---|---|
| ‖h‖ reference | 1748 | 1757 | 1789 | 2077 | **1517** | 1665 | 6412 |
| rel-L2, int4-g128 | 0.0847 | 0.0840 | 0.0830 | 0.0921 | **0.2412** | 0.2239 | 0.1315 |
| rel-L2, bf16 | 0.00350 | 0.00347 | 0.00344 | 0.00304 | 0.00450 | 0.00570 | 0.00610 |
| abs err, int4-g128 | 148.1 | 147.6 | 148.4 | 191.3 | 365.8 | 372.7 | 843.2 |

1. **Layers 9→21 are transparent.** Absolute error moves 148.1 → 191.3 over 13 layers = **1.29x total, 1.020x per
   layer**; bf16's relative error *decays* (0.00350 → 0.00304). The residual stream is a near-lossless channel for
   injected perturbations — **not chaotic, not exponentially divergent**. Transformers do not blow up on activation noise.
2. **Layer 22 is where the damage happens.** One layer takes rel-L2 from 0.0921 to 0.2412 — **2.62x**, more than the
   previous 13 combined. The reference norm *drops* 2077 → 1517 there: layer 22 cancels residual content, so the
   same absolute error becomes a far larger fraction of what remains.
3. **Overall gain is codec-independent, ~2–3x** (§1: 2.05 / 2.35 / 2.75 / 3.16 / 3.24); only int8-tensor breaks it
   at 5.84, because its error is a few catastrophic tokens, not a distributed perturbation.
   *Budget against 2.5x: whatever rel-L2 you inject, expect ~2.5x of it at the logits.*

### 3b. Cut placement is asymmetric — and backwards from intuition (measured, 3x512 tok)

Same codec, same total bytes, only the hop changes:

| configuration | avg B/tok/hop | KL mean | KL p99 | top-1 | Δppl% |
|---|---|---|---|---|---|
| int4-g128 on **hop1 only** (L7, 16 layers downstream) | 462 | **0.0732** | 0.368 | 0.8874 | +3.8 |
| int4-g128 on **hop2 only** (L15, 8 layers downstream) | 462 | 0.1047 | 0.634 | 0.8425 | +6.5 |
| int4-g128 on both | 462 | 0.1471 | 0.972 | 0.8190 | +10.4 |
| int8-tok hop1 + int4-g128 hop2 | 680 | 0.1062 | 0.670 | 0.8366 | +7.3 |
| **int4-g128 hop1 + int8-tok hop2** | **680** | **0.0796** | 0.438 | **0.8770** | +4.6 |

**The late cut is 1.43x more damaging than the early one despite half the remaining depth** — more layers
downstream means *more* averaging, not more compounding. Therefore:

> **v1, free, one config line: aggressive codec on hop 1, conservative on hop 2.**
> Identical wire bytes, **1.33x lower KL and +4.0 points of top-1** purely from ordering.

## 4. Why int8 *per-tensor* dies — massive activations (measured)

Per-token L2 norms of the layer-7 hidden state: **mean 17.1, max 1717.9 — a 100.2x outlier, at token index 0.**

One absmax scale for the whole [512,896] block is therefore set by a token 100x larger than typical. int8 has
127 levels; a typical token receives `127/100 ≈ 1.3` of them, so it rounds to 0 or ±1. Measured: cosine 0.0387,
KL 10.03, ppl 411041, **per-token mean rel-L2 0.933** — the typical token is erased. Per-token scaling contains
it (worst token 1.88x the mean, still token 0). This is the attention-sink / massive-activation phenomenon
(Sun et al. 2024; Xiao et al. StreamingLLM 2024; the outliers that motivated LLM.int8()).

> **Per-token scaling is not an optimization, it is the difference between working and not working.**
> 2 extra bytes per 896-element row = **+0.22% wire overhead** buys a **328x KL reduction** (10.029 → 0.031).

## 5. The calibration law — a free control signal, no reference forward needed

Fitting `KL_mean = c · (rel-L2)²` across every healthy codec (bf16, int8/6/4-tok, int4/3-g128, gauss):
**c = 27.2 (median), range 12.9 – 49.2.** The gauss control lands at c = 31.8, *inside* the quantizer range — so
**for well-scaled quantizers only the error's *magnitude* matters, not its structure.** That is precisely what
makes the sender's local rel-L2 a sufficient controller input.

| KL budget (nats) | implied rel-L2 budget | cheapest codec that clears it |
|---|---|---|
| 1e-3 | 0.0061 | bf16 (0.0021) |
| 1e-2 | 0.0192 | none below bf16 |
| 5e-2 | 0.0429 | int8-tok (0.0287) |

**int8-tensor sits 10.5x above the line** (c = 284). The law is therefore also the *pathology detector*: a codec
whose measured KL exceeds `27·e²` by more than ~8x has an outlier problem, and the probe catches it in one sample.

## 6. The margin gate — flips are not uniformly distributed (measured, 512 tok)

fp32 top-1 minus top-2 logit, deciles p10/p25/p50/p75/p90 = 0.146 / 0.410 / 1.278 / 2.715 / 5.551.

| codec | flip rate | flip rate, margin < p25 | flip rate, margin > p25 | share of all flips in bottom quartile |
|---|---|---|---|---|
| bf16 | 0.39% | 1.56% | **0.00%** | **100%** |
| int8-tok | 6.84% | 26.6% | 0.26% | **97.1%** |
| int6-tok | 11.5% | 38.3% | 2.60% | 83.1% |
| int4-g128 | 17.4% | 50.0% | 6.51% | 71.9% |

Mean margin of flipped tokens 0.187 vs kept 2.361 — **12.6x separation**, and node2 already computes both logits,
so the signal costs zero.

## 7. Is any of this worth it? The arithmetic that says "stop at bf16"

Measured compute: full 24-layer fp32 forward, 512 positions, 8 threads = **0.45 s → 0.88 ms per token-position**
(the PoC's 2-CPU containers are ~4x slower, ~3.5 ms/token, modelled).
bf16 → int8-tok saves `(1792−898) × 2 hops = 1788 B/token`.

| link | time saved per token | vs 0.88 ms compute | verdict |
|---|---|---|---|
| 1 GbE (125 MB/s) | 14.3 µs (modelled) | **1.6%** | not worth 7.3% token flips |
| 10 GbE | 1.4 µs (modelled) | 0.16% | noise |
| 100 Mbit/s WAN | 143 µs (modelled) | 16% | marginal |
| 10 Mbit/s WAN | 1.43 ms (modelled) | 163% | int8 clearly pays |

**Crossover: sub-bf16 compression starts to matter below ~163 Mbit/s per hop** (1788 B ÷ 88 µs = 20.3 MB/s,
taking 10% of compute as the threshold; modelled from the measured 0.88 ms/token).

> **AUDIT CORRECTION (90-AUDIT F01) — the 0.88 ms anchor is a prefill cost, not a decode cost, and the
> 163 Mbit/s figure is ~70x too high.** `0.45 s / 512 positions` is the *amortised per-position cost of a
> batched 512-token prefill*. A single-token **decode** step — which is the only regime in which a
> 1792 B/hop activation is sent — costs **72.9 ms** (T2-A2, measured, 24 layers, 2 threads) or **123.94 ms**
> across the 3-node chain (T1-A1, measured). Re-running the same 10%-of-compute criterion against the
> measured decode step gives a crossover of **~2 Mbit/s**, not 163 Mbit/s; 30-PERF-MODEL §5 derives
> **2.3 Mbit/s** independently. Every "vs 0.88 ms compute" percentage in the table above (1.6%, 0.16%,
> 16%, 163%) is therefore ~80–140x too large, as are the §9 guardrail-overhead figures (0.11% / 0.39%).
> **The conclusion — "stop at bf16 on a LAN" — gets stronger, not weaker.** Do not put 163 Mbit/s on a slide.

> Quality-police verdict: on a LAN the PoC is compute-bound, and per VERIFIED FINDING 2/3 the real wins are the
> KV cache (271x) and moving argmax to node2 (607744 B → 4 B). **Trading a 7.3% top-1 flip rate for 14 µs is a bad
> deal.** Ship bf16. Keep everything below it behind the controller, armed only for WAN peers.

## 8. Adaptive rate controller

Ladder (all thresholds measured, from §1):

| rung | codec | B/tok/hop | e\* block-F | e\* per-token mean | KL\* |
|---|---|---|---|---|---|
| r0 | int4-g128 | 462 | 0.0967 | 0.2583 | 0.1489 |
| r1 | int6-tok | 674 | 0.0756 | 0.1158 | 0.0525 |
| r2 | int8-tok | 898 | 0.0441 | 0.0283 | 0.0306 |
| r3 | **bf16 (v1 default on LAN)** | 1792 | 0.0031 | 0.0033 | 0.000057 |
| r4 | fp32 (floor) | 3584 | 0 | 0 | 0 |

`c = 27` (§5) was fitted on **block-Frobenius** e, so the controller must use that definition — which resolves
itself in the target design: **once a KV cache lands (VERIFIED FINDING 3) the payload is a single [1,896]
position, where block-Frobenius and per-token rel-L2 are the same number.**

**The design point: the inner loop needs no reference forward.** By §5 the sender predicts `KL̂ = 27·e²` from a
quantity it already has (it must decode to validate anyway — one int4 dequant of 896 values ≈ 1 µs). The fp32
shadow probe is demoted from *controller* to *auditor*: it only re-validates `c` and catches the int8-tensor
pathology. That drops the probe period from K=32 (3.1% overhead) to **K=256 (0.39% overhead)**.

`KL_hi` is the **SLO, an operator input**, not a constant: `1e-3` = "indistinguishable from fp32" (only bf16 clears
it, §5), `5e-2` = "WAN survival mode" (int8-tok clears it). The controller climbs until the SLO is met.
```
state:  r = r_default   d = 0 (dwell)   k = 0 (probe ctr)   breaches = 0
policy: KL_hi = 1e-3 (SLO)   KL_lo = KL_hi/4   K = 256   D = 64 tok   W = 256 tok

per token t:
  e = ||dec(enc(a)) - a|| / ||a||                     # sender-local, ~1 us, FREE
  if 27*e^2 > KL_hi:            ESCALATE(1)           # (A) SLO breach, predicted
  elif e > 1.5 * e*[r]:         ESCALATE(1)           # (B) anomalous token for THIS codec
  k += 1
  if k >= K:                                          # AUDIT, not control
      k = 0
      KL = shadow_fp32_probe(t)                       # 1 extra full pipeline forward
      if KL > 8 * 27 * e^2:     ESCALATE(2); breaches += 1; k = K-1   # law broke -> re-probe now
      elif KL > KL_hi:          ESCALATE(1); breaches += 1
      elif KL < KL_lo and d >= D: DEESCALATE(1)
  d += 1

ESCALATE(n):   r = min(r+n, r4); d = 0        # fast, on any breach
DEESCALATE(1): r = max(r-1, floor); d = 0     # slow, one rung, only after dwell D
PIN:           if breaches >= 2 within W:  floor = r3 (bf16) for the rest of the request
```

Two distinct guards, deliberately not merged: **(A)** is the SLO — on an over-aggressive rung it fires every
token and drives `r` up until the budget is met. **(B)** is the anomaly detector — it catches the massive-activation
token (§4) even on a rung that is nominally within budget. Shape is **AIMD run backwards** (Jacobson 1988):
escalate fast on any breach, relax one rung after a 64-token quiet period, pin after two breaches so a
pathological prompt cannot oscillate. `ESCALATE(2)` on a law violation is deliberate — a violated law means the
*proxy itself* failed, so do not trust it for the next step.

Costs: inner loop ~1 µs/token/hop (**0.11%** of the measured 0.88 ms/token). Auditor at K=256 = **0.39%**.
Total guardrail overhead **≈0.5%** — cheaper than the 1.6% that dropping bf16→int8 would have won on 1 GbE.

**v2 upgrade — bounded loss instead of monitored loss.** Escalation cannot un-emit a token already sent. Replace
monitoring with verification: compressed pipeline = *draft*, fp32 = *verifier*, accept/reject per token
(Leviathan et al. 2023; Chen et al. 2023), gated on margin (§6) — at bf16, verifying only the bottom-quartile-margin
tokens catches **100%** of flips for 25% of the verify cost. A hard guarantee, not a threshold.

## 9. Recommendations

| # | tag | recommendation | measured impact |
|---|---|---|---|
| 1 | **v1** | Ship **bf16** on both hops. Stop there. | 2x wire, KL 5.7e-5, top-1 0.9974, Δppl −0.03%, greedy output bit-identical |
| 2 | **v1** | If any int8 is used, **per-token scaling, never per-tensor** | 328x KL reduction for +0.22% bytes |
| 3 | **v1** | **Asymmetric rungs**: aggressive codec on hop1, conservative on hop2 | same bytes, 1.33x lower KL, +4.0 pts top-1 |
| 4 | **v1** | Wire `t2a4_quality_harness.py` into CI as the merge gate: block any codec with KL_p99 > 1e-3 or top-1 < 0.999 | runs in ~4 min on CPU |
| 5 | **v1** | Report **KL p99 + top-1 agreement**. Delete cosine similarity from every table and slide. | cos 0.9996 hid a 7.3% flip rate |
| 6 | **v2** | Adaptive controller §8, armed only for peers under ~163 Mbit/s | ≈0.5% overhead |
| 7 | **v2** | Margin-gated speculative verification (§8) for a hard bound | 100% flip coverage at 25% verify cost (bf16) |
| 8 | **v2** | Re-run this harness on the real target model before trusting any of it | see limitations |

## 10. Limitations — state these before anyone quotes the table

- **3072 wikitext-2 tokens, 4 chat prompts.** Ranks codecs by orders of magnitude; not a ±0.1% claim. `exact`/`div@` are demos, not benchmarks.
- **Qwen2.5-0.5B is small.** Activation outliers worsen above ~6.7B params (Dettmers et al. 2022) — int8 gets *harder* at scale, not easier.
- The layer-22 spike is **this model's** geometry. Re-run `--trace` per model; do not assume it.
- Greedy only. Sampling hides flips (they land on low-margin tokens that were coin flips anyway), so top-1 agreement is a **conservative** bound there.
- Only dense quantizers measured. Sparsity/low-rank/learned codecs (T2-A1/A2) must be re-measured — `c = 27` is fitted on quantization error alone.

## 11. Citations

| system / work | why it matters here |
|---|---|
| Borzunov et al., **Petals** (arXiv 2209.01188, ACL 2023 demo) | The direct precedent: 8-bit dynamic blockwise quantization of activations **between pipeline servers** over the internet, ~0.4 pt accuracy change on OPT-175B (75.3 → 74.9 avg over HellaSwag/LAMBADA/WinoGrande). Validates int8-on-the-wire *for WAN*. |
| Dettmers et al., **LLM.int8()** (NeurIPS 2022); Sun et al., **Massive Activations in LLMs** (arXiv 2402.17762, 2024); Xiao et al., **StreamingLLM / attention sinks** (ICLR 2024) | Emergent outlier features and attention sinks: they name the 100x token measured in §4 and explain why per-tensor scaling dies. |
| Xiao et al., **SmoothQuant** (ICML 2023) | Migrating activation outliers into weights — the v2 fix for §4. |
| Elhage et al., **A Mathematical Framework for Transformer Circuits** (Anthropic, 2021); nostalgebraist, **logit lens** (2020) | The residual stream as a *communication channel* whose intermediate states are already decodable — the framing §3 measures, and consistent with layers 9–21 being transparent. |
| Heimersheim & Turner, **Residual stream norms grow exponentially over the forward pass** (2023) | GPT-2-XL grows 1.045x/layer; the measured Qwen norm curve (1748 → 2077, then the layer-22 drop) is the same phenomenon, and it is what makes *relative* error norm-driven. |
| **VERIFIED FINDING 2 / 3** (`01-VERIFIED-FACTS.md`) | The context for §7: logits (607744 B) and the missing KV cache (271x) dwarf anything a hidden-state codec can win. |
| Leviathan et al., **Speculative Decoding** (ICML 2023); Chen et al., **Speculative Sampling** (2023) | The v2 verify-and-rollback design in §8. |
| Jacobson, **Congestion Avoidance and Control** (SIGCOMM 1988) | AIMD; §8 is AIMD inverted onto a precision ladder. |
