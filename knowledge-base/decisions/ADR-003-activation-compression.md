---
id: ADR-003
title: Activation compression — a dtype ladder, not a codec
status: v1 accepted (bf16); int8+outliers v1 gated; byte codecs rejected; structural compression rejected
date: 2026-09-01
sources: teams/T2-A1, T2-A2, T2-A3, T2-A4, T2-A5, T4-A1, T1-A1
---

# ADR-003 — Activation compression: dtype ladder + gated codec policy

## Context

The brief asked for "a compression algorithm for transferring embeddings/activations". Five agents measured it
on the real Qwen2.5-0.5B and converged on an answer the brief did not expect: **the winning compressor is a
dtype cast, and no byte codec ever pays on the decode path.**

The governing physical fact is IEEE-754, not the model. fp32 byte-plane order-0 entropy (LSB→MSB) is
`[7.842, 8.000, 7.970, 2.838]` bits = 26.65/32 → a lossless floor of **r = 0.833**. Best measured ratio over
21 codecs is **0.843** (blosc2 zstd+shuffle) — against **0.850 for i.i.d. Gaussian noise of the same shape**.
Real activations compress no better than white noise; all the redundancy is one low-entropy exponent byte.
bf16 deletes those two noise planes outright for **r = 0.500 at ~0 CPU**.

The second governing fact is a 972x outlier. Channel 62 alone carries **97.07%** of activation energy at the
node0→node1 cut (96.93% at node1→node2); max |1701.9| vs a 1.75 median per-channel max; excess kurtosis 62,180.
The outlier index set `[62, 490, 570, 53, 262, 591, 450, 208]` is **identical at both cut points and across
unrelated prompts** — so it can be a baked-in constant, no runtime detection.

Third: on a LAN this is all nearly free money. Post-KV the whole bf16 hop payload is **0.06% of a decode step**;
bf16→int8 saves 1,788 B/token = **14.3 µs on 1 GbE against 0.88 ms/token of compute (1.6%)**. Saturating 1 GbE
needs batch ~349 at 100 steps/s. Compression is a **WAN feature**, not the headline.

## Options considered

| option | bytes/tok/hop | quality (measured) | verdict |
|---|---:|---|---|
| fp32 + base64 (v0) | 4,779 | exact | **rejected.** base64 is a codec with r=1.333 — deleting it is worth +36.25 ms/hop on a 7.34 MB prefill vs +1.00 ms for the best real compressor. ADR-002. |
| fp32 raw | 3,584 | exact | baseline |
| **bf16** | **1,792** | KL_mean 5.7e-5, KL_p99 4.97e-4, top-1 0.9974, dppl −0.031%, greedy output **bit-identical on 4/4 prompts**, both hops quantised | **ACCEPTED v1 default.** 3.5 µs/frame. |
| int8 **per-tensor** | 898 | KL 10.029, top-1 **0.00684**, perplexity **411,041** vs 18.64, emits "declaration declaration declaration" | **BANNED in code review.** Error amplification 1.60x per segment, 15.3x through lm_head. |
| int8 per-token (absmax) | 898 | KL 0.0306, top-1 0.9268 → **7.32% of tokens flip**, dppl +3.22% | rejected as a default; per-token scaling costs +0.22% bytes and buys a **328x KL reduction** over per-tensor, so it is the mandatory *floor* for any int8. |
| **int8 per-token + 8 fp16 outlier channels** | **906** | relerr 0.00234, agreement 0.9865, KL 0.00205, **20/20 exact greedy match**, amplification 0.99x | **v1, gated.** ~15 lines of numpy/side. The 8 fp16 channels are what make int8 shippable. |
| fp8 e4m3 + per-tensor fp16 scale | 898 | relerr 0.00439, 20/20 greedy match (but only 0.9459 per-position agreement) | alternative to the above if the index list is unwanted. **Raw e4m3 cast NaNs on the first tensor** — 1701.9 > e4m3 max 448. |
| int4 (any variant) | 450–462 | best variant 8/20 greedy, relerr 0.078 | rejected at H=896. |
| Lossless byte codecs (zstd/lz4/snappy/brotli/blosc2, 21 tested) | ≥3,020 | exact | **rejected on the decode path.** At 1 GbE only **2 of 60** codec/payload combinations are net-positive, both 2048-token prefills; at 10 GbE and 25 GbE **zero** are. LZ4 and Snappy *expand* activations (r=1.000–1.006) without a shuffle filter — **retire LZ4 from the DLP enum before anyone implements it**. |
| Low-rank / PCA projection | 448 @ k=224 | **100% top-1** at k=224 = H/4 | **rejected, measured.** Projection costs 27.31 µs to save 10.75 µs of 1 GbE wire — a 2.5x loss on 1 GbE, 26x on 10 GbE, 100x on loopback. The only rank cheap enough to win (k=56) scores 62.5% top-1. And it gets **18x worse as H grows** (crossover 394 Mbit/s at H=896 → 21.6 Mbit/s at H=8192 on CPU), because cost ∝ rH² and saving ∝ (1−r)H. |
| Learned autoencoder bottleneck | — | — | **rejected a fortiori.** A 896→k linear encoder is the identical matmul to PCA plus a nonlinearity — strictly dominated by a technique already shown to lose, at weeks of cost and a third versioned artifact. |
| Top-k sparsification | 560–896 | 87.4% energy retained | **rejected — triply dominated.** More bytes than PCA at equal coefficients while discarding 12.6% of energy, and PCA is itself dominated by int8. For a *typical* token energy is not concentrated: top-128 of 896 holds only 77.5%. |
| Error feedback (DGC/PowerSGD), delta coding, mean-centring, batch-centroid coding | — | — | **rejected, measured.** Median ‖h_t − h_{t−1}‖/‖h_t‖ = **1.09** — consecutive states are not near each other, so the delta is the same size as the signal and there is nothing to warm-start. EF assumes a stationary objective and a re-appliable residual; h_t is not an estimate of anything, and with a KV cache a compressed h_t writes corrupted K/V that every later position reads forever. **Mean-centring is actively harmful** — µ is dominated by the rare 1725-norm tokens, so it makes the typical token 2.6–3.6x larger. |

## Decision

1. **bf16 on the wire at both hops, fp32 compute retained.** `dtype=2` in the DLP header (ADR-002). Stop there
   on a LAN. This is also the demo-day safety net.
2. **Per-token (per-row) absmax scaling is mandatory for any int8.** Per-tensor scaling is banned in review.
3. **int8 + 8 static fp16 outlier channels** ships as `dtype=3`, **off by default**, armed only when measured
   link goodput is WAN-class (< ~163 Mbit/s per hop, where the 1,788 B saving exceeds 10% of compute).
4. **No byte codec in the demo.** Hard-disable any codec below 1 MiB. If one is ever armed it is blosc2
   LZ4+BITSHUFFLE (`clevel=5`, `nthreads=1`, `typesize` matched to dtype) at payload ≥ 1 MiB **and** EWMA link
   ≤ 120 MB/s. This ADR removes work rather than adding it.
5. **Asymmetric rungs if sub-bf16 is ever used:** aggressive codec on hop 1, conservative on hop 2. At an
   identical 680 B/tok/hop that is **1.33x lower KL and +4.0 points of top-1, free** — the late cut is 1.43x
   more damaging than the early one.
6. **Quality gate in CI:** `bench/t2a4_quality_harness.py`, blocking on KL_p99 > 1e-3 or top-1 < 0.999.
   **Delete cosine similarity from every table, slide and README** — int8 scores 0.99958 while flipping 7.3% of
   tokens. Report per-token mean and p99 rel-L2, never block-Frobenius (understates typical-token error 2.67x–
   4.31x, worse as the codec gets more aggressive). Every frame asserts `len(buf) == expected` before decode:
   a bf16/fp32 mismatch reshapes *silently* when the sizes divide evenly.

## Consequences

**Good.** 2.00x wire at zero measured quality cost for 3.5 µs; a 3.96x path for WAN; a measured reason to say
no to five techniques the audience will ask about; LZ4 retired before implementation.

**Bad.**
- **Compression is not the headline and the deck must say so.** On a LAN the pipeline is compute-bound; the
  935x wire figure is KV caching + not shipping logits + dropping base64, **not a codec**. Framed otherwise,
  the whole T2 workstream gets attacked as a solution to a non-problem.
- The 8 outlier indices are **model- and cut-point-specific**. Moving the split to 11/11/2 (ADR-007) or
  swapping the model invalidates them — hence the model hash in the handshake and the hard length assert.
  **Do not fit any basis or index list before the split is finalised.**
- Evidence base is one model, 3,072 wikitext-2 tokens, 4 chat prompts, greedy only, single-step teacher-forced.
  It ranks codecs by orders of magnitude; it does not support a ±0.1% claim. Qwen2.5-0.5B is **below** the
  ~6.7B emergent-outlier threshold, so these int8 results are *optimistic* for larger models.
- **Corpus trap:** tiled/repetitive prompts inflate every ratio ~2.5x (int8@2048 read 0.268 on tiled prose vs
  0.677 on wikitext-2) — any quoted ratio must name its corpus. blosc2's Python API silently ignores a
  misspelled `filter=` (vs `filters=`) and a wrong `typesize` silently degrades bitshuffle to noise.

## Status

**v1 accepted** (bf16 default; int8+outliers gated off). **v2 proposed:** QuaRot Hadamard rotation (removes the
outlier channels entirely, opening int4), native fp8 e4m3 on Hopper/Ada/Blackwell, KIVI-style per-channel keys /
per-token values when the KV cache is quantised, per-peer online calibration of `B_cross = (1−r)/(1/T_c + 1/T_d)`,
and a rate controller (ADR-005) driven by the sender's free rel-L2 via the measured law `KL ≈ 27·e²`.
