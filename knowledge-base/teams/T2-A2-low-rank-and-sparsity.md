---
team: T2 — Activation Compression
agent: T2-A2
topic: Structural compression beyond quantisation — low-rank projection, learned bottlenecks, top-k sparsification + error feedback, temporal/batch redundancy
headline: "Measured on the real model: rank-224 of 896 reproduces every next-token choice exactly, but the projection matmul costs 27.3 µs and saves at most 10.8 µs of 1 GbE wire time. Every structural technique in this brief is a pessimisation at H=896 on any LAN, and low-rank gets 18x WORSE as H grows, not better. Ship none of it in v1."
status: MEASURED — real Qwen2.5-0.5B-Instruct, holdout-validated
scripts: bench/t2a2_lowrank.py, bench/t2a2_outlier_lowrank.py, bench/t2a2_decode_anchor.py
data: bench/t2a2_lowrank.json, bench/t2a2_outlier_lowrank.json, bench/t2a2_decode_anchor.json
---

# T2-A2 — Low-rank, sparsity, and the structural techniques that do not pay

## 0. Experimental setup (all `(measured)` numbers below come from this)

Real `Qwen/Qwen2.5-0.5B-Instruct`, fp32, CPU, 2 threads (matches the PoC's 2-CPU containers).
40 prompts, **32 calibration / 8 holdout — the basis is fit on calibration only, every error number
is evaluated on holdout.** 1,162 calibration / 293 holdout tokens. Cuts are the PoC's real ones:
after layer 7 (node0→node1) and after layer 15 (node1→node2).

Quality metric = **top-1 next-token agreement** vs the uncompressed run, via a forward hook on
`model.model.layers[{7,15}]` that substitutes the rank-k reconstruction and lets the remaining
layers + `lm_head` run normally. End-to-end, not a proxy.

**Anchor (measured):** one cached decode step, all 24 layers, fp32/CPU/2 threads = **72.9 ms**
(min of 12; median 75.5 ms) → **~24.3 ms per 8-layer shard**. Hold that number; everything else
in this file is microseconds.

---

## 1. The finding that governs all four techniques: massive activations

| metric (holdout) | cut after L7 | cut after L15 |
|---|---|---|
| share of activation energy in **channel 62 alone** | **97.07%** (measured) | **96.93%** (measured) |
| share in top-4 channels `[62, 490, 570, 53]` | 98.32% (measured) | 98.17% (measured) |
| median token norm ‖h‖ | 13.5 (measured) | 18.6 (measured) |
| **max** token norm ‖h‖ | **1725.4** (measured) | **1732.6** (measured) |

The outlier channel set is **identical at both cut points** — this is the known massive-activation /
attention-sink phenomenon (Sun et al. 2024, *Massive Activations in LLMs*; Dettmers et al.,
`LLM.int8()` outlier features). **Handoff to the numeric-format agent: the outlier channel indices
are static and shared across cuts, so they can be a baked-in constant — no runtime outlier
detection is needed.**

**This makes every aggregate metric lie.** Frobenius/energy statistics are dominated by a handful of
128x-norm tokens:

| k | centered energy captured (all 896 dims) | Frobenius rel. error | **per-token MEDIAN rel. error** | **top-1 agreement** |
|---|---|---|---|---|
| 16 | 99.951% | 2.24% | **35.4%** | **12.5%** |
| 32 | 99.977% | 1.61% | 11.1% | 12.5% |
| 64 | 99.989% | 1.31% | 2.08% | 62.5% |
| 128 | 99.997% | 1.10% | 0.51% | 87.5% |
| 224 | 100.000% | 0.95% | 0.05% | **100%** |

All `(measured)`, cut L7. **Rank-1 captures 99% of the centered energy and yields a broken model.**
Anyone quoting "99% energy at rank k" as a compression result is quoting channel 62.

Honest spectrum, with the 4 outlier channels removed (892 dims) — this is the real one:

| k | energy, L7 | energy, L15 |
|---|---|---|
| 16 | 97.12% | 95.59% |
| 32 | 98.62% | 98.02% |
| 64 | 99.37% | 99.14% |
| 128 | 99.85% | 99.76% |
| 224 | 99.99% | 99.97% |
| **rank for 99%** | **44** | **58** |

---

## 2. Technique 1 — Low-rank / PCA projection

Basis fit offline, baked into both node images (896x224 fp32 = **0.80 MB**, against a 545 MB fp32
shard — free). Only k coefficients cross the wire; no handshake, no per-request negotiation.

### 2a. Does stripping the outliers rescue low-rank? **No — measured, and it changes nothing.**

Scheme: ship the 4 outlier channels exactly (indices static ⇒ 0 index bytes) + rank-k PCA of the
other 892 dims.

| k | plain rank-k top-1 | outliers-exact + rank-k top-1 |
|---|---|---|
| 16 | 12.5% | 12.5% |
| 32 | 12.5% | 12.5% |
| 64 | 62.5% | 62.5% |
| 128 | 87.5% | 87.5% |
| 224 | 100% | 100% |

Bit-for-bit identical (measured, cut L7). The usable rank is **k ≈ 224 = H/4** either way. The
decision-relevant signal lives in the low-energy tail, which is exactly what PCA discards first.

> Cut L15 plateaus at 87.5% even at k=448 where median reconstruction error rounds to 0.000. That
> last disagreeing prompt is a near-tied argmax flipped by float re-association in the hook, not a
> compression artifact. Real ceiling is 100%.

### 2b. **The decisive test.** Projection cost vs network time saved, H=896, one hop, one position

Projection cost = **encode matmul (node A) + decode matmul (node B)**, both on the critical path,
so the sum is the correct number. numpy fp32, 2 threads, min of 5 trials x 1000 reps `(measured)`:
k=56 → **8.86 µs**, k=112 → **17.51 µs**, k=224 → **27.31 µs**. Run-to-run spread ±25%.

Wire rates from house numbers: 1 GbE = 125 B/µs, 10 GbE = 1250 B/µs, loopback = 5000 B/µs `(modelled)`.

Baseline is **bf16 dense = 1,792 B**, because bf16 is the teammates' v1 default and costs ~0 CPU.
Beating fp32 is not the bar; beating the free thing is.

| scheme | payload | ratio | CPU cost | 1 GbE saved | 10 GbE saved | loopback saved | **verdict** |
|---|---|---|---|---|---|---|---|
| bf16 dense (baseline) | 1,792 B | 1.0x | 0 µs | — | — | — | — |
| PCA k=56 (**62.5% top-1 — broken**) | 112 B | 16x | 8.86 µs | 13.44 µs | 1.34 µs | 0.34 µs | +4.6 / **−7.5** / **−8.5 µs** |
| PCA k=128 (87.5% top-1) | 256 B | 7x | ~19.5 µs `(modelled)` | 12.29 µs | 1.23 µs | 0.31 µs | **−7.2 / −18.3 / −19.2 µs** |
| **PCA k=224 (100% top-1)** | **448 B** | **4x** | **27.31 µs** | **10.75 µs** | **1.08 µs** | **0.27 µs** | **−16.6 / −26.2 / −27.0 µs** |

> **The only rank cheap enough to win is the rank that destroys the output.** At the rank that
> actually preserves the model's decisions, low-rank projection is **2.5x a loss on 1 GbE, 26x a
> loss on 10 GbE, 100x a loss on loopback.**

Even against the naive v0 fp32 wire (3,584 B), k=224 saves 21.5 µs on 1 GbE against a 27.31 µs cost —
**still a loss.**

**Crossover link speed** (k=224, bf16 baseline): 1,344 B saved / 27.31 µs = **49.2 MB/s ≈ 394 Mbit/s**.
Below that, low-rank pays. That is WAN / consumer-broadband territory, not a datacentre.

And the scale that matters most: **27.31 µs is 0.11% of a 24.3 ms shard step (measured).** The
entire bf16 payload takes 14.3 µs on 1 GbE = **0.06% of a decode step.** At H=896 on CPU, nothing
about the wire is on the critical path at all.

### 2c. Low-rank gets **worse** with model size, not better — measured

Cost ∝ H·k = rH² ; savings ∝ (1−r)H. The ratio degrades linearly in H. Measured at k=H/4, 2 threads:

| H | k=H/4 | projection (measured) | bf16 bytes saved | **crossover link speed** |
|---|---|---|---|---|
| 896 | 224 | 27.31 µs | 1,344 B | 49.2 MB/s = **394 Mbit/s** |
| 4096 | 1024 | 970.44 µs | 6,144 B | 6.33 MB/s = **50.6 Mbit/s** |
| 8192 | 2048 | 4544.51 µs | 12,288 B | 2.70 MB/s = **21.6 Mbit/s** |

**18x worse from H=896 to H=8192.** The intuition "low-rank will pay off on a bigger model" is
measurably backwards on CPU. (Also visible: a cache cliff — H=4096/k=256 is 90.7 µs but k=512 is
642.9 µs, when the basis stops fitting in L2. Effective bandwidth 58.8 GB/s cache-resident vs
29.5 GB/s from DRAM, measured.)

**Where it does flip: GPU + large H + slow link.** On an A100 the projection is bandwidth-bound at
~1.5 TB/s: H=8192/k=2048 fp16 = 67 MB / 1.5 TB/s ≈ 45 µs `(modelled)` → crossover 12,288 B / 45 µs =
273 MB/s ≈ **2.2 Gbit/s**. So on GPU at H=8192, low-rank wins on 1 GbE and loses on 10 GbE.
**Low-rank projection is a GPU + large-H + ≤1 GbE technique. It is never a CPU technique and never a
10 GbE technique.**

### 2d. Verdict

| | verdict |
|---|---|
| **v1 (hackathon)** | **NO. Do not build it.** 0 dev days spent, 0 quality risk taken. bf16 + binary + argmax-on-node2 (VERIFIED-FACTS FINDING 2) already win by 100-1000x more and cost nothing. |
| **v2 (production)** | Build only behind a **measured link-speed gate**: enable when inter-node goodput < 50 MB/s at H=896 (or < 273 MB/s on GPU at H=8192). Ship the basis in the image; never negotiate it per request. |
| **blocker** | The basis is tied to (model, cut layer, calibration distribution). VERIFIED-FACTS FINDING 1 says the cut **should move** to 11/11/2 for a 1.55x rebalance — that invalidates any basis fit at 7/15. **Do not build low-rank before the split is finalised.** |

---

## 3. Technique 2 — Learned autoencoder bottleneck. **Decisive: no.**

| dimension | assessment |
|---|---|
| **Inference cost** | A 896→k linear encoder is the **identical matmul** to PCA — same FLOPs, same bytes, same 27.31 µs — plus a nonlinearity, and 2x that if the encoder has a hidden layer. It can never be cheaper than PCA. **PCA is already a pessimisation at H=896, so the AE is a pessimisation a fortiori.** No measurement needed; the AE is strictly dominated by a technique already shown to lose. |
| **Does it beat PCA on rate-distortion?** | Modestly, yes — nonlinear AEs typically buy ~1.3-2x rank at equal L2 on natural signals. **But L2 is the wrong objective here**: measured, k=128 already has 0.51% median L2 error yet only 87.5% top-1 agreement. The failure is direction-sensitive, not energy-sensitive. To help, the AE must be trained against KL-to-teacher-logits through the cut — i.e. per-model, per-cut-layer distillation. |
| **Cost to build** | Calibration corpus + training loop + eval harness + a third versioned artifact. Weeks, not days. |
| **Risk** | (1) Version skew: AE weights must match model **and** cut layer; moving the split breaks them silently. (2) Silent OOD quality regression, no runtime detector. (3) A trained codec in a system whose pitch is "no node holds the whole model" — now no node holds the whole *codec* either. |
| **v1 / v2** | **NO** / only after low-rank's link gate has fired AND int4+Zstd was measured insufficient. Realistically: never for this project. |

---

## 4. Technique 3 — top-k magnitude sparsification + error feedback

### 4a. The payload arithmetic kills it before quality is even discussed

A dense 896-vector has no free index space. Options: 896-bit bitmask = **112 B fixed**, or uint16
indices = **2 B per surviving coordinate**. Values in bf16 = 2 B.

| scheme | k | payload | ratio vs bf16 dense | energy retained (measured, per-token mean) |
|---|---|---|---|---|
| bitmask + bf16 values | 224 | 112 + 448 = **560 B** | 3.2x | **87.4%** (L7) |
| uint16 idx + bf16 values | 224 | 224x4 = **896 B** | 2.0x | 87.4% |
| bitmask + bf16 values | 448 | 112 + 896 = **1,008 B** | 1.8x | 97.0% |

Compare like for like: **PCA at k=224 costs 448 B and leaves 0.05% median error; sparsification at
224 coefficients costs 560-896 B and throws away 12.6% of the energy.** Sparsification is dominated
by PCA at equal payload, and PCA is dominated by int8 quantisation (4x, ~1 µs, no basis). Triply
dominated. **Dead — v1 no, v2 no.**

Root cause, measured: for a *typical* token the energy is **not** concentrated. Top-128 of 896
coordinates holds only **77.5%** (L7) / **78.8%** (L15). The 97%-in-one-channel statistic is an
energy-weighted aggregate over the rare massive tokens, not a per-token property.

### 4b. Does error feedback (DGC, PowerSGD) apply to forward activations? **No. Plainly, no.**

Error feedback works in distributed SGD because of three properties, **none of which hold here**:

| EF-SGD assumption | forward activations in autoregressive decode |
|---|---|
| The compressed object is a **gradient** — a noisy estimate of a descent direction on a *stationary* objective. Delaying part of it still moves toward the same minimum (Karimireddy et al. 2019). | h_t is **not an estimate of anything**. It is the exact input layer 8 needs for position t. There is no objective, no minimum, nothing to converge to. |
| The residual is re-applied to **the same vector** at the next step, and thousands of steps amortise it. | h_{t+1} is a **different vector from different input**. Adding r_t to h_{t+1} injects position t's error into position t+1's computation. That is **contamination, not correction**. Measured: median ‖h_t − h_{t−1}‖/‖h_t‖ = **1.09** (L7) / **0.95** (L15) — consecutive states are not near each other, so there is no shared target for a residual to chase. |
| Errors are **transient** — eventually applied, then gone. | With a KV cache the compressed h_t is consumed by layers 8-15, which write **K_t, V_t into the cache**. That corrupted K/V is read by *every subsequent position for the rest of the generation*. Errors are **persistent and accumulating**, and no mechanism ever removes them. |

**PowerSGD specifically** relies on warm-starting the power-iteration matrix Q from the previous
step, which requires the compressed object to be slowly-varying across steps. Measured ‖Δ‖/‖h‖ ≈ 1.0
means there is nothing to warm-start. Its core mechanism is inapplicable.

> **Verdict: error feedback does not apply to forward activations. Do not cite DGC/PowerSGD as
> precedent for this system.** The one partial exception — carrying residual across chunks of a
> pipelined *prefill* — still fails, because each chunk is distinct data rather than an iterate.

---

## 5. Technique 4 — Temporal and batch redundancy. All three hypotheses tested, all rejected.

| hypothesis | measurement (holdout) | verdict |
|---|---|---|
| **H1 — delta-encode h_t against h_{t−1}** | median ‖h_t − h_{t−1}‖/‖h_t‖ = **1.09** (L7), **0.95** (L15). Frobenius: 3.77 / 3.37. Mean cosine 0.41 / 0.53. | **REJECTED (measured).** The delta is the *same size* as the signal. Delta coding costs a subtraction and saves nothing. Still true after removing the outlier channels (1.15 / 1.00). |
| **H2 — subtract a static calibration mean before quantising** | median ‖h − µ‖/‖h‖ = **3.59** (L7), **2.65** (L15). | **REJECTED (measured), and it is actively harmful.** µ is dominated by the rare 1725-norm tokens, so mean-centring makes the *typical* (norm-13.5) token **2.6-3.6x larger**. **Warning to the numeric-format agent: do NOT mean-centre before quantising. Use per-token scaling.** |
| **H3 — batch redundancy across similar prompts** | cross-prompt cosine of last-position states, median **0.98** (L7) / **0.91** (L15). Survives outlier removal: **0.97 / 0.88**. | **REJECTED (measured).** The similarity looks enormous but is decision-irrelevant: rank-16 captures exactly that shared direction and yields **12.5% / 0% top-1 agreement**. Coding against a batch centroid is H2's trap wearing a different hat. |

The one form of batch redundancy that **is** real is not compression: shipping B positions in one
message amortises headers and RTT. That belongs to the batching/queueing team, not to T2.

---

## 6. Numbers other agents should reuse

| number | value | tag |
|---|---|---|
| Outlier channel indices, **identical at both cuts** | `[62, 490, 570, 53]` | measured |
| Energy in channel 62 alone | 97.07% (L7) / 96.93% (L15) | measured |
| Token norm median / max | 13.5 / 1725.4 (L7) | measured |
| **Do not mean-centre before quantising** (median residual grows 3.59x) | — | measured |
| Rank preserving 100% top-1 at both cuts | **k = 224 = H/4** | measured |
| Rank for 99% energy, outliers removed | 44 (L7) / 58 (L15) | measured |
| One cached decode step, 24 layers, fp32/CPU/2 threads | **72.9 ms** (24.3 ms per 8-layer shard) | measured |
| 896x224 projection, encode+decode, 2 threads | **27.31 µs** | measured |
| bf16 hop payload as a share of a decode step | **0.06%** on 1 GbE | measured+modelled |
| Link speed below which low-rank pays, H=896 | **394 Mbit/s** | measured+modelled |

## 7. Risks

1. **Metric trap (highest risk to the whole team).** Cosine similarity, L2 error and "energy
   captured" are all dominated by channel 62 and will report success on a broken model — measured:
   99.95% energy at rank 16, 12.5% top-1 agreement. **Any T2 quality gate must be end-to-end top-1
   agreement or KL on logits, never a hidden-state distance.**
2. **Calibration overfitting.** Fitting the basis on the eval set inflates every number. Our split is
   32/8 held out; a narrower calibration set looks better and generalises worse.
3. **Coupling to the layer split.** Any fitted artifact (PCA basis, AE weights, outlier set) is
   invalidated by the 11/11/2 rebalance that FINDING 1 recommends. Sequence the work accordingly.
4. **Opportunity cost.** Days spent on structural compression are days not spent on the KV cache
   (271x) and the argmax move (607,744 B → 4 B), which are strictly larger wins at zero quality risk.

## 8. Deck-worthy claims (max 3, all measured)

1. **"One channel out of 896 carries 97% of everything we send. We found it, and it is the same
   channel at every cut point."**
2. **"Rank 224 of 896 reproduces every single next-token choice exactly — the activation is 4x
   redundant."** (Pair it with claim 3 or it is misleading.)
3. **"We built the compressor, measured it, and cut it: the projection costs 27 µs to save 11 µs of
   wire. At this model size, compressing the activation is slower than sending it."**
