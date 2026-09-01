---
status: ADVERSARIAL NUMERIC AUDIT — read before any number leaves this repo
scope: every file under knowledge-base/ (6 root docs, 13 ADRs, 25 team files, bench/, assets/)
method: re-derived every load-bearing figure; cross-checked each number against every other file that quotes it
supersedes: the specific rows it corrects, in 01-VERIFIED-FACTS, 40-PITCH, ADR-013, and the named teams/ files
---

# 90 — AUDIT

**23 findings: 8 high, 10 medium, 5 low. 11 fixed in place. Zero invented credentials.**
24 `<FILL:>` markers in `40-PITCH.md` / `T5-A5` are intact — nothing was fabricated into them.
No invented benchmark was found: every number traces to a script, a team measurement, or a citation.
What *was* found is worse in a different way — **measured numbers used against the wrong denominator,
and ratios built by dividing one team's run by another team's run.**

Legend: **FIXED** = the file was edited and the correction noted inline. **OPEN** = needs a decision or a re-run.

---

## HIGH — these change a number that is currently on a slide

| # | File | Claim | Problem | Correction | Status |
|---|---|---|---|---|---|
| **F01** | `teams/T2-A4-quality-guardrails.md` §7 → `40-PITCH` Slide 3, `10-ARCHITECTURE` §5, `20-INFRA` §7, `ADR-003` | "Sub-bf16 compression only starts to matter **below ~163 Mbit/s** per hop"; "14.3 µs vs **0.88 ms/token** compute = 1.6%" | The 0.88 ms anchor is `0.45 s ÷ 512 positions` — the **amortised per-position cost of a batched prefill**, used as if it were a **decode-step** cost. A decode step is the only regime in which a 1792 B/hop activation is sent, and it measures **72.9 ms** (T2-A2) / **123.94 ms** across the chain (T1-A1). The anchor is ~80–140x too small. The same section's guardrail overheads (0.11%, 0.39%) inherit the error. | Crossover is **~2 Mbit/s**, not 163 Mbit/s. `30-PERF-MODEL` §5 independently derives **2.3 Mbit/s** — a 70x disagreement with the team file that no synthesis doc caught. **The conclusion ("stop at bf16 on a LAN") gets stronger.** | **FIXED** — correction block in T2-A4; Slide 3 label and speaker note changed in `40-PITCH` |
| **F02** | `teams/T3-A1-kv-cache-across-shards.md` §2 → `40-PITCH` Slide 4, `ADR-013` ledger, `T5-A3` storyboard | "512-token generation **2,908 s → 101.6 s = 28.6x** (modelled)" | Both halves contradict measured data already in this repo. v0 side: modelled at **50 GFLOP/s**; T1-A1's measured stage times (165.81/253.68/712.54/3551.25 ms at seq 16/128/512/2048) imply a linear fit `100.7 ms + 1.195 ms·n`, so the same generation is **~227 s**, not 2,908 s — **12.8x apart**. v1 side: modelled at **198 ms** per cached decode step against T1-A1's measured **123.94 ms** — 1.6x apart. | On the measured basis the whole-generation figure is **~227 s → ~64 s = ~3.6x**. It is *lower* than the seq=512 per-token 6.3x because a generation spends most of its tokens at short sequence lengths, where there is little redundancy to remove. **Quote per-token at a stated seq_len; never quote a whole-generation speedup.** (The linear fit is taken in the 128–512 regime, which is the only one a P=32/G=512 run visits; it under-predicts at seq=2048, so ~227 s is a *lower* bound on v0 and ~3.6x a lower bound on the ratio — nowhere near 28.6x either way.) | **FIXED** — row removed from Slide 4, retired in `ADR-013` (twice) |
| **F03** | `teams/T3-A3-continuous-batching.md` §4 | "**32.9x** aggregate throughput at B=16" and "**2.76x** from microbatch pipelining" | Divides *this file's* ctx=128 pipeline throughput (265.4 tok/s) by *T1-A1's* seq=512 serial baseline (8.07 tok/s). On T3-A3's own measured stages the serial baseline is `23.63 + 23.63 + (23.63 + 21.23 lm_head) = 92.1 ms = 10.86 tok/s`. This is exactly the cross-run mixing `ADR-013` forbids — but it forbade only *multiplying 32.9x into 19x*, not the 32.9x itself. | Same-run figures: **pipelining 2.05x**, **combined 24.4x at B=16** (265.4/10.86). Quote 24.4x, or attach "against T1-A1's seq=512 serial baseline" to 32.9x. | **FIXED** — correction block added |
| **F04** | `teams/T5-A1-problem-and-market.md` §3.2 + numbers list | Row-sharding the tied matrix gives "**75,968× fewer bytes** on the return hop (607,744 B → 3 × 8 B partials = 24 B)" | Straight arithmetic error: 607,744 / 24 = **25,322.7**. 75,968 is 607,744 / 8 — divided by one partial instead of three. | **25,323×.** (The competing fix — argmax on node2, 4 B — is 151,936×, which is the larger number anyway.) | **FIXED** |
| **F05** | `teams/T1-A2-wired-lan-fast-path.md` headline + deck_worthy | "**114 ms per token** of the current system is transport and base64 CPU, not model compute — the model has not started yet when that clock runs out" | Directly contradicts T1-A1's measurement that **89% of v0's per-token wall clock is compute** and transport is **72.8 ms of 785.3 ms = 9.3%** (cross-checked against a symmetric echo to 0.3 ms). 114.2 = 3 × 38.06 ms, where 38.06 ms is a *symmetric 1.79 MB* round trip; v0's three POSTs are not symmetric (POST0's request is 3.6 KB of ids; POST2's response is the 810 KB logit blob), so ×3 double-counts. Both lines are in `deck_worthy` lists — one of them will be spoken. | **72.8 ms / 9.3%** is authoritative for share-of-wall-clock. T1-A2's 38.7x per-hop ratio and its **454x** composite are valid only as *protocol-choice* evidence. | **FIXED** — correction block added |
| **F06** | `teams/T3-A4-queue-admission-slo.md` §3 | "N=3 → 3.237 tok/s = **2.54x**"; balanced split "4.210 tok/s = **3.31x** v0" | The `vs measured 1.2734` column divides a **compute-only** ceiling (`X(N)=min(N/D, 1/D_max)`, transport excluded) by a **transport-inclusive** measured baseline. It silently credits the concurrency lever with ~1.10x of transport saving owned by the connection-pool and codec levers. T3-A2 computes the identical change as **2.31x / 3.00x**. Two team files, two answers, one change. | Concurrency-only: **2.31x** (3.237/1.403) and **3.00x** (4.210/1.403). Use T3-A2's figures. | **FIXED** — correction block added |
| **F07** | `T2-A2` §0 vs `T3-A3` §2 vs `T1-A1` §7 vs `T2-A3` | Three unreconciled **(measured)** values for *one cached decode step*: **72.9 ms** (24 layers, 1 process, 2 threads), **~92 ms** (T3-A3's own stages summed, ctx=128), **123.94 ms** (3-node chain, seq=512). Per 8-layer shard: 17.1 (T2-A3) / 23.63 (T3-A3) / 24.3 (T2-A2) / ~41 (T1-A1 ÷3). | A **1.7x spread on the single load-bearing v1 denominator.** Everything downstream — 8.06 tok/s, `D_max`, `N*=3`, the 19x, the queueing model — rests on 123.94 ms. The prototype HTML was quoting 72.9 ms for the same thing. The gap is plausibly per-call framework overhead ×3, but **nobody measured it**, so it is an assumption wearing a `(measured)` tag. | Re-measure one cached decode step **on the real 3-node chain**, and pin one number. Until then, state the configuration beside every decode-step figure. Prototype HTML switched to 123.94 ms (with the 72.9 ms single-process figure named). | **OPEN** (prototype **FIXED**) |
| **F08** | `bench/` vs `teams/T1-A1, T1-A2, T1-A3, T1-A4, T3-A2, T3-A3` | Every stage time, per-hop latency and transport figure in those six files is tagged **(measured)** | `bench/` contains scripts and raw JSON for **T2-A2, T2-A3, T2-A4, T5-A4, 30-PERF-MODEL, verify_constants, parity_check** only. There is **no script and no raw data** for 712.54 / 123.94 / 205.81 / 197.76 / 308.97 ms (T1-A1), 8.483 → 0.089 ms (T1-A4), 10,406 → 30.1 µs (T1-A3), 38.06 → 0.98 ms (T1-A2), or 23.63 / 44.81 ms (T3-A3). The 19x headline, `N*=3`, `D_max`, ADR-005 and ADR-007 all sit on the T1-A1 set. | Land those scripts in `bench/` with raw output, or downgrade the tag to *reported by T1-A1, not reproducible here*. A judge asking "can I run it?" currently gets "no" for the numbers that matter most. | **OPEN** |

---

## MEDIUM — internally inconsistent, or a tag that overstates

| # | File | Claim | Problem | Correction | Status |
|---|---|---|---|---|---|
| **F09** | `01-VERIFIED-FACTS.md`; `T1-A5`; `T3-A1`; `T5-A1` | "total params **493,961,216**", "one transformer layer is **14,909,440** params", status *VERIFIED, computed from the real config.json* | Both figures count weight matrices only — they drop the QKV biases (1,152/layer) and the RMSNorm gains (1,792/layer). Real per-layer is **14,912,384**; real total is **494,032,768** (which is the published 0.49B). `T1-A5` independently computed 14,912,384 and 494,031,872 (itself missing the final norm's 896), so two files that both claim to read the same config disagree. | Ratios are safe: 9.13 layer-eq, 27.6%, 51.7%, 33.3% all move by <0.02%. **Do not print 493,961,216 as "the parameter count."** | **FIXED** — note added to `01-VERIFIED-FACTS` |
| **F10** | `01-VERIFIED-FACTS` F2, `T3-A1`, `T4-A5`, `T5-A3`, `assets/split-model-bench.html`, `40-PITCH` Slide 4 vs `30-PERF`, `T5-A4` | Logit blob base64 length quoted as **810,325 B** (5 files) / **810,328 B** (2 files) / **810,346 B** (JSON-wrapped, 3 files); return-path factor quoted as both **151,936×** (raw→4 B) and **202,581×** (base64→4 B) | 810,325 is `607744 × 4/3` truncated; correct base64 is `4·⌈607744/3⌉ = **810,328**`, so the derived factor is **202,582×**, not 202,581×. Separately, the deck currently carries *two different factors for the same two-line fix* — Slide 4 says 202,581×, ADR-002/ADR-010/ADR-013/T5-A2 say 151,936×. | Use **607,744 B → 4 B = 151,936×** everywhere (raw-to-raw, no base64 on either side of the comparison). Reserve 810,328 B for describing what v0 actually puts on the wire. | **FIXED** in `01-VERIFIED-FACTS` and the prototype; **OPEN** in `T3-A1`, `T4-A5`, `T5-A3` |
| **F11** | `01-VERIFIED-FACTS` F1 → 6 downstream files | "The current split is **1.55x** slower than a balanced one" | 1.55 = 17.131 / **11.044**, the *fractional* ideal, which no integer layer split can reach. The achievable split (0-11 / 11-22 / 22-24) is 11.131 eq, so the realisable gain is **1.539x** — which is what `T3-A5` computes. And `T3-A2` measured the wall-clock ratio at **1.30x** (308.97/237.51). `20-INFRA` §7 #8 conflates the first two ("1.55x / 1.539x is the layer-equivalent ratio (17.13/11.13)" — that division is 1.539). Three numbers, one claim, all on slides. | **1.539x** for achievable layer-equivalents, **1.30x** for anything wall-clock, **1.55x** only when explicitly naming the unachievable fractional bound. | **OPEN** |
| **F12** | `T2-A4` vs `T2-A5` vs `T2-A1`; quoted in `40-PITCH` Slide 3 and `ADR-013` | bf16 quality: top-1 **0.9974** / KL **5.7e-5** (T2-A4) vs **99.41%** / KL **2.6e-5** (T2-A5) vs KL **8e-5** (T2-A1) | Three measured runs, three answers, no reconciliation. Slide 3 prints T2-A5's 99.41%; the ADR-013 ledger prints T2-A4's 0.9974; `20-INFRA` §6 prints "99.41% top-1, KL 2.6e-5 (**measured, T2-A4/T2-A5**)" — attributing T2-A5's pair to both files. | Pick one run, name it, use it everywhere. All three support "bf16 is free"; only the decimals disagree. Fix the `20-INFRA` mis-attribution. | **OPEN** |
| **F13** | `T2-A1` vs `T2-A4` vs `T2-A5` | int8-per-token top-1 agreement: **0.8919** / **0.92676** / **93.16%** | Same codec, same model, three measured values spanning 4.0 points. Any slide quoting "int8 flips X% of tokens" is quoting one of three. | Quote as a **range, 89–93%**, or name the run and its corpus. | **OPEN** |
| **F14** | `40-PITCH` + `ADR-013` vs `30-PERF-MODEL` | The 19x decomposes as "**6.8x** single-stream × **2.8x** concurrency" (pitch) vs "**6.3x** × **3.0x**" (perf model) | Not just rounding. The 6.8x uses T5-A4's v1 of **116.0 ms/token** (which credits the −8 ms `output_hidden_states` win); the 24.21 tok/s endpoint uses **123.94 ms** (which does not). So the two halves of the published decomposition are computed on **different v1 baselines**, and 6.8 × 2.8 = 19.0 only by coincidence. `30-PERF`'s 6.3 × 3.0 is the internally consistent pair. | Publish **6.3x × 3.0x**. Both reach 19.0x; only one is self-consistent. | **OPEN** |
| **F15** | `10-ARCHITECTURE` §4.2 vs `30-PERF-MODEL` §4e | Per-token wire reduction quoted as **2,826x** (10-ARCH) and **2,738x** (30-PERF) | Different routing assumptions on the v1 side — 10-ARCH prices chain routing (2 activation crossings, bf16, 3,752 B); 30-PERF deliberately keeps star routing (4 crossings, int8, 3,872 B). Neither is wrong; both are unlabelled, in two synthesis docs, for the same headline. | State the routing beside the factor. Prefer **2,826x** (v1 ships chain routing per ADR-002). | **OPEN** |
| **F16** | `01-VERIFIED-FACTS` F4 → the 935x on every artifact | "1,821.7 MB → 1.948 MB = **935x** fewer bytes" | The v0 side counts **2** activation crossings per token; T1-A1 read `coordinator.py` and found v0 star-routes with **4**. v0's real hidden traffic is ~2,813.7 MB, so the true factor is ~**1,657x**. `ADR-013` C1 rules to keep 935x as a deliberate lower bound (understating our own baseline errs honestly) — a defensible call, but the *reason* is not printed anywhere the number appears. | Keep 935x. Add "conservative: v0's star routing makes the real figure ~1,657x" wherever the derivation is shown, so a judge who counts the POSTs does not think it is an error. | **OPEN** (decision recorded, disclosure missing) |
| **F17** | `teams/T4-A5-observability-and-sre.md` | "Return-hop bytes after F2's fix: 1760 B / 40 tokens = **44 B/token** (measured)"; "dllm:imbalance = **1.562** (measured)" | The file's own risks section says: *"I ran no code against the PoC. The containers were not started."* These are **derived from other teams' measurements**, tagged measured. 44 B/token is derived from a design that does not exist yet. | Retag as **(derived)**. The house rule in `00-SHARED-CONTEXT` is explicit and this is the clearest breach of it in the corpus. | **OPEN** |
| **F18** | `T4-A2`, `T4-A3`, `T4-A4` | llama.cpp RPC **91.8 → 52.7 tok/s** decode / **76.1 → 317.7** prefill; MLX "27B across 4× M3 Ultra ≈3x"; exo **47.2k** stars; Bittensor Chutes **>9.1 trillion** tokens; **$1.07/(GB/s)** for a used 3090 | Presented to 3–4 significant figures and load-bearing (the llama.cpp pair is `20-INFRA`'s headline recommendation and `T4-A2`'s deck line). Sourced to one third-party GitHub benchmark repo, one WWDC session, and price estimates the file itself calls "estimates, not verified quotes". None reproduced here. | Keep, but every occurrence must carry **"third-party, not reproduced"**. `T4-A2` already says this in its risks; the synthesis docs dropped the qualifier. | **OPEN** |

---

## LOW — cosmetic, or already self-flagged

| # | File | Claim | Problem | Correction | Status |
|---|---|---|---|---|---|
| **F19** | `T1-A1` §5 | Decomposition "closes to within **0.3 ms**" of the 72.8 ms transport figure | The components sum to 712.54 + 17.63 + 55.14 + 0.90 = **786.21 ms** against the 785.3 ms being decomposed — a 0.91 ms residual, not 0.3 ms. | Say "within 1 ms". Conclusion unaffected. | OPEN |
| **F20** | `30-PERF-MODEL` §4d | TPOT column reads 124.1 / 124.1 / **161.4** / 215.3 / 430.5 ms | The column is `R · D_max` for the **8/8/8** split only. In the rebalanced column (`D_max = 41.37`) TPOT at R=3 is **124.1 ms**, not 161.4 — the table's own headline result (`R* = S`) is where latency is still free. | Label the column, or split it in two. | OPEN |
| **F21** | `assets/split-model-bench.html` | "an fp16 hop on 1 GbE takes 14 µs — **0.06%** of it" | 14 µs / 72.9 ms = 0.019%; against the 123.94 ms chain figure it is 0.011%. The printed 0.06% was ~3–5x wrong in the direction that weakens the argument. | Now reads **0.011%** against 123.94 ms. | **FIXED** |
| **F22** | `T4-A4` BOM | Tier A = 3 × Mac mini M4 @ **$599** = $1,887 → **$4.37/1M tokens**, beating Claude Haiku 4.5 at $5.00 | The file's own risk list says Apple's US list is **$799** as of Aug 2026. At $799 the BOM is $2,487 and Tier A becomes **~$5.70/1M** — it crosses to the *losing* side of the only comparison the slide makes. | Use $799. The "we are not cheaper" thesis (`T5-A2`) survives either way; the Tier A row does not. | OPEN (self-flagged) |
| **F23** | `30-PERF-MODEL` §4 | `b64json(n) = 4·⌈n/3⌉ + **25**` used for activations, but the logits row uses **+18** (810,346 = 810,328 + 18) | Two different JSON-overhead constants in one byte model, undocumented. It matches T1-A1's measured 810,346, so it is a fit to reality rather than an error — but it makes the stated formula false for one row. | Note the two field-name lengths, or use one constant and accept 7 B of drift on 10.6 MB. | OPEN |

---

## Verified clean

Re-derived and correct, to the digit: FINDING 3's `147,200 → 543 = 271.1x`; FINDING 4's byte totals
(1,406.8 / 414.9 / 1,821.7 / 1.946 / 1.948 MB and 935x); `lm_head = 9.131` layer-equivalents and the
8.00 / 8.00 / 17.13 shard table; the GQA cache sizing (512 B/token/layer bf16, 12 KB whole model,
25.2 MB @2048); `30-PERF-MODEL`'s entire §4 byte ladder (all six rows, and its independent reconstruction
of T1-A1's 10,602,865 B/token to the byte); every `U = min(1, R/S)`, `X(N)`, `N* = D/D_max` and M/M/1
identity; T5-A2's 70B memory-wall table (47.0 / 17.6 / 11.8 / 8.8 GB) and its $5.76-vs-$1.73/day
arithmetic; the 75.85% two-of-three coalition share and the 33.33% row-sharded fix; `1-(0.95)^32 = 80.6%`
spot-check catch rate; T4-A4's `$/(GB/s)` column. `bench/parity_check.out` is a genuine run against the
real checkpoint and its three asserts pass, including the negative control.

**No invented credential, employer, hackathon placing, repo or benchmark was found anywhere in the corpus.**

---

## MUST NOT GO ON A SLIDE

1. **"163 Mbit/s"** as the sub-bf16 crossover. It is ~2 Mbit/s. (F01)
2. **"28.6x"** and **"2,908 s → 101.6 s"** for a 512-token generation. Measured basis says ~3.6x. (F02)
3. **Any whole-generation speedup at all.** Quote per-token at a stated `seq_len`. (F02)
4. **"32.9x"** from continuous batching without "against T1-A1's seq=512 baseline" attached. Same-run is 24.4x. (F03)
5. **"75,968x"** for the row-sharded return path. It is 25,323x. (F04)
6. **"114 ms per token is transport, the model hasn't started yet."** Measured: transport is 9.3%, compute is 89%. This line and T1-A1's headline cannot both be spoken. (F05)
7. **"454x"** (T1-A2's transport+KV composite) — built on the same inflated 114.2 ms baseline. (F05)
8. **"2.54x" / "3.31x"** for concurrency. Use 2.31x / 3.00x. (F06)
9. **"72.9 ms"** as *the* cached decode step in a distributed context — it is a single-process 24-layer number. (F07)
10. **"493,961,216 parameters."** The checkpoint has 494,032,768. (F09)
11. **"202,581x"** for the return path. Use 151,936x (raw→raw) and use it everywhere. (F10)
12. **"1.55x"** for anything wall-clock. Achievable layer-equivalents is 1.539x; measured wall clock is 1.30x. (F11)
13. **"6.8x × 2.8x"** as the decomposition of 19x — the two halves use different v1 baselines. Use 6.3x × 3.0x. (F14)
14. **llama.cpp / MLX / exo / Bittensor / GPU-price figures** without "third-party, not reproduced". (F18)
15. **Mac mini Tier A at $4.37/1M.** At list price it is ~$5.70 and loses the comparison it is making. (F22)
16. **Any `(measured)` tag on a v1 number.** Not one cell of the v1 ladder has been run — this is the corpus's own §7b finding and it is the first thing a hostile judge will test.

## The one experiment that retires most of this

A single integrated v1 run on the target hardware, at `seq_len` 512, at R=1 and R=4, with
`bench/perf_model_micro.py`'s estimator, would convert F02, F07, F08, F14 and half of the
"must not" list from arguments into measurements. **A measured 3x beats a modelled 19x in front
of anyone who asks one follow-up question** (`30-PERF-MODEL` §7c — the corpus already knows this).
