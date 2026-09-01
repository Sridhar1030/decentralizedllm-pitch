---
status: SYNTHESIS — the deliverable. Supersedes the pitch fragments in teams/T5-*.
inputs: 00-SHARED-CONTEXT.md, 01-VERIFIED-FACTS.md, all 25 teams/ files
rule: every number carries (measured) / (derived) / (modelled) / (cited). No untagged number ships.
---

# 40-PITCH — elevator, demo script, 5 slides, Q&A, submission copy

## THE HEADLINE CLAIM (one sentence, everything else supports it)

> **Same three containers, 19x the tokens per second — 1.27 → 24.2 tok/s (modelled) — and we found every
> one of those levers by profiling our own baseline first.**

Caveat that ships attached to it, every time, in the same size type:
*Modelled by composing per-stage times measured on one laptop. 6.8x is single-stream latency; 2.8x needs
three concurrent requests. v1 has never been run as an integrated system — this is a design target, not a
result.* (T5-A4 §7)

---

## 0. CONTRADICTIONS RESOLVED (read before quoting any number)

Five teams disagreed. These are the rulings, not averages.

| # | Conflict | Ruling | Why |
|---|---|---|---|
| 1 | Headline speedup: **19.0x** (T5-A4) vs **28.6x** (T3-A1) vs **32.9x** (T3-A3) | **Quote 19x.** | T5-A4 composes the *measured* 785.3 ms/token v0 wall clock (T1-A1 §5) against measured stage times. T3-A1's 28.6x is FLOP-arithmetic against a modelled 50 GFLOP/s clock, not the measured baseline. T3-A3's 32.9x needs continuous batching (not in v1) and was run at ctx=128, a different measurement — T3-A3 itself says do not add it. Never sum these. |
| 2 | v0 is **3 hops** (house numbers) vs **6 wire crossings** (T1-A1, measured) | **6 crossings, 4 activation-sized.** v0 is star-routed through the coordinator. | T1-A1 counted the bytes. FINDING 4's 935x already assumes chain routing on the v1 side, so it stands; anyone quoting a *doubled v0* byte figure is double-counting the same correction. Draw the star in scene 03. |
| 3 | Transport is **346x / 95x / 38.7x** improvable (T1-A2/3/4) vs **transport is 9–18% of the clock** (T1-A1) | **Both true; the ratio is per-hop, the fraction is per-token.** | Ratios are real and measured. But 89% of v0's wall clock is compute (T1-A1). Present the fast path as the lever that pays at WAN, at 10 GbE, and once nodes are GPU-backed — not as the demo-box win. Anyone opening with RDMA is optimising 0.9% of the clock on 10 GbE. |
| 4 | Compression: **int8+outliers, 906 B** (T2-A1) vs **bf16 and stop** (T2-A4, T2-A5) vs **no codec ever** (T2-A3) | **bf16 is the v1 default; int8+8-outlier-channels is the WAN toggle; no byte codec ships.** | T2-A4 measured the crossover: sub-bf16 only starts to matter below ~163 Mbit/s per hop. T2-A3 measured that entropy coding never wins on a decode frame at any LAN speed. T2-A1's int8 codec is correct and shippable — it is just aimed at a link we do not have on stage. |
| 5 | Parameter share after the fix: **33.3%** (T5-A1) vs **24.1%** (T4-A3) vs **51.7%** (T1-A5) | **51.7% today. Two different fixes, name both.** | Pre-slicing safetensors alone leaves 51.7% because `tie_word_embeddings` duplicates the 136M matrix. Row-sharding that matrix 3 ways → 33.3%. Moving embed+lm_head to the client (Petals' design) → 24.1%. Different fixes, different numbers; do not blend. |

Notation normalised: **R** = in-flight requests, **S** = pipeline stages, **U = min(1, R/S)**.
(T3-A5 used C and P for the same thing.)

---

## 1. THE 30-SECOND ELEVATOR PITCH

74 words, ~30 s at 150 wpm. (T5-A5 §6, verbatim.)

> A hospital consortium wants one model trained on all of their data, and no member — and no vendor — may
> ever hold the whole thing. So we cut the model up. Twenty-four layers, three machines, eight layers each.
> Every machine computes its slice and passes a hidden state along; the last one emits the token. Take any
> machine away and the model is gone. It runs today, on three CPU containers, no GPUs.

**Delivery:** pause after *"so we cut the model up"* — that is the idea. Land the last sentence flat, no
emphasis; it is the credibility line. Never say "revolutionary", "decentralized AI", or "democratize".

---

## 2. THE 2-MINUTE DEMO SCRIPT

Keyed to `assets/split-model-bench.html` — 6 scenes, 71 s of animation. **Drive it with the scene buttons;
do not let it autoplay.** 297 spoken words = 123 s at ~145 wpm (±3 s).

| Time | Scene | Say this, verbatim | w |
|---|---|---|---|
| **0:00–0:14** | **01 · One model, three machines** | "This is one language model — twenty-four transformer layers. We cut the stack into three shards, one per machine, each in its own container. No single node holds the full model. Everything else follows from that." | 35 |
| **0:14–0:32** | **02 · One token, three hops** | "Here's one token. The prompt hits the coordinator. Node zero embeds it, runs layers zero through seven, hands a hidden state to node one, which hands it to node two. Node two owns the output head — so node two is where a token is born." | 43 |
| **0:32–0:58** | **03 · What v0 actually costs** | "Now the honest part. We measured our own baseline. There's no KV cache, so the whole sequence is re-sent and re-computed every single token — watch the packet grow. It travels as float32, inside base64, inside JSON. And argmax runs on the coordinator, so node two ships six hundred thousand bytes of logits back per token. Meanwhile two of three nodes sit dark." | 62 |
| **0:58–1:30** | **04 · Four levers** *(step all four)* | "Four levers. One: cache K/V per shard, send only the newest position, and move argmax onto node two — the return path becomes a four-byte token id instead of six hundred kilobytes. Two and three: a binary frame on a persistent socket, bf16 instead of float32. Four: the output head is nine layers' worth of compute, so our 'equal' eight-eight-eight split really runs eight-eight-seventeen. Re-cut it, and run three requests at once — one request can never fill a pipeline." | 78 |
| **1:30–1:46** | **05 · When a node disappears** | "Kill node one. A third of the model is gone — today we return a five hundred. With a standby holding the same layers, the coordinator reroutes; the honest cost is that the dead node's cache dies with it." | 38 |
| **1:46–1:58** | **06 · What the levers bought** | "Nine hundred thirty-five times fewer bytes, two hundred seventy-one times less redundant compute. Caveat, same slide: on a fast LAN we're compute-bound — the wall clock comes from the recompute, not the bytes." | 32 |
| **1:58–2:00** | hold on 06 | "One model. Three machines. None of them knows it." | 9 |

**Running long?** Cut *"Two and three"* from the 0:58 row — levers 1 and 4 carry the numbers judges keep.
**Running short?** Restore *"Bytes win on WAN, on one-gig ethernet, and at long context"* to the 1:46 row.
**Live demo instead?** `docker compose up -d && ./demo.sh`. Do **not** run the live stack *and* the animation.
The animation is safer and 30x faster, and cold start is 60–90 s per node (T5-A4 risks) — a stack that boots
on stage dies on stage.

---

## 3. THE DECK — 5 slides, fully specified

### SLIDE 1 — "Why split a model at all"

**Bullets** (≤12 words each)
- Five hospitals, one shared model; no member may hold it. *(T5-A1 §2.2)*
- 70B fp16 across 3 nodes: 47 GB per shard — fits nothing. *(T5-A2 §Q5, derived)*
- Across 16 nodes: 8.8 GB — fits an ordinary laptop. *(T5-A2 §Q5, derived)*
- Below ~13B this is theatre. The memory wall is the product. *(T5-A2 headline)*
- European sovereign cloud IaaS: $12.6B in 2026, from $6.9B. *(Gartner, 9 Feb 2026, cited)*

**Visual.** Full-width horizontal bar chart, one row per configuration, x-axis = GB per shard on a linear
scale ending at 50 GB. Four bars, top to bottom: `70B fp16, N=3 → 47.0 GB`; `70B fp16, N=8 → 17.6 GB`;
`70B int4, N=3 → 11.8 GB`; `70B fp16, N=16 → 8.8 GB`. Three vertical dashed "device ceiling" lines drawn
*through* the bars and labelled at the top: **16 GB laptop**, **24 GB RTX 4090**, **80 GB H100**. Bars
right of the 16 GB line are grey and hatched (does not fit); bars left of it are solid in `--n0` teal.
A single caption under the chart in mono: `derived from Llama-3.3-70B config shapes — 70.552B params`.
No icons, no clip-art hospitals, no globe.

**Speaker notes** (97 words)
> Open with the hospital sentence — a named buyer inside ten seconds, before any architecture. Then the
> chart, and be blunt about what it says: at half a billion parameters, splitting a model is theatre, and
> our own demo proves it by loading the whole thing on every node. The interesting line is the 16 GB one.
> Left of it, a room of ordinary laptops can hold a frontier-class model. Right of it, they cannot, and no
> amount of quantization fixes that — int4 of a 7B is still a 7B. That boundary is the entire product.

---

### SLIDE 2 — "One token, three hops, today"

**Bullets**
- Qwen2.5-0.5B, 24 layers, split 8/8/8 across three Docker containers. *(00-SHARED-CONTEXT)*
- FastAPI nodes, CPU-only, 2 vCPU each; gateway, coordinator, Prometheus, Grafana. *(PoC source)*
- Node2 owns lm_head, so node2 is where a token is born. *(FINDING 1)*
- Pipeline parallelism is not ours: GPipe 2019, Petals 2022. *(T4-A3, cited)*
- What is new: the shard boundary is a trust boundary. *(T5-A2 §0)*

**Visual.** A left-to-right architecture strip, one row, five boxes with arrows:
`client → gateway :8080 (api-key, circuit breaker) → coordinator :8081 → [node0 | node1 | node2] → token`.
The three node boxes are drawn as one grouped panel; inside each, a vertical stack of small layer bars —
8 teal bars labelled `embed + L0–7`, 8 indigo bars labelled `L8–15`, 8 magenta bars labelled `L16–23 + norm
+ lm_head`. The `lm_head` bar is drawn **9.13x taller than a layer bar** and outlined in `--bad` red, with
a callout: `lm_head = 9.13 layer-equivalents`. Between node boxes, label the wire `hidden state [seq, 896]
fp32 · base64 · JSON`. Bottom-left corner, small mono footnote: `runs today — docker compose up`.

**Speaker notes** (94 words)
> This runs. Three containers, CPU only, no GPUs, and a real completion comes back. Say the prior art out
> loud before anyone asks: GPipe defined pipeline parallelism in 2019, Petals shipped it decentralized in
> 2022, and vLLM ships it today behind one flag. We did not invent the mechanism, and claiming we did is
> how you lose a technical judge in the first minute. What is different is that in all three of those, the
> shard boundary is a performance boundary. Here it is a trust boundary — and that changes what you build.

---

### SLIDE 3 — "Three bets: wire, bytes, queue"

**Bullets**
- Wire: 8.483 → 0.089 ms per hop, binary frame on persistent TCP. *(T1-A4, measured)*
- 4.0 ms/hop of that was TLS cert parsing on plain http. *(T1-A1, measured)*
- Bytes: bf16 halves the wire, 99.41% top-1, 3.5 µs. *(T2-A5, measured)*
- One channel is 972x the median — naive int8 outputs garbage. *(T2-A1, measured)*
- Queue: U = min(1, R/S). One request can never fill three stages. *(T3-A2, derived)*

**Visual.** Three stacked horizontal panels, each a before→after pair with the number between them.
**Panel A (Wire, teal):** two bars, `HTTP + JSON + base64 8.483 ms` vs `DLP binary frame 0.089 ms`, with
`95x` between them, and beneath in small mono: `40-byte header · persistent socket · TCP_NODELAY`.
**Panel B (Bytes, indigo):** a 4-step descending staircase labelled `fp32 3584 B → bf16 1792 B → int8+outliers
906 B → (WAN only)`, with a red X over the step below bf16 and the label `LAN crossover: ~2 Mbit/s`.
*(90-AUDIT F01: T2-A4's published 163 Mbit/s used a prefill per-position cost, 0.88 ms, as the decode-step
compute anchor. Against the measured decode step — 72.9 ms (T2-A2) / 123.94 ms (T1-A1) — the crossover is
~2 Mbit/s; 30-PERF-MODEL §5 independently derives 2.3 Mbit/s. Do not print 163 Mbit/s.)*
**Panel C (Queue, magenta):** two 3-lane Gantt strips over the same time axis. Top strip R=1: one diagonal
staircase of work, two lanes dark at every instant, labelled `33%`. Bottom strip R=3: all three lanes solid,
labelled `~100%`, `1.40 → 4.21 tok/s at unchanged per-request latency (modelled, T3-A2)`.

**Speaker notes** (98 words)
> Three bets, and one of them is a negative result we kept on the slide. The wire: ninety-five times per hop,
> and the funny part is that four milliseconds of every hop was parsing X.509 certificates for a connection
> that never used TLS. The bytes: bf16 is free and we stop there, because we measured the crossover — below
> bf16 you only win on a link slower than a couple of megabit. The queue is the one people miss: a single request
> cannot fill a pipeline, because token t+1 waits on token t. Only concurrency can, and three is the knee.

---

### SLIDE 4 — "The scoreboard, and the caveats"

**Bullets**
- 1.27 → 24.2 tok/s, 19x, same three containers. *(T5-A4 §5, modelled)*
- 147,200 → 543 position-forwards per node: 271x redundant compute. *(FINDING 3, derived)*
- 1,821.7 MB → 1.948 MB per generation: 935x fewer bytes. *(FINDING 4, derived)*
- Caveat: bytes, not seconds. On a fast LAN we are compute-bound. *(FINDING 4)*
- Caveat: v1 has never run as an integrated system. *(T5-A4 §7)*

**Visual.** A six-row scoreboard table, three columns `v0 | v1 | factor`, mono figures, right-aligned:
`bytes on the wire 1,821.7 MB | 1.95 MB | 935x` · `redundant position-forwards 147,200 | 543 | 271x` ·
`return path / token 810,325 B | 4 B | 202,581x` · `throughput (R=3, balanced) 1.27 tok/s | 24.21 tok/s | 19.0x` ·
`pipeline utilisation 33% | ~100% | 3x`.
**The `512-token generation 2,908 s → 101.6 s = 28.6x` row is REMOVED from the slide (90-AUDIT F02):**
T3-A1 modelled it at 50 GFLOP/s and 198 ms per cached decode step, both contradicted by measured numbers
already in this knowledge base (T1-A1's stage times imply ~227 s for the same v0 generation and 123.94 ms
per decode step). On the measured basis the whole-generation figure is **~227 s → ~64 s = ~3.6x**, which is
*lower* than the seq=512 per-token 6.3x because a generation spends most of its tokens at short sequence
lengths where there is little redundancy to remove. Quote per-token at a stated seq, never whole-generation.
Each row carries a right-margin tag chip: `measured` / `derived` / `modelled`. **Directly beneath the table,
in the same font size as the table body — not smaller** — a boxed caveat in `--bad`: *"935x is wire bytes,
not wall clock. 6.8x of the 19x is single-stream; 2.8x needs three concurrent requests. Not yet run
end-to-end."*

**Speaker notes** (97 words)
> The caveat is in the same size type as the numbers, on purpose. If I shrink it, you stop believing the
> table. Two-seventy-one-x is the number I would defend hardest — it is pure arithmetic on the model config,
> not a benchmark. Nine-hundred-thirty-five-x is the biggest number and the weakest claim, because it is
> bytes and on this LAN we are compute-bound; it wins on WAN, on one-gig ethernet, and at long context.
> Nineteen-x is modelled from stage times we measured on one laptop. We have not run v1 end to end. That is
> the honest state.

---

### SLIDE 5 — "The memory wall, and the ask"

> **Have this number ready before you show this slide.** Every memory-wall figure in this knowledge base
> is a **weights** figure. The first thing an infra judge does with "11.8 GB per shard" is add the KV cache
> to it, and until now nobody had. Derived below (§Q10) — **the short answer is that the int4 shard fits a
> 16 GB laptop with 4.2 GB of headroom, which is 5 concurrent 8k sessions, or one 41k-token session.**
> Say that unprompted; it is the difference between a slide and an engineered answer.

**Bullets**
- IS: the option when weights exceed the device you may use. *(T5-A2 §Q5)*
- IS NOT: cheaper than an API — 3.3x underwater on electricity alone. *(T5-A2 §Q8, modelled)*
- IS NOT: encryption — ~35% token recovery at an 8-layer split. *(arXiv:2602.16760, cited)*
- Next: pre-sliced weight files, so no node *can* load the whole model. *(T1-A5, ~40 LOC)*
- Ask: `<FILL: the specific ask — pilot partner, judges' pick, compute credits>`

**Visual.** Two columns on one slide, no chart. **Left, headed `IS` in `--ok` green:** three short lines,
each a checkmark plus the memory-wall figure repeated once as an anchor — `70B int4, N=3 → 11.8 GB/shard`.
**Right, headed `IS NOT` in `--bad` red:** three short lines with the arithmetic beside each —
`$5.76/day electricity vs $1.73/day of tokens`, `~35% of tokens recoverable from an 8-layer activation`,
`node.py:36 loads the whole checkpoint today`. Across the full width beneath both columns, a single-line
roadmap strip with four ticks: `pre-sliced shards → KV cache + local argmax → binary bf16 frame →
reshard-on-failure`, the first two marked `v1, days`, the last marked `~63 net LOC`. The ask goes in a
box at bottom-right, one line, largest type on the slide.

**Speaker notes** (99 words)
> I want to end on what this is not, because every line is a question you were going to ask. It is not
> cheaper than an API — a sixteen-node fleet burns five seventy-six a day in electricity to make a dollar
> seventy-three of tokens. It is not encryption; split inference raises the cost of recovering a prompt, it
> does not prevent it. And today every node still loads the whole checkpoint, which is forty lines from being
> false. What it is, is the only option when the weights do not fit the machine you are allowed to use.

---

## 4. OBJECTION HANDLING — the hostile-judge script

Tightened from T5-A2 §2. Lead sentence is the whole answer; the rest is only if pressed.

**Q1 · "Isn't this just Petals / exo / vLLM pipeline parallelism?"**
> Mechanically, yes — and we say so in the first ten seconds. GPipe 2019, Petals 2022. What differs is a
> property, not a mechanism: in all three, the shard boundary is a *performance* boundary. Here it is a
> *trust* boundary — no node can materialise full weights (pre-sliced shards, ~40 LOC), no plaintext logit
> vector ever crosses the wire (~2 LOC), and a lying node is caught by spot-checking at 1.7% extra compute.
> **Concede first:** today `node.py:36` loads the whole checkpoint on every node. Peak RSS ~3.95 GB. Say it
> before they find it.

**Q2 · "Pipeline parallelism over a network is strictly slower than one box."**
> Correct, and unarguable — for a model that fits. The same weight bytes are read per token however you cut
> them: 70B int4 at ~50 GB/s is 706 ms/token on one machine or sixteen (modelled). PP buys throughput via
> concurrency and *existence* via memory, never single-stream latency. Against the real alternative on a
> 16 GB laptop — llama.cpp streaming 35.3 GB from NVMe at 0.14 tok/s — a 16-node pipeline at 1.42 tok/s is
> ~10x faster (modelled). And the network is not the problem: 15 hops of bf16 activation is 1.97 ms on
> 1 GbE against 706 ms of compute — 0.3% of the budget.

**Q3 · "You're sending hidden states in the clear. Privacy hole?"**
> Yes, and worse than you think — but the biggest hole is two lines from closed. vec2text recovers 92% of
> 32-token inputs exactly from an embedding (arXiv:2310.06816); 27% of prompts exactly from next-token
> logits alone (arXiv:2311.13647) — and node2 currently ships that exact oracle, 607,744 bytes, every token.
> Move argmax onto node2: 4 bytes, oracle deleted. **The honest sentence: split inference is not encryption.
> It raises the cost of recovery; it does not make recovery impossible.** Cryptographic privacy is a TEE
> (2–8% overhead, arXiv:2409.03992) or MPC (BOLT: 3.18 s/token — three orders too slow). Both are v2.

**Q4 · "What happens when a node dies mid-generation?"**
> Today: total outage, HTTP 500, up to 60 s to notice. Here is the silver lining nobody expects — **our worst
> performance bug is currently our failover mechanism.** No KV cache means no distributed state to lose, so
> recovery costs exactly one extra forward pass. A KV cache *ends* that, which is why the boundary-activation
> journal (3.67 MB at n=512, ~5 LOC) ships with it. Rung 1 is reshard-onto-survivors: ~63 net LOC, 3–8 s,
> the request survives, HTTP 200 `degraded: true`. Residual we concede unprompted: the coordinator is a SPOF
> as total as any node, and no rung fixes it — that needs a stateless coordinator, v2.

**Q5 · "0.5B on CPU is a toy. Does this hold at 70B?"**
> The 0.5B demo *is* a toy, and 70B is the only size where this stops being a stunt. fp16: 47.0 GB/shard at
> N=3 (fits nothing consumer), 17.6 at N=8, **8.8 at N=16 — the first configuration where a room of ordinary
> laptops holds a frontier-class model at full precision.** Two things get *easier* at scale: Llama-3 does
> not tie embeddings, so the 27.6% duplication tax vanishes, and lm_head drops from 27.6% of params to 1.5%,
> which is why our split is 1.55x imbalanced today and would not be at 80 layers.

**Q6 · "Why not just quantise a small model and run it locally?"**
> For most people you should, and we will say so: one laptop, llama.cpp, go away. Three cases where it fails.
> **Capability floor** — quantisation trades bits, not parameters; int4 of a 7B is still a 7B. **The 4-bit
> floor is real** — int3-g128 measured +91.1% perplexity, 57% top-1 (T2-A4); there is no 2-bit escape hatch.
> **Sovereignty** — a local 7B is useless if policy demands 70B-class output *and* forbids egress.

**Q7 · "How do you stop a malicious node returning garbage?"**
> Today, nothing. The v1 ladder is three items and ~60 lines: norm/NaN sanity bounds, BLAKE3 commit-reveal
> per hop (32 B, ~1 µs), and 5% random shard re-execution. **5% catches an always-cheating node inside a
> single 32-token completion 80.6% of the time, for 1.7% extra compute** (derived). State the trap before
> they do: re-execution across heterogeneous hardware does not bit-match, so naive hash comparison
> false-positives on *honest* nodes — compare under tolerance (rel-L2 < 1e-3), calibrated against the
> measured honest-noise floor of 0.0021 for a bf16 round trip. And on ZK: 2,646 s/token proving for
> LLaMA-2-7B is 26,460x too slow. ZK is the 2030 answer; TEEs are the 2026 answer; spot-checks are the
> answer this weekend.

**Q8 · "What's the business model?"**
> Start by killing the wrong one, with arithmetic. A marketplace of idle consumer hardware selling cheap
> tokens: 16 nodes make 1.96M tokens/day = $1.73 at Together's $0.88/1M, and burn $5.76/day of electricity.
> **3.3x underwater on power alone**, before hardware or ops. So never compete on price per token. The model
> that survives is the sovereign inference appliance: the buyer is a compliance budget, not a compute budget,
> the alternative is *not running the model at all*, and the unit of value is **egress avoided** — which is
> measurable: bytes of prompt that left the perimeter = 0.

**Q9 (bonus) · "Why not RDMA?"**
> Because we measured it. RDMA saves ~27 µs per hop; deleting Python's HTTP+JSON+base64 stack saves 10,376 µs
> (T1-A3, measured). 99.7% of a v0 hop is software. Post-KV-cache the 1 GbE link carries 14 KB/token against
> ≥116 ms of compute — about 1,000x of headroom. RDMA earns its keep at 70B-class tensor parallelism, not here.
> We also benchmarked the "proper" answer and rejected it: torch.distributed gloo send/recv measured 219 µs
> against 30.1 µs for a plain framed socket — 7.3x slower than the simple thing.

**Q10 · "Your shard is 11.8 GB of weights. Where does the KV cache live?"**

The question the memory-wall slide invites and that nothing in this knowledge base answered until now.
Every 70B figure we quote — 47.0 GB fp16, 11.8 GB int4 per shard at N=3 — is **weights only**. KV is a
second, *concurrency-scaled* claim on the same RAM, and on the laptop tier it is what actually binds.

Llama-3.3-70B, 80 layers, **8 KV heads** (GQA), head_dim 128, fp16 KV. Per token, whole model:
`2 (K,V) × 80 × 8 × 128 × 2 B` = **327,680 B = 320 KiB/token**, so **106.7 KiB/token per 3-way shard**
(all **derived**; the same arithmetic as F3, different config):

| context | KV, whole model | KV, per 3-way shard | + int4 weight shard (11.8 GB) |
|---|--:|--:|--:|
| 4k | 1.25 GiB | 0.42 GiB | 12.2 GiB |
| 8k | 2.50 GiB | **0.83 GiB** | **12.6 GiB** |
| 32k | 10.00 GiB | 3.33 GiB | 15.1 GiB |
| 128k | 40.00 GiB | **13.33 GiB** | 25.1 GiB |

Three consequences, and the third is the one worth saying out loud:

1. **The laptop claim survives, with a stated bound.** 16 GB − 11.8 GB = **4.2 GB of headroom** =
   **5 concurrent 8k sessions, or one 41,287-token session.** Quote the bound, not just the fit.
2. **Concurrency, not context, is what breaks it first.** 10 concurrent 8k sessions is 8.33 GiB of KV per
   shard — 20.1 GiB total, off the laptop tier entirely. This is why ADR-005 caps in-flight requests at a
   number and ADR-001 exports `node_kv_bytes`: admission control is a *memory* controller here, not just a
   latency one.
3. **Past ~116k tokens of context the wall stops being a weights wall.** KV per shard (13.3 GiB at 128k)
   exceeds the int4 weight shard (11.8 GiB) — crossover at **115,998 tokens**. At long context the thing
   worth sharding is the cache, not the model. That is the honest v2 statement of our own thesis, and it
   is the strongest available answer to "why will this still matter at 70B."

GQA is doing most of the work: with plain MHA (64 KV heads) the same table is **8x larger** — 20 GiB at 8k
whole-model — and the laptop tier would not exist. Same discount that makes our 0.5B cache free (F3).

---

## 5. SUBMISSION-FORM COPY

**`<FILL: ...>` markers are preserved verbatim from T5-A5 and must not be invented.** Two of them
(PRIOR BUILDS, STAND OUT) are load-bearing: submitting them unfilled is worse than submitting short.
Delete any placeholder you cannot honestly fill — an empty line beats a vague one.

### IDEA TITLE

**"No Node Knows"** (13 chars) — **conditional.** Ship the pre-sliced-safetensors fix (~1 day, ~40 LOC)
before submitting, or the name is a claim `node.py:36` disproves. If that fix does not land, submit
**"Shardmind"** and use "No Node Knows" as the deck subtitle, where an aspirational phrase is allowed and
a product name is not.

### DESCRIPTION — default, 146 words

> We are building **No Node Knows**: one language model sharded by layer across N machines, where no single
> machine can hold or reconstruct the whole model. A prompt enters the coordinator; node 0 embeds it and runs
> its layers, hands a hidden state to node 1, then to node 2, which owns the output head and returns the token.
>
> **For whom:** hospital consortia, EU banks under DORA, and any team whose data may not leave the building
> but whose model no longer fits on one device.
>
> **How:** a working PoC already runs Qwen2.5-0.5B-Instruct across three CPU containers today. From that
> measured baseline we are landing four fixes — a per-shard KV cache (271x less redundant compute), argmax
> moved onto the tail node (607,744 bytes to 4 per token), a binary bf16 frame replacing JSON and base64, and
> a rebalanced split that deletes a 1.55x pipeline stall.

### DESCRIPTION — 60-word fallback (use only if the field truncates)

> No Node Knows splits one LLM across several machines so that no single machine ever holds the whole model.
> Our working prototype runs Qwen2.5-0.5B across three Docker containers: node 0 owns layers 0–7, node 1
> layers 8–15, node 2 layers 16–23 plus the output head. Kill any one and the model stops. For teams whose
> data cannot leave the building.

**Optional device-forward swap** (only if this hackathon is Android-judged) — replace the *For whom*
paragraph with:
> **For whom:** anyone holding three ordinary devices and a model that fits on none of them. An 8B int4 model
> is 4.0 GB — too much for one phone's app heap; 1.3 GB per device across three. Same trick, consumer scale.

*(1.3 GB/shard is derived: 8e9 × 0.5 B int4 = 4.0 GB ÷ 3. Do **not** claim a phone build you have not run.)*

### THE OTHER FIELDS

| Field | Put this | Note |
|---|---|---|
| VIDEO WALKTHROUGH URL | `<FILL: unlisted YouTube/Loom URL>` | Optional on the form, not optional in practice. Record §2 verbatim over the prototype. 2:00 hard cap. |
| PROTOTYPE URL | `<FILL: URL of the published animated prototype>` | `assets/split-model-bench.html` — self-contained, publishable as-is. |
| DECK / DOCUMENT | The 5 slides in §3 | Required. |
| ANDROID PROFICIENCY | **Leave "Basic"** unless untrue | Do not inflate. A judge forgives "Basic"; nobody forgives a demo contradicting a claimed "Advanced". |
| LLM PROFICIENCY | "Deployed local LLMs on-device" — **keep only if true** | `<FILL: confirm this is accurate — which model, which device, which runtime>` |

### PRIOR BUILDS & HACKATHONS — draft, first person

> I built the prototype in this submission end to end: **DecentralizedLLM**, a layer-sharded inference stack
> that splits `Qwen/Qwen2.5-0.5B-Instruct` across three Docker containers. Each container is a FastAPI layer
> node holding a contiguous slice of the 24 transformer layers; a coordinator drives the forward pass hop by
> hop; an API gateway in front adds key auth and a circuit breaker; Prometheus and Grafana are wired in. It
> works today — `docker compose up`, one curl, a completion comes back, with an SSE streaming path for live
> token flow. Stopping node 1 takes the model down, which is exactly the property I am pitching.
>
> `<FILL: hackathons — event name, year, what you built, placing. One line each. "Won X with Y" only if you won X. Delete this paragraph entirely if you have none — no hackathon history is a neutral fact, a vague one is a red flag.>`
>
> `<FILL: shipped at work — one line each: what shipped, at what scale, what was YOURS. Do not list team output as personal output.>`
>
> `<FILL: OSS — github.com/<user>/<repo>, one line on what it does. Include stars/downloads only if the number helps you.>`
>
> `<FILL: on-device / Android work — any app you shipped, any local model you have actually run on a phone (llama.cpp, MediaPipe LLM Inference API, MLC-LLM, ONNX Runtime Mobile), naming the device and the model. Delete if none — the STAND OUT section already handles the gap.>`

### WHAT MAKES YOU AND YOUR TEAM STAND OUT — draft, first person

> **I brought a working system, not a slide.** The three-node split runs before the pitch starts, so the demo
> is the product rather than a mock of it.
>
> **I profiled my own build before defending it, and the deck leads with what I found wrong.** The 8/8/8 layer
> split looks balanced and is not: `lm_head` is 136M parameters — 9.13 transformer layers of compute per token
> — so node 2 really carries 17.13 layer-equivalents against 8.00, a 1.55x throughput loss. The coordinator
> runs `argmax`, so node 2 ships a 607,744-byte fp32 logit vector back per token when a 4-byte token id would
> do. There is no KV cache, so a 32-token prompt generating 512 tokens performs 147,200 position-forwards per
> node where 543 suffice — 271x redundant. I would rather a judge hear those three numbers from me than find
> them in my repo.
>
> **I know exactly what this is not.** It is not cheaper than a hosted API and I will not claim it is. Below
> ~13B parameters, splitting a model across machines is a stunt; the architecture earns its place at the point
> where the weights stop fitting on the device you are allowed to use. That boundary is the product.
>
> `<FILL: your domain edge — the industry, regulation, or infrastructure you have actually worked in that makes THIS problem yours. Distributed systems? Healthcare or fintech data handling? On-device ML? One specific sentence beats three general ones.>`
>
> `<FILL: team — who else, what they own, and whether you have shipped together before. Delete this block if you are solo, and say "solo" plainly; solo is not a weakness on a working prototype.>`
>
> `<FILL: Android specifically — if your Android proficiency is Basic, say so here in one line and state the plan: the node is a plain HTTP service today, and the v2 path to a phone node is a known, bounded piece of work, not a mystery. Owning the gap is stronger than hiding it.>`

---

## 6. PRE-SUBMISSION CHECKLIST

| # | Item | Owner | Blocking? |
|---|---|---|---|
| 1 | Pre-sliced safetensors in `node.py` (~40 LOC) — otherwise the title is false | code | **Yes, for title A** |
| 2 | Both load-bearing `<FILL>` blocks filled or deleted | you | **Yes** |
| 3 | 935x never spoken or printed without its caveat clause | you | **Yes** |
| 4 | Prototype defects D1–D8 applied (T5-A3 §2): 40-byte header, star topology, honesty clause, scene-04 clock | code | High |
| 5 | `assets/demo.mp4` recorded — PowerPoint strips SVG animation | you | High |
| 6 | Slide 4's caveat box is the same font size as the table body | design | High |
| 7 | Do not quote EU AI Act high-risk as 2 Aug 2026. It is **2 Dec 2027** | you | **Yes** |
| 8 | Do not claim HIPAA forbids calling a US API — every major vendor signs BAAs | you | **Yes** |
| 9 | Never invent a TAM for "decentralised LLM inference". Say the category is unmeasured | you | **Yes** |
