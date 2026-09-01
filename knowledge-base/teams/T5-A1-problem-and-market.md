---
team: T5 — Product, Narrative & Deliverables
agent: T5-A1
topic: Problem statement and market case — why split an LLM across mutually-distrusting nodes
headline: Lead with data sovereignty (named laws, named dates, no security claim required); land model-weight
  confidentiality second, with the collusion limit stated by us before a judge asks. Our v0 does NOT have the
  property it claims — node0 already holds 51.7% of parameters because embed/lm_head are tied — and that is a
  one-day fix that takes max single-node share to 33.3%.
---

# T5-A1 — The problem and the market

## 0. Scope discipline

No invented TAM. Two market figures below are named-source/dated; everything else is argued qualitatively or from
arithmetic. Every number tagged (measured) / (derived) / (modelled) / (cited, <source> <year>).

---

## 1. Ranking, and the one to lead with

Judge-believability score = can it be demoed in 5 min + does it need the judge to accept an unproven security
claim + is there a named buyer *today*.

| # | Driver | Named buyer today | Demoable in 5 min | Needs unproven security claim | Score | Verdict |
|---|---|---|---|---|---|---|
| 1 | **(a) Data sovereignty / regulation** | Yes — EU hospital consortium, EU bank under DORA, Indian fintech under RBI localisation | Yes — label shards by jurisdiction, kill one, show outage | **No** | **9/10** | **LEAD** |
| 2 | **(b) Model-weight confidentiality** | Yes — model owner serving on a customer's tin | Yes — `curl` node1, show it cannot generate | **Yes** (and the honest answer is "partly") | 7/10 | Beat 2, with caveat pre-empted |
| 3 | (d) Edge / air-gapped / intermittent | Yes — ship, forward hospital, factory line | Yes — pull the network cable | No | 6/10 | Supporting slide |
| 4 | (c) Cost / idle heterogeneous compute | Weak — the buyer is a CFO, not a user | Arithmetic only | No | 5/10 | One line in the appendix |
| 5 | (e) Censorship-resistance / availability | No enterprise buyer | Yes | No | 3/10 | Cut |

**Lead with (a), phrased in the buyer's language, and reveal (b) as the mechanism that makes (a) real.**
One sentence: *"A consortium of hospitals wants a model fine-tuned on all of their data, and no member — nor any
vendor — may ever hold the whole thing."* That sentence is simultaneously (a) and (b), has a real buyer, needs no
crypto claim, and is the only framing where "just run vLLM on one box" genuinely fails rather than merely being
inconvenient. Do **not** open on (b) alone: the first judge question is "what if two nodes collude" and you have
30 seconds to answer it. Answer it on your own slide instead (§3.4).

---

## 2. (a) DATA SOVEREIGNTY — the lead

### 2.1 The regulatory surface, verified

| Instrument | Status / date | What it actually forces | Overclaim to avoid |
|---|---|---|---|
| GDPR Art. 44–49 | In force since 2018 | Transfers outside the EEA need an adequacy decision, SCCs, or a derogation | It does **not** ban US processing |
| EU–US Data Privacy Framework | Adequacy 2023; upheld by the General Court in **Latombe**, 3 Sep 2025; **appealed to the CJEU Oct 2025, no hearing scheduled as of mid-2026** (cited, Hunton / eucrim 2025-26) | Makes US processing lawful **today** | Do not say "illegal". Say **"legally durable for a 5-year contract? Procurement won't sign."** Safe Harbor and Privacy Shield both died at the CJEU |
| EU AI Act — prohibited practices | In force since 2 Feb 2025 | Bans, unchanged | — |
| EU AI Act — GPAI obligations | In force since **2 Aug 2025** | Documentation, copyright policy, training-data summary | — |
| EU AI Act — high-risk (Annex III) | **Deferred from 2 Aug 2026 to 2 Dec 2027** by the Digital Omnibus (provisional agreement 7 May 2026; EP 16 Jun 2026; Council 29 Jun 2026) (cited, 2026) | Logging, human oversight, data governance for e.g. clinical triage | Getting this date wrong in front of a judge is the fastest way to lose the room. It is **2027**, not 2026 |
| India DPDP Act 2023 + **DPDP Rules 2025** | Rules notified **13 Nov 2025** (G.S.R. 846(E)); full compliance **13 May 2027** (cited, 2025-26) | Consent, breach notice, Significant Data Fiduciary duties | The Act permits transfers except to notified blocked countries — it is **not** a blanket localisation law |
| RBI payment-data localisation | Directive DPSS.CO.OD No.2785/06.08.005/2017-18, Apr 2018 | Payment system data **stored only in India** | Sector-specific, not general |
| **DORA** | Applies since **17 Jan 2025** | ICT third-party concentration-risk management + documented **exit strategies** for EU financial entities | This is the sharpest financial-sector hook — it targets *dependence on one vendor*, which is exactly what we remove |
| HIPAA | In force | A BAA with the processor | **HIPAA does not forbid calling a US API.** OpenAI, Anthropic, AWS Bedrock, Azure OpenAI and Vertex all sign BAAs at enterprise tier (cited, 2026). Claiming otherwise is a factual error a healthcare judge will catch |

### 2.2 Buyer / workload / why the alternative fails

| Buyer | Workload | Why "just call an API" fails | Why "just run vLLM on one box" fails |
|---|---|---|---|
| **5-hospital EU research consortium** | Discharge-letter summarisation + cohort search over pooled records; ~8,000 docs/day, ~1,200 in / 300 out tokens | Art. 9 special-category data; DPF is under CJEU appeal → 5-year procurement risk; national health-data rules stack on top | **This is the case where it genuinely fails.** The model is fine-tuned on all five hospitals' data. Whichever hospital hosts the box owns a extractable asset derived from the other four. No member will host it, and no member will let another |
| **EU bank, DORA-regulated** | Internal policy/controls Q&A over confidential credit files | Concentration risk on one US GPAI provider is now a named regulatory finding; exit strategy must be *demonstrable* | Single box = single ICT dependency, just self-hosted. Does not answer the resilience test |
| **Indian payments company** | Chargeback-dispute triage on payment records | RBI Apr 2018: payment data must be stored in India, full stop | Works — but the fleet is already idle branch/office hardware; see (c) |
| **Defence / govt integrator** | Classified document QA | No cross-border processing at any price | Works technically; fails on "no single admin can walk out with the model" |

### 2.3 Market figures — named source only

| Figure | Source |
|---|---|
| Worldwide sovereign cloud IaaS spending **$80B in 2026, +35.6% YoY** | Gartner press release, 9 Feb 2026 |
| **European** sovereign cloud IaaS: **$12.6B in 2026, up from $6.9B in 2025** | same Gartner release |
| H100 market-median rental **$2.95–$3.46 /GPU-hr, mid-2026**; published rates $1.49–$12 | IntuitionLabs 2026 cross-provider survey; CloudZero 2026 |

There is **no defensible TAM for "decentralised LLM inference"** — the category does not exist as a measured
segment. Say that out loud instead of inventing one; it costs nothing and buys credibility. The sovereign-cloud
line is the honest proxy: it is the budget line this would be bought out of.

---

## 3. (b) MODEL-WEIGHT CONFIDENTIALITY — the sharpest framing, and its honest limits

### 3.1 The threat model, stated precisely

Assets: `W` = the full weight set; prompts `x`; outputs `y`.
Principals: model owner `M`; node operators `N0,N1,N2` (mutually distrusting, possibly the *customer*); a
network observer `E`; the coordinator `C`.

| Adversary | Capability in our v0 | Wants |
|---|---|---|
| A1 — one honest-but-curious node operator | Its own shard's weights; every activation crossing it, in cleartext | `W`, `x` |
| A2 — coalition of 2 of 3 node operators | Two shards + query access to the third | `W` |
| A3 — passive wire observer `E` | Every hidden state and the full fp32 logit vector, base64 over **plaintext HTTP/1.1** on a Docker bridge network (measured, `node.py:97-113`, `coordinator.py:43-72`) | `x`, `y`, `W` |
| A4 — the coordinator | Token ids in, tokens out | `x`, `y` |

Security goal that is actually achievable: **no principal short of the full set can obtain `W`, and obtaining `W`
requires an active conspiracy that leaves evidence.** Not: privacy of `x`. See §3.4.

### 3.2 Our v0 does not have the property it claims (derived, from `01-VERIFIED-FACTS.md` + `node.py`)

`config.json` has `tie_word_embeddings: true`. `node.py` gives node0 `embed_tokens` and node2 `lm_head` — the
**same 136,134,656-parameter matrix**, 27.6% of the model, materialised twice.

| shard | params held | share of 493,961,216 |
|---|---|---|
| node0 = embed (136,134,656) + layers 0–7 (119,275,520) | 255,410,176 | **51.71%** |
| node1 = layers 8–15 | 119,275,520 | 24.15% |
| node2 = layers 16–23 + lm_head (136,134,656) | 255,410,176 | **51.71%** |
| **any 2 of 3, distinct params** | 374,685,696 | **75.85%** |

> **A single node in v0 holds a majority of the model's parameters.** Any 2 of 3 hold 75.85% — identical for all
> three pairs, because the tied matrix is double-counted.

**v1 fix (1 day):** row-shard the tied embed/lm_head matrix 3 ways — 151,936 vocab rows / 3 = 50,645 rows =
45,378,219 params per node. Per node: 45,378,219 + 119,275,520 = **164,653,739 = 33.33%**; any two = 66.67%.
This also collapses FINDING 2's 607,744-byte logit payload into three 8-byte (local max, local argmax) partials
plus a coordinator-side `max` — **25,323× fewer bytes on the return hop** (607,744 B / 24 B; the
90-AUDIT F04 correction — the file previously said 75,968×, which divided by one 8-byte partial, not three), and it removes the last dense matmul
from node2's critical path, helping FINDING 1's 1.55× imbalance. One change, three wins.

### 3.3 Buyer / workload / why the alternative fails

| Buyer | Workload | Why "just call an API" fails | Why "just run vLLM on one box" fails |
|---|---|---|---|
| Vertical-AI vendor (radiology, legal, claims) with a 70B model fine-tuned on years of proprietary labels | Serve inside the customer's VPC because the customer will not export data | Reversed: it is the *customer* who refuses the API. The vendor must go on-prem | A single `.safetensors` on the customer's disk is the vendor's entire company. Weight exfiltration by an admin is one `scp` |
| Frontier lab licensing a capability to a defence/health partner | Attested on-prem serving | Partner cannot use a hosted endpoint | Same — hand over the box, hand over the model |
| Hospital consortium (§2.2) | Jointly fine-tuned model | n/a | No member may unilaterally hold the joint asset |

### 3.4 The honest limitation — put this on a slide, in our own words, first

Three separate results, all real, all against us:

| Attack | Result | Hits which adversary |
|---|---|---|
| Morris et al., *Text Embeddings Reveal (Almost) As Much As Text*, EMNLP 2023 (arXiv 2310.06816) — vec2text | Recovers **92% of 32-token inputs exactly**, BLEU 97.3, from a dense embedding | A1, A3 |
| *Language Models are Injective and Hence Invertible*, ICLR 2026 (arXiv 2510.15511) — SIPIT | Decoder-only transformers are almost-surely injective; prompts recovered **exactly from internal activations in provably linear time** | A1, A3 — this is our exact wire format |
| Carlini et al., *Stealing Part of a Production Language Model*, ICML 2024 best paper (arXiv 2403.06634) | Recovered the full embedding-projection matrix of OpenAI `ada`/`babbage` and confirmed hidden dims 1024/2048 **for under $20**, from black-box top-k logprobs alone | A2 |
| Borzunov et al., *Petals*, NeurIPS 2023 (arXiv 2209.01188) | The authors state plainly that peers serving the first layers can recover input tokens; their only mitigation is "use trusted servers or run your own swarm" | prior art punts on this too |

Three consequences we must state, not hide:

1. **The hidden state *is* the prompt.** Any node, and any wire observer, reads `x`. mTLS fixes A3 and does
   nothing for A1. **We must not claim prompt privacy.**
2. **A2 is strictly stronger than Carlini's setting.** A coalition holding 2 shards has exact intermediate
   activations for the missing block — recovering it is supervised regression with unlimited labelled data, not
   black-box extraction. Assume 2-of-3 collusion ⇒ full model.
3. Therefore sharding is a **cost-and-evidence multiplier, not a proof**: it converts "one admin runs `scp`" into
   "three organisations conspire". That is a real, sellable control (it is how key ceremonies and HSM quorums are
   sold) — but it is a *governance* control, not a cryptographic one. Say the word "governance".

### 3.5 What actually closes the gap, with honest cost

| Mechanism | Closes | Cost | Tag |
|---|---|---|---|
| mTLS between nodes + binary framing | A3 only | ~0 | **v1** |
| Row-shard tied embed/lm_head (§3.2) | Raises A1 from 51.7% → 33.3% | 1 day | **v1** |
| Sample/argmax on node2, ship a 4-byte token id | Removes the logit vector Carlini's attack feeds on | hours (FINDING 2) | **v1** |
| Periodic shard re-permutation across a larger pool | Raises A2's cost (must corrupt a *changing* majority) | days–weeks | v2 |
| **GPU TEE — NVIDIA H100 Confidential Computing + Intel TDX, remote attestation, encrypted weight delivery** | A1 and A2 **properly** — the host operator cannot read weights it is running | **2–8% throughput** (arXiv 2409.03992 benchmark; NVIDIA quotes 2–5%; overhead is I/O-bound and shrinks with model and sequence size) | v2 |
| Secure MPC (PUMA, BumbleBee, BOLT) | Everything, cryptographically | BumbleBee: **>13 min per token**, LLaMA-7B, 8-token prompt. BOLT: **3.18 s/token** on a 30 Gbps / 0.8 ms LAN (cited, Nimbus arXiv 2411.15707) | **v2 / never for interactive** |

> Honest conclusion for the deck: **TEE is the production answer for weight confidentiality; MPC is 3–4 orders of
> magnitude too slow; our sharding is the governance layer that works on hardware you already own, today, with no
> H100 and no attestation infrastructure.** A judge will respect that we know which one we are.

---

## 4. (d) EDGE / air-gapped / intermittent — supporting

| Buyer | Workload | Why API fails | Why one box fails |
|---|---|---|---|
| Naval vessel / offshore rig | Maintenance-manual QA, incident logs | No link, or a 600 ms satellite link metered per MB | Mostly works. Splitting wins only when the biggest model that fits on any single onboard box is too small, and several boxes exist |
| Forward/rural hospital, intermittent WAN | Triage note drafting | Link unavailable when it matters most | Same caveat |
| Factory floor / defence enclave | Ops QA on classified SOPs | Air gap | Same caveat |

**Be honest:** the counter here is strong. Splitting is justified at the edge only by *aggregate memory* — N idle
boxes hold a model none of them fits — not by sovereignty. State the condition, don't overclaim the driver.

---

## 5. (c) COST — appendix line only

70B-class model, bf16 = 70e9 × 2 B = **140 GB** of weights.

| Option | Config | Monthly cash | Aggregate memory BW |
|---|---|---|---|
| Rent | 2 × H100 80GB, 24/7, @ $3.20/GPU-hr (mid-point of the cited $2.95–$3.46 median) × 730 h | **$4,672** (modelled) | 2 × 3,350 = 6,700 GB/s |
| Idle fleet | 8 office workstations, RTX 4090 24 GB = 192 GB aggregate; marginal electricity 8 × 450 W × 730 h = 2,628 kWh @ $0.18 | **~$473** (modelled) | 8 × 1,008 = 8,064 GB/s aggregate, but **only 1,008 GB/s effective** without microbatching, because a naive pipeline leaves 7 of 8 stages idle |

**9.9× cheaper cash-out, ~6.6× slower per single stream** until T3-A2's microbatching lands. A typical office fleet
is idle 128 of 168 weekly hours = **76% idle** (modelled, 9-5 M-F). This is a *sunk-cost utilisation* argument, not
a throughput argument, and the counter — "rent an H100 spot instance at $2.50/hr" — is easy. One line, no slide.

---

## 6. (e) CENSORSHIP-RESISTANCE — cut

No enterprise buyer, no procurement line item, and Petals (NeurIPS 2023) already occupies the ground. It reads as
ideology to a judge evaluating an engineering submission. Drop it entirely rather than dilute the lead.

---

## 7. The claim ladder — hand this to T5 slide authors verbatim

| Claim | Can we say it? |
|---|---|
| "No single node can serve the model alone" | **Yes** — verifiable live: stop node1, the pipeline dies |
| "No single node holds the whole model" | **Yes, after the v1 tying fix.** Today it is 51.7%; after, 33.3% |
| "No single node can read the prompt" | **No.** node0 receives raw token ids; hidden states are exactly invertible (SIPIT, ICLR 2026) |
| "A wire-tapper cannot read the prompt" | Only with mTLS — and it is irrelevant against a malicious node |
| "A colluding majority cannot reconstruct the model" | **No.** 2 of 3 = 75.85% of params + exact activations for the rest |
| "Compromising the model requires conspiracy across N organisations, and leaves evidence" | **Yes** — this is the real claim, and it is the one worth selling |
