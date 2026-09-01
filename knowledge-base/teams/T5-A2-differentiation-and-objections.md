---
team: T5 — Product, Narrative & Deliverables
agent: T5-A2
topic: Competitive positioning vs Petals / exo / vLLM-PP / hosted APIs, and the eight-objection script for a hostile judge
headline: >
  We cannot win on price and must never claim to — 16 idle consumer machines burn $5.76/day of electricity
  to produce $1.73/day of Together-priced tokens (3.3x underwater, modelled). The defensible claim is the
  memory wall: at 70B fp16 a 3-node shard is 47.0 GB and fits nothing; at 70B int4 a 3-node shard is 11.8 GB
  and fits a laptop. Below ~13B this architecture is a stunt; above ~70B it is the only local option. Also:
  the pitch's own headline claim is FALSE in v0 — `node.py:36` loads the entire checkpoint on every node.
---

# T5-A2 — Differentiation and the objection script

`(measured)` = a teammate's harness produced it; `(derived)` = falls out of `config.json` arithmetic;
`(modelled)` = everything else. Every §2 number traces to a teams/ file or a cited paper.

---

## 0. The differentiation paragraph (memorise this; it is the answer to Q1 and Q8 both)

> Petals, exo and vLLM each solve one third of this. **vLLM v0.27.0** has the best pipeline-parallel engine in existence and assumes the one thing we deny: a trusted, homogeneous, co-located cluster on a single control plane. **exo** solves zero-config discovery beautifully but is Apple-Silicon-and-Thunderbolt shaped — one owner, one room, one trust domain. **Petals** is the true ancestor and got the hard part right, BitTorrent-style block servers over the open internet, and then explicitly told you it cannot defend you: its own paper says peers serving the first layers can recover your input tokens, and its answer is "use a private swarm." We are building the private swarm as the *product*: a layer-sharded pipeline where the shard boundary is a **trust boundary and an enforced memory boundary**, not a convenience. Concretely, three things nobody above ships together — no node can materialise the full checkpoint (pre-sliced safetensors, not a runtime slice of a full load); no node ever sees a plaintext logit vector (argmax moves to the tail shard, 607,744 B → 4 B per token); a node returning garbage is caught by commit-reveal plus spot-checking at **1.7% extra compute**, not by trusting it. We are not cheaper than an API. We are the option that exists when the weights do not fit on your laptop and the prompt is not allowed to leave your building.

---

## 1. Positioning table

| | **Petals** (bigscience-workshop) | **exo** (exo-explore) | **vLLM PP** v0.27.0 | **Hosted API** (Together, Llama-3.3-70B) | **DecentralizedLLM** |
|---|---|---|---|---|---|
| Shape | BitTorrent block servers, libp2p, WAN | Auto-discovered LAN cluster, Thunderbolt-5 RDMA, MLX | `--pipeline-parallel-size N` inside one cluster | Someone else's GPUs | Layer-sharded pipeline, explicit trust boundary |
| Trust domain | Many, untrusted | **One** (your own devices) | **One** (your own cluster) | **One** (theirs) | **Many, mutually distrusting** — the design centre |
| Transport | plain OS TCP over libp2p | Thunderbolt RDMA / TCP | NCCL / Ray, requires homogeneity | HTTPS | HTTP/1.1+b64 today → binary bf16 + keepalive (T1-A4) |
| Activation compression | dynamic blockwise int8, −0.4 pt on OPT-175B | none documented | none (assumes fat interconnect) | n/a | bf16 default, int8 only under ~160 Mbit/s (T2-A4, measured) |
| Node death mid-gen | reroute to another block server (replicas exist) | cluster reforms; request lost | **fatal** — PP rank loss kills the engine | their problem | reshard onto survivors, request survives (T1-A5 rung 1) |
| Byzantine node | **not addressed** | n/a (you own the hw) | n/a | n/a | commit-reveal + 5% spot-check (§2.7) |
| Prompt privacy | paper: first-layer peers can recover input tokens | n/a | n/a | contractual only | argmax on tail node; client-side embedding (v2) |
| Price/1M tok | free (donated) | your electricity | free (your GPUs) | **$0.88 in/out** (Jul 2026) | your electricity — **more than $0.88** (§2.8) |
| Throughput datum | **0.83 tok/s** geo-distributed (T4-A3, measured) | Thunderbolt-5 RDMA, Apple-silicon only | fastest of the four, on one cluster | ~30+ tok/s | LAN target, sub-ms hop (T1-A2) |
| Honest verdict | closest ancestor; steal the routing | steal the discovery UX | **steal the engine**, not the fabric | the price floor we must not fight | the fabric + the trust story |

**One-line steal list:** vLLM's `EngineCore`/continuous batching (T4-A1), Petals' shortest-path chain routing
(T3-A5), exo's zero-config discovery. Build only the trust boundary and the failover — that is ~103 net LOC
across `node.py` + `coordinator.py` (T1-A5 §6 + §7.4).

---

## 2. The eight questions

### Q1 — "Isn't this just Petals / exo / vLLM pipeline parallelism?"

**Honest answer: mechanically yes, and we should say so in the first ten seconds.** The forward-pass shape is
identical and we did not invent it. GPipe (Huang et al., NeurIPS 2019) defined it; Petals (Borzunov et al.,
ACL 2023 demo, arXiv:2209.01188) shipped it decentralized in 2022. Claiming novelty in the *mechanism* is how
you lose a technical judge in the first minute.

What differs is a property, not a mechanism: **in all three prior systems the shard boundary is a performance
boundary; here it is a trust boundary.** Three testable consequences, none of which Petals, exo or vLLM-PP
ships: (a) a node *cannot* materialise full weights — pre-sliced shard files, **v1, ~40 LOC**; (b) no plaintext
next-token distribution ever crosses the wire — **v1, ~2 LOC**; (c) Byzantine-node detection — **v1, ~30 LOC**.

**Concede first, loudly:** today `layer-nodes/node.py:36` calls `AutoModelForCausalLM.from_pretrained(MODEL_NAME)`
on *every* node and slices afterwards; `Qwen2ForCausalLM(config)` allocates a full random 24-layer model first,
so peak RSS ≈ 2 × 1976 MB = **3.95 GB/node** (derived). The slide claim "no node holds the full model" is
**false in v0**, true only after pre-slicing. Say it before the judge finds it.

### Q2 — "Pipeline parallelism across a network is strictly slower than one box. Why would anyone do this?"

**Correct, and unarguable for a model that fits.** Pipeline parallelism does not accelerate a single stream:
the same total weight bytes get read per token whichever way you cut them. 70B int4 = 35.3 GB; at ~50 GB/s
consumer memory bandwidth that is **706 ms/token whether it is one machine or sixteen** (modelled). PP buys
throughput via concurrency, and existence via memory — never single-stream latency.

The correct comparison is not against a box that doesn't exist. Against the real alternative on a 16 GB laptop:

| Option for 70B int4 (35.3 GB) on a 16 GB machine | Mechanism | tok/s (modelled) |
|---|---|---|
| Run locally | doesn't fit; llama.cpp mmap streams from NVMe at ~5 GB/s | 35.3/5 = **0.14** |
| **16-node pipeline, 2.2 GB/shard, all in RAM** | 16 × 44 ms sequential | **1.42** |
| One H100 80 GB | fits in VRAM | ~30+ |

**≈10x faster than the honest local alternative** (modelled). And the network is not the problem: at H=8192,
bf16 activations are 16,384 B/token/hop; 15 inter-node hops = 245,760 B/token = **1.97 ms on 1 GbE, 0.20 ms on
10 GbE** (derived) against 706 ms of compute — **0.3% of the budget.** Corroborated at 0.5B: transport is only
**9–16% of v0 per-token wall clock**, and 17.6 ms/token of *that* is httpx client construction, not wire time
(T1-A1, measured). The interconnect is not why this is slow; consumer memory bandwidth is.

### Q3 — "You are sending hidden states in the clear. Isn't that a privacy hole?"

**Yes. It is worse than the judge thinks, and the biggest hole is one we can close in two lines.**

| Attack | Source | Result |
|---|---|---|
| Invert a sentence embedding | Morris et al., EMNLP 2023, arXiv:2310.06816 (vec2text) | **92% of 32-token inputs recovered exactly**, BLEU 97.3 |
| Invert *next-token logits alone* | Morris et al., ICLR 2024, arXiv:2311.13647 | Llama-2-7b: BLEU 59, token-F1 78, **27% of prompts exact** |
| Invert a split-inference activation | arXiv:2602.16760 (2026) | **~59% @ 2-layer split, ~35% @ 8-layer split** |
| Prior art conceding the same risk | Petals, arXiv:2209.01188 | "peers serving the first layers can use their inputs to recover input tokens" |

Who leaks what in v0 (T1-A5 §7.2): **coordinator holds plaintext** (it tokenizes); **node0 receives raw
`input_ids`** (`coordinator.py:46`) — hop 0 is literally plaintext; node1 sees an 8-layer activation (~35%
recovery by analogy, modelled); **node2 emits the full 607,744 B fp32 logit vector every token — precisely the
oracle arXiv:2311.13647 inverts.** Fix ladder, in order of return:

| Fix | Effect | Cost | Tag |
|---|---|---|---|
| argmax/sampling on node2, return token id | 607,744 B → **4 B**, 151,936x; inversion oracle deleted | ~2 LOC | **v1** |
| top-k=50 as (id, logit) pairs if sampling needed | 400 B, still 1,520x | ~5 LOC | v1 |
| mTLS between nodes (uvicorn `--ssl-*`) | wire confidentiality; free *after* connection pooling | ~15-line openssl script | **v1** |
| Client holds `embed_tokens`, sends `h` not ids | kills the plaintext hop-0; Petals' own recommendation | moves 544.5 MB to client | v2 |
| Deeper split point before the first untrusted node | 59% → 35% recovery at 8 layers | placement policy | v2 |

**The honest sentence:** "split inference is not encryption. It raises the cost of recovery; it does not make
recovery impossible. Our claim is a smaller blast radius per node plus deletion of the strongest oracle — not
cryptographic privacy. Cryptographic privacy is homomorphic encryption or a TEE, and both are v2."

### Q4 — "What happens when a node disappears mid-generation?"

**Today: total outage, HTTP 500, and the failure takes up to 60 s to notice** (`timeout=60`,
`coordinator.py:46`). Worse, the gateway circuit breaker is reset by the very failure it exists for
(T1-A5 bug B1, `gateway/app.py`, 2-LOC fix).

**Silver lining nobody expects: v0's worst performance bug is currently its recovery mechanism.** The
coordinator resends the whole `gen_ids` every token (no KV cache), so **there is no distributed state to
lose** — failover costs one extra forward pass = **one token of latency.** A KV cache ends that; sequence the
roadmap accordingly.

| Rung | Mechanism | Cost | Recovery | Tag |
|---|---|---|---|---|
| 1 | Reshard onto survivors (node0→`0-12`, node2→`12-24`) | +238.6 MB RAM/survivor | 3–8 s, request survives | **v1** |
| 2 | Boundary-activation journal (keep the `h` you already relay) | 3.67 MB @ n=512, ~5 LOC | ~0 beyond rung 1 | v1.5 |
| 3 | Hot standby of the hottest shard | +1 node (477.2 MB) | ≤1.5 s | v2 |
| 4 | N+1 chained replication, 50% of model per node | 2x weights fleet-wide | ≤1.5 s | v2 |

Demo: `docker compose stop node1` live on stage; generation pauses ~5.6 s, reshard SSE events fire, the
completion finishes coherently, **HTTP 200 with `degraded: true, nodes: 2`**, `docker ps` shows two containers.
**~63 net LOC** (T1-A5 §6). Residual to concede: the coordinator is an equally total SPOF that no rung fixes —
that needs etcd 3.6.6 and a stateless coordinator (v2).

### Q5 — "0.5B on CPU is a toy. Does this hold at 70B?"

**The 0.5B demo is a toy, and 70B is the only size at which this architecture is not a stunt.** Arithmetic
for Llama-3.3-70B (L=80, H=8192, I=28672, V=128256, 8 KV heads, head_dim 128 ⇒ **70.552B params**, derived
from config shapes):

| Precision | Total | **N=3** | **N=8** | **N=16** |
|---|---|---|---|---|
| fp16 (2 B) | 141.1 GB | **47.0 GB** | **17.6 GB** | **8.8 GB** |
| int4 pure (0.5 B) | 35.3 GB | **11.8 GB** | **4.4 GB** | **2.2 GB** |
| int4 realistic (4.5 bpw with group scales) | 39.7 GB | 13.2 GB | 5.0 GB | 2.5 GB |
| + KV cache @ 4096 ctx (320 KB/tok, GQA) | 1.34 GB | +0.45 GB | +0.17 GB | +0.08 GB |

Fit grid: **fp16 N=3 (47.0 GB)** fits nothing consumer; **fp16 N=8 (17.6 GB)** fits only a 24 GB 4090 or
32 GB Mac; **fp16 N=16 (8.8 GB)** and **int4 N=3 (11.8 GB)** fit a 12 GB GPU or 16 GB laptop.

**This is the slide.** At 0.5B, splitting is theatre — the whole model is 1.98 GB fp32 and v0 *proves* it fits
everywhere by loading it whole on all three nodes. At 70B fp16 it fits no consumer node below N=8, and
**N=16 (8.8 GB/shard) is the first configuration where a room of ordinary laptops can hold a frontier-class
model at full precision at all.** Not an optimisation of local inference — the enabling condition for it.

Caveats to state unprompted: (a) 0.5B's `tie_word_embeddings: true` duplicates the 544.5 MB embedding matrix
across node0 and node2, so v0's fleet footprint is **127.6% of the model** (T1-A1) — Llama-3 does not tie, so
this tax vanishes at 70B; (b) layer balance gets *easier* at 80 layers — lm_head is 1.05B of 70.6B = 1.5%,
versus **27.6% at 0.5B**, which is why v0's split is 1.55x imbalanced today (FINDING 1).

### Q6 — "Why not just quantise a small model and run it locally?"

**For most people, you should — and we should say so.** Qwen2.5-7B int4 (~4 GB) on one laptop beats a
16-machine 70B pipeline on latency, ops complexity and cost for almost every task. Our recommendation to a
user with one machine and a general task is *llama.cpp, one box, go away.* Three cases where it is not the
answer — and they are the entire addressable market:

| Case | Why quantising doesn't solve it |
|---|---|
| **Capability floor** | Quantisation trades bits, not parameters. int4 of a 7B is still a 7B; 70B int4 (35.3 GB) ≫ 7B fp16 (14 GB). You cannot quantise your way to 70B knowledge. |
| **The 4-bit floor is real** | Below 4 bits quality collapses — measured activation analogue: int3-g128 = **+91.1% ppl, 57% top-1**; int4-tok = **+119% ppl** (T2-A4). No 2-bit escape hatch. |
| **Sovereignty** | A local 7B is useless if policy requires 70B-class output *and* forbids egress. Options: a private fleet, or a compliance exception. |

### Q7 — "How do you stop a malicious node returning garbage hidden states?"

**Today: nothing. Zero verification.** Here is the ladder, priced, with the ZK answer given honestly.

| Scheme | Detects | Overhead | Verdict |
|---|---|---|---|
| **Sanity bounds** (‖h‖, NaN/Inf, cosine vs EWMA of prior steps) | crude corruption only; a smart attacker stays in-distribution | ~0 (one norm per hop) | **v1**, 10 LOC, catches the demo case |
| **Commit-reveal** — each node BLAKE3-hashes its output and publishes the 32 B digest *before* seeing downstream state | equivocation / after-the-fact rewriting; makes cheating attributable | 32 B/hop + ~1 µs on a 3.5 KB payload | **v1**, ~20 LOC |
| **Spot-check** — coordinator re-executes one shard on a random p fraction of steps | *any* deviation, probabilistically | at p=5%, one of three shards ⇒ **+1.7% compute** | **v1**, ~30 LOC |
| **Redundant execution k=2** | non-colluding malice, deterministically | **+100% compute**; and see the nondeterminism trap below | v2 |
| **ZK proof of inference** | everything, cryptographically | see below | **v3+ / not now** |

Catch probability `1−(1−p)^G` against a node that always cheats (derived) — G=32 / G=512:
p=1% → **27.5% / 99.42%**; p=2% → **47.6% / 99.997%**; **p=5% → 80.6% / ~100%**; p=10% → 96.6% / ~100%.

**5% spot-checking catches a garbage node inside a single 32-token demo completion 4 times out of 5, for 1.7%
compute.** That is the honest v1 answer and it is enough for a hackathon.

**The redundancy trap, stated before the judge finds it:** re-execution across heterogeneous hardware does not
bit-match — different BLAS kernels, thread counts and reduction orders differ in the last bits, so naive hash
comparison **false-positives on honest nodes**. Exactly the problem *DiFR* (arXiv:2511.20621) exists for.
v1 fix: compare under tolerance (rel-L2 < 1e-3, cos > 0.9999), calibrated against T2-A4's measured honest-noise
floor — bf16 round-trip is rel-L2 **0.0021** at injection, so the honest band is known, not guessed.

**ZKML, honestly:** 4–5 orders of magnitude too slow; nobody should pretend otherwise.

| System | Model | Proving cost | vs our ~100 ms/token v1 target |
|---|---|---|---|
| Chen et al. 2025 (parallel proof accumulation) | LLaMA-2-7B | **2,646 s/token** | **26,460x** |
| zkLLM (Sun et al., arXiv:2404.16109) | LLaMA-2-13B | 986 s commit + 803 s prove **per forward pass** | ~18 days for a 2,000-token generation |
| NANOZK (arXiv:2603.18046), best-case 2026 | GPT-2-Small, 12 layers | ~6.2 s/block; **8.6 min end-to-end**, 3.2 min on 12 workers | extrapolated to 80 blocks ≈ **496 s/token** |

Circuit expansion for an N-parameter model runs 10³–10⁴x. Verification is cheap (<25 ms); *proving* is the
wall. The credible near-term path is not ZK but **TEE attestation** — SGX/SEV-SNP/H100 CC, ~5–10% overhead —
or optimistic TEE-rollup hybrids (arXiv:2512.20176). Say "ZK is the right answer in 2030; TEEs are the right
answer in 2026; spot-checks are the right answer this weekend."

### Q8 — "What is the business model?"

**Start by killing the wrong one, with arithmetic.** The obvious pitch — "a marketplace of idle consumer
hardware selling tokens cheaper than the cloud" — is underwater at retail API prices:

| Line | Value | Basis |
|---|---|---|
| 16-node fleet, 70B int4, 2.2 GB/shard @ ~50 GB/s | 44 ms/shard ⇒ **~22.7 tok/s aggregate** with 16 streams in flight | modelled |
| Daily output | 22.7 × 86,400 = **1.96 M tokens/day** | derived |
| Revenue-equivalent @ Together Llama-3.3-70B **$0.88/1M** (Jul 2026) | **$1.73/day** | derived |
| Electricity: 16 machines × ~100 W incremental × 24 h = 38.4 kWh @ $0.15 | **$5.76/day** | modelled |
| **Result** | **3.3x underwater on power alone.** Hardware, bandwidth, ops are extra. | — |

**Therefore: never compete on price per token.** The three models that survive the arithmetic:

| # | Model | Buyer | Why price doesn't apply |
|---|---|---|---|
| **1 (primary)** | **Sovereign inference appliance** — "your 40 idle laptops run a 70B overnight; no packet leaves the building" | regulated enterprise: health, legal, defence, EU data-residency | the alternative is *not running the model*, not running it cheaper. Compliance budget, not compute budget |
| 2 | **Open core + control plane** — fabric Apache-2.0; sell registry, attestation, audit log, SLA | platform teams | Petals' unmonetised path; the control plane is where trust lives (Q4, Q7) |
| 3 | **Edge / air-gapped** — ships, mines, forward bases, factory floors | industrial / defence | no API exists to be cheaper than |

Model 1 is the pitch. The unit of value is **egress avoided**, not FLOPs sold, and it is measurable: bytes of
prompt that left the perimeter = 0.

---

## 3. Pre-emptive concession slide (put these on stage before the judge finds them)

| # | Concession | The fix, and its size |
|---|---|---|
| 1 | v0's headline claim is false: every node loads the whole checkpoint (`node.py:36`), peak RSS ~3.95 GB | pre-sliced safetensors, ~40 LOC, **v1** |
| 2 | "No node holds the full model" is 51.7%, not 33% — tied embeddings duplicate 544.5 MB (VERIFIED FINDING 1) | untied at 70B; state the real number |
| 3 | The 8/8/8 split is **1.55x imbalanced**; node2 carries 17.13 layer-equivalents of 33.13 | one env-var change to `0-10 / 11-21 / 22-23` |
| 4 | Split inference is not encryption; ~35% token recovery at an 8-layer split | argmax on node2 + mTLS (v1); client-side embed (v2) |
| 5 | We are not cheaper than a hosted API and never will be | compete on egress, not on $/token |
| 6 | The coordinator is a SPOF as total as any node | stateless coordinator + etcd 3.6.6, **v2** |

---

## 4. Sources

External: [exo-explore](https://github.com/exo-explore/) · [Petals releases](https://github.com/bigscience-workshop/petals/releases) · [Petals v2.0.0](https://github.com/bigscience-workshop/petals/releases/tag/v2.0.0.post1) · [zkLLM arXiv:2404.16109](https://arxiv.org/pdf/2404.16109) · [NANOZK arXiv:2603.18046](https://arxiv.org/pdf/2603.18046) · [DiFR arXiv:2511.20621](https://arxiv.org/pdf/2511.20621) · [TEE-rollups arXiv:2512.20176](https://arxiv.org/pdf/2512.20176) · [Together 70B pricing](https://www.together.ai/models/llama-3-1-70b) · [ZKML EuroSys '24](https://dl.acm.org/doi/10.1145/3627703.3650088). Also Petals arXiv:2209.01188, vec2text arXiv:2310.06816, LM inversion arXiv:2311.13647, split-inference arXiv:2602.16760, GPipe (NeurIPS 2019).

Internal: `01-VERIFIED-FACTS.md` (FINDINGS 1–4) · `T1-A1` (transport 9–16%, 127.6% footprint) · `T1-A5` (§5 failover ladder, §7 security, ~63 LOC) · `T2-A4` (measured codec quality) · `T3-A3` (batch-32 ≈ batch-1 on CPU) · `T3-A5` (Petals shortest-path routing) · `T4-A1` (vLLM v0.27.0).
