---
team: T4 — Infrastructure & Serving Runtimes
agent: T4-A3
topic: Prior art in decentralized / untrusted-node LLM inference — Petals, hivemind, exo, split learning, Prime Intellect, Bittensor, and the DePIN tier; plus the honest differentiation answer
headline: "Layer-sharded inference across machines is a solved, eight-year-old idea (SplitNN 2018 → Petals 2022 → exo 2024) and we must never claim it — the three things nobody has shipped together are (a) a defense against activation-inversion on the shard boundary, which Petals' own paper concedes it does not have, (b) continuous batching across untrusted shards, and (c) a sub-millisecond commodity-Ethernet fast path, since Petals is tuned for 100 ms WAN (0.83 tok/s measured, geo-distributed) and exo 1.0's fast path is Thunderbolt-5 RDMA on Apple silicon only."
---

# T4-A3 — Decentralized LLM inference: the competitive landscape

Status verified 2026-09-01 via repo/paper fetch. Numbers tagged **(measured)** = from the cited paper's own
experiments, **(reported)** = vendor/press claim I did not verify, **(derived)** = my arithmetic from
`01-VERIFIED-FACTS.md`. No invented benchmarks.

## 1. The taxonomy — three axes that sort every system below

Most "decentralized AI" projects are not competitors. They differ on three orthogonal axes:

| Axis | Options | Where we sit |
|---|---|---|
| **What is split** | (a) nothing — whole model per node, load-balanced; (b) **layers across nodes (PP)**; (c) tensors across nodes (TP) | **(b)** |
| **What is untrusted** | (a) nothing — one owner, one datacenter; (b) the network; (c) **the node operator** | **(c)** aspirationally |
| **Workload** | training / RL post-training / **inference** | **inference** |

> Axis 1(a) covers Akash, io.net, Nosana, Render, Golem, Kuzco, Hyperbolic, most Bittensor subnets, and every
> "decentralized GPU cloud". **They rent you a whole GPU. They do not split a model.** They are a supply-side
> marketplace, not an inference architecture. Judges conflate them with us; the answer is one line: *"those are
> Airbnb for GPUs; we are the thing you'd run on top of them."*

## 2. Master comparison

| System | Split | Topology / routing | Trust model | Incentive | Status (2026-09) | What we do differently |
|---|---|---|---|---|---|---|
| **Petals** (BigScience/Yandex/HSE) | layers (PP), blocks over WAN | Kademlia DHT (hivemind); client picks server chain by **D\* Lite** shortest-path on measured per-block latency; servers greedily claim the lowest-throughput block spans | Honest-but-curious *assumed*, not enforced. Paper concedes: *"peers serving the first layers of the model can use their inputs to recover input tokens."* Mitigation offered = "run a private swarm" | Proposed only — "points" spendable on priority. **Never implemented** | Repo live, not archived, 10.5k★, 522 commits. Public swarm long past its BLOOM-era peak | LAN + sub-ms, small model, batching across shards, an actual activation-privacy mechanism |
| **hivemind** (`learning-at-home`) | n/a — it's the substrate | Kademlia DHT, libp2p, averaging primitives, DMoE | none (library) | none | MIT, active, 229★ on the core repo | We use ~0 of it at N=3; a DHT for three docker containers is malpractice (see §6) |
| **exo 1.0** (exo labs) | layers, **ring memory-weighted partitioning** — each device gets layers ∝ its RAM | UDP/mDNS auto-discovery, **p2p, no master node**, OpenAI/Claude/Ollama-compatible API on :52415 | Fully trusted LAN. Remote-code opt-in for custom HF models is the only control | none | **The closest competitor.** 47.2k★, Apache-2.0. v1.0 ships day-0 **RDMA over Thunderbolt 5** (macOS 26.2+), *"99% reduction in latency between devices"* (reported). Old python/tinygrad repo archived as `ex-exo` 2025-12-17 | exo's fast path is Apple-silicon + TB5-only; ours is commodity wired Ethernet on heterogeneous x86/ARM **CPU**. exo trusts every node; we don't. exo does not batch |
| **Distributed Llama** (b4rtaz) | **tensors** (TP), root + 2^n−1 workers | static config, Ethernet sync | fully trusted | none | active, C++, ARM + x86 AVX2 | TP needs 2 collectives per block. Per T4-A1, TP moves **12× more bytes** than PP at N=3 (86,016·B vs 7,168·B per decode step, modelled). Wrong parallelism for an untrusted/slow link |
| **Cake** (evilsocket) | transformer blocks across heterogeneous devices | zero-config mDNS clustering or manual topology | fully trusted | none | active, Rust/Candle, CUDA/Metal/Vulkan/CPU, iOS+Android. Self-described experimental | Same gap: trust, batching, queueing |
| **GPUStack** + `llama-box` | layers via llama.cpp RPC; also vLLM backend | central manager schedules workers | single-owner cluster | none | active, production-ish | It's a *cluster manager* for hardware you own. Orthogonal — a plausible v2 control plane, not a rival |
| **llama.cpp RPC** (`rpc-server`) | layers over TCP | manual `--rpc host:port` list | fully trusted, no auth | none | in-tree, explicitly "not secure, do not expose" | Baseline to beat, and the honest CPU comparison point |
| **Prime Intellect** | **training**, not inference | DiLoCo/`prime-rl`; SHARDCAST for weight broadcast; TOPLOC for verifying untrusted *inference* workers in the RL rollout loop | Untrusted workers, verified cryptographically | Announced p2p compute protocol w/ crypto-economic primitives; **no token launched** | INTELLECT-2 = 32B globally-distributed RL (arXiv 2505.07291). **INTELLECT-3 (106B MoE) was trained on a centralized 512-GPU cluster** — the flagship decentralized lab shipped its flagship model centrally | Different problem entirely. **Steal TOPLOC** (§5), don't compete |
| **Together AI** | was PP over heterogeneous WAN (DT-FM, NeurIPS'22, arXiv 2206.01288 — evolutionary tasklet scheduler, **4.8× faster than SOTA on networks up to 100× slower**, 8 cities / 3 continents, measured) | — | — | — | **Abandoned the thesis.** Now a conventional neocloud: $800M Series C Jul-2026, $8.3B valuation, owns its own high-bandwidth GPU clusters | The strongest team that tried decentralized inference at scale bought datacenters instead. Slide-worthy, and we must address it, not hide it |
| **Gensyn** | training | Verde optimistic verification (refereed delegation, safe with ≥1 honest party); OP-Stack L2 | untrusted, economically secured | token + slashing | Mainnet live **2026-04-22**; >5,000 H100-equivalents day one (reported) | Training. Verification design is the transferable part |
| **Bittensor / TAO subnets** | none — whole models, competitive scoring | Yuma consensus, validators score miners | untrusted miners, economically scored | TAO emissions | **Chutes (SN64)**: >9.1T tokens, >400k users, >$100M cumulative inference volume. **Targon (SN4)**: TEE-based verifiable compute, ~$10.4M ARR, $10.5M Series A (all reported) | Whole-model routing marketplace. Real revenue, real users, **zero model splitting**. Not a competitor; a potential distribution channel |
| **Akash / io.net / Nosana / Render / Golem / Spheron / Flux** | none | job marketplace | untrusted host, no compute verification worth the name | tokens | live, real volume | Rent-a-GPU. See §1 note |
| **Ritual** | none | Infernet oracle network (Phase 1 live); Ritual Chain (Phase 2) | on-chain verifiable *results* | token | testnet→prod, io.net partnership | On-chain inference oracle. Different product |
| **Kuzco/Inference.net, Hyperbolic, Bagel** | none | GPU/serverless inference networks; Hyperbolic uses Proof-of-Sampling-style spot checks | untrusted, sampled verification | tokens | **status not re-verified 2026-09 — do not quote on a slide** | — |

## 3. The academic ancestor — split learning / SplitNN, and why it matters more than Petals

Split learning (Gupta & Raskar 2018; Vepakomma et al., "NoPeek") is *literally this architecture*: a network cut at
layer *k*, client runs `[0,k)`, server runs `[k,L)`, only activations cross the boundary. Every claim about
"the server never sees your data" that a decentralized-inference pitch makes was made first here — and was broken.

| Attack | Threat model | Result | Consequence for us |
|---|---|---|---|
| **UnSplit** (Erdoğan et al., WPES'22, eprint 2021/1074) | **Honest-but-curious** server, knows only the client's *architecture* | Recovers input samples **and** steals a functionally equivalent copy of the client model. Client cannot detect it | Our node1/node2 operators are exactly this adversary. Passive. Undetectable. |
| **FSHA — Feature-Space Hijacking** (Pasquini et al., CCS'21) | **Malicious** server, steers the training objective | Hijacks the shared feature space into an invertible one, then inverts | Training-phase; less relevant to pure inference, but kills "just add DP" |
| **FSHA vs DP** (arXiv 2201.04018) | + client-side DP optimizer | **Differential privacy does not stop it** | Do **not** put "we add DP noise" on a slide |
| **SplitOut** (Springer 2024) | detection, not prevention | Outlier-detection on client-side gradients catches training-hijacking | A v2 monitoring idea, not a v1 defense |
| **Black-box feature inversion for split DNNs** (arXiv 2511.15316) | data-efficient, black-box | Still inverts | The literature is not trending our way |

> **Honest framing for the deck:** the activation on the wire is *not* privacy. Hidden states from early layers are
> close to invertible; that is a published, replicated result, and Petals' own paper concedes it in one sentence.
> Anyone claiming otherwise is selling. What is defensible is **parameter secrecy** (no operator can walk off with
> the whole model) and **compartmentalisation**, not input secrecy — unless we add a real mechanism (§5).

## 4. Petals in detail — the measured numbers a judge will check

All from the Petals/SWARM follow-up, arXiv 2312.08361 (measured, their hardware):

| Setup | Sequential decode | Parallel forward (batch 1×128) | Parallel forward (64×128) |
|---|---|---|---|
| Llama 2 70B, 3×T4, 1 Gbit/s, 100 ms RTT | **2.29 steps/s** (2.02 @ 2048 ctx) | 45.4 tok/s | 155.1 tok/s |
| BLOOM 176B, 3×A100, 1 Gbit/s | **1.71 steps/s** (1.54 @ 2048) | 70.0 tok/s | 253.6 tok/s |
| **Real geo-swarm, 14 heterogeneous servers, EU + NA** | **0.83 steps/s** | 32.6 tok/s | 179.4 tok/s |
| RAM-offloading baseline, same HW | 0.139 (Llama 2) / 0.0495 (BLOOM) steps/s | — | — |

Derived: 2.29 / 0.139 = **16.5×** over offloading for Llama 2; 1.71 / 0.0495 = **34.5×** for BLOOM. The paper's own
headline claim is the conservative "up to 10×".

Two facts about Petals that we must state before a judge does:

1. **Petals already solved our FINDING 2.** *"clients store the model's token embeddings … locally and rely on
   servers to run Transformer blocks."* The client owns embeddings **and** the LM head, so the swarm never ships a
   `V×4 = 607,744 B` logit vector — it ships a hidden state and the client computes logits locally. Our v0 ships the
   full logit vector every token. **Petals' return path is already 170× better than ours** (derived). Fixing this is
   a catch-up item, not an innovation. Frame it that way.
2. **Petals has no incentive layer.** The paper proposes "points"; nothing shipped. The swarm ran on goodwill and
   thinned out accordingly. Any pitch of ours that hand-waves "operators will join" repeats their exact failure.

## 5. What is worth stealing, tagged

| # | Steal from | Idea | Tag | Why |
|---|---|---|---|---|
| 1 | Petals | Client (or coordinator) holds embeddings + `lm_head`; nodes hold only transformer blocks | **v1** — hours | Kills FINDING 2's 607,744 B → ~4 B return path. Also fixes the `tie_word_embeddings` duplication (§7) |
| 2 | exo | **Memory-weighted partitioning** generalised to **compute-weighted** using `lm_head` = 9.13 layer-equivalents (FINDING 1) | **v1** — one env-var + a cost table | Our 8/8/8 split is 1.55× slower than balanced, for free |
| 3 | exo | mDNS/UDP discovery, **no master**, OpenAI-compatible API on a fixed port | **v1** for discovery, **v2** for masterless | Our coordinator is a single point of failure (defect #9). mDNS in docker-compose is ~40 lines |
| 4 | Petals | **D\* Lite** shortest-path routing over measured per-hop latency, with live re-planning on node loss | **v2** | Only pays off at N≫3 with churn |
| 5 | Prime Intellect | **TOPLOC** (arXiv 2501.16007) — LSH commitment over intermediate activations; up to **100× faster to validate than to generate**, ~1000× smaller proofs, 100% detect rate on their eval (reported) | **v2**, but a **v1 slide** | This is *the* answer to "how do you know node1 didn't cheat?" It is designed exactly for our shape: verifying an untrusted worker's activations |
| 6 | Gensyn | **Verde**-style optimistic verification: assume honest, dispute-narrow on challenge, safe with ≥1 honest party | **v2** | Cheaper than re-executing everything |
| 7 | Bittensor / Targon | **TEE attestation** of the node process | **v2** | The only deployed technique that actually addresses UnSplit-class inversion |
| 8 | Split-learning lit. | Push the cut deeper — client runs layers `0..k` locally for small *k* | **v1** — the cheap, honest privacy story | Inverting layer-8 activations is materially harder than inverting layer-1. Not a proof, but a real, cheap mitigation Petals also suggests |
| 9 | vLLM / Orca (see T4-A1) | continuous batching **across shards** | **v1** | Nothing in this landscape does it. Petals batches within a server, not across the chain. Real differentiation |

## 6. Where the prior art says we are wrong

Be ready for these. Each is a fair hit.

| Objection | Honest answer |
|---|---|
| "Just use exo." | Correct for an Apple-silicon home lab. exo is 47.2k★, Apache-2.0, and better than our v0 today. We differ on hardware (commodity CPU/Ethernet), on trust (exo assumes a friendly LAN), and on throughput (exo does not do continuous batching). If those three don't matter to you, use exo. |
| "A DHT for 3 nodes?" | We should **not** ship a DHT. hivemind's Kademlia is right for a 1000-peer WAN swarm and pure overhead at N=3. mDNS + a static manifest is the lazy correct answer. Say this before someone else does. |
| "Together AI abandoned this." | They abandoned *WAN-decentralized training economics*, having first proven the scheduling works (4.8× on 100×-slower networks, measured). Capital made owning datacenters the better move for them. That is an argument about capital, not about the architecture — and it does not apply to the on-prem / edge / regulated-data case, which is ours. |
| "Isn't the network the bottleneck?" | On WAN, yes — Petals measures 0.83 tok/s. On a LAN it is **compute**-bound, which is why our wins come from FINDING 3 (271× recompute) and FINDING 1 (1.55× rebalance), not from bytes. Do not oversell the 935× wire-bytes number. |
| "Your privacy claim is broken (UnSplit)." | Concede it immediately. Claim **parameter secrecy**, not input secrecy, and point at §5 items 7–8 as the roadmap. |

## 7. The honest parameter-secrecy arithmetic (derived from `01-VERIFIED-FACTS.md`)

`tie_word_embeddings: true`, so node0's `embed_tokens` and node2's `lm_head` are the **same 136,134,656-param matrix**.

| Split | node0 | node1 | node2 | max single node | note |
|---|---|---|---|---|---|
| v0, 8/8/8 | 255,410,176 (**51.7%**) | 119,275,520 (24.1%) | 255,410,176 (**51.7%**) | **51.7%** | node0 ∩ node2 = the tied 136M matrix |
| balanced-compute (0–10 / 11–21 / 22–23+head) | 300,138,496 (**60.8%**) | 164,003,840 (33.2%) | 165,953,536 (33.6%) | **60.8%** | compute-balanced ⇒ *worse* parameter spread |
| **+ steal #1** (embed + `lm_head` both on the client/coordinator; nodes hold blocks only) | 119,275,520 (24.1%) | 119,275,520 (24.1%) | 119,275,520 (24.1%) | **24.1%** | with the head off the pipeline, 8/8/8 is *also* compute-balanced again. Removes the 545 MB fp32 duplication |

> **"No node holds the full model"** is true, but v0's honest number is *"no node holds more than 51.7% of the
> parameters, and the two that hold 51.7% hold the same matrix."* Applying steal #1 makes it **24.1%** and removes the
> duplication. State the real number on the slide; a judge who reads `config.json` will find the tie.

## 8. The differentiation paragraph — "isn't this just Petals?"

> Yes — the same way a database is just a file. Layer-sharded inference across machines is eight years old: split
> learning proposed it in 2018, Petals shipped it over the open internet in 2022, exo brought it to a LAN in 2024.
> **We did not invent the topology and we will not claim to.** Petals solves a specific problem — volunteers on
> residential WAN, 100 ms hops, a 176B model that fits nowhere — and it solves that problem at **0.83 tokens/sec
> geo-distributed** (their measurement, 14 servers, Europe + North America). Every design choice in Petals follows
> from that: a Kademlia DHT, D\* Lite re-routing around churn, block-span rebalancing. We are solving the *inverse*
> problem — a small, fixed set of wired nodes on a sub-millisecond LAN, where the network is not the bottleneck and
> the *operator* is the adversary. In that regime a DHT is pure overhead and the wins move to three places nobody
> in this landscape has put together: **continuous batching across shards** (Petals batches inside one server, never
> across the chain; exo does not batch at all), a **wire protocol built for a 0.05 ms hop rather than a 100 ms one**,
> and an **explicit activation-privacy mechanism**. On that last one we are honest: Petals' own paper concedes that
> *"peers serving the first layers of the model can use their inputs to recover input tokens"*, and the split-learning
> literature has broken every weaker claim — UnSplit reconstructs inputs from an honest-but-curious server that knows
> only the architecture, and differential privacy does not stop feature-space hijacking. So we claim **parameter
> secrecy, not input secrecy**, we push the cut past the shallow layers to raise the inversion cost, and we point at
> TOPLOC-style activation commitments and TEE attestation as the v2 path. Finally, the honest scoreboard: exo has
> 47.2k stars and today it is better than our PoC; Together AI proved the scheduling and then bought datacenters
> instead; Prime Intellect trained its flagship 106B model on a centralized 512-GPU cluster. The field's own results
> say WAN-decentralized inference loses to a datacenter on economics. **The case that survives is on-prem: data that
> is not allowed to leave, hardware you already own, and a model no single box in the building can hold.** That is
> the pitch, and it is not Petals'.
