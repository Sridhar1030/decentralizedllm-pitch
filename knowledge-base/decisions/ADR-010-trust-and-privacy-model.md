---
id: ADR-010
title: Trust and privacy model for hidden states on the wire
status: v1 accepted (governance claim + mTLS + delete the logits oracle); TEE v2 proposed
date: 2026-09-01
sources: teams/T5-A1, T5-A2, T1-A5, T4-A3, T2-A4
---

# ADR-010 — What sharding does and does not protect

## Context

"Decentralized" invites the privacy question, and the literature does not favour us. The honest position must
be stated by us, first, on our own slide.

| result | source | number |
|---|---|---|
| Text recovered from a sentence embedding | Morris et al., EMNLP 2023, arXiv:2310.06816 (vec2text) | **92% of 32-token inputs recovered exactly**, BLEU 97.3 |
| Prompt recovered from next-token probabilities **alone** | Morris et al., ICLR 2024, arXiv:2311.13647 | Llama-2-7b: **27% of prompts exact**, BLEU 59, token-F1 78 |
| Tokens recovered from a split-inference activation, by depth | Cunningham 2026, arXiv:2602.16760 | ~59% at a 2-layer split, **~35% at an 8-layer split** — ours is an 8-layer split |
| Prompts recovered from internal activations in provably linear time | SIPIT, ICLR 2026, arXiv:2510.15511 | decoder-only transformers are almost-surely injective |
| Embedding-projection matrix extracted black-box from top-k logprobs | Carlini et al., ICML 2024 best paper, arXiv:2403.06634 | full matrix of OpenAI `ada`/`babbage` for **under $20** |
| Petals' own concession | Borzunov et al., ACL 2023, arXiv:2209.01188 | "peers serving the first layers can use their inputs to recover input tokens" |

Who sees what in *our* topology: the **coordinator tokenizes the prompt itself and receives the full logits —
total leakage**; **node0 receives `{"input_ids": [...]}`, raw token ids — hop 0 is plaintext**; node1 sees `h`
after 8 layers; node2 sees `h` after 16 layers, emits logits, and holds the tied embedding matrix.

And the parameter arithmetic, which is checkable from `config.json` in thirty seconds:

| | node0 | node1 | node2 | any 2-of-3 |
|---|---:|---:|---:|---:|
| v0 (8/8/8, `tie_word_embeddings: true`) | 255,410,176 = **51.71%** | 119,275,520 = 24.15% | 255,410,176 = **51.71%** | 374,685,696 = **75.85%** |
| after row-sharding the tied 136,134,656-param matrix 3 ways | 164,653,739 = **33.33%** | 33.33% | 33.33% | **66.67%** |

## Options considered

| option | verdict | why |
|---|---|---|
| Claim **input/prompt privacy** | **rejected — would be dishonest** | Hidden states are exactly invertible in provably linear time, and node0 gets plaintext token ids anyway. Claiming it and being caught by a judge who knows the inversion literature costs the whole pitch. |
| Claim **model-weight confidentiality as the lead** | **rejected as the opener** | The first question is "what if two nodes collude", the honest answer is "75.85% plus exact activations for the rest", and answering that in the first 30 seconds frames the pitch as broken. Pre-empt it on our own slide instead. |
| **Claim governance + a smaller blast radius; lead the pitch with data sovereignty** | **ACCEPTED v1** | Sharding converts "one admin runs `scp`" into "three organisations conspire, and leave evidence". Sovereignty needs **no security claim at all** — named laws, named dates — and sits on a real budget line (Gartner, 9 Feb 2026: European sovereign cloud IaaS **$12.6B in 2026**, up from $6.9B in 2025; worldwide $80B, +35.6% YoY). |
| Secure MPC for interactive serving | **rejected, cite the number** | BumbleBee: **>13 minutes per token** on LLaMA-7B with an 8-token prompt. BOLT: 3.18 s/token on a 30 Gbps / 0.8 ms LAN. Three to four orders of magnitude too slow. Put it on the slide as rejected rather than leaving it open. |
| ZK proofs of correct inference | **rejected, cite the number** | 2,646 s/token proving for LLaMA-2-7B vs a ~100 ms/token target = **26,460x too slow**; zkLLM needs 986 s commit + 803 s prove per forward pass on LLaMA-2-13B (~18 days for a 2,000-token generation). |
| **GPU TEE (H100 Confidential Computing + Intel TDX) with remote attestation** | **v2 proposed — the real answer** | The only mechanism that properly closes the malicious-node and colluding-coalition adversaries. **2–8% throughput cost** (arXiv:2409.03992; NVIDIA quotes 2–5%), and the overhead is I/O-bound so it *shrinks* with model size and sequence length. |
| TOPLOC LSH commitments over intermediate activations | **v2 proposed** | arXiv:2501.16007: up to **100x faster to validate than to generate**, ~1000x smaller proofs. Name it on a slide now, implement in v2 — it answers "how do you know node1 didn't cheat?" with a citable scheme designed for exactly this shape. |
| Verde-style optimistic verification (Gensyn), TEE attestation (Targon/SN4) | v2 proposed | Safe with ≥1 honest party; cheaper than verifying every hop. |
| Differential privacy on activations | **rejected** | FSHA (arXiv:2201.04018) defeats client-side DP via feature-space hijacking. |

## Decision

1. **Lead with data sovereignty, land model-weight confidentiality second, and state the collusion limit
   ourselves before anyone asks.** The claim is **governance, not cryptography.**
2. **Move argmax/sampling onto node2; return a 4-byte token id.** 607,744 B → 4 B (**151,936x**). This is
   simultaneously the largest bandwidth win in the system (ADR-002) and the deletion of the strongest known
   prompt-inversion oracle (arXiv:2311.13647). ~2 lines. **Non-negotiable on security grounds alone.**
3. **Row-shard the tied `embed_tokens`/`lm_head` matrix**, or move it off the pipeline entirely (ADR-011).
   Max single-node share 51.71% → 33.33%; any-2-of-3 75.85% → 66.67%.
4. **mTLS between nodes** via uvicorn's native flags (`--ssl-certfile --ssl-keyfile --ssl-ca-certs
   --ssl-cert-reqs 2`), certs from a ~15-line openssl script with CN = node id. **Sequence it AFTER ADR-002's
   connection pooling** — before, it costs ~0.2 s per completion (2 ms × 3 hops × 32 tokens) and reads as
   "security costs performance"; after, it is 3 handshakes total, then unmeasurable.
5. **Deepen the cut as the cheap honest mitigation** — run layers 0..k on the client for small k. Raises
   inversion cost materially versus layer-1 activations. Not a proof; the only v1-affordable answer.
6. **Never say "no node holds the full model" without the asterisk.** Today every node calls
   `AutoModelForCausalLM.from_pretrained()` and materialises the entire checkpoint (ADR-011). Until that is
   fixed, shard isolation is a **runtime convention, not an enforced property**.

## Consequences

**Good.** The pitch survives a hostile question because we asked it first; the argmax fix is a bandwidth win,
a latency win and a security fix in one diff; sovereignty is the only driver that needs no unproven claim.

**Bad.**
- **Do not claim HIPAA forbids calling a US API.** OpenAI, Anthropic, AWS Bedrock, Azure OpenAI and Vertex all
  sign BAAs at enterprise tier in 2026. A healthcare-literate judge catches this instantly.
- **Do not say the EU AI Act high-risk deadline is 2 Aug 2026.** The Digital Omnibus deferred Annex III to
  **2 Dec 2027** (Council approval 29 Jun 2026). GPAI obligations have been in force since 2 Aug 2025.
- **Do not invent a TAM** for "decentralised LLM inference" — the category is not a measured segment. Use the
  Gartner sovereign-cloud line and say out loud that no TAM exists.
- The ~35% token-recovery figure at our 8-layer boundary is an **analogy** from a different model and split.
  Order-of-magnitude only; do not put a precise percentage on a slide without running the attack ourselves.
- Saying "51.7% per node, not 33%" is the honest move but hands a judge an attack line unless immediately
  paired with the tied-embedding explanation and the fix.
- Sharding does not survive a 2-of-3 coalition, and no v1 rung changes that. TEE is the production answer.
- The cost driver is weak and invites "rent an H100 spot instance": it is a sunk-cost utilisation argument,
  not a throughput one, and our pipeline is ~6.6x slower single-stream. One appendix line, no more.

## Status

**v1 accepted.** **v2 proposed:** GPU TEE + attestation-gated shard admission, TOPLOC commitments, Verde
optimistic verification, SPIFFE/SPIRE SVIDs (or a WireGuard mesh at ~0.03 ms as the zero-app-change
alternative), client-side embedding (moves 544.5 MB to the client and fixes the plaintext hop-0 leak), and
periodic shard re-permutation across a pool larger than 3.
