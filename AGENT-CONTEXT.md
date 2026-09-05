# DecentralizedLLM: agent briefing

Paste this whole file to any agent picking up work on this project. It is deliberately short.
Everything here is checked. Do not restate a number from memory; take it from this file.

## What this is

One LLM split by layer across several devices a small team already owns, so that no single device holds
the whole model. Phone, laptop, desktop with a GPU. Each device holds a contiguous range of layers, the
partial result is passed to the next over LAN or VPN, the last device produces the token.

**Status: designed, not built.** There is a research PoC (three Docker containers, CPU) but the target
design has never been run as an integrated system. Write in the planning tense. Do not say or imply that
anything is implemented.

## Where things are

| | |
|---|---|
| Planning repo | `~/CODING/DecentralizeLLMs`, branch `planning`, public at `github.com/Sridhar1030/decentralizedllm-pitch` |
| Research PoC (separate) | `~/CODING/DecentralizedLLM` |
| Landing page | `decentralizedllm-pitch.vercel.app` (redeploy: `vercel --prod --yes`) |
| Deck | `knowledge-base/assets/DecentralizedLLM-deck.pptx`, rebuild with `build_deck.py` |
| Regenerate the facts | `python3 knowledge-base/bench/verify_constants.py` |

## Model constants (Qwen2.5-0.5B-Instruct, from its config.json)

`hidden_size 896` · `24 layers` · `14 attn heads` · `2 KV heads (GQA)` · `head_dim 64` ·
`vocab 151,936` · `tie_word_embeddings true` · `494,032,768 params`

## The four design decisions, and the cost that forced each

1. **Placement is cost aware, not layer counting.** `lm_head` is 136M params = **9.13 layers' worth of
   compute per token**. An even 24-layer split across three devices really runs 8/8/17. Cost-aware gives
   11/11/11. Worth **1.539x** on layer-equivalents, **1.30x** measured wall clock.
2. **Sampling runs on the last device.** The logit vector is **607,744 B**. Sampling where it is produced
   sends back **4 B**, a factor of **151,936x**.
3. **Each device caches its own K and V.** Without it a 512-token reply does **147,200** position forwards
   per device instead of **543** (271x). GQA keeps the cache at **512 B per token per layer**, 12 KB for
   the whole model, so there is no memory reason to skip it.
4. **Devices hand off directly.** Relaying every hop through a coordinator costs **6 wire crossings per
   token**, 4 of them carrying activations, against 4 and 2.

## Do not build these. They were measured and rejected.

- **int8 activations.** Per-tensor destroys the model (cosine 0.039, perplexity 411,041 against 18.6).
  Per-token flips 7-11% of tokens. Cause: **channel 62 carries 972x the median magnitude**. Stop at bf16,
  which is free (KL 5.7e-5, greedy output unchanged).
- **Byte codecs on activations.** LZ4 achieves ratio **1.0042**. Entropy coding never pays on the decode
  path at any LAN speed.
- **Low-rank projection.** Rank 224 of 896 is exact, but the projection costs **27.31 µs** to save **11 µs**
  of 1 GbE time. Only pays below **394 Mbit/s**.
- **RDMA, custom binary protocol, automatic DP placement.** All deferred. Persistent connections capture
  most of the transport win.

Metric trap worth knowing: rank 16 keeps **99.95% of activation energy** and yields **12.5%** top-1
agreement. Gate quality on end-to-end top-1 or KL, never on a distance between hidden states.

## Two facts that contradict the headline claim

Both are in the PoC and must be fixed before "no device holds the whole model" is literally true. Say them
before a judge finds them.

1. `layer-nodes/node.py` calls `from_pretrained(MODEL_NAME)`, so **every node loads the entire checkpoint**
   then discards the layers it does not own. Fix is pre-sliced safetensors, ~40 LOC (ADR-011). Until it
   lands, do not use a product name that asserts the property.
2. `node.py` has **no outbound HTTP client at all**. Nodes never talk to each other; the coordinator relays
   every hop. The PoC's own README diagram describes a system its code does not implement.

## Rules

- **`knowledge-base/90-AUDIT.md` gates every number.** It lists 23 findings against our own work and a
  "must not go on a slide" list. Check it before quoting anything.
- **Never quote these**, all retired by the audit: `28.6x`, `163 Mbit/s`, `202,581x`, `1.55x` for anything
  wall clock, `6.8x x 2.8x`. Quote per-token at a stated `seq_len`, never a whole-generation speedup.
- **Never invent credentials.** `40-PITCH.md` has 24 `<FILL: ...>` placeholders for hackathon results,
  employers and repos. Leave them.
- **Tag every number** `(measured)`, `(derived)` or `(modelled)`. No `(measured)` tag on anything the
  design has not actually run.
- **Prose style:** simple and conversational, no em dashes.
- **Never claim speed.** Distributed decode is slower than one device; llama.cpp's RPC path measures
  1.74 to 1.88x slower even over 10 GbE. The argument is capacity and fit, never cost or latency.

## Current focus

A 30-hour hackathon build, with **phones as nodes**. Priority order: pre-sliced shards, sampling on the
tail node, KV cache, balanced split. Phone-specific ideas worth building: token relay lit up across
physical screens, airplane mode as the kill switch, battery and thermal aware placement (the one genuinely
novel piece, extends ADR-007), and a hotspot-only cluster with no router.

## Deeper reading, in order

`01-VERIFIED-FACTS.md` → `10-ARCHITECTURE.md` → `decisions/` (13 ADRs) → `90-AUDIT.md` → `teams/` (25 reports)
