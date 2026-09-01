---
status: VERIFIED — computed from the real HF config.json, reproducible
script: knowledge-base/bench/verify_constants.py
supersedes: any conflicting number in a teams/ file
---

# VERIFIED FACTS — every number here is (measured/derived), not modelled

Run `python3 knowledge-base/bench/verify_constants.py` to regenerate. Raw output in `bench/verify_constants.out`.
Config pulled from `huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json`.

| key | value |
|---|---|
| hidden_size H | 896 |
| num_hidden_layers L | 24 |
| num_attention_heads | 14 |
| num_key_value_heads | 2 (GQA) |
| head_dim | 64 |
| vocab_size V | 151936 |
| intermediate_size | 4864 |
| max_position_embeddings | 32768 |
| **tie_word_embeddings** | **true** |
| total params | 493,961,216 (see note) |

> **Note (90-AUDIT F09):** 493,961,216 counts weight matrices only. The real checkpoint also carries QKV
> biases and RMSNorm gains: per-layer 14,912,384 (not 14,909,440), total **494,032,768** — which is the
> published 0.49B. Every ratio in this file (9.13 eq, 27.6%, 51.7%) moves by <0.02% and is unaffected;
> do not quote 493,961,216 as "the parameter count" on a slide.

## FINDING 1 — the 8/8/8 layer split is 1.55x imbalanced, and node2 is the bottleneck

`lm_head` is 896 x 151936 = **136,134,656 params = 27.6% of the whole model**, and it is a dense matmul executed
once per generated token. One transformer layer is 14,909,440 params. So:

> **lm_head costs 9.13 transformer layers' worth of compute per token.**

| shard | contents | layer-equivalents | share of pipeline |
|---|---|---|---|
| node0 | embed_tokens (a lookup, ~0 FLOPs) + layers 0-7 | 8.00 | 24.1% |
| node1 | layers 8-15 | 8.00 | 24.1% |
| node2 | layers 16-23 + norm + **lm_head** | **17.13** | **51.7%** |

A pipeline runs at the speed of its slowest stage. Bottleneck = 17.13 eq; perfectly balanced = 11.04 eq.
**The current split is 1.55x slower than a balanced one, for free.**

Balanced cut (this is the fix, and it is a one-line env-var change):
`node0 = layers 0-10 (11.00 eq)`, `node1 = layers 11-21 (11.00 eq)`, `node2 = layers 22-23 + lm_head (11.13 eq)`.

Corollary: `tie_word_embeddings: true` means node0's `embed_tokens` and node2's `lm_head` are the **same 136M-param
matrix**, so 545 MB (fp32) is duplicated across two nodes. Relevant to both the memory budget and to the
"no node holds the whole model" claim — state it honestly.

## FINDING 2 — the biggest payload on the wire is the LOGITS, not the hidden state

`coordinator.py` does `next_id = int(np.argmax(logits))` — the argmax runs on the **coordinator**, so node2 ships the
entire fp32 logit vector back: V x 4 = **607,744 B**, base64-inflated to **810,328 B** (`4*ceil(607744/3)`; 810,325 appears in several teams/ files and is
the unpadded 4/3 truncation — off by 3 B, and it makes the derived factor 202,581x rather than 202,582x),
**per generated token**.

The hidden state is only `seq_len x 3584 B`. Therefore:
- the logit vector is the largest single payload on the wire **until seq_len > 170 tokens**;
- once a KV cache lands (hidden state = 3584 B for one position), **logits are 170x larger than the activation**.

Fix: move `argmax`/sampling into node2 and return a 4-byte token id (plus optional top-k logprobs).
**607,744 B -> 4 B on that hop.** Hours of work. Nobody's compression scheme can beat simply not sending it.

## FINDING 3 — no KV cache means 271x redundant compute

For a prompt of P=32 and a generation of G=512, `coordinator.py` resends the growing sequence every step, so each
node performs `G*P + G(G-1)/2 = 147,200` position-forwards. With a KV cache it is `P + (G-1) = 543`.

> **271x redundant position-forwards per node.**

GQA makes the cache almost free: 2 KV heads x 64 head_dim x 2 (K and V) x 2 B fp16 = **512 B per token per layer**,
i.e. **12 KB per token for the whole 24-layer model**; a 2048-token context is **25.2 MB total, 8.4 MB per 8-layer
shard**. There is no memory reason not to cache.

## FINDING 4 — combined wire reduction, one 512-token generation (P=32, G=512)

| | hidden hops (x2) | return path | total |
|---|---|---|---|
| **v0** fp32 + base64 + no cache + full logits | 1,406.8 MB | 414.9 MB | **1,821.7 MB** |
| **v1** bf16 + binary + KV cache + argmax on node2 | 1.946 MB | 0.002 MB | **1.948 MB** |

> **935x fewer bytes on the wire.**

Honest caveat, must accompany the number on any slide: this is a **wire-bytes** reduction, not an end-to-end latency
reduction. On a fast LAN the v0 pipeline is compute-bound, so the wall-clock win comes mostly from FINDING 3's 271x
recompute elimination and FINDING 1's 1.55x rebalance, not from the bytes. Bytes dominate on WAN / 1 GbE / long context.

## FINDING 5 — v0 does not chain. It is a star, and the repo's own README says otherwise.

`layer-nodes/node.py` contains **no outbound HTTP client of any kind** — no `httpx`, no `requests`. Nodes
therefore never talk to each other. `coordinator.py` issues all three POSTs itself, so every hop round-trips
through the coordinator:

```
1. coord -> node0   input_ids          4. node1 -> coord   hidden
2. node0 -> coord   hidden             5. coord -> node2   hidden
3. coord -> node1   hidden             6. node2 -> coord   logits
```

> **3 POSTs = 6 wire crossings per token, and the activation crosses the wire 4 times, not 2.**

Consequences:
- The README's `node0 -> node1 -> node2` flow diagram describes a system the code does not implement.
- FINDING 4's 935x is therefore **conservative**. Counting v0's real 4 activation crossings puts v0 at
  ~2,813.7 MB and the true reduction at **~1,657x**. We keep quoting **935x** deliberately: understating
  our own baseline is the honest direction to err. Say so whenever the derivation is shown, so a judge who
  counts the POSTs does not think it is a mistake.
- **Chain routing is a real v1 change, not a given.** It halves activation crossings (4 -> 2) and takes the
  per-token crossing count from 6 to 4. It belongs in the DLP protocol work (ADR-002), not assumed away.

Verify in two commands:

```bash
grep -c "httpx\|requests" layer-nodes/node.py        # 0
grep -n "client.post" layer-nodes/coordinator.py      # 3 POSTs, all from the coordinator
```
