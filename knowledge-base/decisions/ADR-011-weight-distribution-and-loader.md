---
id: ADR-011
title: Weight distribution — pre-sliced shards, and the tied embedding matrix
status: v1 accepted
date: 2026-09-01
sources: teams/T1-A5, T3-A5, T5-A1, T5-A2, T5-A5, T3-A1, T4-A4, 01-VERIFIED-FACTS F1
---

# ADR-011 — Every node currently loads the whole model. Fix that first.

## Context

This is the **highest-risk item in the project**, and it is not a performance issue.

`layer-nodes/node.py:36` calls `AutoModelForCausalLM.from_pretrained(MODEL_NAME)` on **every node** and
materialises the complete checkpoint before slicing. Worse, it then constructs `Qwen2ForCausalLM(config)` — a
fresh, randomly-initialised, full 24-layer model — so both are resident simultaneously:

> **Peak RSS ≈ 2 × 1,975.8 MB = 3,951.7 MB per container.**

That is what the commented-out `memory: 4G` in `docker-compose.yml` is quietly telling you, and why every
memory limit in the file is disabled. The pitch's headline claim — "no node holds the full model" — is
**disproved by 40 lines of source that a judge can read in thirty seconds.**

Second, `tie_word_embeddings: true` means node0's `embed_tokens` and node2's `lm_head` are the **same
136,134,656-param matrix** — 544.5 MB fp32 genuinely duplicated across two nodes. Fleet footprint is
**2,520.6 MB = 127.6% of the model**, i.e. 27.6% replication overhead, and `embed_tokens` alone is 27.56% of
all parameters while contributing **zero FLOPs**.

| shard footprint, fp32 | node0 | node1 | node2 | fleet |
|---|---:|---:|---:|---:|
| 8/8/8 | 1,021.7 MB = 51.7% | 477.2 MB = 24.1% | 1,021.7 MB = 51.7% | 2,520.6 MB = 127.6% |

## Options considered

| option | verdict | why |
|---|---|---|
| Status quo (`from_pretrained` everywhere, slice, `del full`) | **rejected** | Makes the product name and the core claim false, costs 3.95 GB peak, and re-parses 1.98 GB per node at every cold start. |
| Mutate `full` in place instead of allocating a second model (~5-line diff: `full.model.layers = full.model.layers[a:b]`; `full.model.embed_tokens = torch.nn.Identity()`; `model = full`) | **ACCEPTED v1, do it today** | Peak RSS **3,951.7 → 1,975.8 MB**. Unblocks the commented-out `memory:` limits and is the hard gate on any multi-pipeline demo (P=2 is 23.7 GB today, 11.9 GB after this). |
| **Pre-slice offline into `shard{0,1,2}.safetensors`** via `safetensors.torch.save_file`; ship only the shard; load with `load_file` into a `config` whose `num_hidden_layers` is that shard's count | **ACCEPTED v1, ~40 LOC** | Peak RSS → **~1.1 GB per node**. Makes shard isolation an **enforced property rather than a runtime convention** — this is what turns the pitch's headline from aspiration into fact. |
| `safetensors.safe_open` + per-tensor `get_slice`, materialising only owned blocks | **v2 proposed** | Peak RSS → the shard itself, 663.8–1,200.6 MB. Makes ADR-007's DP `mem_i` constraint mean something for the first time, and makes P=2 fit in 5.04 GB. |
| **Row-shard the tied matrix 3 ways** | **v1 accepted** | Max single-node parameter share **51.71% → 33.33%**, any-2-of-3 **75.85% → 66.67%** (ADR-010). |
| **Move `embed_tokens` and `lm_head` off the pipeline onto the client/coordinator** (Petals' own design) | **v1 preferred where the client can hold 544.5 MB** | Strictly better than row-sharding: all three nodes drop to 119,275,520 params = **24.1% each, equal**; removes the 545 MB duplication entirely; kills the plaintext hop-0 leak (client sends `h`, not `input_ids`); and **8/8/8 becomes compute-balanced again as a side effect**, since `lm_head`'s 9.13 layer-equivalents leave node2. |
| Claim "no node holds a duplicated weight" | **rejected — checkable and false** | 544.5 MB = 21.6% of the deployed footprint is genuinely duplicated. State the number. |

## Decision

1. **Fix the double-allocation loader before anything else.** ~5-line diff, halves peak RSS, gates every
   memory claim in ADR-001, ADR-007 and ADR-009.
2. **Ship pre-sliced `shard{n}.safetensors`.** Until this lands, do **not** use a product name or a slide line
   that asserts shard isolation — T5-A5's recommended title "No Node Knows" is explicitly conditional on it.
3. **Prefer client-side embeddings; row-sharding is the fallback** when the client cannot hold 544.5 MB.
   Note the ordering interaction with ADR-007: compute-balancing *alone* makes parameter spread **worse**
   (node0 → 60.8%), so if only one of the two ships, ship this one.
4. **Mount `HF_HOME` on a named docker volume** shared across nodes, and raise the Docker Desktop VM to
   16 GB / 8 CPU. Cold start ~255 s → ~65 s (**~4x**, modelled); also removes the OOM risk from
   3 × 3.95 GB = 11.9 GB on a default 8 GB VM. ADR-009's reshard depends on this cache being warm.
5. **Set `mem_limit: 6G`, not the commented-out `4G`** — at 3.95 GB peak, 4G OOM-kills at startup. Drop to
   the real limit only after decision 1 lands.
6. Add `OMP_NUM_THREADS=2` / `MKL_NUM_THREADS=2` per node: `deploy.resources.limits.cpus` is a **cgroup
   quota, not a core mask**, so torch reads the host's 10 cores and spawns 10 OpenMP threads that thrash
   inside 2 CPUs of quota.
7. **Do not remove the compose `sleep 45` / `depends_on` chain** until decisions 1 and 4 are in — the hack is
   load-bearing.

## Consequences

**Good.** The product's central claim becomes true; peak RSS 3.95 GB → ~1.1 GB; cold start ~4x faster;
multi-pipeline (P=2) becomes arithmetically possible; and the parameter-share table stops being an attack
surface (ADR-010).

**Bad.**
- **A build step now sits between the HF checkpoint and a running node.** Pre-sliced shards must be
  regenerated whenever the model *or the layer split* changes — and ADR-007 changes the split. Sequence:
  finalise the split, then slice, then fit anything (ADR-003's outlier indices have the same dependency).
- A shard file is a new artifact to version, distribute and checksum. A silent model/shard mismatch produces
  plausible wrong text, so the model hash belongs in ADR-002's handshake.
- **Client-side embeddings move 544.5 MB to the client** — acceptable for a fat client, not for a browser.
  Row-sharding is the fallback precisely because of this.
- Even after every fix, node0 and node2 legitimately both need the embedding matrix if it stays on the
  pipeline. `tie_word_embeddings` cannot be sharded away, only relocated.
- Nothing here makes the demo visibly faster. It is table stakes for honesty, and it must still be done.

## Status

**v1 accepted.** **v2 proposed:** `safe_open` + `get_slice` shard-only loading; content-addressed shard
distribution; per-shard signature so a node can prove which slice it serves (feeds ADR-010's
attestation-gated admission).
