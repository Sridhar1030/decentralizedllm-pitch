# SHARED CONTEXT — read this first (every agent, every team)

Project: **DecentralizedLLM** — one LLM split across N physical nodes, no node holds the full model.
Hackathon submission. Deliverable = pitch deck (4-5 slides) + animated prototype + architecture docs.

## Where things live
- Existing working PoC code: `/Users/srpillai/CODING/DecentralizedLLM/`
- This knowledge-base (shared, all teams write here): `/Users/srpillai/CODING/DecentralizeLLMs/knowledge-base/`
  - `teams/<TEAM-ID>-<AGENT-ID>-<topic>.md`  ← each agent writes exactly ONE file here
  - `decisions/`  ← ADRs, written by synthesis phase only
  - `bench/`      ← measured numbers, scripts
  - `assets/`     ← diagrams, prototype

## Current PoC (v0) — ground truth, verified by reading source

Model: `Qwen/Qwen2.5-0.5B-Instruct`, 24 transformer layers, hidden_size 896, vocab 151936, fp32.
Split: node0 = embed + layers 0-7 ; node1 = layers 8-15 ; node2 = layers 16-23 + norm + lm_head.
Each node = Docker container, FastAPI, CPU-only (2 CPU / 2 GB).
Path: client -> gateway (:8080, api-key + circuit breaker) -> coordinator -> node0 -> node1 -> node2 -> logits.
Transport: HTTP/1.1 + JSON, hidden states as **base64-encoded fp32 numpy bytes**.
Generation: greedy argmax, coordinator loop, SSE streaming variant exists for live UI.

### v0 measured/derived defects (do NOT re-discover these — build on them)
1. **No KV cache anywhere.** `coordinator.py` resends the ENTIRE token sequence through all 3 nodes for every
   new token. Cost is O(n^2) in sequence length. Biggest single win available.
2. **Full hidden-state tensor shipped each hop**, shape [seq_len, 896] fp32 — grows linearly with seq_len.
   With a KV cache only the LAST position's [1, 896] vector needs to cross the wire.
3. **fp32 on the wire.** 3584 B/token/hop. bf16 halves it; fp8/int8 quarters it.
4. **base64** adds 33% + encode/decode CPU + JSON string parse of a ~MB payload.
5. **`async with httpx.AsyncClient()` is constructed INSIDE every forward call** — new TCP connection,
   new TLS-less handshake, no keepalive, no connection pool. 3 fresh TCP handshakes per generated token.
6. **Zero batching, zero microbatching.** Strict sequential chain: at any instant 2 of 3 nodes are idle.
   Pipeline utilization ceiling = 1/3.
7. **`output_hidden_states=True`** materialises every intermediate hidden state; only the last is used.
8. No admission control, no queue, no backpressure, no priority. Single in-flight request assumed.
9. Node failure = total outage (node1 down => model incomplete). No replicas, no re-sharding.

## What the user asked for (the brief)
- **Wired-connection fast path**: custom logic/protocol so node-to-node communication is faster.
- **Compression algorithm** for transferring embeddings/activations between layers on distributed systems.
- **Queueing system** for optimization + maximizing compute utilization.
- **Infra & tech recommendations**: vLLM and other popular LLM serving runtimes.
- **Animated prototype** for the demo.
- **PPT deck, max 4-5 slides.**

## Rules for every agent
- Be concrete and numeric. "Faster" is worthless; "3584 B -> 896 B per hop, 4x" is useful.
- Show the arithmetic. State assumptions inline.
- Cite real systems/papers/APIs by exact name (vLLM, SGLang, Petals, NCCL, UCX, RoCEv2, Zstd, LZ4,
  Orca continuous batching, GPipe/1F1B, PagedAttention, DeepSpeed-Inference, TensorRT-LLM, llama.cpp RPC).
- Distinguish **v1 (hackathon-demoable, ~days)** from **v2 (production, ~months)**. Tag every recommendation.
- No invented benchmark numbers. If a number is modelled, write `(modelled)`. If measured, write `(measured)`.
- Do not edit files outside your own `teams/` file unless explicitly told to.
- Keep your file under ~250 lines. Dense. Tables over prose.

## House numbers (use these, consistently)
- hidden_size H = 896, layers L = 24, vocab V = 151936, dtype fp32 (4 B)
- activation per token per hop = H * 4 B = 3584 B  (bf16: 1792 B, fp8/int8: 896 B)
- 3 hops per token in a 3-node chain (node0->1, 1->2, plus coordinator->node0 ingress)
- 1 GbE = 125 MB/s, 10 GbE = 1.25 GB/s, 25 GbE = 3.125 GB/s
- typical LAN RTT: 1 GbE ~0.2-0.5 ms, 10 GbE ~0.05-0.1 ms, loopback ~0.02 ms, RDMA ~0.002-0.005 ms

> **Before using any number, read `01-VERIFIED-FACTS.md`.** It contains four verified findings (shard imbalance, the logits return path, the KV-cache recompute factor, wire-byte totals) that supersede any conflicting figure.
