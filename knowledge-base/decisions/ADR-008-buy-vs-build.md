---
id: ADR-008
title: Buy vs build — where vLLM, SGLang and llama.cpp RPC fit, and where they do not
status: v1 accepted (build the fabric, benchmark against the runtimes); buy the engine in v2
date: 2026-09-01
sources: teams/T4-A1, T4-A2, T4-A3, T5-A4, T5-A2
---

# ADR-008 — Buy the engine, build the fabric

## Context

**Say this before a judge finds it: vLLM already ships cross-node layer-split pipeline parallelism**
(`--pipeline-parallel-size` with ray, or `mp --headless` across machines). "One LLM split across N nodes" is
not a new capability. So does llama.cpp (`--rpc`), so does mlx-lm (`--pipeline`), so does Petals, so does exo
(47.2k stars, Apache-2.0, day-0 RDMA over Thunderbolt 5 on macOS 26.2+). SplitNN dates to 2018.

Two external measurements set the honest expectation for what distribution costs:

| system | workload | decode | prefill |
|---|---|---|---|
| llama.cpp `--rpc`, Qwen2.5-7B Q4_K_M, M2 Ultra + DGX Spark, 10 GbE measured at 9.41 Gbps | vs single machine | 91.8 → 52.7 tok/s = **0.574x (1.74x SLOWER)** | 76.1 → 317.7 tok/s = **4.17x faster** |
| same rig, Qwen2.5-72B Q4_K_M | vs single machine | 11.1 → 5.9 tok/s = **0.532x** | 28.2 → 29.5 = 1.05x |
| SGLang chunked PP, DeepSeek-V3.1 | PP1 → PP8 | no gain | **TTFT 55.5 s → 10.5 s = −81.1%** |

Both are third-party/lab-reported. **Pipeline parallelism is a prefill/TTFT/capacity technology, not a decode
throughput technology.** Any slide claiming "decentralized = faster" is falsifiable in five minutes.

The architectural number that follows: PP with N stages costs **(N−1) × RTT/2 per decode step**, so at N=3 the
floor is exactly **one full RTT per token** and the ceiling is `1000/RTT_ms` — independent of model size,
compute, dtype and any compression: 10 GbE 10,000 tok/s · 1 GbE 2,500 · metro WAN 100 · coast-to-coast 16.7 ·
intercontinental 4.0 (modelled).

## Options considered

| option | verdict | why |
|---|---|---|
| **Run vLLM PP=3 as the measured baseline; keep `node.py` as the product** | **ACCEPTED v1** | A custom stack that cannot state its number against vLLM is not credible; one that can — losing on tok/s, winning on "one node killed" — is. Both stacks speak `/v1/chat/completions`, so it is drop-in behind the existing gateway via `VLLM_URL`. |
| Adopt vLLM PP as the product | **rejected for v1** | vLLM PP is a *cluster* feature: **one `torch.distributed` group, no authn, no authz, no attestation, no encryption on the inter-stage channel, static boot-time membership, and collectives that hang the whole server when any rank dies.** Heterogeneous TP-per-stage is an open feature request (#27239). Our premise is not a configuration of vLLM PP; it is a different system. |
| Wrap vLLM per node ("hybrid") | **v2, and scope it honestly** | **There is no supported API.** vLLM exposes no "run only my layers on this activation and return the hidden state" surface; `IntermediateTensors` P2P is internal to the `torch.distributed` group with no HTTP path. This needs a custom `Executor`/`GroupCoordinator` shim or a `NixlConnector`-style plugin — a **fork-or-plugin project, not an integration**. Scoping it as an integration is a multi-month estimate error. |
| Adopt llama.cpp's RPC backend as the substrate | **rejected** | Upstream self-describes it as "in a proof-of-concept development stage… fragile and insecure" and warns against open networks: `rpc-server` executes arbitrary remote compute graphs against local memory with **zero authentication**. Use it as a baseline (3 local `rpc-server`s + single-process `llama-cli` as the ceiling), not as a foundation. |
| Adopt mlx-lm `--pipeline` / exo | **rejected for v1, strongest v2 track on Apple silicon** | mlx-lm already implements our exact dataflow (`mx.distributed.send` / `recv_like` between adjacent ranks) and MLX's JACCL backend gives RDMA over Thunderbolt on macOS 26.2+. Adopting any of them **for the hackathon deletes the demo's premise** — at that point we are demoing someone else's project. |
| Reimplement PagedAttention / continuous batching / prefix caching / a sampler | **rejected** | That is the 18-month part and it is free. Copy the *idea* (Orca iteration-level scheduling, ADR-006), not the code. |
| Cite HuggingFace TGI or NVIDIA Triton as live alternatives | **rejected** | TGI's repo is archived / maintenance-mode as of ~Mar 2026; Triton was folded into Dynamo and is now Dynamo-Triton. Citing either as current is the cheapest way to lose an infra-literate judge. |
| TensorRT-LLM | **rejected** | NVIDIA-only, per-GPU-SKU engine build — directly contradicts the commodity-heterogeneous-nodes thesis. Cite only as the performance ceiling. |

## Decision

1. **Build the inter-node fabric. Buy the per-node execution engine, in v2.** The defensible claim is the
   trust / heterogeneity / churn / fault model, **not** the layer split. Lead the deck with that delta.
2. **Run three baselines before the pitch**, publishing all numbers next to ours: (a) vLLM v0.27.0 CPU
   `--pipeline-parallel-size 3` (`--distributed-executor-backend mp`, `--enforce-eager`,
   `VLLM_CPU_OMP_THREADS_BIND=nobind`, no quantisation — `PPMissingLayer` stubs break quant state setup);
   (b) llama.cpp with 3 local `rpc-server`s (`-DGGML_RPC=ON`), plus single-process `llama-cli` as the ceiling;
   (c) single-node vLLM as the no-network control. Reference target: **within 2x of llama.cpp RPC**.
3. **Set `VLLM_PP_LAYER_PARTITION="11,11,2"` on every baseline run, and run once without it as the control.**
   vLLM reproduces ADR-007's imbalance exactly — naive PP=3 gives 8/8/17.13 layer-equivalents because
   `lm_head` lands on the last rank. The control run proves the imbalance is **generic, not a PoC bug**.
4. **Re-frame the pitch from speed to capacity + TTFT**, corroborated by llama.cpp's 4.17x-prefill /
   0.574x-decode split and SGLang's −81.1% TTFT at PP8.
5. **Steal, do not import:** llama.cpp's wire discipline (`[uint64 LE length][raw bytes]`, one persistent
   socket, shape/dtype sent once per session — its `RPC_CMD_GRAPH_RECOMPUTE` collapses a 205 KiB graph
   description to **4 bytes** on every decode step after the first; `sizeof(rpc_tensor) = 296 B` exactly);
   its FNV-1a content-addressed dedup above a 10 MiB threshold (`SET_TENSOR_HASH`) as a free prefix cache on
   the wire; vLLM V1's `SchedulerOutput` as the model for ADR-006's `BatchDescriptor`; SGLang's RadixAttention
   and dynamic chunking. All of these are patterns, not dependencies.

## Consequences

**Good.** The one comparison number the pitch currently lacks; a defensible novelty claim; a clear line
between what we write (fabric) and what we will never write (engine).

**Bad.**
- **Novelty risk is the highest-severity item in the whole project.** A judge who knows vLLM will ask why this
  is not `vllm serve --pipeline-parallel-size 3`. There is exactly one good answer and it is not performance.
- **Fault tolerance is claimed as our advantage but is NOT implemented** — the PoC also dies when node1 dies.
  The difference is that stateless HTTP request/response makes recovery *addressable* while a standing
  NCCL/Gloo collective does not. That is an architecture argument, not a demo. **Do not demo a failover that
  does not exist** — build ADR-009 first, or drop the claim.
- `vllm-metal` (Apple Silicon MLX plugin) documents **no TP, no PP, no multi-node**. The repo's own
  `sprint.md` already hit this and degraded to 3 proxies in front of one vLLM, which is not model splitting.
  Any Apple-GPU path must be assumed single-node.
- Baseline-validity traps that will silently invalidate the comparison: `--enforce-eager` on both stacks or a
  compile step is being compared; `VLLM_CPU_OMP_THREADS_BIND=nobind` (Docker exposes no NUMA nodes); and a
  **concurrency-1 comparison measures nothing**, because PP=3 idles 2 of 3 stages on *both* stacks.
- bf16 on ARM without the BF16 ISA extension may be emulated and slower than fp32; fp16 is unstable on torch
  CPU and rejected outright on AMD Zen. Benchmark both dtypes or the baseline number is noise.
- Adopting llama.cpp RPC requires pinning **one commit** across all three `rpc-server`s and the client — it
  does strict `{major,minor,patch}` version checking in HELLO and otherwise hangs at the handshake.

## Status

**v1 accepted** (build fabric, run three baselines). **v2 proposed:** vLLM or SGLang as the per-node engine
behind a fork/plugin shim; NIXL as the transport connector (ADR-004); NVIDIA Dynamo's disaggregated
prefill/decode + KV-aware routing as the north star; AIBrix v0.7.0 over llm-d at K8s scale, because AIBrix is
multi-engine and that matches a heterogeneous-node thesis.
