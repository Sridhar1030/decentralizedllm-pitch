---
status: SYNTHESIS — merges T4-A1..A5, T1-A2/A3/A4, T2-A3/A4/A5, T3-A1..A5, T5-A2
supersedes: nothing; defers to `01-VERIFIED-FACTS.md` on every conflicting number
scope: the user's explicit ask — "give the infra, tech required etc like vLLM, or some popular LLM serving runtimes"
---

# 20 — INFRASTRUCTURE & TECHNOLOGY STACK

Every number is tagged `(measured)` / `(derived)` / `(modelled)` / `(reported)` = third-party, unverified here.
Versions checked 2026-09-01. Nothing in this file invents a benchmark.

---

## 0. The four sentences that decide the whole stack

1. **We are a pipeline-parallel (PP) project, not tensor-parallel.** At N=3, PP moves **7,168·B bytes**
   per decode step against TP's **86,016·B** — **12x fewer** (modelled, T4-A1 §2) — and needs no collective.
   TP requires NVLink/InfiniBand; PP survives a real network. Every stack choice below follows from this.
2. **On a LAN we are compute-bound, not network-bound.** 89% of v0's per-token wall clock is compute
   (measured, T1-A1). PP saturates 1 GbE only at **batch ~349 at 100 steps/s** (modelled, T4-A1).
   So: buy an execution engine, do not buy a NIC.
3. **The layer split is not novel — vLLM ships it.** `vllm serve --pipeline-parallel-size 3` with
   `mp --headless` or `ray` splits layers across physical boxes today. The defensible product is the
   **trust / heterogeneity / churn / fault model**, which is not a vLLM flag; it is a different system.
4. **The runtime landscape's own scoreboard says distribution costs decode speed.** llama.cpp `--rpc`
   — the mature C++ binary-wire version of our architecture — measures **1.74x–1.88x slower on decode**
   and **4.17x faster on prefill** over a 9.41 Gbps link (measured, third-party). Pitch capacity and TTFT.

---

## 1. Serving-runtime comparison

`Split?` = can it place layer ranges on physically separate machines. Maturity as of 2026-09.

| Runtime | Core idea | Pipeline / multi-node | Hardware | Licence | Maturity | Fit verdict |
|---|---|---|---|---|---|---|
| **vLLM** v0.27.0 (2026-08-10) | PagedAttention (16-tok blocks, <4% waste) + Orca continuous batching + V1 engine (`AsyncLLM`↔`EngineCore` over ZeroMQ) | **Yes, native.** `--pipeline-parallel-size N`, executor `mp --headless` or `ray`; `VLLM_PP_LAYER_PARTITION="11,11,2"` for uneven splits | CUDA/ROCm/CPU (x86 AVX512, ARM NEON, Apple via container); `vllm/vllm-openai-cpu:latest-arm64` | Apache-2.0 | Production, the default | **BUY as per-node engine (v2). USE AS BASELINE (v1).** Never reimplement PagedAttention/continuous batching/sampler — that is the 18-month part, and it is free. |
| **SGLang** ≥0.5.7 (0.5.8, Jan 2026) | **RadixAttention** (radix trie over token-id runs, LRU-from-leaves, cache-aware queue reorder) + **chunked pipeline parallelism** | Yes. `--pp-size --nnodes --node-rank --dist-init-addr`, `--enable-dynamic-chunking` | NVIDIA/AMD GPU | Apache-2.0 | Production, top-tier | **STEAL TWO IDEAS.** Its chunked PP is the real fix for our 33% bubble; RadixAttention is the right v2 prefix cache. Not runnable on the CPU demo box. |
| **TensorRT-LLM** v1.0 | AOT kernel fusion + in-flight batching in a C++ `BatchManager` | Yes (TP/PP/hybrid), MPI/NCCL ranks | **NVIDIA only**, per-GPU-SKU engine build | Apache-2.0 | Production, fastest on NVIDIA | **REJECT.** NVIDIA-only + per-SKU engine build directly contradicts "commodity heterogeneous nodes". Cite only as the perf ceiling. |
| **HF TGI** | Rust router + Python shards; popularised continuous batching in OSS | TP only, sharded launcher | GPU | Apache-2.0 | **Archived / maintenance mode ~Mar 2026** | **REJECT, and never name it on a slide as "the alternative"** — it dates the deck instantly. |
| **DeepSpeed-Inference / MII** | **Dynamic SplitFuse** (FastGen): split long prompts, fuse with decodes into fixed-size forward batches | TP (+ZeRO-Inference offload), MPI launcher | GPU | Apache-2.0 | Public dev slowed 2025→2026 | **REJECT as runtime, STEAL SplitFuse.** Cleanest published statement of chunk-and-fuse, which is our chunked-prefill design. |
| **LMDeploy** | Two backends in one CLI: TurboMind (hand-written CUDA, persistent batch) + PyTorch | TP (`--tp`) | NVIDIA (TurboMind) | Apache-2.0 | Production (InternLM) | **Ignore.** No layers-across-hosts story, nothing unique for us. |
| **llama.cpp + `rpc-server`** | ggml backend scheduler treats a remote TCP peer as **just another `ggml_backend`**; graph serialised and shipped | **Yes, native.** `--rpc h0:p,h1:p,h2:p`; split by `--tensor-split` or auto-by-free-memory | CPU/Metal/CUDA/Vulkan/SYCL | MIT | Core mature; **RPC self-labelled "fragile and insecure"** | **BENCHMARK AGAINST — this is our C++ twin.** Steal 5 mechanisms (§1.1). Never build the product on the RPC path. |
| **Ollama** | UX layer; since 2025 its own Go engine over ggml (no longer llama.cpp) | **No** | CPU/Metal/CUDA | MIT | v0.30.x, huge install base | **Distribution channel, not architecture.** Relevant only as "what a node operator already has installed". |
| **MLX / `mlx-lm`** | Unified-memory arrays; `mx.distributed` with 4 backends incl. **JACCL = RDMA over Thunderbolt** (macOS 26.2+) | **Yes, native.** `mlx.launch --hostfile`, `--pipeline`; ranks `send`/`recv_like` to adjacent ranks | **Apple Silicon only** | MIT | Active, Apple-backed (WWDC26 s233) | **BUILD ON (Apple track, v2).** `mlx-lm --pipeline` is *literally our dataflow*. Best latency ceiling in this table. |
| **Ray Serve / Ray Data** | Actor + task placement over a heterogeneous cluster; `ray.serve.llm` wraps vLLM engines | Orchestrates, does not split | Any | Apache-2.0 | Mature; **already scaffolded in the repo** (`ray-vllm/`) | **BUILD ON — control plane only (v2).** See §3 and §6: Ray is not a data plane. |
| **KServe** v0.20.0 (6 Aug 2026) | K8s CRD for inference; v0.20 adds an LLM CRD backed by llm-d + disaggregated serving + Envoy AI Gateway | Orchestrates, K8s-native | K8s | Apache-2.0 | CNCF incubating, mature | **v2 only.** Overkill for a 3-container demo; correct once there are ≥2 pipeline replicas per region. |
| **Triton Inference Server** | Multi-framework model server, C-API backends, ensembles | No | Any | BSD-3 | **Folded into Dynamo → "Dynamo-Triton"**; new investment goes to Dynamo | **REJECT.** Superseded. Citing it as current in 2026 is a free credibility loss. |
| **NVIDIA Dynamo** 1.0 GA (16 Mar 2026) | **Disaggregated prefill/decode**, KV-aware router, hierarchical KV offload, **NIXL** transfer | Orchestrates engines, datacenter scale | NVIDIA-centric | Apache-2.0 | 1.0 GA | **ARCHITECTURAL NORTH STAR (v2).** P/D disaggregation *is* heterogeneous-node scheduling. Steal the pattern, not the implementation. |
| **llm-d** | K8s-native vLLM at scale; NIXL for KV transfer between prefill and decode pods | Orchestrates, K8s + Inference Gateway | GPU + K8s | Apache-2.0 | Red-Hat-led, shipping (KServe 0.20 integration) | v2, **and only if K8s.** Loses to AIBrix for us: vLLM-only. |
| **AIBrix** v0.7.0 (16 Jun 2026) | ByteDance control plane (now under `vllm-project`); LLM-aware autoscaling, LoRA mgmt, KV-cache-centric P/D data plane | Orchestrates, K8s | K8s | Apache-2.0 | 242 PRs in the v0.7 cycle | **v2 preferred over llm-d** — it is **multi-engine** (vLLM/SGLang/TRT-LLM), which matches a heterogeneous-node thesis. |
| **ExecuTorch 1.0** | PyTorch export → flatbuffer `.pte`, AOT memory planning, no Python at runtime | No | Phone / MCU / edge | BSD-3 | 1.0 GA | Out of scope now. **v2 relevance: this is what a phone-class node runs.** |
| **ONNX Runtime + onnxruntime-genai** 0.15.2 | Graph-level execution-provider abstraction; one model, many accelerators | No (partitions to EPs, not hosts) | Anything | MIT | Mature | **Ignore for serving.** Only if Windows/ARM/WASM nodes ever matter. |

### 1.1 The five mechanisms to steal from llama.cpp RPC (all v1, ≤1 day each)

Read from `ggml/src/ggml-rpc/ggml-rpc.cpp`, master. Wire format: `[uint64 LE length][payload]`, 18 commands,
`HELLO == 14` with a strict `{major,minor,patch}` check. **No JSON, no base64, anywhere.**

| # | Mechanism | Our fix | Win |
|---|---|---|---|
| 1 | `[u64 len][payload]` binary framing | Drop JSON + base64 | 1.33x bytes (measured, exactly 4/3) + no parse of a ~MB string |
| 2 | One persistent socket per peer, `send_async` request pipelining | Hoist `httpx.AsyncClient` to module scope, `verify=False` | −17.6 ms/token (measured, T1-A1/T5-A4); kills 3 TCP handshakes/token |
| 3 | `GRAPH_RECOMPUTE`: `cgraph->uid` match ⇒ client sends **4 bytes** instead of re-serialising 205 KiB | Send shape/dtype header once per session, then raw bytes | Header amortises to ~0 |
| 4 | `SET_TENSOR_HASH` + FNV-1a 64 above `HASH_THRESHOLD = 10 MiB` | Content-address repeated prompt prefixes at node0 ingress | A free prefix cache on the wire, ~20 lines |
| 5 | `HELLO` strict version check | Version field in our node handshake | Prevents the silent-corruption bug class |

> The reason `GRAPH_RECOMPUTE` exists is the same reason our v0 is slow: for an 8-layer Qwen2.5-0.5B shard the
> ggml **graph description is 209,452 B ≈ 205 KiB (modelled)** — *larger than the activation it describes*.
> Both systems hit the identical wall: the metadata, not the tensor, is the payload.

---

## 2. Decentralized prior art

| System | What it splits | Topology / routing | Trust model | Incentive | Status 2026-09 | Delta vs us |
|---|---|---|---|---|---|---|
| **Petals** (BigScience/Yandex/HSE) | **Layers (PP)**, blocks over WAN | Kademlia DHT (hivemind); client picks the server chain by **D\* Lite** shortest path on measured per-block latency | Honest-but-curious **assumed, not enforced**. Paper concedes: peers on the first layers can recover input tokens | Proposed "points", **never implemented** | Live, 10.5k★, 522 commits; public swarm past its BLOOM-era peak | **0.83 decode steps/s** on a real 14-server EU+NA swarm (measured, arXiv 2312.08361). Tuned for 100 ms WAN; we are the inverse problem — sub-ms LAN, operator is the adversary. **Petals already solved our FINDING 2** (client holds embeddings + `lm_head`), so its return path is already 170x better than v0's (derived). |
| **exo 1.0** (exo labs) | **Layers**, **ring memory-weighted** partitioning (layers ∝ device RAM) | UDP/mDNS auto-discovery, **p2p, no master**, OpenAI-compatible API on :52415 | Fully trusted LAN | none | **47.2k★, Apache-2.0.** v1.0 ships day-0 **RDMA over Thunderbolt 5** (macOS 26.2+), "99% latency reduction" (reported) | **The closest competitor, and today it beats our v0 on Apple silicon.** Its fast path is Apple+TB5-only; ours is commodity wired Ethernet on heterogeneous CPU. **exo does not batch.** |
| **hivemind** (`learning-at-home`) | n/a — it is the substrate | Kademlia DHT, libp2p, averaging primitives, DMoE | none (library) | none | MIT, active, 229★ | We use **~0 of it at N=3**. A DHT for three docker containers is malpractice — say it before a judge does. |
| **Distributed Llama** (b4rtaz) | **Tensors (TP)**, root + 2ⁿ−1 workers | Static config, Ethernet sync | Fully trusted | none | Active, C++, ARM + x86 AVX2 | **Wrong parallelism.** TP = 2 collectives per block and 12x the bytes of PP at N=3 (modelled). Fine on a fast fabric, fatal on an untrusted/slow link. |
| **Cake** (evilsocket) | Transformer blocks across heterogeneous devices | Zero-config mDNS or manual topology | Fully trusted | none | Active, Rust/Candle, CUDA/Metal/Vulkan/CPU, iOS+Android; self-described experimental | Same three gaps: trust, batching, queueing. |
| **GPUStack** + `llama-box` | Layers via llama.cpp RPC; also a vLLM backend | Central manager schedules workers | Single-owner cluster | none | Active, production-ish | **Orthogonal — a plausible v2 control plane**, not a rival. |
| **llama.cpp `rpc-server`** | Layers over TCP | Manual `--rpc host:port` list | Fully trusted, **no auth at all** | none | In-tree, explicitly "not secure, do not expose" | **The honest CPU baseline to beat** (§1, §5). |
| **Prime Intellect** | **Training**, not inference | DiLoCo / `prime-rl`; SHARDCAST weight broadcast; **TOPLOC** for verifying untrusted inference workers | Untrusted, cryptographically verified | p2p compute protocol announced, **no token launched** | INTELLECT-2 = 32B globally-distributed RL (arXiv 2505.07291). **INTELLECT-3 (106B MoE) was trained on a centralized 512-GPU cluster** | Different problem. **Steal TOPLOC** (arXiv 2501.16007): LSH commitment over intermediate activations, up to **100x faster to validate than to generate**, ~1000x smaller proofs (reported). |
| **Together AI** | Was PP over heterogeneous WAN (DT-FM, NeurIPS'22, arXiv 2206.01288) | Evolutionary tasklet scheduler; **4.8x faster than SOTA on networks up to 100x slower**, 8 cities / 3 continents (measured) | — | — | **Abandoned the thesis.** $800M Series C Jul-2026 at $8.3B; owns its own high-bandwidth clusters | The strongest team that tried this bought datacenters. **Address it, do not hide it** — it is an argument about capital, not architecture, and it does not apply on-prem. |
| **Gensyn** | Training | Verde optimistic verification (refereed delegation, safe with ≥1 honest party); OP-Stack L2 | Untrusted, economically secured | Token + slashing | Mainnet live **2026-04-22**, >5,000 H100-equivalents day one (reported) | Verification design is the transferable part. |
| **Bittensor / TAO subnets** | **None** — whole models, competitive scoring | Yuma consensus, validators score miners | Untrusted miners, economically scored | TAO emissions | **Chutes (SN64)**: >9.1T tokens, >400k users, >$100M cumulative volume. **Targon (SN4)**: TEE-verified compute, ~$10.4M ARR (all reported) | Real revenue, real users, **zero model splitting.** A distribution channel, not a competitor. |
| **Akash / io.net / Nosana / Render / Golem / Flux** | **None** | Job marketplace | Untrusted host, no meaningful compute verification | Tokens | Live, real volume | *"Those are Airbnb for GPUs; we are the thing you'd run on top of them."* |
| **Ritual** | None | Infernet oracle network; Ritual Chain | On-chain verifiable results | Token | Testnet→prod | Different product (on-chain inference oracle). |

**Academic ancestor, and it matters more than Petals.** Split learning / SplitNN (Gupta & Raskar 2018) is
literally this architecture, and every privacy claim it made was broken: **UnSplit** (WPES'22, eprint 2021/1074)
recovers inputs *and* steals a functionally equivalent client model from an **honest-but-curious** server that
knows only the architecture; **FSHA** (CCS'21) survives client-side differential privacy (arXiv 2201.04018).
Corroborating inversion results: **vec2text** recovers **92% of 32-token inputs exactly**, BLEU 97.3
(arXiv 2310.06816); **LM inversion** recovers **27% of prompts exactly** from next-token probabilities alone
(arXiv 2311.13647). ⇒ **Claim parameter secrecy, never input secrecy.**

---

## 3. The chosen stack, layer by layer

### 3.1 v1 — hackathon (days, CPU, docker-compose, 3 nodes)

| Layer | Choice | One-sentence justification |
|---|---|---|
| **Runtime** | **Keep the custom Python chain** (`node.py` FastAPI + torch eager), harden it | Adopting any runtime that already splits across hosts deletes the demo's entire premise — at that point we are demoing llama.cpp; every runtime that *doesn't* split can't do the job at all. |
| **Model / dtype** | Qwen2.5-0.5B-Instruct, fp32 compute, **bf16 at the wire boundary only** | bf16 is 2.00x wire for **3.5 µs/frame** at **99.41% top-1 agreement, KL 2.6e-5**, greedy output bit-identical on 4/4 prompts (measured, T2-A4/T2-A5) — a dtype cast beats every byte codec measured. |
| **Layer split** | `NODE_LAYERS "0-11" / "11-22" / "22-24"` — three env-var edits | FINDING 1: `lm_head` = **9.13 layer-equivalents**, so 8/8/8 is really 8/8/17.13 and the pipeline is **1.55x** slower than balanced, for free (derived). |
| **KV cache** | Per-node per-session `DynamicCache` keyed by `session_id`, with a `position` fence + LRU/TTL | FINDING 3: **271x** redundant position-forwards (147,200 → 543) and **77%** of all wire bytes removed; **28.6x wall clock** (modelled). Hard prerequisite: renumber sliced layers to local `layer_idx` or RoPE and the causal mask break **silently**. |
| **Transport** | **DLP**: 40-byte fixed header over one **persistent TCP** socket per peer, `TCP_NODELAY=1`, single `sendmsg([hdr, mv])` | 8.483 ms → **0.089 ms per hop, 95.4x** (measured, loopback); header overhead **40/3584 = 1.12%**. Do the one-line `AsyncClient` hoist **first** — it alone is 8.483 → 1.103 ms (7.7x, measured) even if DLP slips. |
| **Serialisation** | `struct.pack('<4sBBHIIIIIIBBHI')` header + `memoryview` of the tensor's own storage; `np.frombuffer` zero-copy on receive | O(1) instead of O(n): **25.4 µs → 1.08 µs at seq=1, 23,429 µs → 1.80 µs at seq=1024** (measured). base64 is a compressor with ratio **1.333** that also burns **7.18 ms/MB** (measured). |
| **Compression** | **bf16 cast, and NOTHING else.** Hard-disable any byte codec below 1 MiB payload | Two of 60 measured codec/payload combinations are net-positive at 1 GbE; **zero** at 10 GbE (measured, T2-A3). LZ4 **expands** activations (ratio 1.0036–1.0056, measured). Byte-codec ceiling on fp32 is r=0.843 vs **r=0.850 for white Gaussian noise** — there is nothing to compress. |
| **Return path** | **argmax/sampling on node2**, return a 4-byte token id | FINDING 2: **607,744 B → 4 B**, and it simultaneously deletes the LM-inversion oracle that recovers 27% of prompts exactly. ~3-line diff. |
| **Queue** | `asyncio.Semaphore(3)` at the coordinator + `Semaphore(1)` per node + `asyncio.Queue(maxsize=6)` → HTTP 429 with `Retry-After: 8` | N\* = D/D_max = 0.71254/0.30897 = **2.31 → 3**, and exactly **3.00 after the rebalance** — the whole scheduler is one integer. K=6 comes from Little's law (K = λ·W_SLO), not a guess. |
| **Threading** | `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and anyio's thread limiter set to **1** | `limits.cpus: "2"` is a **cgroup quota torch cannot see** — it spawns 10 OpenMP threads that thrash 2 CPUs; and Starlette runs the sync `def forward` in a **40-slot** anyio pool (verified, anyio 4.12.1), which is processor sharing, not a queue. |
| **Control plane** | Self-registering table: nodes `POST /register` with their layer range + state `{LOADING, READY, DRAINING, DEAD}`; coordinator polls **500 ms**, greedy cover of `[0,24)` | Failure detection **60 s → 1.5 s**; **net −20 LOC** in `coordinator.py` despite adding the feature, because it deletes the hardcoded r0/r1/r2 blocks. Overhead 1.2 KB/s = **0.001%** of 1 GbE. |
| **Discovery** | **mDNS/UDP + a static manifest. Explicitly NOT a DHT.** | Kademlia beats O(n²) polling only above ~n=50 (modelled); we have **n=3**, where 6 probes/s of direct polling is strictly better. |
| **Orchestration** | `docker-compose`, one user-defined bridge, **6G** mem limit per node, shared `hf-cache` volume + `HF_HOME`, Docker Desktop VM ≥16 GB/8 CPU | The loader allocates the model **twice** ⇒ **3.95 GB peak/container**, so `4G` OOM-kills at startup and 3×3.95 = 11.9 GB blows the default 8 GB VM. Shared cache: cold start **~255 s → ~65 s** (modelled). |
| **Observability** | Prometheus **`scrape_interval: 1s`** + `prometheus_client` **Histograms** (not f-string counters) + Grafana + Tempo, namespace `dllm_`, names mirroring vLLM's | 15 s scraping yields **12 samples in a 3-minute demo**; two counters can only produce a mean, and a mean hides the p99 the SLO is written against. Whole mesh ≈ **420 series** provided `session_id`/`request_id`/`token_idx` are **never** labels. |
| **Tracing** | OpenTelemetry SDK 1.44.0 + a **32-byte `F_TRACE` DLP header extension** (flags bit 4), head-sample `token_idx < 8` | 32/3624 = **0.88%** of a decode frame when sampled, **0 bytes** when not; `40+32 = 72`, `72 % 8 == 0`, so payload alignment survives. Untamed, one 512-token generation is **2,049 spans**. |
| **CI** | GitHub Actions running (a) `bench/t2a4_quality_harness.py` as the **codec merge gate** — block `KL_p99 > 1e-3` or top-1 < 0.999; (b) `bench/verify_constants.py`; (c) a `torch.allclose` stateless-vs-cached equivalence test | The 10-codec sweep runs in **~4 min on CPU** and reproduced byte-identically across two runs (measured). The allclose check is **non-negotiable**: a stale-cache bug produces plausible **wrong text**, not an error. |
| **Load / chaos** | `locust` (SSE-aware, closed-loop X(N) sweep at N ∈ {1,2,3,4,6,8,12}) + `vegeta` (open-loop, 0.1→1 req/s) + `bench/chaos_failover.sh` | A closed-loop VU test **structurally cannot observe 429 behaviour** (coordinated omission); against v1's μ_req = **0.252 req/s** (modelled), rates >0.3/s must produce 429 or admission control is not wired. |
| **Security** | Per-node API key + mTLS via uvicorn's native flags, **sequenced strictly after the connection-pool fix** | mTLS costs **~0.2 s per completion** if added before pooling (3 handshakes × 32 tokens), and **~0 after** (modelled) — order matters, or security reads as "security costs performance". |

### 3.2 v2 — production (months). Two tracks; **say which one on the slide.**

| Layer | x86 / NVIDIA fleet | Apple Silicon mesh | Justification |
|---|---|---|---|
| **Runtime** | **vLLM** (or SGLang) as the per-node execution engine | **`mlx-lm --pipeline`** | Buy PagedAttention, continuous batching, prefix caching, the sampler — 18 months of work, free. `mlx-lm`'s `send`/`recv_like` between adjacent ranks is *already* our dataflow. |
| **Transport** | DLP over TCP → **NIXL** (`--kv-transfer-config`) once RDMA exists | **JACCL = RDMA over Thunderbolt** (macOS 26.2+) | NIXL's backends are RDMA/IB, RoCE-via-UCX, TCP fallback, NVMe-oF, S3 — the *same code path* runs on today's TCP and tomorrow's RDMA with a config change. TB5 at 80 Gb/s puts 3 hops of bf16 at **~0.5 µs of wire time** (modelled). |
| **Serialisation** | DLP unchanged; `google-crc32c` swapped in for `zlib.crc32` | MLX arrays, no serialisation | The 4-byte header field is already named `crc32c` for exactly this swap; **Python stdlib has no CRC32C**, so v1 fills it with CRC-32/ISO-HDLC — do not claim hardware CRC32C in v1. |
| **Compression** | Negotiate wire dtype **once at pipeline setup**, not per frame; **QuaRot** (NeurIPS 2024, arXiv 2404.00456) Hadamard rotation to delete outlier channels; fp8 e4m3 on Hopper/Ada/Blackwell | same | Quantisation carries the compression: r=0.500 (bf16) and 0.250 (fp8) at ~0 CPU beat the **0.843** lossless ceiling of every byte codec measured. QuaRot is the only route to int4 that survives the byte-plane entropy argument. |
| **Queue** | **Continuous batching** with a coordinator-minted **immutable `BatchDescriptor`** (`step_id`, `epoch`, per-slot `req_id/slot/kind/emit/n_new/ctx_len`) executed verbatim by all 3 stages | same | A stage **cannot admit or evict on its own** — row order is the contract. Descriptor is **343 B at n_seq=16 = 1.20%** of the bf16 activation. Get it wrong and there is no exception, just row *i* attending to another user's KV. |
| **Control plane** | **etcd 3.6.6** with lease TTLs + watches, so the coordinator becomes stateless and can run 2+ behind the gateway | thin coordinator over `mlx.launch` | Not because 3 nodes need consensus — because **the coordinator is an undiscussed SPOF that no rung of the failover ladder addresses**. |
| **Orchestration** | **Ray Serve** first (already scaffolded in `ray-vllm/`), **AIBrix v0.7.0** at K8s scale; one **StatefulSet per stage** | `mlx.distributed_config` + `mlx.launch --hostfile` | AIBrix over llm-d because it is **multi-engine** (vLLM/SGLang/TRT-LLM), matching the heterogeneous-node thesis. One STS *per stage*, not one STS of 3 pods: the stages have different layer counts and resource shapes. A Deployment's LB Service would route decode step *n+1* to a pod that does not hold R's KV cache ⇒ full re-prefill. |
| **Observability** | DCGM exporter → Prometheus; **native histograms**; OTel Collector doing **tail-based** sampling; Alertmanager with multiwindow burn-rate rules | same | Replaces the hand-rolled `/metrics` counters; tail-based sampling replaces v1's head-based per-token hack. Make the coordinator OpenAI-streaming-compatible so **`vllm bench serve`** can drive it — it computes TTFT/TPOT/ITL percentiles natively and puts our tok/s on the same axis as published vLLM numbers. |
| **CI** | Same gates + a nightly **3-stage vLLM PP baseline** and **llama.cpp `--rpc`** run, published next to ours | + `mlx-lm --pipeline` on 2 Macs | *A custom stack that cannot state its number against vLLM is not credible.* |
| **Security** | **TEE attestation** (H100 Confidential Computing + Intel TDX, **2–8% throughput**, arXiv 2409.03992) + **SPIFFE/SPIRE** X.509 SVIDs with ~1 h rotation + **TOPLOC** activation commitments | same | TEE is the only *deployed* mechanism that addresses UnSplit-class inversion; SPIFFE makes identity drive authorization ("node1 may serve layers 8–16 and nothing else"), not just encryption. WireGuard (~0.03–0.1 ms, kernel-space) is the lazy alternative if you only want transport security. |
| **Scaling unit** | **A whole pipeline, never a stage.** PDB `minAvailable: R-1`; HPA on queue depth/TTFT via KEDA, **never CPU**; keep one warm spare pipeline | same | Stage 1 alone is useless — the model is incomplete. CPU is ~100% on every stage *by construction* and carries no signal. A 40 GB pull + weight load is 3–10 min, far beyond HPA reaction time. |
| **Multi-region** | **Never stripe a pipeline across regions.** Replicate whole pipelines per region; replicate only *weights* (S3 CRR); KV never leaves its region | same | us-east-1↔us-west-2 ≈ 60–70 ms × 2 hops = **+130 ms/token** = **66.6 s** on a 512-token response. Fatal for chat; fine for offline batch once ⌈65/19⌉+1 = **5 microbatches** are in flight per stage. |

---

## 4. Hardware BOM and $/1M tokens

All 3-node clusters sized for **Llama-3.3-70B Q4_K_M ≈ 40 GB** — the smallest model that does *not* fit one
cheap node, i.e. the smallest model for which this architecture has a reason to exist.
Prices checked 2026-09-01; `(list)` / `(street)` / `(est.)`.

| Tier | Line items | Total |
|---|---|---|
| **A — 3× Mac mini M4** (16 GB/256 GB) **over Thunderbolt, no switch** | 3 × $599 + 3 × TB4 cable @$30 | **$1,887** |
| A' — same, over a 10GbE switch | 3 × $599 + 3 × $100 10GbE BTO + TP-Link TL-SX105 $200 + Cat6a $24 | **$2,321** |
| **B — 3× consumer GPU box** | 3 × [used RTX 3090 24 GB ~$1,000 (est.) + mobo/CPU/64 GB/PSU/case ~$700] + TL-SX105 $200 + NICs/cables ~$114 | **$5,414** |
| B' — the **same 3 GPUs in ONE chassis** | host $900 + 3 × $1,000 GPU. Same 72 GB, same 2,808 GB/s, **no network at all** | **$3,900** |
| **C — cloud** | 3 × `g5.xlarge` (A10G 24 GB) @ $1.006/hr = **$3.018/hr**; single `g6e.xlarge` (L40S 48 GB) = **$1.861/hr** | hourly |

> **Price risk, flagged:** the BOM uses Mac mini M4 at **$599** (launch list). **Apple US list moved to $799 in
> Aug 2026.** At $799 Tier A is **$2,487** and its $/1M rises from **$4.37 to ~$5.70**, which puts it *above*
> Claude Haiku 4.5. Quote both or the table is stale.

### 4.1 The table that is the whole hardware argument — $ per unit of memory bandwidth

LLM decode is memory-bandwidth-bound, so `$/(GB/s)` is the real price of throughput.

| Device | GB | GB/s | $ | $/GB | **$/(GB/s)** |
|---|---:|---:|---:|---:|---:|
| **RTX 3090 24 GB (used)** | 24 | 936 | 1,000 | 41.7 | **1.07** |
| 3× 3090, **one** box (B') | 72 | 2,808 | 3,900 | 54.2 | **1.39** |
| 3× 3090, three boxes (B) | 72 | 2,808 | 5,414 | 75.2 | **1.93** |
| RTX 5090 32 GB | 32 | 1,792 | 5,769 (street) | 180.3 | 3.22 |
| RTX PRO 6000 96 GB | 96 | 1,792 | 8,500 (est.) | 88.5 | 4.74 |
| Mac mini M4 16 GB | 16 | 120 | 599 | 37.4 | 4.99 |
| 3× Mac mini M4 (A) | 48 | 360 | 1,887 | 39.3 | 5.24 |
| Mac mini M4 Pro 64 GB | 64 | 273 | 1,999 | 31.2 | 7.32 |
| Mac Studio M3 Ultra 512 GB | 512 | 819 | 9,499 | 18.6 | 11.60 |

**Three readings, all load-bearing.** (1) **The 2026 GPU market is the pitch:** a used 3090 is **3.0x** cheaper
per GB/s than a 5090 and **4.4x** cheaper than an RTX PRO 6000, and cheap bandwidth is only sold in **24 GB
units** — splitting the model is the only way to spend it. (2) **On Apple silicon, clustering loses:** Apple's
marginal memory is **$12.50/GB** (the +$200-per-16 GB BTO) against **$37.4/GB** for a whole extra Mac — buy RAM,
not Macs, until you exceed the largest single box. (3) **Even on GPUs, one chassis beats three:**
**$1.39 vs $1.93 per GB/s, 1.39x.**

### 4.2 $/1M output tokens — 70B Q4, decode, pipeline full

Assumptions stated: 3-year straight-line amortization (26,280 h), $0.17/kWh, 100% utilization,
`tok/s ≈ 0.65–0.75 × BW / 40 GB` per stage, **labour excluded**. All tok/s **(modelled)**.

| Tier | tok/s | $/hr | h per 1M | **$/1M tok** | Breakeven util. vs Haiku 4.5 |
|---|---:|---:|---:|---:|---:|
| A — 3× Mac mini M4 | 5.86 | 0.0922 | 47.4 | **$4.37** | **87%** |
| **B — 3× used 3090** | 52.7 | 0.4355 | 5.27 | **$2.30** | **46%** |
| C — 3× `g5.xlarge` on-demand | 33.8 | 3.018 | 8.22 | **$24.80** | never |
| C' — 1× `g6e.xlarge` on-demand | 16.2 | 1.861 | 17.15 | **$31.91** | never |
| Claude Haiku 4.5 (API, output) | — | — | — | **$5.00** | — |
| Claude Sonnet 5 (API, output) | — | — | — | $10.00 | — |
| Claude Opus 5 (API, output) | — | — | — | $25.00 | — |

Cross-AZ egress is **noise**: 70B bf16 hidden+residual = 32,768 B/hop × 2 hops = 65.5 KB/token → 65.5 GB per
1M tokens × $0.02/GB = **$1.31**. This independently confirms that PP is latency-bound, not bandwidth-bound.

**Four caveats that must travel with this table.** (1) **$/token across model qualities is meaningless** — the
only defensible claim is *if a 70B-class open model is good enough*, Tier B is ~2.2x cheaper than the cheapest
frontier API **at 100% utilization**. (2) **Utilization is the whole game** — at 10% utilization Tier B is
$23/1M and loses to every API listed. (3) **Labour is excluded and it dominates:** one engineer-week ≈ $4–6k ≈
**2,000 M tokens** of Haiku 4.5. (4) **In the cloud, owning loses outright** — every cloud row is 5–14x the
on-prem rows.

### 4.3 The crossover condition — when does the cluster win?

Let `P(m)` = price of the cheapest single device with `m` bytes; `U` = pipeline utilization.

> **The cluster wins iff `P(M) / (N · P(C)) > 1/U`.**

Without microbatching a 3-stage chain caps at `U = 1/3`, so the big box must be **3.0x** more expensive.
With microbatching `U → ~0.9`, `1/U → 1.11`, and an **11% price edge suffices** — a **2.7x swing**.
**This is why the queueing work, not the compression work, makes the economics real.**

| Scenario | M | C | 1-box `P(M)` | 3-node `3·P(C)` | Ratio | Verdict |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B fp32 (the PoC)** | 2.0 GB | 16 GB | $599 | $1,887 | **0.32x** | **Cluster loses 3.5x.** `M < C`, so the split buys nothing and costs 2 hops. **Say this on the slide.** |
| 70B Q4, Apple | 40 GB | 16 GB | $1,999 (M4 Pro 64 GB) | $1,887 | 1.06x | Dead heat. Buy the M4 Pro — simpler. |
| **70B Q4, NVIDIA** | 40 GB | 24 GB | $8,500 | $3,900 / $5,414 | **2.18x / 1.57x** | **Cluster wins.** No consumer GPU ships >32 GB; above it the price curve goes vertical. |
| DeepSeek-V3 671B fp8 | 671 GB | 512 GB | none exists | 8× Mac mini | ∞ | **Cluster is the only option.** |

---

## 5. Wired-cluster setup runbook — 3 machines, a switch, static IPs, MTU, iperf3

**If all three machines are Macs, buy no switch.** Each Mac mini M4 has 3 Thunderbolt 4 ports; three machines +
three cables = a full triangle, every pair a direct point-to-point link. N ≤ 4 needs no switch. TB4 sustains
**~20–26 Gbit/s**, TB5 on M4 Pro **~60 Gbit/s** (both reported) — 2–6x a 10GbE switch, for $90 of cable and
**$434 less** than the switched build.

### Step 1 — Static IPs. No DHCP, no mDNS in the data path.

```
mac0/node0  10.42.0.1/24      mac1/node1  10.42.0.2/24      mac2/node2  10.42.0.3/24
```
On the **dedicated** NIC only; leave Wi-Fi/DHCP on the other interface for internet. Put the **literal IPs** in
`NODE0_URL` / `NODE1_URL` / `NODE2_URL`. Rationale: `.local` mDNS costs **1–5 ms on first resolve**,
`mDNSResponder` re-resolves on cache expiry, and Docker containers do not see host mDNS by default.
macOS: `System Settings → Network → Thunderbolt Bridge → Configure IPv4: Manually`.

### Step 2 — Publish the ports; split the compose file three ways.

The compose file already does `ports: 8001:8001` etc. The 3-machine version is the same file split into three:
one node service per machine, `coordinator` + `gateway` + Prometheus on machine 0. **No code change.**

### Step 3 — MTU 9000 on all three NICs **and** the switch **and** the Docker bridge.

```bash
sudo ifconfig en1 mtu 9000                 # macOS
sudo ip link set en1 mtu 9000              # Linux
```
```yaml
networks:
  decentralized-net:
    driver_opts:
      com.docker.network.driver.mtu: "9000"   # miss this and containers still emit 1500 B segments
```
Payload efficiency **1448/1538 = 94.1% → 8948/9038 = 99.0%**. A 458,760 B prefill goes from **317 packets
(MSS 1448) to 52 (MSS 8948) — 6.1x fewer** (modelled); with a KV cache a 3,584 B hop goes from **3 segments to 1**.
**Jumbo must be set everywhere or PMTUD silently black-holes traffic — this is the most likely way the
3-machine demo fails on stage.**

### Step 4 — Verify with `iperf3` BEFORE blaming the model. Do not skip this.

```bash
node1$ iperf3 -s
node0$ iperf3 -c 10.42.0.2 -t 10 -P 4      # bulk bandwidth
node0$ ping -c 100 -i 0.01 10.42.0.2       # RTT
```

| Link | iperf3 expected | RTT expected | Tag |
|---|---|---|---|
| 1 GbE, MTU 1500 | **940–990 Mbit/s** (theoretical ceiling 1000 × 1448/1538 = **941**) | 0.20–0.35 ms | derived |
| 10 GbE, MTU 1500 / 9000 | 9.4 / 9.8 Gbit/s | 0.06–0.12 ms | modelled |
| TB4 bridge | 15–26 Gbit/s | 0.10–0.20 ms | reported |
| TB5 bridge (M4 Pro) | ~60 Gbit/s | <0.10 ms | reported |

> **Gate: < 900 Mbit/s on a 1 GbE link means a duplex mismatch or a bad cable. Fix that before anything else.**
> BDP for buffer sizing: 1 GbE @0.3 ms = **37.5 KB**; 10 GbE @0.1 ms = **125 KB**; 25 GbE @0.06 ms = **187.5 KB**.
> Size `SO_SNDBUF`/`SO_RCVBUF` at **2× BDP** (4 MB is a safe default) — **mandatory** for UNIX domain sockets,
> whose macOS default `net.local.stream.sendspace` is **8192 bytes** (measured): a 1.79 MB UDS transfer goes
> **2,732.4 µs → 467.6 µs, 5.8x**, just from the buffer (measured).

### Step 5 — Then run the test that actually matters.

`iperf3` measures **bulk bandwidth**; our v1 decode payload is **3,584 B**. At 10 GbE that transfer is
`3584 / 1.25e9` = **2.9 µs** against a 60–120 µs RTT — **the wire is 100% latency, 0% bandwidth.** Use a
3,584 B ping-pong against T1-A2's measured **0.084 ms/hop** frame server, not `iperf3`.

**Do not buy 10GbE for a 0.5B model.** 1 GbE saturates only at (big model × big batch): 70B, H=8192, bf16
hidden+residual = 32,768 B/hop; at B=64 that is **2.097 MB = 16.8 ms on 1 GbE** against a ~19 ms GPU stage
(88% of the stage). On 10 GbE, 1.68 ms.

### Step 6 — Demo-day landmines, in the order they will bite you.

| # | Landmine | Fix |
|---|---|---|
| 1 | Docker Desktop VM default 8 GB; 3 × 3.95 GB peak RSS = **11.9 GB** | Raise the VM to ≥16 GB / 8 CPU. `mem_limit: 6G`, **not** 4G. |
| 2 | Cold HF cache ⇒ each node re-downloads 1 GB and the demo stalls **indefinitely** | Shared `hf-cache:/root/.cache/huggingface` volume + `HF_HOME` on all three. |
| 3 | Gateway breaker opens after **3 failures for 30 s**; a slow first cold-CPU request kills the demo | `CIRCUIT_FAILURE_THRESHOLD=10`, `CIRCUIT_COOLDOWN_SEC=5`. Also fix bug B1: the breaker resets `failures=0` after any successful exchange, so a coordinator 500 caused by a dead node **resets** it instead of tripping it (2-line fix). |
| 4 | Cold start is **60–90 s per node** | Boot before the audience arrives. Never boot on stage. |
| 5 | `demo.sh` sends **exactly one curl**, so utilisation is 33% and every concurrency number on the slide is invisible | Drive at concurrency ≥3, or the queueing work shows nothing. |
| 6 | Keep the `sleep 45` + `depends_on: service_healthy` chain until #1 and #2 land | The hack is **load-bearing** — it serialises the peak allocation. |

---

## 6. What we would NOT use, and why

This section is where credibility lives. Each row is a thing a judge might expect us to reach for.

| Rejected | Why, with the number |
|---|---|
| **RDMA / RoCEv2 / InfiniBand (v1)** | RDMA saves **27 µs per hop**; deleting Python's HTTP+JSON+base64 stack saves **10,376 µs** (measured). **99.7% of a v0 hop is software** (10,406 µs of stack vs 28.7 µs on the wire). Post-KV the 1 GbE link carries 14 KB/token = 0.11 ms against ≥120 ms of compute — **~1,000x headroom.** RoCEv2 also needs PFC on every switch port, per-queue ECN/WRED thresholds, DCQCN firmware tuning and deadlock watchdogs, and its queue-pair model is **O(N²) in memory** with no NAT traversal — architecturally opposed to a swarm. Revisit only for 70B-class TP. |
| **`torch.distributed` gloo / NCCL / MPI** | We benchmarked the "proper" answer and it lost: **gloo `send`/`recv` = 219 µs vs 30.1 µs for a plain framed socket at seq=1 — 7.3x SLOWER** (measured, torch 2.10.0). Worse, gloo/MPI impose a **static `world_size` with fail-stop ranks**, which contradicts the join/leave premise. NCCL is unavailable on this host (`nccl: False`, measured). One dead rank hangs the collective to timeout and kills the server — this is the strongest architectural argument *for* the custom stack. |
| **A DHT (hivemind / Kademlia)** | Gossip beats O(n²) polling only above **~n=50** (modelled). We have **n=3**, where 6 probes/s of direct polling is strictly better. A DHT for three docker containers is malpractice. |
| **Any byte codec on the decode path (zstd, LZ4, brotli, blosc2)** | **2 of 60** measured codec/payload combinations are net-positive at 1 GbE; **zero** at 10 GbE or 25 GbE. **LZ4 EXPANDS activations** (ratio 1.0036–1.0056 in all 12 realistic cases, measured). The lossless ceiling on fp32 activations is **r=0.843 — barely better than white Gaussian noise at r=0.850** — because the only compressible thing is one low-entropy IEEE-754 exponent byte (2.84 of 8 bits). Best-possible saving on a 3,584 B hop at 1 GbE = **4.5 µs**, against a 200–500 µs LAN RTT. |
| **base64** | It is already a codec we are running, with ratio **1.333** and a CPU cost of **7.18 ms/MB** (measured). Deleting it is worth **+36.25 ms/hop** on a 7.34 MB prefill; the best compressor in the entire study is worth **+1.00 ms/hop**. **36x.** |
| **int8 per-**tensor** scaling** | Catastrophic and it is our own measurement: perplexity **411,041 vs 18.64**, top-1 agreement **0.00684**, output degenerates to `" time declaration declaration declaration"`. Cause: channel 62 of our residual stream is **972x** the median channel max. Per-**token** scaling costs **+0.22% wire bytes** and buys a **328x KL reduction**. |
| **int4 activations** | At H=896 the best variant (group-128, 462 B) manages **8/20 greedy prefix match**, rel-err 0.078 (measured). Unusable without QuaRot. |
| **Cosine similarity as a quality metric** | The vanity metric of activation compression: int8 scores **0.99958** — "basically identical" — while silently flipping **7.3% of generated tokens**. Report **KL p99 + top-1 agreement**. Likewise report per-token rel-L2, never block-Frobenius, which understates typical-token error by **2.67x–4.31x**. |
| **gRPC / protobuf** | Rejected on **no zero-copy in Python**, Python-list materialisation for `repeated float`, and HTTP/2 framing overhead. **CORRECTION to a claim circulating in this project:** "protobuf varint pessimises a float tensor" is **factually wrong** — `float` is wire type 5 (I32) fixed-width and `repeated float [packed=true]` is byte-identical to a raw fp32 buffer on little-endian. Do not put that argument on a slide. (Varint *would* pessimise a quantised `repeated int32` — encode quantised tensors as `bytes`.) |
| **Apache Arrow Flight** | A genuinely strong contender rejected on **payload shape, not quality**: the KV-cached hot payload is **3,584 B**, which is control-plane-shaped, and `pyarrow` is a **~120 MB wheel**. This decision **flips** if frames exceed ~1 MB or nodes become polyglot. |
| **UCX-Py / `torch.distributed.rpc`** | UCX-Py is **discontinued** (last release 0.45, RAPIDS 25.08); its successor UCXX ships CUDA-flavoured wheels with no macOS-arm64 story. `torch.distributed.rpc` has been **deprecated since PyTorch 2.0**. Do not build on either. |
| **A service mesh on the tensor path** | Istio's own published benchmark: **p90 +0.63 ms/hop sidecar, +0.16 ms ambient L4**. Two inter-stage hops = **+1.26 ms/token = +645 ms on a 512-token response** — 6.3% of a 20 ms/token budget, **25% of a 5 ms GPU budget**, paid on every token forever. Envoy is userspace, so it also adds **two extra memcpys of every tensor per hop** — exactly the copy our zero-copy `frombuffer` path deleted. **Mesh the control plane; never the tensor path.** Exclude it explicitly (`traffic.sidecar.istio.io/excludeOutboundPorts`). |
| **Ray as a data plane** | Ray's object path is Plasma → gRPC → Plasma with **~0.5–1 ms remote `ray.get` overhead** (documented) vs a **0.084 ms measured** raw TCP frame; Plasma is zero-copy **same-node only**. Ray is a control plane. This is vLLM's own split (`--distributed-executor-backend ray` for orchestration, NCCL/gloo for tensors). |
| **Kafka on the inference path** | It is a partition-ordered log with **no per-message ack**, so head-of-line blocking is architectural; `acks=all` + `min.insync.replicas=2` adds ms-to-tens-of-ms p99 to a **41 ms** per-token budget; and durability of a deadline-bound message is a contradiction. Kafka for telemetry/audit/billing **only**. NATS JetStream 2.10+ is the correct durable inference queue if one is ever needed. |
| **`shm_size` / `ipc: host`** | `/dev/shm` matters only for `DataLoader` worker IPC and NCCL shared-memory transports. **We have neither** — one process, sockets between containers. Stated explicitly so nobody spends a day on it. |
| **`vmsplice` / `sendfile` / `splice` zero-copy** | Saves **exactly one user-to-kernel copy = 96.9 µs at 1.79 MB** (measured), against a **38 ms** problem. Explicitly not worth building. |
| **eBPF / XDP** | Optimises **packet rate**; a 3-node request/response chain is **latency**-bound, not pps-bound. |
| **PagedAttention as the first KV lesson** | It solves **fragmentation**, which we do not have — we have **no cache at all**. A plain `dict[req_id, DynamicCache]` is sufficient at B≤32. Build paged blocks for **concurrency and reservation waste**, not for realloc cost: `torch.cat` regrowth is **0.1%** of generation time here (6.8% even at 32k context). |
| **Speculative decoding at 0.5B** | A negative result and it should be presented as one: `lm_head` is **9.13 of 33.13 layer-equivalents**, so every draft pays it. Layers 0–7 early-exit costs **51.7% of target cost = 0.75x (slower)**. Only n-gram/prompt-lookup (c=0) helps: 1.33x on free-form chat, 2.31x on quoting/code/JSON. Revisit at 7B+ where the draft/target ratio reaches the ~10x spec-decode needs. |
| **Interleaved 1F1B / virtual pipeline stages / zero-bubble ZB-H1/ZB-H2** | An **active trap**: 1F1B doubles hops 3→6 with **zero bubble reduction at R=1**, and zero-bubble schedules work by splitting the **backward pass**, which inference does not have. Any 1F1B implementation degenerates to plain round-robin without a backward. |
| **ZKML / zero-knowledge proofs of inference** | **26,460x too slow**: 2,646 s/token proving for LLaMA-2-7B vs a ~100 ms/token target; zkLLM needs 986 s commit + 803 s prove per forward pass on LLaMA-2-13B (**~18 days** for a 2,000-token generation). Use TEE attestation (2–8%) or TOPLOC (100x faster to validate than generate) instead. |
| **Secure MPC for interactive serving** | BumbleBee: **>13 minutes per token** on LLaMA-7B with an 8-token prompt; BOLT: **3.18 s/token** on a 30 Gbps / 0.8 ms LAN. Three to four orders of magnitude too slow. **Say so on the slide** rather than leaving it as an open question. |
| **The llama.cpp RPC backend as our substrate** | Upstream self-describes it as *"currently in a proof-of-concept development stage… fragile and insecure"* and warns against open networks: `rpc-server` executes **arbitrary remote compute graphs against local memory with zero authentication.** Benchmark against it; never ship on it. |
| **"We add differential privacy"** | **FSHA defeats client-side DP** (arXiv 2201.04018). Do not put DP on a slide as a privacy answer. |
| **TGI, Triton, TensorRT-LLM as "the alternative"** | TGI is **archived** (~Mar 2026); Triton is folded into **Dynamo-Triton**; TensorRT-LLM is NVIDIA-only with a per-SKU engine build. Citing any of them as current is the cheapest available way to lose an infra-literate judge. |

---

## 7. Contradictions between teams, resolved explicitly

Not averaged. Each has a winner and a reason.

| # | Conflict | Resolution |
|---|---|---|
| 1 | **"3 hops per token"** (shared context, most teams) vs **"6 wire crossings, 4 activation-sized"** (T1-A1, measured) | **T1-A1 is right about v0's topology**: v0 is **star-routed** — the coordinator relays every activation, so each logical hop costs two crossings. But **`01-VERIFIED-FACTS.md` FINDING 4 supersedes for the 935x figure**, which counts 2 activation crossings under the v1 chain-routed design. Use "3 POSTs, 6 crossings" when *describing v0*; use FINDING 4's totals when quoting the reduction. Anyone anchoring on "3 hops" for v0 **undercounts wire bytes ~2x**. |
| 2 | **T4-A1: "compression buys nothing on a LAN"** vs **T2: "bf16 is a 2x win, ship it"** | **Not a conflict once the terms are split.** T4-A1's claim is about **byte codecs and sub-bf16 quantisation**, and T2-A3/T2-A4/T2-A5 independently agree: no byte codec is net-positive on a decode frame at any LAN speed, and sub-bf16 only pays **below ~163 Mbit/s per hop**. **bf16 is a dtype cast, not a codec** — 3.5 µs, 2.00x, KL 2.6e-5 — and it ships. Everything below bf16 is a **WAN toggle**, off by default. |
| 3 | **Wire cost per token per hop: `H` vs `2H`** | For **our** stack it is `H` (3,584 B fp32). For a **vLLM PP baseline** it is **`2H`** — vLLM ships `IntermediateTensors{hidden_states, residual}` because it keeps the residual un-added for fused add-RMSNorm. So vLLM PP bf16 = **3,584 B/hop = exactly our fp32 figure**, and the vLLM **fp32 CPU path is 7,168 B = 2x worse**. Never claim "vLLM PP is more efficient on the wire per token" — it is not; its win is KV caching and sampling on the last rank. |
| 4 | **T1-A3: "defer RDMA"** vs **T4-A2: "JACCL Thunderbolt RDMA is the Apple fast path"** | Different things. **RDMA over Ethernet (RoCEv2) = defer** — 27 µs of headroom, and PFC/DCQCN tuning on every switch port. **RDMA over Thunderbolt via MLX JACCL = free** — it is a backend flag on hardware the user already owns, zero fabric engineering. Track-dependent, not contradictory. |
| 5 | **Headline speedup: 19.0x (T5-A4) vs 28.6x (T3-A1) vs 32.9x (T3-A3)** | **Quote 19.0x** (1.27 → 24.21 tok/s, modelled by composing seq=512 measured stage times) as the integrated figure — it is the conservative one and it does not credit the −8 ms node-local win. **28.6x** is T3-A1's wall-clock KV-cache-only figure on a different basis. **32.9x is from a separate ctx=128 measurement run and must NOT be merged with the seq=512 ladder** — that would double-count. And **271x is FLOPs, not wall clock**: quote them separately or the stage under-delivers. |
| 6 | **Notation collision: T3-A2's `R` vs T3-A5's `C`/`P`** | Normalised here: **`R`** = in-flight requests, **`S = 3`** = stages, **`P`** = pipeline replicas. **`U = min(1, R/(P·S))`.** T3-A5's formula *generalises* T3-A2's `R ≥ S`; it does not contradict it. Corollary worth saying: **replicas at fixed load make utilisation worse** (P=3, R=3 ⇒ 33%). Concurrency fills a pipeline; replication only buys capacity and fault tolerance. |
| 7 | **Ray: "build on it" (T4-A2) vs "0.5–1 ms overhead" (T4-A4)** | Agreement, differently scoped. **Ray for placement groups, actor lifecycle, health, autoscaling. Never for tensors.** Actors talk over the DLP socket. |
| 8 | **T3-A5: "rebalance is worth 1.539x" vs T3-A2: "1.30x"** | Both correct on different axes. **1.55x / 1.539x is the layer-equivalent (compute) ratio** (17.13/11.13). **1.30x is the measured wall-clock ratio** (308.97/237.51 ms). **Use 1.30x for any wall-clock claim**; use 1.55x only when explicitly talking layer-equivalents. |
| 9 | **"No node holds the full model"** | **Currently false twice over.** `node.py:36` calls `AutoModelForCausalLM.from_pretrained()` on **every** node, so every node loads the whole checkpoint at boot (~3.95 GB peak RSS). And `tie_word_embeddings: true` makes node0's `embed_tokens` and node2's `lm_head` the **same 136,134,656-param matrix**, so max single-node parameter share is **51.7%**, not 33%, and any 2-of-3 coalition holds **75.85%**. Fixes: pre-sliced `shard{0,1,2}.safetensors` (peak RSS → ~1.1 GB) and moving embed + `lm_head` to the client (Petals' own design) ⇒ **24.1% per node, all three equal**. **State the real number on the slide; a judge who opens `config.json` will find the tie.** |

---

## 8. Three baselines to run before any of this reaches a slide

A custom stack that cannot state its number against a production runtime is not credible.

| # | Baseline | Command | What it proves |
|---|---|---|---|
| 1 | **vLLM PP=3, one container** | `docker run … -e VLLM_PP_LAYER_PARTITION="11,11,2" -e VLLM_CPU_OMP_THREADS_BIND=nobind vllm/vllm-openai-cpu:latest-arm64 --model Qwen/Qwen2.5-0.5B-Instruct --pipeline-parallel-size 3 --distributed-executor-backend mp --dtype bfloat16 --enforce-eager` | The production-runtime pipeline cost **with no network in it**. Run once **without** `VLLM_PP_LAYER_PARTITION` as the control — that run is the 1.55x-imbalanced baseline that proves FINDING 1 applies to vLLM too. |
| 2 | **llama.cpp `--rpc`, 3 local `rpc-server`s** | `llama-cli -ngl 99 --rpc 127.0.0.1:50052,127.0.0.1:50053,127.0.0.1:50054` (build `-DGGML_RPC=ON`) | The honest peer: same split, C++, binary wire, persistent sockets. **Within 2x of it is a result.** Pin one commit across all three servers and the client — `HELLO` does a strict `{major,minor,patch}` check and mismatched builds hang or crash mid-inference. |
| 3 | **Single-process llama.cpp, same box** | `llama-cli -ngl 99` | The ceiling. Quantifies our distribution tax the way the 1.74x/1.88x third-party numbers do. |

Both stacks speak `/v1/chat/completions`, so drive them through the **existing gateway** by swapping `VLLM_URL`.
Record at concurrency **1 / 4 / 16**: TTFT p50/p99, tok/s, and — the row we actually win — **tok/s with one node
killed.** Pitfalls that otherwise corrupt the comparison: `nobind` is mandatory (Docker exposes no NUMA nodes);
`--enforce-eager` on both sides or you are comparing a compile step; **a concurrency-1 comparison measures
nothing**, because PP=3 idles 2 of 3 stages on *both* stacks.

---

## 9. Bill of technologies — the copy-paste list

**v1 (hackathon):** Python 3.12 · PyTorch 2.10 (eager, `set_num_threads(2)`) · transformers 5.5 · FastAPI/uvicorn
(control plane only) · **raw `asyncio` TCP + DLP 40-byte frames** (data plane) · numpy `frombuffer` zero-copy ·
bf16 wire cast · `asyncio.Semaphore` + `asyncio.Queue` admission · docker-compose · Prometheus (`scrape_interval: 1s`) +
`prometheus_client` histograms · Grafana + Tempo · OpenTelemetry SDK 1.44.0 · locust + vegeta · GitHub Actions ·
per-node API key + uvicorn mTLS.

**v2 (production, x86/NVIDIA):** vLLM v0.27.0 per node (or SGLang 0.5.8) · DLP → NIXL over RDMA/UCX ·
`google-crc32c` · fp8 e4m3 / QuaRot · coordinator-minted `BatchDescriptor` continuous batching · etcd 3.6.6 ·
Ray Serve → AIBrix v0.7.0 on K8s (StatefulSet per stage) · Karpenter + KEDA · DCGM exporter + OTel Collector
(tail sampling) + Alertmanager · `vllm bench serve` in CI · SPIFFE/SPIRE + H100 CC/Intel TDX + TOPLOC.

**v2 (production, Apple mesh):** `mlx-lm --pipeline` · `mx.distributed` JACCL (RDMA over Thunderbolt, macOS 26.2+) ·
`mlx.distributed_config` + `mlx.launch --hostfile` · thin coordinator · same observability and CI.
