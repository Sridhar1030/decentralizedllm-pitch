---
team: T4 — Infrastructure & Serving Runtimes
agent: T4-A2
topic: Serving-runtime landscape excluding vLLM — what to demo on, build on, benchmark against
headline: llama.cpp's ggml RPC backend is our PoC's already-shipped twin, and its own third-party benchmark shows distribution costs 1.74–1.88x on decode even over 10 GbE — so the pitch must be capacity, not speed; demo on the custom Python chain, build v2 on vLLM+Ray/KServe (x86/NVIDIA) or mlx-lm --pipeline over JACCL/Thunderbolt (Apple mesh), benchmark against llama.cpp --rpc.
---

# T4-A2 — Serving-runtime landscape (vLLM excluded, see T4-A1)

**One-line verdict up front:** nothing in this landscape does what our PoC claims *as a product*. Two things do it
as a *feature* — `llama.cpp` RPC backend and `mlx-lm --pipeline`. Both split by layer range across boxes. Both are
the honest baseline. Neither markets it as "decentralized", because both know the decode-side truth in §2.4.

---

## 1. The scoreboard

`Split?` = can it put layer ranges on physically separate machines. `CPU?` = usable CPU-only on a laptop.

| Runtime | Distinguishing technical idea | Split? | Multi-node story | HW floor | License | Maturity (Sep 2026) | Fit verdict |
|---|---|---|---|---|---|---|---|
| **llama.cpp + `rpc-server`** | ggml backend-scheduler treats a remote TCP peer as just another `ggml_backend`; graph is serialized and shipped | **yes, native** | `--rpc h1:p,h2:p`, split by `--tensor-split` or auto-by-free-memory | CPU / Metal / CUDA / Vulkan / SYCL | MIT | core mature; **RPC self-labelled "proof-of-concept… fragile and insecure"** | **BENCHMARK AGAINST.** Closest peer. Steal 5 ideas (§2.5). Don't build the product on the RPC path. |
| **MLX + `mlx-lm`** | unified-memory arrays; `mx.distributed` with 4 backends incl. **JACCL = RDMA over Thunderbolt** (macOS 26.2+) | **yes, native** | `mlx.launch --hostfile`, `--pipeline` (depth) or tensor-parallel (width) | Apple Silicon only | MIT | active, Apple-backed; WWDC26 session 233 | **BUILD ON (Apple track).** User is on a Mac. Best latency ceiling of anything here. |
| **SGLang** ≥0.5.7 (v0.5.8, Jan 2026) | **RadixAttention** (§4.1) + **chunked pipeline parallelism** (§4.2) | yes | `--pp-size N --nnodes N --node-rank i --dist-init-addr` | NVIDIA/AMD GPU | Apache-2.0 | production, top-tier | **STEAL IDEAS.** Its PP-with-chunking is the real answer to our bubble problem. Not runnable on our CPU demo box. |
| **TensorRT-LLM** v1.0 | AOT kernel fusion + in-flight (continuous) batching in a C++ `BatchManager`; PyTorch backend now default | yes (TP/PP/hybrid) | MPI/NCCL ranks | **NVIDIA only**, engine build per GPU SKU | Apache-2.0 | production, fastest on NVIDIA | **REJECT for us.** NVIDIA-only kills "commodity heterogeneous nodes", our entire thesis. Cite as the perf ceiling. |
| **HF TGI** | Rust router + Python shards; popularized continuous batching in OSS | TP only | sharded launcher | GPU | Apache-2.0 | **archived / maintenance mode (~Mar 2026)** | **REJECT.** Dead end. Naming it as "the alternative" on a slide dates the deck. |
| **DeepSpeed-Inference / MII** | **Dynamic SplitFuse** (FastGen): split long prompts + fuse with decodes into fixed-size forward batches | TP (+ ZeRO-Inference offload) | MPI/`deepspeed` launcher | GPU | Apache-2.0 | public dev slowed 2025→2026 | **REJECT as runtime, STEAL SplitFuse.** It is the cleanest published statement of the chunk-and-fuse idea T3 needs. |
| **LMDeploy** | two backends in one CLI: **TurboMind** (hand-written CUDA, persistent batch) + PyTorch backend | TP | `--tp` | NVIDIA (TurboMind) | Apache-2.0 | production (InternLM) | Ignore. No layer-across-hosts story; nothing unique for us. |
| **Ollama** | UX layer; **since 2025 its own Go engine over ggml**, no longer llama.cpp | **no** | none | CPU/Metal/CUDA | MIT | v0.30.x, huge install base | **Distribution channel, not architecture.** Relevant only as "what a node operator already has installed". |
| **ExecuTorch 1.0** | PyTorch export → flatbuffer `.pte`, ahead-of-time memory planning, no Python at runtime | no | none | phone / MCU / edge | BSD-3 | 1.0 GA | Out of scope now. **v2 relevance: this is what a phone-class node would run.** |
| **ONNX Runtime + onnxruntime-genai** 0.15.2 | graph-level EP abstraction; one model, many accelerators | no (partitions to EPs, not hosts) | none | anything | MIT | mature | Ignore for serving. Useful only if we ever need Windows/ARM/WASM nodes. |
| **Ray Serve + Ray Data** | actor/task placement over a heterogeneous cluster; `ray.serve.llm` wraps vLLM engines | orchestrates, doesn't split | native, its whole point | any | Apache-2.0 | mature; already in our repo (`ray-vllm/`) | **BUILD ON (control plane, v2).** Already scaffolded in the PoC. |
| **KServe** v0.20.0 (6 Aug 2026) | K8s CRD for inference; v0.20 adds an **LLM-specific CRD backed by llm-d** + disaggregated serving + Envoy AI Gateway | orchestrates | K8s-native | K8s | Apache-2.0 | CNCF incubating, mature | v2 only. Overkill for a 3-container demo. |
| **Triton Inference Server** | multi-framework model server, C API backends, ensembles | no | K8s | any | BSD-3 | **folded into Dynamo, now "Dynamo-Triton"**; new investment goes to Dynamo | **REJECT.** Superseded. Do not put on a 2026 slide as current. |
| **NVIDIA Dynamo** 1.0 GA (16 Mar 2026) | **disaggregated prefill/decode**, KV-aware router, hierarchical KV offload, NIXL transfer | orchestrates engines | datacenter scale | NVIDIA-centric | Apache-2.0 | 1.0 GA | **Architectural north star for v2.** P/D disaggregation *is* heterogeneous-node scheduling. Too heavy to demo. |
| **llm-d** | K8s-native vLLM at scale; **NIXL** for KV transfer between prefill and decode pods | orchestrates | K8s + Inference Gateway | GPU + K8s | Apache-2.0 | Red-Hat-led, shipping (KServe 0.20 integration) | v2, and only if we go K8s. |
| **AIBrix** v0.7.0 (16 Jun 2026) | ByteDance control plane, now under `vllm-project`; LLM-aware autoscaling, LoRA mgmt, **KV-cache-centric P/D data plane**, multi-engine (vLLM/SGLang/TRT-LLM) | orchestrates | K8s | K8s | Apache-2.0 | 242 PRs in the v0.7 cycle | v2 alternative to llm-d. Its **multi-engine** stance matches our heterogeneous-node story better than llm-d's vLLM-only stance. |

---

## 2. Deep dive — `llama.cpp` RPC backend (the thing we are re-inventing)

### 2.1 Architecture: it is a *backend*, not a server

ggml talks to hardware through pluggable backends (CPU, Metal, CUDA, Vulkan, SYCL). `GGML_RPC=ON` adds one more:
`RPC`, whose "device" is a TCP endpoint. `ggml-rpc-server` on the remote box exposes *its* local ggml devices.
The consequence is the single biggest difference from our PoC:

> **llama.cpp does not chain three model servers. It builds ONE compute graph on the client, lets the ggml
> backend scheduler cut it at the layer boundary, and ships the far side of the cut to a remote device.**

Weights upload **once at load time** into remote buffers (`ALLOC_BUFFER` + `SET_TENSOR`); `rpc-server -c` caches
them on the server's disk so a restart skips the re-upload. Per token only the *graph* and the *boundary tensor*
move. Layer assignment: weights **and KV cache** spread across all devices, local and remote, in proportion to
free memory; `--tensor-split` overrides, `-ngl` bounds how many layers leave the CPU. The split unit is the
**repeating layer block** — exactly our node0/1/2 split, chosen by a memory heuristic instead of an env var.

### 2.2 Wire format (read from `ggml/src/ggml-rpc/ggml-rpc.cpp`, master)

Framing is deliberately dumb, which is why it is fast:

```
every message:  [ uint64 little-endian length ][ payload ]      (send_msg / recv_msg)
handshake:      RPC_CMD_HELLO (== 14, static_assert'd) -> {major, minor, patch, conn_caps}
```

18 commands: `ALLOC_BUFFER, GET_ALIGNMENT, GET_MAX_SIZE, BUFFER_GET_BASE, FREE_BUFFER, BUFFER_CLEAR, SET_TENSOR,
SET_TENSOR_HASH, GET_TENSOR, COPY_TENSOR, GRAPH_COMPUTE, GET_DEVICE_MEMORY, INIT_TENSOR, GET_ALLOC_SIZE, HELLO,
DEVICE_COUNT, GRAPH_RECOMPUTE, MEMSET_TENSOR`.

`ggml_tensor` → `rpc_tensor`, `#pragma pack(push,1)`, `static_assert(sizeof % 8 == 0)`:

| field | bytes | | field | bytes |
|---|---|---|---|---|
| `id` | 8 | | `flags` | 4 |
| `type` | 4 | | `src[10]` | 80 |
| `buffer` | 8 | | `view_src` | 8 |
| `ne[4]` | 16 | | `view_offs` | 8 |
| `nb[4]` | 16 | | `data` | 8 |
| `op` | 4 | | `name[64]` | 64 |
| `op_params[16]` | 64 | | `use_count` | 4 |
| | | | **total** | **296 B** (derived, 296 % 8 = 0 ✓) |

`GRAPH_COMPUTE` payload:
`| device u32 | n_nodes u32 | nodes[n_nodes] u64 | n_tensors u32 | tensors[n_tensors] × 296 B |`

**No JSON. No base64. No text parsing anywhere.** That alone is the delta our PoC needs (see FINDING 4).

### 2.3 The arithmetic that matters — and why `GRAPH_RECOMPUTE` exists

For an 8-layer Qwen2.5-0.5B shard, a llama-family decode graph is ≈35 ggml nodes/layer, ≈2.5 tensors touched per
node **(modelled)**:

```
n_nodes   = 35 × 8            = 280
n_tensors = 280 × 2.5         = 700
payload   = 4 + 4 + 280×8 + 4 + 700×296 = 209,452 B ≈ 205 KiB   (modelled)
```

Compare our PoC's per-hop hidden state at seq_len 32: 114,688 B raw / 152,920 B base64 **(derived)**.

> **The graph description is bigger than the activation.** llama.cpp hit this wall and fixed it with
> `RPC_CMD_GRAPH_RECOMPUTE`: `cgraph->uid` is compared against `last_graph_uid`; on a match the client sends
> `{device: u32}` — **4 bytes** — instead of re-serializing 205 KiB. Steady-state decode re-uses the same graph
> shape every step, so this fires on essentially every token after the first.

Second dedup: `HASH_THRESHOLD = 10 MiB`. Above it, `SET_TENSOR_HASH` sends an FNV-1a 64 hash first; the server
answers "already have it" and the bytes never cross. Content-addressed transfer, ~20 lines.

Third: `send_async` on `SET_TENSOR`/`GET_TENSOR`/`GRAPH_COMPUTE` — request pipelining on a persistent socket, no
stop-and-wait. Our PoC constructs `httpx.AsyncClient()` *inside* every forward call (defect #5): 3 fresh TCP
handshakes per token vs. llama.cpp's one socket per peer for the process lifetime.

Transport: TCP by default; built with `GGML_RPC_RDMA` it prints `transport : TCP (RDMA auto-negotiate enabled)`
and auto-negotiates RDMA, `GGML_RPC_NO_RDMA=1` forces TCP. Debug via `GGML_RPC_DEBUG=1`.

### 2.4 The measured number that must go in the deck

Third-party benchmark, Mac Studio M2 Ultra (Metal) + DGX Spark GB10 (CUDA), 10 GbE direct link measured at
9.41 Gbps, llama.cpp RPC **(measured, third-party — reproduce before quoting on a slide)**:

| model | mode | prefill tok/s | decode tok/s |
|---|---|---|---|
| Qwen2.5-7B-Instruct Q4_K_M | local Metal only | 76.1 | **91.8** |
| | RPC Metal+CUDA | **317.7** (4.17x) | 52.7 (**0.574x → 1.74x slower**) |
| Qwen2.5-72B-Instruct Q4_K_M | local Metal only | 28.2 | **11.1** |
| | RPC Metal+CUDA | 29.5 (1.05x) | 5.9 (**0.532x → 1.88x slower**) |

Author's conclusion, and ours: *RPC's value is capacity, not speed.* Per-token round trip ≈0.17 ms; at 1 GbE
expect ~10x that network overhead on decode.

> **Slide implication.** The mature, C++, binary-protocol, persistent-socket, zero-copy version of our PoC still
> loses 1.7–1.9x on decode. Any slide claiming "decentralized = faster" is falsifiable in 5 minutes by a judge
> who knows this repo. **Claim: run a model that does not fit on one node, at a fraction of the latency penalty
> you'd expect** — and back it with the 4.17x *prefill* win, which is real and which our pipeline also gets.

### 2.5 Five things to steal, verbatim (all **v1**, all ≤1 day each)

| # | llama.cpp mechanism | Our fix | Win |
|---|---|---|---|
| 1 | `[u64 len][payload]` binary framing | drop JSON+base64 | 1.33x + no parse of a 2.4 MB string |
| 2 | one persistent socket per peer, `send_async` | hoist `httpx.AsyncClient` to module scope, `http2=True` | kills 3 TCP handshakes/token (defect #5) |
| 3 | `GRAPH_RECOMPUTE` (uid match → 4 B) | send shape/dtype header once per session, then raw bytes | header ~0 |
| 4 | `SET_TENSOR_HASH` + FNV-1a over `HASH_THRESHOLD` | dedup repeated prompt prefixes across requests | free prefix cache |
| 5 | `HELLO` with strict `{major,minor,patch}` check | version field in our node handshake | prevents the silent-corruption class of bug ("mismatched builds hang at handshake or crash mid-inference") |

**Do not adopt the RPC backend itself.** Upstream text: *"the functionality is fragile and insecure"*, do not run
on open networks. A `rpc-server` executes arbitrary remote graphs against local memory with no authentication.

---

## 3. Deep dive — MLX / `mlx-lm` (the Mac track, and the user is on a Mac)

`mx.distributed.init(backend=...)`, four backends:

| backend | transport | note |
|---|---|---|
| `mpi` | OpenMPI | mature; `mlx.launch --backend mpi` fixes the `libmpi.dyld` path pain |
| `ring` | **TCP sockets, linear neighbour topology** | "always available and usually faster than MPI"; **`send`/`recv` restricted to adjacent ranks** |
| `jaccl` | **RDMA over Thunderbolt, macOS 26.2+** | needs a full mesh of direct TB cables; RDMA enable requires recovery-mode config |
| `nccl` | CUDA | not our path |

`mlx-lm` pipeline parallelism is **literally our PoC's dataflow**: ranks after the first `mx.distributed.recv_like`
from `rank+1`; ranks before the last `mx.distributed.send` to `rank-1`; `all_gather` to sync when `pipeline_size>1`.
In server mode only rank 0 runs the HTTP server, shares the request via `_share_object`, and returns the response —
our coordinator, minus JSON. Tooling: `mlx.distributed_config` auto-configures Thunderbolt interfaces and writes the
hostfile; `mlx.launch --hostfile hosts.json script.py`; `--pipeline` shards by depth.
WWDC26 s233 **(vendor-reported, unverified)**: 27B Qwen on 4× M3 Ultra ≈3x single-machine generation rate; a
1T-param Kimi 2.6 demo across 4 Macs.

**Why this beats everything else in the table for us:** the ring backend's adjacent-ranks-only restriction is not a
limitation for a layer pipeline — a layer pipeline *is* a ring. And Thunderbolt 5 at 80 Gb/s = 10 GB/s makes our
per-hop payload (bf16 single position 1792 B, 3 hops = 5376 B/token) **~0.5 µs of wire time (modelled)**. The link
stops being the story; only per-hop latency remains.

---

## 4. SGLang — the two ideas worth stealing

### 4.1 RadixAttention — what it actually is

PagedAttention (vLLM) answers *"how do I allocate KV memory without fragmentation?"* → non-contiguous blocks.
RadixAttention answers a **different** question: *"how do I reuse KV across requests?"*

Mechanism: the KV cache for **prompts and completions** is retained after a request finishes, indexed in a
**radix tree (compressed trie)** whose edges carry *variable-length token-id runs* rather than single tokens.
A new request walks the tree from the root along its token ids; the matched prefix's KV blocks are reused
verbatim and only the divergent suffix is computed. Eviction is **LRU from the leaves** — evicting a leaf can
make its parent a leaf and thus a future candidate — and a **cache-aware scheduler** reorders the pending queue
to group requests sharing a prefix, raising the hit rate rather than merely exploiting it.
Configurable via `--radix-eviction-policy` (LRU/LFU).

**Relevance to us (v2):** in a layer pipeline every shard holds *its own* KV for the *same* token prefix. One
radix tree per shard, keyed on the identical token-id path, gives coherent cross-shard prefix reuse for free —
the coordinator only needs to broadcast the matched prefix length. Nothing about the tree is GPU-specific.

### 4.2 Chunked pipeline parallelism (LMSYS, 15 Jan 2026; SGLang ≥0.5.7)

The direct answer to our defect #6 (2 of 3 nodes idle). Long prompts are cut into 4K–12K chunks that flow
stage-by-stage, so stage 1 starts chunk 2 while stage 2 works chunk 1. Two mechanisms:
**async P2P** (`async_send` returns a `P2PWork` handle, sync deferred, CPU metadata work overlaps the transfer)
and **dynamic chunking** (chunk size shrunk along a quadratic model of sequence length, so per-stage times stay
aligned as attention cost grows). Flags: `--pp-size`, `--chunked-prefill-size`, `--enable-dynamic-chunking`,
`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR` (0.6–0.85).

Reported **(vendor/lab-reported)**: PP4·TP8 beats PP1·TP32 by 30.5% throughput on DeepSeek-V3.1; 82.8% strong-
scaling efficiency at PP4, 76.9% at PP8 (Qwen); TTFT 55.5 s → 10.5 s (−81.1%) from PP1 to PP8.

Note the shape of that result: **the pipeline win is in prefill/TTFT, not decode.** Same shape as §2.4. Two
independent systems, same conclusion. Our story should be built on it, not against it.

---

## 5. Three-tier recommendation

### Tier 1 — DEMO ON (v1, days, CPU, docker-compose, 3 nodes)

**Keep the custom Python chain.** Adopting a real runtime deletes the demo's entire point — every runtime here
either can't split across hosts, or splits across hosts *by itself*, at which point we are demoing llama.cpp.
Harden it with §2.5's five steals plus the KV cache (FINDING 3, 271x) and the balanced split (FINDING 1, 1.55x).
Both are the highest-value hours available and neither needs a new dependency.

Add **one** third-party comparison run so the demo has a credible axis: `vllm serve --pipeline-parallel-size 2`
from the existing `ray-vllm/` scaffold (already written, CPU image, ARM64-native on Apple Silicon).

### Tier 2 — BUILD THE REAL PRODUCT ON (v2, months)

Two forks; pick by target hardware, and say which on the slide.

| track | engine per node | control plane | why |
|---|---|---|---|
| **x86/NVIDIA fleet** | vLLM (T4-A1) or SGLang | **Ray Serve** first (already scaffolded), **AIBrix** at K8s scale | AIBrix over llm-d: **multi-engine** (vLLM/SGLang/TRT-LLM) matches heterogeneous nodes; llm-d is vLLM-only |
| **Apple Silicon mesh** | **`mlx-lm --pipeline`** | thin coordinator over `mlx.launch` | JACCL/Thunderbolt RDMA is the only sub-10 µs hop available to a consumer; `send`/`recv_like` is already our dataflow |

Architecture to converge on regardless of track: **NVIDIA Dynamo 1.0's** disaggregated prefill/decode + KV-aware
routing. P/D disaggregation is the correct generalisation of "heterogeneous nodes with different strengths" —
prefill is compute-bound and parallelises across our chain, decode is latency-bound and does not. Steal the
pattern, don't adopt the (NVIDIA-centric) implementation.

### Tier 3 — BENCHMARK AGAINST (for credibility; this is what a judge will ask)

| # | baseline | command | what it proves |
|---|---|---|---|
| 1 | **llama.cpp `--rpc`, 3× `rpc-server`** | `llama-cli -ngl 99 --rpc h0:50052,h1:50053,h2:50054` | the honest peer. Same split, C++, binary wire. If we are within 2x of it we have a result. |
| 2 | single-process llama.cpp, same box | `llama-cli -ngl 99` | the ceiling. Quantifies our distribution tax the way §2.4 does. |
| 3 | vLLM `--pipeline-parallel-size 2/3`, one box | existing `ray-vllm/start-head.sh` | what a production runtime's pipeline costs without a network |
| 4 | `mlx-lm --pipeline`, 2 Macs | `mlx.launch --hostfile hosts.json` | the Apple-track ceiling |
| 5 | Petals | — | the WAN-scale peer (T1's territory) |

**Do not benchmark against TGI or Triton.** TGI is archived; Triton is folded into Dynamo. Citing either as
"current" in Sep 2026 is the cheapest way to lose an infra-literate judge.

---

## 6. Sources

**llama.cpp** [tools/rpc README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md) · [ggml-rpc.cpp, master, read directly](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp) · [DeepWiki 8.4](https://deepwiki.com/ggml-org/llama.cpp/8.4-multi-gpu-and-distributed-inference) · [measured RPC benchmarks](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks) · [discussion #9136](https://github.com/ggml-org/llama.cpp/discussions/9136).
**MLX** [distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) · [launching](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html) · [WWDC26 s233](https://developer.apple.com/videos/play/wwdc2026/233/) · [mlx-lm distributed](https://deepwiki.com/ml-explore/mlx-lm/7.5-distributed-execution).
**SGLang** [RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) · [chunked PP, Jan 2026](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/) · [PP docs](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pipeline_parallelism.md).
**Rest** [TRT-LLM v1.0](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.0.0) · [DeepSpeed-FastGen](https://arxiv.org/html/2401.08671v1) · [LMDeploy](https://github.com/InternLM/lmdeploy) · [Ollama #9959](https://github.com/ollama/ollama/issues/9959) · [ExecuTorch 1.0](https://pytorch.org/blog/introducing-executorch-1-0/) · [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) · [Ray Serve LLM](https://docs.ray.io/en/latest/serve/llm/index.html) · [KServe releases](https://github.com/kserve/kserve/releases) · [KServe+llm-d](https://developers.redhat.com/articles/2026/04/21/kserve-llm-d-optimized-gen-ai-inference) · [Dynamo](https://github.com/ai-dynamo/dynamo) · [Dynamo-Triton](https://developer.nvidia.com/dynamo-triton) · [AIBrix](https://github.com/vllm-project/aibrix).
