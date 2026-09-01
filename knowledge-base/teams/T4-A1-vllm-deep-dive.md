---
team: T4 — Infrastructure & Serving Runtimes
agent: T4-A1
topic: vLLM in depth — PagedAttention, continuous batching, V1 engine, and whether its pipeline parallelism replaces the DecentralizedLLM custom stack
headline: "Buy vLLM's intra-node engine, build the inter-node fabric: vLLM PP is a correct, fast, layer-splitting implementation that assumes exactly the one thing this project denies — a trusted, homogeneous, co-located cluster on one control plane — and its per-decode-step network floor of (N-1)×RTT/2 makes geo-distributed PP latency-bound, not bandwidth-bound, which is also why activation compression is nearly pointless for a 0.5B model on a LAN."
---

# T4-A1 — vLLM deep dive

Reference version: **vLLM v0.27.0** (released 2026-08-10). All flags below checked against
`docs.vllm.ai/en/stable` and the `main` branch docs, not from memory. Numbers tagged (measured) /
(modelled) / (documented) — *documented* = quoted from vLLM docs, not benchmarked by me.

## 1. What vLLM actually solves

| Mechanism | Problem it kills | Relevance to our PoC |
|---|---|---|
| **PagedAttention** | KV cache stored contiguously ⇒ internal + external fragmentation; you must pre-reserve `max_seq_len` per request. Paging into fixed blocks (default 16 tokens) with a block table cuts waste to <4% and enables copy-on-write sharing across beams/forks. | **We have no KV cache at all** (FINDING 3, 271× redundant compute). We do not have a fragmentation problem yet — we have a *no cache* problem. PagedAttention is the wrong first lesson to copy; a plain per-node dict cache is. |
| **Continuous batching** (Orca-style iteration-level scheduling) | Static batching wastes the tail: whole batch waits for the longest sequence. Continuous batching admits/retires requests every decode step. | Directly fixes shared-context defect #8 (no queue/admission control) and #6 (zero batching). This is the single most valuable idea to steal. |
| **V1 engine** (`AsyncLLM` ↔ `EngineCore` over ZeroMQ IPC, separate processes) | Python GIL contention between HTTP/tokenisation and model execution. V1 also collapses the prefill/decode distinction — the scheduler is a plain `{request_id: num_tokens}` dict, which is what makes chunked prefill and prefix caching fall out for free. | Our coordinator is a single asyncio process doing tokenise + HTTP + argmax. The `AsyncLLM`/`EngineCore` split is the right shape to copy for v2. |
| **Prefix caching** (`--enable-prefix-caching`, on by default in V1) | Shared system prompts re-prefilled per request. Hash-keyed block reuse. | Free win once we have any KV cache. Our chat template prefix is identical across requests. |

## 2. Tensor parallelism vs pipeline parallelism

| | `--tensor-parallel-size` (TP) | `--pipeline-parallel-size` (PP) |
|---|---|---|
| Splits | Each layer, across the hidden dim (Megatron-LM style) | Whole layers, into contiguous stages |
| Comms per layer | 2× **all-reduce** per transformer block | none |
| Comms per stage boundary | — | 1× **point-to-point send/recv** |
| Comms volume, decode step, batch B | `2 × L × B × H × dtype` = for us `2×24×B×896×2` = **86,016·B B** | `(N-1) × B × 2H × dtype` = for us with N=3 **7,168·B B** (modelled) |
| Interconnect need | NVLink / InfiniBand. Docs: *"Efficient tensor parallelism requires fast internode communication, preferably through high-speed network adapters such as InfiniBand."* | Tolerant. Docs recommend PP over TP for nodes without NVLINK. |
| Latency | Scales *down* with more ranks | Scales *up* — stages are sequential |
| Uneven splits | No | **Yes** |

> PP moves **12× fewer bytes** than TP for Qwen2.5-0.5B at N=3 (86,016 / 7,168, modelled) and needs
> no collective. **PP is the only parallelism that survives a real network.** That is why this
> project is a PP project, and vLLM agrees with us on the strategy — it just disagrees on the threat model.

## 3. How vLLM implements PP (the part that matters)

**Sharding.** A model must implement the `SupportsPP` interface. `make_layers()` builds the full layer
list but instantiates only this rank's slice; every other position becomes a `PPMissingLayer` stub —
a real object, so `state_dict` indices stay aligned. Weight loading skips missing layers. Consequences:

- The **embedding lives on rank 0, the `lm_head` on rank N-1**. This reproduces our FINDING 1 imbalance
  *exactly*: `lm_head` is 136.1 M of 494 M params = 9.13 layer-equivalents, so a naive PP=3 on 24 layers
  gives stages of 8 / 8 / 17.13 eq → **1.55× slower than balanced** (derived, same arithmetic as FINDING 1).
- vLLM ships the escape hatch: **`VLLM_PP_LAYER_PARTITION="11,11,2"`** (env var, comma-separated layer counts
  per stage, must sum to `num_hidden_layers`). 11 / 11 / (2 + 9.13 lm_head) = 11.00 / 11.00 / 11.13 eq.
  **Any PP baseline run without this env var is a 1.55× unfair comparison and must be reported as such.**
- Models whose layers don't divide evenly are fine — docs: PP *"splits the model along layers and supports
  uneven splits."* The "layers must be divisible" folklore is **false** for vLLM. Divisibility only matters
  in that the *default* partition is even.

**What crosses the wire.** Not the logits, and not a raw tensor — an `IntermediateTensors`, a dict sent
via `GroupCoordinator.send_tensor_dict` / `recv_tensor_dict`. For Qwen2 the dict is
`{"hidden_states": [T, H], "residual": [T, H]}` — **two** H-vectors, because vLLM keeps the residual stream
un-added for fused add-RMSNorm. So the real per-token per-hop cost is `2 × H × dtype`, not `H × dtype`:

| dtype | per token per hop | vs our v0 (`H × fp32` = 3,584 B, un-base64'd) |
|---|---|---|
| bf16 | **3,584 B** | 1.00× — identical |
| fp32 (the macOS/CPU path) | **7,168 B** | 2.00× **worse** |

> Do not put "vLLM PP is more efficient on the wire per token" on a slide. It is **not**. Its win is
> that it sends *one* position instead of the whole sequence (because it has a KV cache), and that it
> never sends logits back (sampling runs on the last rank). Those are FINDING 3 and FINDING 2 — both
> already on our fix list, both achievable without adopting vLLM.

**Transport.** `GroupCoordinator` wraps a `torch.distributed` ProcessGroup and holds *two* channels: a
device channel (**NCCL** on CUDA, via `PyNcclCommunicator`) and a CPU channel (**Gloo**). On the CPU
backend the PP send/recv is **Gloo over TCP**. There is no compression, no quantisation of the
inter-stage tensor, and no application-level framing you can hook — it is `torch.distributed` P2P.

**Executor.**

| Backend | Flag | Scope | How it works |
|---|---|---|---|
| multiprocessing | `--distributed-executor-backend mp` | Default when `PP×TP ≤ local devices` | One process per rank, fork/spawn on the local host. Also supports **multi-node** via `--nnodes` / `--node-rank` / `--master-addr` / `--headless` on the workers. |
| Ray | `--distributed-executor-backend ray` | Default for multi-node; needs `pip install "ray[cgraph]"` | Ranks are Ray actors placed by a placement group. `examples/ray_serving/run_cluster.sh` starts head/worker containers. |

**Cross-NODE PP: yes, supported, and it is a first-class documented path.** Both `mp --headless` and `ray`
work across machines. So the raw capability "one LLM, layers split across 3 physical boxes" **already exists
off the shelf.** We must say this out loud in the deck rather than let a judge discover it.

**V0 vs V1 PP.** V0 used *virtual engines*: PP_size independent schedulers + block managers, so N batches
could be in flight and the pipeline stayed full. V1 replaced this with a single global scheduler and
async microbatch dispatch (RFC #11945) — better global optimisation, but PP was the last feature to land
on V1 and remains the roughest edge. Known frictions to expect when benchmarking:

| Limitation | Effect |
|---|---|
| PP couples activation transfer with input scheduling metadata | ~17% of execution time in CPU-side input prep (published third-party measurement, SqueezeBits — not measured by me) |
| Pipeline bubble with 1 request in flight | Utilisation = **1/N = 33%** at N=3 (modelled). Identical to our PoC's ceiling. PP only pays off *under concurrency*. |
| PP + chunked prefill | A chunked prefill occupies the pipeline for multiple steps; interacts badly with in-flight microbatch accounting. Set `--enforce-eager` and fix `--max-num-batched-tokens` when benchmarking to remove this variable. |
| `PPMissingLayer` + quantisation | Known class of bugs (e.g. bitsandbytes PR #10200) — quant state setup trips on stub layers. Avoid quant + PP in a baseline. |
| No fault tolerance | Gloo/NCCL collectives assume all ranks live. One rank dies ⇒ the group hangs to timeout, then the whole server dies. |

## 4. The rest of the feature surface (one line each)

| Feature | Flag / API (v0.27.0) | Verdict for us |
|---|---|---|
| OpenAI server | `vllm serve <model>` → `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/metrics` | Our gateway already speaks this. **Baseline is drop-in**: point `VLLM_URL` at it. |
| Quantisation | AWQ, GPTQ, FP8 (W8A8, needs Hopper+ for native), bitsandbytes, GGUF; `--quantization` | Irrelevant at 0.5 B. Also collides with PP (above). Skip. |
| Speculative decoding | `--speculative-config '{"method":"ngram","num_speculative_tokens":3}'`; methods: `ngram`, `eagle`, `eagle3`, `medusa`, `mlp_speculator`, `draft_model`, `deepseek_mtp`, `qwen3_next_mtp`. The old `--speculative-model` flag is **gone**. | **Genuinely interesting for us**: spec-decode amortises N accepted tokens over *one* pipeline traversal. On a 60 ms-RTT link, 3 accepted tokens per traversal ≈ 3× effective tok/s. Best idea in vLLM for a geo-distributed pipeline. **v2.** |
| LoRA | `--enable-lora --max-loras N`; per-request adapter id | Multi-tenant story for v2. Not a hackathon feature. |
| Structured output | `--structured-outputs-config.backend {auto,xgrammar,guidance,outlines}`, default `auto`; `guided_json` per request | Orthogonal. Ignore. |
| Disaggregated prefill | `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both",...}'`; NIXL runs over RDMA/UCX-TCP/NVMe-oF/S3 | **The closest vLLM primitive to what we want** — a pluggable, network-transported KV/state connector between *separate vLLM instances*. It moves KV caches, not layer activations, so it does not do our job, but `NixlConnector` is the right thing to cite as prior art for T1's fast path. |

## 5. CPU / Apple Silicon — the honest answer

| Question | Answer |
|---|---|
| Does vLLM run on CPU? | Yes. Supported: x86 AVX512 (recommended) / AVX2 (limited), ARM AArch64 + NEON, Apple silicon (macOS Sonoma+), IBM Z s390x. |
| Prebuilt images? | `vllm/vllm-openai-cpu:latest-arm64` and `:latest-x86_64`. |
| Does the CPU backend do PP? | **Yes — documented verbatim:** *"vLLM CPU supports data parallel (DP), tensor parallel (TP) and pipeline parallel (PP) to leverage multiple CPU sockets and memory nodes."* |
| Native macOS (not container)? | Build from source; **fp32 and fp16 only** on the macOS CPU path. Prefer the Linux-arm64 container on Docker Desktop, which runs natively on M-series. |
| Apple GPU? | **`vllm-metal`** — official `vllm-project` community hardware plugin, MLX backend, zero-copy on unified memory, paged attention (experimental), GQA, spec-decode, GGUF. Requires **native arm64 Python 3.12** (Rosetta unsupported). v0.2.0 claims 83× TTFT / 3.6× throughput over v0.1.0 (vendor-reported, not measured by me). **It does not document TP or PP or any distributed/multi-node support — treat cross-node PP on `vllm-metal` as unavailable.** The PoC's `sprint.md` already went down this road; that is why it ended up as 3 proxies in front of *one* vLLM, which is not model splitting. |
| Key env vars | `VLLM_CPU_KVCACHE_SPACE` (GiB), `VLLM_CPU_OMP_THREADS_BIND` (core list \| `auto` \| `nobind`), `VLLM_CPU_NUM_OF_RESERVED_CPU`. Docker reports no NUMA nodes ⇒ **`nobind` is required**, as the PoC's `start-head.sh` already discovered. |
| dtype gotchas | fp16 is unstable on torch CPU; bf16 preferred but on ARM without the BF16 ISA extension it may be emulated and *slower* than fp32. **Benchmark both.** AMD Zen rejects fp16 outright. |

## 6. The decisive analysis — what each stack gives that the other cannot

| Capability | vLLM PP | DecentralizedLLM custom stack |
|---|---|---|
| PagedAttention, continuous batching, prefix caching, spec-decode | ✅ years of work | ❌ none of it |
| Cross-node layer split | ✅ `mp --headless` or `ray` | ✅ (that's the whole PoC) |
| Uneven / capability-proportional split | ✅ `VLLM_PP_LAYER_PARTITION` (static, boot-time) | ✅ and could be **dynamic** |
| **Trust model** | **Single trust domain.** All ranks join one `torch.distributed` group; any rank can `send`/`recv` arbitrary tensors and read the full weight-loading path. There is no authn, no authz, no attestation, no encryption on the PP channel. | The premise. Per-node API keys today; attestation/verification is the v2 product. |
| **Homogeneity** | Required: same model, same vLLM version, same dtype, same CUDA/CPU platform on every rank. Heterogeneous TP-per-stage is an **open feature request** (issue #27239). | Heterogeneous by design — HTTP nodes, any hardware. |
| **Membership** | Static. Ranks fixed at boot. Adding/removing a node = full restart. | Can be dynamic (T1-A5's topology discovery). |
| **Fault tolerance** | **None.** One rank down ⇒ collective hangs ⇒ server dies. No replicas, no re-shard. | Also none today — but it is *addressable*, because the transport is stateless HTTP request/response, not a standing collective. **This is the single strongest architectural argument for the custom stack.** |
| Membership churn cost | Restart the world | Re-route one hop |
| WAN / NAT / firewall traversal | Gloo/NCCL over raw TCP between all ranks; needs `VLLM_HOST_IP`, open ports, flat routable network | HTTP/1.1 — traverses anything |
| Engineering cost to reach parity | — | 18+ months |

### The number that decides the architecture

PP with N stages costs **(N-1) one-way network crossings per decode step**. One-way ≈ RTT/2, so for N=3
the floor is exactly **1 × RTT per token**, and the token-rate ceiling is `1000 / RTT_ms` — *independent of
model size, compute, dtype, and any compression scheme* (modelled):

| Link | RTT | Floor per decode step (N=3) | tok/s ceiling |
|---|---|---|---|
| loopback | 0.04 ms | 0.04 ms | 25,000 |
| 10 GbE LAN | 0.10 ms | 0.10 ms | 10,000 |
| 1 GbE LAN | 0.40 ms | 0.40 ms | 2,500 |
| metro WAN (~50 km) | 10 ms | 10 ms | **100** |
| US coast-to-coast | 60 ms | 60 ms | **16.7** |
| intercontinental | 250 ms | 250 ms | **4.0** |

Now the bandwidth side, same configuration. Per link per decode step = `B × 2H × 2 B` = `B × 3,584 B`.
Saturating 1 GbE (125 MB/s) needs `B × steps_per_sec = 34,877` — i.e. **batch 349 at 100 steps/s** (modelled).

> **Conclusion, and this is the deck slide:** for a 0.5 B model, inter-stage PP traffic is
> **latency-bound at batch 1 and does not become bandwidth-bound until batch ~349 on 1 GbE.**
> Compression of PP activations therefore buys **nothing** on a LAN at demo concurrency. It matters only
> (a) on WAN uplinks, (b) at high batch, or (c) for large-H models — Qwen 70B's H=8192 is 9.1× our
> per-token cost. Our v0's 935× wire reduction (FINDING 4) is real but is almost entirely *not* about
> compression: it is KV caching + not shipping logits + dropping base64. **State this honestly or T2's
> compression work will be attacked as a solution to a non-problem.**

### Buy vs build — explicit call

- **BUY vLLM as the per-node execution engine, for v2.** Do not reimplement PagedAttention, continuous
  batching, prefix caching, or a sampler. That is the 18-month part and it is free.
- **BUILD the inter-node fabric.** vLLM PP is a *cluster* feature: one trust domain, one control plane,
  static homogeneous membership, fail-stop. Our premise — untrusted, heterogeneous, geo-distributed,
  churning nodes — is not a configuration of vLLM PP; it is a different system whose failure and trust
  model vLLM has no representation for. There is no flag to buy.
- **The blocker for the hybrid (v2, be honest about it):** vLLM exposes no "run only my layers on this
  activation and hand back the hidden state" API. `IntermediateTensors` P2P is internal to the
  `torch.distributed` group; there is no HTTP surface for it. Wrapping vLLM per node requires either a
  custom `Executor`/`GroupCoordinator` shim, or a new KV-connector-style plugin modelled on
  `NixlConnector`. That is a real fork-or-plugin project, not an integration.
- **For v1 (hackathon): keep `node.py`. Use vLLM PP as the measured baseline, not as the product.**
  A custom stack that cannot state its number against vLLM is not credible; one that can — and that
  wins on *fault tolerance and heterogeneity* while losing on raw tok/s — is.

## 7. Baseline commands — 3-stage PP vLLM to benchmark against

**(A) Apples-to-apples throughput baseline, one container, 3 PP stages** (v1, ~1 hour):

```bash
docker run --rm -p 8000:8000 --shm-size=4g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_CPU_KVCACHE_SPACE=4 \
  -e VLLM_CPU_OMP_THREADS_BIND=nobind \
  -e VLLM_PP_LAYER_PARTITION="11,11,2" \
  vllm/vllm-openai-cpu:latest-arm64 \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --pipeline-parallel-size 3 \
    --distributed-executor-backend mp \
    --dtype bfloat16 --max-model-len 4096 --enforce-eager \
    --host 0.0.0.0 --port 8000
```
Run it a second time with `--dtype float32` and a third with `VLLM_PP_LAYER_PARTITION` unset — the
unset run is the 1.55×-imbalanced control that proves FINDING 1 applies to vLLM too.

**(B) 3 separate containers, matching the PoC topology** (v1, ~half a day). Extends the PoC's existing
`ray-vllm/` — which currently hardcodes `--pipeline-parallel-size 2`; change it to 3:

```bash
# head (rank 0)
docker run -d --name pp-head --network decentralized-net -p 8000:8000 --shm-size=4g \
  -e VLLM_HOST_IP=pp-head -e VLLM_CPU_OMP_THREADS_BIND=nobind \
  -e VLLM_PP_LAYER_PARTITION="11,11,2" \
  decentralizedllm-ray-vllm:latest \
  bash -c 'ray start --head --port=6379 && sleep 20 && \
    vllm serve Qwen/Qwen2.5-0.5B-Instruct \
      --pipeline-parallel-size 3 --distributed-executor-backend ray \
      --dtype bfloat16 --max-model-len 4096 --enforce-eager --host 0.0.0.0 --port 8000'

# workers (ranks 1,2) — repeat for pp-w2
docker run -d --name pp-w1 --network decentralized-net --shm-size=4g \
  -e VLLM_HOST_IP=pp-w1 -e RAY_HEAD_ADDRESS=pp-head:6379 \
  decentralizedllm-ray-vllm:latest ray start --address=pp-head:6379 --block

docker exec pp-head ray status     # must list 3 nodes BEFORE vllm serve binds
```

**(C) Drive both stacks with the same client.** Both expose `/v1/chat/completions`, so point the existing
gateway at each in turn (`VLLM_URL=http://pp-head:8000` vs `http://coordinator:8080`) and record, at
concurrency 1 / 4 / 16: TTFT p50/p99, tok/s, and — the row we actually win — **tok/s with one node killed.**

Pitfalls that will otherwise corrupt the baseline: `nobind` is mandatory (Docker exposes no NUMA nodes);
`--enforce-eager` on both sides or you are comparing a compile step; PP=3 with concurrency 1 idles 2 of 3
stages on *both* stacks, so a concurrency-1 comparison measures nothing about batching.

## Sources
[vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) ·
[vLLM CPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/) ·
[RFC: Pipeline-Parallelism for vLLM V1 (#11945)](https://github.com/vllm-project/vllm/issues/11945) ·
[vLLM V1 core architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) ·
[vllm-metal](https://docs.vllm.ai/projects/vllm-metal/en/latest/) ·
[Speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) ·
[Structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) ·
[NixlConnector usage](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/) ·
[Heterogeneous TP per PP stage (#27239)](https://github.com/vllm-project/vllm/issues/27239) ·
[SqueezeBits: vLLM vs TensorRT-LLM parallelism](https://blog.squeezebits.com/vllm-vs-tensorrtllm-9-parallelism-strategies-36310)
