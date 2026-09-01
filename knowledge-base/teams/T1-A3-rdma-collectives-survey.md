---
team: T1 — Transport & Protocol
agent: T1-A3
topic: High-performance transport survey (RDMA, RoCEv2, libfabric, UCX, NCCL/gloo, torch.distributed, MPI, DPDK, XDP, SR-IOV, GPUDirect, NVLink) — ruled in or out for DecentralizedLLM
headline: RDMA would save 27 µs per hop; deleting the Python HTTP+JSON+base64 stack saves 10,376 µs per hop. RDMA is a v2 answer to a problem we do not have yet. The v1 substitute is a persistent length-prefixed binary TCP socket — measured 346× cheaper per hop than v0 on this exact hardware, and it needs zero new hardware.
---

# T1-A3 — Transport layer survey and verdicts

## 0. Hardware reality check (measured, this machine, 2026-09-01)

This decides half the survey, so it goes first.

| Fact | Value | Source |
|---|---|---|
| Host | Apple M1 Pro (`T6000`), `arm64`, Darwin 25.6.0 | `uname -a`, `sysctl` (measured) |
| Wired NICs present | 2× USB 10/100/1000 LAN (`en4`, `en8`) = **1 GbE** | `networksetup -listallhardwareports` (measured) |
| Also present | **Thunderbolt Bridge** (`bridge0`, TB4) | `ifconfig bridge0` (measured) |
| PyTorch | 2.10.0 | (measured) |
| `torch.distributed` backends available | gloo ✅ · **nccl ❌** · **mpi ❌** | `dist.is_*_available()` (measured) |
| Docker | daemon not running; Desktop-on-macOS runs containers **inside a Linux VM** | (measured / architectural) |

Consequences, non-negotiable:

- **macOS has no `libibverbs`/RDMA stack, and an M1 Pro laptop has no PCIe slot for an RDMA HCA.** Every verbs-based option (IB, RoCEv2, iWARP, UCX-over-RDMA, GPUDirect) is not "hard" here — it is *physically unavailable*.
- **No CUDA ⇒ no NCCL, no NVLink, no GPUDirect.** `is_nccl_available() == False` is measured, not assumed.
- **Docker Desktop containers get a virtual NIC in a VM.** DPDK (needs to unbind a physical NIC from the kernel), XDP (needs a real NIC driver), and SR-IOV (needs IOMMU + NIC VFs) all have nothing to bind to.
- The one genuine **wired fast path this hardware already owns is Thunderbolt Bridge**: ~15–16.4 Gbps TCP peer-to-peer measured by third parties on TB4, vs 1 GbE on the USB adapters — **~120×** the bandwidth, cost £0. That is the "wired connection" story for the demo, not RoCEv2.

## 1. Where the time actually goes (measured, loopback, M1 Pro, payload = `[seq,896]` fp32, median of 100–200 reps)

Reproduces `node.py:/forward` wire format exactly, **transport only, no model compute**. Bench scripts in this session's scratchpad (`t1a3_bench.py`, `t1a3_gloo.py`, `t1a3_http.py`, `t1a3_httpraw.py`).

| # | Per-hop transport, seq=1 (3584 B) | RTT | vs v0 | Notes |
|---|---|---|---|---|
| 1 | **v0 as written**: new `httpx.AsyncClient()` per call + JSON + base64 | **10 406 µs** | 1.0× | shared-context defect #4 + #5 |
| 2 | pooled `AsyncClient` + JSON + base64 | 3 494 µs | 3.0× | keepalive alone |
| 3 | pooled `AsyncClient` + raw binary body | 2 577 µs | 4.0× | httpx client is now the cost |
| 4 | hand-rolled HTTP/1.1 keepalive + raw body → same uvicorn | **381 µs** | 27× | ⇒ **~2.2 ms of #3 is httpx itself** |
| 5 | `torch.distributed` **gloo** `send`/`recv` | 219 µs | 48× | slower than a plain socket |
| 6 | **raw length-prefixed TCP socket, `TCP_NODELAY`** | **30.1 µs** | **346×** | ← the v1 answer |
| 7 | UNIX domain socket (default buffers) | 20.8 µs | 500× | wins at seq=1 only — see §5 |
| 8 | RDMA verbs, small message | ~2–5 µs *(modelled)* | ~2600× | **not achievable on this hardware** |

Same ladder at longer context (v0 has no KV cache, so payload grows with `seq`):

| Per-hop RTT | seq=1 (3.5 KB) | seq=128 (448 KB) | seq=512 (1.75 MB) |
|---|---|---|---|
| v0 (new client + JSON + b64) | 10 406 µs | 16 028 µs | 35 374 µs |
| pooled + JSON + b64 | 3 494 µs | 8 801 µs | 30 428 µs |
| pooled + raw binary | 2 577 µs | 2 863 µs | 6 121 µs |
| hand-rolled HTTP/1.1 + raw | 381 µs | 1 070 µs | 2 737 µs |
| gloo `send`/`recv` | 219 µs | 381 µs | 1 060 µs |
| **raw framed TCP socket** | **30.1 µs** | **219 µs** | **722 µs** |

Serialization CPU **with no network at all** (round-trip encode+decode, measured):

| seq | JSON+base64 bytes | raw framed bytes | size ratio | JSON+b64 codec | raw codec | **speedup** |
|---|---|---|---|---|---|---|
| 1 | 4 805 | 3 592 | 1.338× | 24.8 µs | 0.7 µs | **35×** |
| 128 | 611 697 | 458 760 | 1.333× | 2 745 µs | 8.2 µs | **334×** |
| 512 | 2 446 705 | 1 835 016 | 1.333× | 11 145 µs | 42.1 µs | **265×** |

Size ratio 1.333 is exactly the base64 4/3 expansion — confirms shared-context defect #4 numerically.

### The arithmetic that kills the RDMA case

Wire time for one 3584 B activation *(modelled, from house numbers)*:

```
1 GbE  : 3584 B / 125e6 B/s   =  28.7 µs
10 GbE : 3584 B / 1.25e9 B/s  =   2.87 µs
TB4    : 3584 B / ~1.9e9 B/s  =   1.9  µs   (15 Gbps measured third-party)
```

So on the 1 GbE we actually have, at seq=1:

- **software stack (v0) = 10 406 µs ; wire = 28.7 µs ⇒ software is 99.7 % of the hop.**
- v0 → raw framed socket saves **10 376 µs/hop**.
- raw framed socket → ideal RDMA saves at most **30.1 − 3 ≈ 27 µs/hop = 0.26 % of the v0 hop.**
- Per generated token (3 hops): v0 burns **31.2 ms of pure transport**; framed sockets burn **90 µs**. RDMA would take that to ~9 µs.

**Honest statement for the deck: RDMA is a 0.26 % optimisation of a problem that is 99.7 % Python.** It becomes interesting only after v1 lands *and* the model moves to GPUs *and* the payload grows — i.e. v2.

## 2. Verdict table

Latency class = small-message RTT order of magnitude. Effort = to working state in *this* codebase.

| Technology | Latency class | Hardware required | Integration effort | Verdict |
|---|---|---|---|---|
| **InfiniBand verbs** | 2–5 µs | IB HCA (ConnectX-6/7) + IB switch + subnet manager (`opensm`) | weeks; C verbs API, no Python story | **never (v1) / v2 only if a real cluster is bought.** Impossible on macOS/M1. |
| **RoCEv2** | 2–5 µs | RDMA-capable Ethernet NIC + **lossless DC-grade switch** (see §3) | weeks + network-engineering time | **v2, data-centre only.** Not a LAN-on-a-desk technology. |
| **iWARP** | 10–20 µs | Chelsio T6-class NIC; works on *lossy* Ethernet (no PFC) | weeks | **never.** The one RDMA flavour that tolerates a dumb switch, but slower than RoCE and the vendor ecosystem has effectively collapsed. |
| **libfabric (OFI)** | provider-dependent | same as provider | weeks; C API, no maintained Python binding | **never directly.** Reach it *indirectly* via NIXL in v2. |
| **UCX / UCXX** | 1.5–3 µs (IB), ~TCP otherwise | none strictly (has TCP provider) but the point is RDMA | days–weeks; C++ build. **UCX-Py is discontinued** (last release 0.45, RAPIDS 25.08) → UCXX, whose wheels are CUDA-flavoured (`libucxx-cu12`). No macOS-arm64 story. | **v2**, and only as the backend under NIXL — not hand-wired. |
| **NCCL `ncclSend`/`ncclRecv`** | 5–10 µs | **NVIDIA GPUs**, mandatory | days *if* you have GPUs | **v2.** p2p send/recv (since NCCL 2.7; current docs 2.30.7) is exactly the right primitive for pipeline-parallel hops — but **NCCL only moves GPU tensors**. `is_nccl_available()==False` here. Dead for v1. |
| **gloo** (CPU equivalent of the above) | **measured 219 µs** | none — pure TCP, works on macOS/arm64 | hours | **v1: NO — ruled out by measurement.** 7.3× *slower* than a plain socket (219 vs 30.1 µs) and it forces a static `world_size` with fail-stop ranks, which contradicts "nodes join and leave". Keep as v2 portability layer only. |
| **`torch.distributed` backends generally** | see above | — | hours | gloo = only usable one here; nccl/mpi both unavailable (measured). |
| **`torch.distributed.rpc`** | ≳ gloo | none | days | **never.** Deprecated since PyTorch 2.0 and emits a deprecation warning pointing at the public Distributed API; building a 2026 demo on it is a liability. |
| **MPI (OpenMPI/MPICH)** | 1–5 µs (IB) / 30–50 µs (TCP) | none for TCP | days; also `is_mpi_available()==False` ⇒ rebuild PyTorch from source | **never.** MPI's world is static and fail-stop: one dead rank aborts the job. That is the *opposite* of a decentralized swarm. |
| **DPDK / kernel bypass** | 5–20 µs | dedicated NIC unbound from the kernel, hugepages, Linux | weeks; you re-implement TCP | **never.** No NIC to own (Docker VM), no macOS support, and it optimises packet *rate* — we send ~3 small messages per token. |
| **eBPF / XDP** | saves ~30–50 % of kernel RX path | Linux ≥ 4.8 + XDP-capable NIC driver | weeks | **v2, and marginal.** XDP shines at millions of pps (DDoS, LB). A 3-node request/response chain is latency-bound, not pps-bound. |
| **SR-IOV** | removes ~10–20 µs vswitch hop | NIC with VFs + IOMMU/VT-d + hypervisor config | days–weeks | **v2 only if nodes become Linux VMs.** Irrelevant to bare-metal or Docker-on-macOS. |
| **GPUDirect RDMA** | 2–5 µs, GPU-mem ↔ NIC, no host bounce | NVIDIA GPU + RDMA NIC on the same PCIe root complex | weeks | **v2.** The correct end-state for cross-node activation transfer once we are on GPUs. |
| **NVLink / NVSwitch** | 0.5–1 µs, 900 GB/s (NVLink4) | NVIDIA GPUs in one server / NVL72 rack | n/a | **v2 / arguably never.** NVLink is *intra-node*. Using it means the "decentralized across physical nodes" premise is gone. Mention on the deck only as "what the centralized incumbent uses". |

## 3. Why RoCEv2 is a data-centre-only answer

RoCEv2 puts RDMA in a UDP/IPv4 packet, so it looks routable — but the verbs layer assumes a **lossless** fabric. Delivering that is a switch-configuration project, not a cable:

| Requirement | What it is | Why a hackathon LAN cannot supply it |
|---|---|---|
| **PFC** (802.1Qbb) | Per-priority PAUSE frames; pauses one traffic class instead of dropping | Must be enabled and buffer-tuned on **every switch port along every path**. Unmanaged/consumer switches do not implement it at all. |
| **ECN marking** (WRED thresholds per queue) | Switch marks CE bit before queues fill | Requires per-queue threshold tuning on a managed DC switch. |
| **DCQCN** | NIC-resident congestion control reacting to ECN (Mellanox) | Lives in NIC firmware; tuning is vendor-specific and iterative. |
| **PFC deadlock / storm watchdogs** | PFC is a *back-pressure* protocol: a misconfigured port can pause an entire fabric | Well-documented operational hazard; you need monitoring you do not have. |
| **Single administered L2/L3 domain** | You must own every hop | Contradicts the project's "decentralized, heterogeneous nodes" premise outright. |

Add that RDMA's queue-pair model is **connection-oriented and O(N²) in memory** as node count grows, and gives no NAT traversal — so RoCEv2 is not merely inconvenient for a decentralized system, it is architecturally opposed to one. Note that Petals/hivemind, the closest shipping analogue to this project (each server holds a subset of transformer blocks, clients chain pipeline-parallel servers), routes over **libp2p and ships activations over plain OS TCP sockets** — not RDMA.

## 4. The v1 substitute — what captures ~99.7 % of the win

**Persistent framed binary TCP.** One long-lived TCP connection per hop; 8–16 B binary header (`struct.pack("<II", seq, hidden)` + dtype/req-id); raw tensor bytes; no JSON, no base64, no HTTP.

| v1 change | Measured effect | Effort |
|---|---|---|
| **1. Kill per-call `httpx.AsyncClient()`** (defect #5) | 10 406 → 3 494 µs/hop @seq=1 (**3.0×**) | ~5 lines — a module-level client |
| **2. Drop base64+JSON for raw bytes** | codec 2 745 → 8.2 µs @seq=128 (**334×**); payload −25.0 % (1.333× → 1.0×) | ~20 lines |
| **3. Drop HTTP entirely → framed TCP + `TCP_NODELAY`** | 10 406 → **30.1 µs/hop** (**346×**); 3 hops/token: 31.2 ms → 90 µs | ~80 lines, `asyncio.start_server` + `readexactly` |
| **4. bf16 on the wire** | 3584 → 1792 B/token/hop (2×); wire on 1 GbE 28.7 → 14.4 µs | ~5 lines |
| **5. Thunderbolt Bridge instead of the USB 1 GbE adapters** | ~1 Gbps → ~15 Gbps *(measured, third-party, TB4)* ≈ **120×** bandwidth | cable + `networksetup`; £0 |
| **Fallback if HTTP must stay** for compatibility | HTTP/1.1 keepalive + `application/octet-stream` body = 381 µs/hop (**27×**) | ~20 lines |

`TCP_NODELAY` is not optional: a 3584 B message under Nagle can sit waiting for an ACK, adding tens of ms. Also worth one line each — `SO_SNDBUF`/`SO_RCVBUF` sizing, and **MTU 9000** on the Thunderbolt/wired link for prefill payloads: 458 760 B at MSS 1448 = **317 packets**, at MSS 8948 = **52 packets**, ≈ **6.1× fewer** packets/interrupts *(modelled)*.

Ordering matters: **the KV cache (shared-context defect #1) beats all of this.** It changes the per-hop payload from `[seq,896]` to `[1,896]` — at seq=512 that is 1 835 016 B → 3 584 B, **512×** less data. Transport work only decides how efficiently we move whatever the cache leaves behind.

## 5. Ruled out with a measurement, not an opinion

| Candidate | Why it looked good | Measured reality | Verdict |
|---|---|---|---|
| **gloo `send`/`recv`** | "Use the real collective library" | 219 µs vs 30.1 µs for a plain socket — **7.3× slower**; plus static `world_size`, fail-stop ranks, awkward inside async FastAPI | v1: no |
| **UNIX domain sockets** for co-located nodes | 20.8 µs @seq=1, beats TCP loopback's 30.1 µs | **loses badly at size**: 815 vs 219 µs @seq=128; 3 397 vs 722 µs @seq=512 (macOS default UDS buffers ⇒ many small syscalls; tunable via `SO_SNDBUF` but not worth it) | v1: no — use TCP loopback everywhere, one code path |
| **httpx as the hop client** | already in `requirements.txt` | at seq=1, httpx costs ~2.2 ms of the 2.577 ms hop (#3 vs #4 in §1) — the client library, not HTTP, is the bottleneck | replace |

## 6. v2 horizon — the correct order to adopt RDMA

Do **not** hand-write verbs, UCX or libfabric. The industry has already packaged this:

1. Move to GPUs; adopt **vLLM** with pipeline/tensor parallelism (its NCCL path gives you `ncclSend`/`ncclRecv` for free).
2. For cross-node activation/KV movement, use **NIXL** (NVIDIA Inference Xfer Library, open-sourced at GTC 2025) through vLLM's `NixlConnector`:
   `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["UCX"]}}'`
   NIXL's backends are RDMA/InfiniBand, RoCE-via-UCX, **TCP fallback**, NVMe-oF and S3 — meaning the same code path runs on our v1 TCP fabric today and on RDMA later with a config change. That is the migration story to put on the slide.
3. Only then spend money on RoCEv2 switch configuration (§3), and only in a DC you administer.

## 7. Caveats on these numbers

- All measurements are **loopback on one M1 Pro, transport only, no model forward pass.** They isolate protocol cost; they are a *lower bound* on real cost (the Docker bridge and a real NIC add more) and an *upper bound* on achievable savings.
- Docker was not running, so container-to-container numbers are not included.
- **Model compute per node is not measured here** — 8 Qwen2.5-0.5B layers on 2 CPUs. If compute is ~30 ms/node, v0's 31.2 ms/token of transport is roughly *half* the budget; if compute is ~200 ms, transport is ~13 %. Either way v1 removes it almost entirely, but the speedup headline must be reconciled with T2/T4's compute measurement before it goes on a slide.
- The Thunderbolt 15–16.4 Gbps figure is third-party measured (iperf3, TB4 peer-to-peer), not measured by us.
