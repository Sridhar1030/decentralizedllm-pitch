---
id: ADR-004
title: Transport tiering — shared memory / UDS / tuned TCP / RDMA
status: v1 accepted (tuned TCP + UDS); RDMA explicitly deferred with a number
date: 2026-09-01
sources: teams/T1-A2, T1-A3, T1-A1, T1-A4, T4-A4, T4-A2
---

# ADR-004 — Transport tiering, and why RDMA is deferred

## Context

ADR-002 decides the *format*. This decides the *medium*, and answers the question a judge will ask in the
first ninety seconds: "why not RDMA?"

The arithmetic that settles it, at seq=1 (the post-KV decode payload, 3,584 B), measured on this hardware:

```
software stack (v0)  = 10,406 µs        wire time on 1 GbE = 28.7 µs
                                        ⇒ software is 99.7% of the hop
v0 → raw framed TCP  saves 10,376 µs/hop      (measured)
framed TCP → RDMA    saves ~27 µs/hop = 0.26% of the v0 hop   (modelled)
```

The full measured ladder, one hop, seq=1, loopback, M1 Pro:

| rung | RTT | vs v0 |
|---|---:|---:|
| v0 as written (new `httpx.AsyncClient()` per call + JSON + base64) | 10,406 µs | 1.0x |
| pooled `AsyncClient` + JSON + base64 | 3,494 µs | 3.0x |
| pooled `AsyncClient` + raw binary body | 2,577 µs | 4.0x |
| hand-rolled HTTP/1.1 keepalive + raw body → same uvicorn | 381 µs | 27x |
| `torch.distributed` **gloo** send/recv | 219 µs | 48x |
| **raw length-prefixed TCP socket + `TCP_NODELAY`** | **30.1 µs** | **346x** |
| UNIX domain socket (default buffers) | 20.8 µs | 500x |
| POSIX shared memory (`multiprocessing.shared_memory`) | 0.6 µs | — |
| RDMA verbs (modelled — **not achievable on this hardware**) | ~2–5 µs | ~2,600x |

## Options considered

| tier | verdict | why |
|---|---|---|
| **Tuned TCP: persistent socket, `TCP_NODELAY`, `SO_*BUF` = 4 MB, single `sendmsg([hdr, mv])`** | **ACCEPTED v1, the default everywhere** | Portable to every deployment we might demo. `TCP_NODELAY` alone is 57.3 → 25.8 µs on loopback (2.2x) and removes a **up-to-40 ms/hop** Nagle-vs-delayed-ACK stall on a real switch (Linux `TCP_DELACK_MIN = HZ/25`). Never a split write. |
| **UNIX domain socket for co-located nodes** | **ACCEPTED v1, opt-in** | 25.8 → 6.6 µs at 3,584 B (**3.9x**), same API, shared bind-mounted dir. **Trap:** UDS wins at seq=1 and *loses badly* at size — 3,397 vs 722 µs for TCP at seq=512 — because macOS `net.local.stream.sendspace` defaults to **8192 B** (measured). Raising `SO_*BUF` to 1 MB takes a 1.79 MB UDS transfer 2,732.4 → 467.6 µs (5.8x). Buffer sizing is not optional here. |
| POSIX shm SPSC ring + eventfd doorbell | **v2 proposed** | 6.6 → 0.6 µs (43x) at 3,584 B, but only pays once the codec cost is gone. macOS has no `/dev/shm`; Docker needs `ipc: shareable` and a raised `shm_size`. |
| Jumbo MTU 9000 end-to-end | **v2, after ADR-001** | 1.79 MB is 1,268 packets at MSS 1448 vs 206 at MSS 8948 (**6.2x fewer**) — a prefill technique, worth nothing on a 3,584 B decode step. Must be set on every NIC, the switch, *and* the Docker bridge or PMTUD silently black-holes traffic. |
| `network_mode: host` / macvlan / ipvlan-l2 | **v2 (Linux only)** | −10–25 µs RTT, +5–15% throughput vs bridge (modelled — Docker was not running on the bench machine). Docker Desktop's host mode (≥4.34.0) is **VM-scoped**: it speeds node↔node, not laptop↔gateway. |
| Thunderbolt 4/5 bridge between Macs | **v1 if the demo is 3 Macs** | ~15–26 Gbps vs 1 GbE USB adapters (reported, third-party iperf3). 3 Macs × 3 TB4 ports = a switchless triangle mesh, **$1,887 vs $2,321** for the 10 GbE alternative. |
| `torch.distributed` **gloo** | **rejected by measurement** | 219 µs vs 30.1 µs for a plain socket = **7.3x slower**, and it forces a static `world_size` with fail-stop ranks — the opposite of nodes that join and leave. |
| MPI, `torch.distributed.rpc`, DPDK, eBPF/XDP, SR-IOV, iWARP, NVLink | **rejected** | MPI is static and fail-stop (`is_mpi_available()==False` here anyway). `rpc` deprecated since PyTorch 2.0. DPDK/XDP optimise packet *rate*; we send ~3 small messages per token. SR-IOV needs Linux VMs. iWARP's vendor ecosystem has collapsed. NVLink is *intra*-node — using it means the premise is gone. |
| `vmsplice` / `sendfile` / `splice` zero-copy | **explicitly not worth building** | Saves exactly one user→kernel copy = 96.9 µs at 1.79 MB (measured), against a 38 ms problem. |
| **RDMA / RoCEv2 / UCX / NCCL / GPUDirect** | **DEFERRED — v2, and say the number out loud** | Post-KV the 1 GbE link carries ~14 KB/token = 0.11 ms against ≥120 ms of compute: **~1,000x headroom.** RoCEv2 also needs PFC enabled and buffer-tuned on every switch port, per-queue ECN/WRED thresholds, DCQCN firmware tuning, and PFC deadlock watchdogs — a network-engineering project, not a cable. Its queue-pair model is O(N²) in memory and offers no NAT traversal, so it is **architecturally opposed** to a decentralized swarm. Petals, the closest shipping analogue, ships activations over plain OS TCP sockets. NCCL moves **GPU tensors only** (`is_nccl_available()==False` here). |

## Decision

1. **Default tier: one persistent TCP socket per peer pair, `TCP_NODELAY=1`, `SO_SNDBUF/SO_RCVBUF = 4 MB`,
   header+body in a single `sendall(hdr+mv)` / `sendmsg([hdr, mv])`.** Never a split write.
2. **Co-located nodes may switch to `AF_UNIX`** via a shared bind-mounted dir, same send/recv code path.
3. **RDMA is a v2 answer to a problem we do not have.** The deck states the number — 27 µs saved per hop
   versus 10,376 µs saved by deleting Python's HTTP+JSON+base64 stack — and moves on. When GPUs arrive, adopt
   it through **NIXL** (vLLM's `NixlConnector`, `--kv-transfer-config`) rather than hand-writing verbs/UCX:
   NIXL's backends are RDMA/IB, RoCE-via-UCX, TCP fallback, NVMe-oF and S3, so the same code path runs on
   today's TCP fabric and on RDMA later with a config change.
4. **Do not build on UCX-Py** (discontinued, last release 0.45 / RAPIDS 25.08; successor UCXX ships
   CUDA-flavoured wheels with no macOS-arm64 story).
5. The transport tier is a config value, not an architecture: DLP's `dtype`/`codec`/framing (ADR-002) are
   unchanged across all four tiers, which is what makes deferring RDMA cheap.

## Consequences

**Good.** ~99.7% of the achievable win with zero new hardware; a numeric answer to "why not RDMA"; a migration
path (NIXL) that does not require rewriting the wire format.

**Bad.**
- **Every number above is loopback on one M1 Pro with no model compute.** They isolate protocol cost correctly
  but are *not* a 3-nodes-over-a-switch measurement: the network component is understated, the Python component
  representative. Re-measure inside compose before anything reaches a slide.
- **The 40 ms Nagle stall — the single largest claimed win — cannot be reproduced on loopback** (loopback ACKs
  immediately). It needs a real switch and a deliberate split write. On the hackathon Mac it will never fire.
  **Do not promise it in a live demo.**
- `TCP_QUICKACK`, `SO_BUSY_POLL`, `MSG_ZEROCOPY`, `TCP_CORK`, `TCP_INFO` and `/dev/shm` are all **absent on
  macOS** (verified by `hasattr` probe). Every Linux-only recommendation here will not demo on the laptop.
- An explicit `setsockopt(SO_RCVBUF)` **disables Linux receive autotuning**. Applying the 4 MB figure on a
  Linux box without raising `net.core.rmem_max`/`wmem_max` is a regression.
- The demo hardware (M1 Pro, macOS, no PCIe slot, no libibverbs, Docker inside a LinuxKit VM) makes every
  kernel-bypass option **physically unavailable, not merely difficult**. Any slide implying we could demo
  RDMA is false.

## Status

**v1 accepted** (tuned TCP default, UDS opt-in, TB bridge if the demo is Macs). **v2 proposed:** shm ring,
jumbo frames, host/macvlan networking, io_uring, NIXL-mediated RDMA, GPUDirect.
