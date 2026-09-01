---
team: T1 — Transport & Protocol
agent: T1-A2
topic: Wired LAN fast path — binary framing, socket tuning, same-host shortcuts, Docker penalty
headline: "Measured on the demo Mac: one hop of v0's HTTP+JSON+base64 at seq_len=512 costs 38.1 ms; the same hop as a 16-byte binary frame over a persistent TCP_NODELAY socket costs 0.98 ms — 38.7x, and 114 ms/token of pure transport disappears from a 3-node chain."
---

# T1-A2 — Wired LAN fast path

All `(measured)` numbers below are from this repo's demo machine: MacBookPro18,1 (Apple M1 Pro, 10 cores),
macOS Darwin 25.6.0 arm64, CPython 3.14.3, loopback. Scripts: `bench_wire.py` / `bench2.py` / `bench3.py`
(scratchpad; reproduce with the frame server in §2). `(modelled)` numbers state their source inline.

## 1. Where the time actually goes today

Full hop = client encodes → wire → server decodes to `np.ndarray` and re-encodes → wire → client decodes —
exactly what `node.py:forward()` + `coordinator.py:forward_chain()` do.

| seq_len | payload | v0 HTTP/1.1+JSON+b64 p50 | 16B binary frame p50 | speedup | wire bytes |
|---|---|---|---|---|---|
| 1 | 3 584 B | 391.4 µs | 83.9 µs | 4.7x | 4 805 → 3 600 B |
| 64 | 229 376 B | 5 003.8 µs | 168.9 µs | 29.6x | 305 861 → 229 392 B |
| 512 | 1 835 008 B | 38 062.6 µs | 984.3 µs | **38.7x** | 2 446 705 → 1 835 024 B |

(all measured). The speedup grows with seq_len because base64+JSON is CPU-bound O(n), not because the network
is slow. **v0 has no KV cache, so it always runs at the seq=512 row, three times per token:**

```
v0 transport+codec per token @ seq 512 = 3 hops x 38.06 ms = 114.2 ms   (measured, model compute excluded)
*** AUDIT CORRECTION (90-AUDIT F05): this 114.2 ms is NOT v0's real per-token transport cost. ***
The 38.06 ms hop is a symmetric 1.79 MB round trip; v0's three POSTs are not all symmetric (POST0's
request is 3.6 KB of token ids, POST2's response is the 810 KB logit blob), so x3 double-counts.
T1-A1 measured v0's actual transport share at seq=512 as 72.8 ms of 785.3 ms = 9.3%, cross-checked
against a symmetric echo to within 0.3 ms. USE 72.8 ms for any share-of-wall-clock claim. The 38.7x
per-hop ratio and the 454x composite in this file are valid ONLY as protocol-choice evidence.
binary frame, same seq                 = 3 hops x  0.98 ms =   2.95 ms  (measured)
binary frame + KV cache (T2), seq=1    = 3 hops x  0.084 ms =  0.25 ms  (measured components, composed = modelled)
                                                             --> 454x on the transport component
```

Component breakdown at 1.79 MB, isolated (measured):

| cost | µs | note |
|---|---|---|
| `json.dumps`+`b64encode`+`json.loads`+`b64decode` roundtrip | 14 363 | pure CPU, no socket involved |
| binary frame pack/unpack (`struct` + `memoryview`) | 96.9 | and `np.frombuffer` on it is zero-copy |
| TCP loopback transfer, NODELAY, 4 MB bufs | 570.6 | |
| fresh `connect()` on loopback | 45.1 (p99 108) | v0 pays this 3x per token |

base64 also inflates the wire 1.333x (4 805 B vs 3 600 B at seq=1 — measured), on top of the CPU.

## 2. Binary frame header (v1 spec)

16 bytes, fixed, little-endian, one cache line's worth of nothing. No length-delimited text, no chunked
encoding, no header parsing.

| off | size | field | values |
|---|---|---|---|
| 0 | 1 | magic | `0xD1` |
| 1 | 1 | version | `1` |
| 2 | 1 | msg_type | 0 fwd_req, 1 fwd_resp, 2 logits, 3 err, 4 ping |
| 3 | 1 | dtype | 0 fp32, 1 bf16, 2 fp8_e4m3, 3 int8 — **hand-off point for T1-A3/T2 compression** |
| 4 | 4 | payload_len | u32 |
| 8 | 4 | req_id | u32 — enables pipelining / out-of-order responses |
| 12 | 2 | seq_len | u16 |
| 14 | 2 | flags | bit0 kv_hit, bit1 compressed, bit2 last_chunk |

`struct.Struct("<BBBBIIHH")`. Payload is the raw tensor bytes; the receiver does
`np.frombuffer(buf, dtype).reshape(seq_len, 896)` — **zero copy**, which is the other half of the 38.7x.

Why not HTTP: per request v0 pays header serialize+parse, a `Content-Length` string, JSON construction over a
2.4 MB string, and a full `httpx` client teardown. Nothing in HTTP is load-bearing for a fixed 3-node chain on a
private LAN — no caching, no proxies, no content negotiation, no auth (the gateway already did it). Keep HTTP
on the gateway's public edge only.

## 3. Nagle x delayed-ACK — the 40 ms killer

Nagle (RFC 896) holds a small segment while any previously-sent data is unacked. The peer's delayed-ACK
timer holds the ACK hoping to piggyback it. Deadlock resolves only on the delack timer:
**Linux `TCP_DELACK_MIN = HZ/25 = 40 ms`** (`include/net/tcp.h`); BSD/macOS uses `net.inet.tcp.delayed_ack`
(this machine: `3` = auto, measured) with a 100 ms ceiling.

Trigger is the **write-write-read** pattern — exactly what `sendall(header)` then `sendall(body)` produces.
Both halves of the fix are required: (1) `TCP_NODELAY=1` on **both** ends, (2) header+body in **one**
`sendall(hdr + mv)` or `sendmsg([hdr, mv])` (scatter-gather, no copy). Measured on loopback, 3 584 B ping-pong
(loopback ACKs immediately, so the 40 ms path never fires — this is the coalescing cost only):

| config | p50 | p99 |
|---|---|---|
| TCP NODELAY=0, single write | 57.3 µs | 120.7 µs |
| TCP NODELAY=1, single write | **25.8 µs** | 80.3 µs |
| TCP NODELAY=0, split write | 44.4 µs | 101.3 µs |
| TCP NODELAY=1, split write | 29.5 µs | 74.1 µs |

Loopback saving: 31.5 µs (2.2x, measured). On a real 1 GbE switch with a split write and Nagle on, the same
pattern stalls up to **40 ms per hop = 120 ms per token** (modelled; source: `TCP_DELACK_MIN` constant above
+ RFC 896 interaction, the classic "Nagle vs delayed ACK" pathology).

`TCP_QUICKACK` (Linux only) disables delayed ACK on the receiver. It is **one-shot** — the kernel re-arms
delayed ACK after activity, so you must set it again after every `recv()`. Confirmed absent on macOS:
`hasattr(socket,'TCP_QUICKACK') == False` (measured). Belt-and-braces on Linux; `TCP_NODELAY` + single write
already removes the stall.

## 4. Buffer sizing and BDP

BDP = bandwidth x RTT. Set `SO_SNDBUF`/`SO_RCVBUF` >= 2 x BDP or the sender stalls waiting for window.

| link | B/s | typical RTT | BDP | recommended buf |
|---|---|---|---|---|
| 1 GbE | 125 MB/s | 0.3 ms | 37.5 KB | 128 KB (Linux/macOS default is already 131 072 — measured) |
| 10 GbE | 1.25 GB/s | 0.1 ms | 125 KB | 512 KB |
| 25 GbE | 3.125 GB/s | 0.06 ms | 187.5 KB | 1 MB |

Two gotchas. **Linux**: an explicit `setsockopt(SO_RCVBUF)` **disables receive autotuning** — prefer raising
`net.core.rmem_max`/`wmem_max` (often 212 992 = 208 KB, below 2xBDP at 25 GbE) and leaving the socket alone.
**UDS has no autotuning and a tiny default**: `net.local.stream.sendspace = 8192` (measured, this machine) —
the difference between UDS being the fastest and the slowest transport:

| transport, 1.79 MB | default bufs | 1 MB bufs | 4 MB bufs |
|---|---|---|---|
| TCP loopback | 662.4 µs | 666.9 µs | 570.6 µs |
| UNIX domain socket | **2 732.4 µs** | **467.6 µs** | 551.0 µs |

(all measured). Raising UDS buffers to 1 MB is a **5.8x** win and a two-line change.

**Jumbo frames (MTU 9000).** Per-packet overhead 14 (Eth) + 20 (IP) + 20 (TCP) + 12 (TS opt) = 66 B.
1.79 MB at MSS 1448 = 1 268 packets; at MSS 8948 = 206 — **6.2x fewer packets/interrupts/ACKs** (exact
arithmetic). Only helps when payload >> MTU, i.e. prefill / no-KV-cache traffic; a 3 584 B decode step is 3
standard frames or 1 jumbo frame, so jumbo buys ~nothing there. Needs MTU 9000 on **every** NIC and switch port
— one 1500 hop causes fragmentation or PMTUD blackholes — and NIC TSO/GRO already coalesces most of it. v2.

## 5. The rest of the socket toolbox

| knob | what it removes | latency removed | avail |
|---|---|---|---|
| persistent connection | TCP handshake per call | 45.1 µs loopback measured; 1 RTT = 0.3 ms on 1 GbE (modelled) x3 hops/token | all |
| `TCP_NODELAY` | Nagle coalescing / delack deadlock | 31.5 µs measured; up to 40 ms modelled | all |
| `SO_SNDBUF/SO_RCVBUF` | window stalls, UDS 8 KB default | 2 264 µs measured on UDS @1.79 MB | all |
| `TCP_QUICKACK` (re-armed) | receiver delayed ACK | up to 40 ms modelled (`TCP_DELACK_MIN`) | Linux only |
| `SO_BUSY_POLL` (e.g. 50 µs) | NAPI interrupt + wakeup path | ~20–50 µs → ~5–10 µs, so ~15–40 µs (modelled; source: kernel `Documentation/networking/napi.rst` busy-polling rationale). Burns a core at 100%. | Linux only |
| IRQ affinity + RSS queue pinning, `tuned-adm profile network-latency` | cross-core wakeups, L3 misses, NUMA hops | ~10–30 µs of p99 jitter (modelled; Red Hat low-latency tuning guidance) | Linux bare metal only |
| `TCP_NOTSENT_LOWAT` | userspace buffering above the socket | bufferbloat on large sends; keeps p99 flat | Linux + macOS (`hasattr` True, measured) |
| pipelining via `req_id` | serialization of the chain | prerequisite for T2's 1F1B microbatching, not a latency win by itself | all |

Confirmed absent on macOS (measured `hasattr` probe): `TCP_QUICKACK`, `SO_BUSY_POLL`, `MSG_ZEROCOPY`,
`TCP_CORK`, `TCP_INFO`. Present: `TCP_NODELAY`, `TCP_NOTSENT_LOWAT`, `AF_UNIX`, `SOCK_SEQPACKET`.

## 6. Same-host shortcuts

| mechanism | 3 584 B p50 | 1.79 MB p50 | verdict |
|---|---|---|---|
| TCP loopback, NODELAY, 4 MB bufs | 25.8 µs | 570.6 µs | v1 baseline, works everywhere |
| UNIX domain socket, 1 MB bufs | **6.6 µs** | **467.6 µs** | 3.9x on small frames, free, v1 |
| POSIX shared memory (`multiprocessing.shared_memory`) | **0.6 µs** | 109.4 µs | 43x on small frames — v2 |

(all measured.)

- **UDS**: same API, `AF_UNIX`, no IP stack, no checksums, no Nagle. Needs a shared bind-mounted dir between
  containers (`volumes: - ./sock:/sock`). Cheapest real win for co-located nodes.
- **Shared-memory ring**: SPSC ring in `/dev/shm` (Linux tmpfs) + `eventfd` doorbell. **macOS has no `/dev/shm`**
  (measured), but `shm_open` via `multiprocessing.shared_memory` works — that is the 0.6 µs above. In Docker,
  containers must share the IPC namespace (`ipc: shareable` / `ipc: "container:node0"`) and raise `shm_size`
  from the 64 MB default. Only pays off once the codec cost is gone.
- **`vmsplice`/`sendfile`/`splice`**: `os.sendfile` (POSIX) and `os.splice` (Linux, py3.10+) are stdlib;
  `vmsplice` needs `ctypes`. They save exactly one user→kernel copy = 96.9 µs for 1.79 MB (measured). Against a
  38 ms problem that is noise. **Skip.**
- **io_uring**: −1–3 µs per syscall x ~4 syscalls/hop → ~5–10 µs/hop (modelled). Python binding is `liburing`
  (Zig wrapper over C liburing 2.5+, **Linux 6.1+**, 6.7+ recommended); CPython has no native asyncio io_uring
  backend (bpo-44738 open). v2, behind a Rust/C sidecar.

## 7. The Docker networking penalty

One hop on the `bridge` driver: A `eth0` → veth → `docker0`/user-bridge → netfilter (`conntrack`, plus
`MASQUERADE`/`DNAT` for egress and published ports) → veth → B `eth0`. Container↔container on a
**user-defined** bridge — which `docker-compose.yml` already creates as `decentralized-net` — is **not** NAT'd,
only egress and published ports are, but still pays two veth traversals + bridge forwarding + conntrack:
**~10–25 µs added RTT, ~5–15 % throughput loss vs host networking** (modelled; standard bridge-vs-host-vs-macvlan
container-networking comparisons).

| mode | what it does | when |
|---|---|---|
| user-defined bridge (current) | veth + bridge, no NAT between services | fine for v1 |
| `network_mode: host` (Linux) | container shares the host net namespace: no veth, no bridge, no conntrack; loopback + UDS become trivial | v1 on a Linux demo box |
| `macvlan` | container gets its own MAC on the physical LAN, bypasses bridge and iptables entirely | v2, multi-host wired LAN. Caveat: the host cannot reach its own macvlan containers without a shim interface |

**Docker Desktop for Mac is different and this matters for the demo.** Containers run inside a LinuxKit VM on
`Virtualization.framework`. `--network host` was added in **Docker Desktop 4.34.0** (beta; enable under
Settings → Resources → Network), but "host" means the **VM's** network namespace, not macOS. So:
node↔node traffic does get the veth/bridge removed; laptop↔gateway traffic still crosses the VM boundary
through the userspace proxy and is unaffected. Docker is not running on this machine right now
(`docker version` → cannot connect to daemon, measured), so the container-path numbers above stay modelled.

## 8. macOS/Docker Desktop (demo) vs Linux bare metal (v2)

| technique | Linux bare metal | Docker Desktop on macOS (this demo) |
|---|---|---|
| binary frame + zero-copy `frombuffer` | yes | **yes — the whole 38.7x is portable** |
| `TCP_NODELAY`, persistent conns, buffer sizing | yes | **yes** |
| UNIX domain socket | yes | yes (shared bind-mounted dir, inside the VM) |
| POSIX shm | `/dev/shm` tmpfs | no `/dev/shm` on the macOS host (measured); works inside containers, 64 MB default |
| `TCP_QUICKACK` / `SO_BUSY_POLL` | yes | no on macOS host; QUICKACK works inside the VM, BUSY_POLL is pointless there (virtio-net) |
| jumbo MTU 9000 | yes, if the switch agrees | VM veth is 1500; macOS `lo0` is already 16384 (measured) |
| IRQ affinity / RSS | yes | no — virtio, no real NIC queues |
| io_uring, `splice`/`vmsplice` | yes | inside the VM only (LinuxKit kernel) |
| `--network host` | yes | Docker Desktop ≥ 4.34.0, VM-scoped only |
| RoCEv2 / RDMA / NCCL / UCX | yes, with the NIC | no |

**Demo rule of thumb: everything in v1 below works on the Mac. Everything that needs a kernel knob is v2.**

## 9. Code — the v1 socket setup, complete

```python
import socket, struct, numpy as np

HDR = struct.Struct("<BBBBIIHH")   # magic,ver,type,dtype,len,req_id,seq,flags
DT  = {0: np.float32, 1: np.float16, 3: np.int8}   # 1 = bf16 slot, reinterpreted by T2

def tune(s, buf=4 << 20):
    if s.family == socket.AF_INET:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # -31.5us measured, -40ms modelled
        if hasattr(socket, "TCP_QUICKACK"):                       # Linux only, one-shot
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
    # UDS default is 8192 (measured) -> 5.8x on 1.79MB. On Linux TCP prefer raising
    # net.core.{r,w}mem_max: an explicit SO_RCVBUF disables receive autotuning.
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buf)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buf)
    return s

def send_tensor(s, arr, req_id, dt=0, mtype=0):
    mv = memoryview(np.ascontiguousarray(arr)).cast("B")
    s.sendall(HDR.pack(0xD1, 1, mtype, dt, len(mv), req_id, arr.shape[0], 0) + mv)
    # ONE write. A split header/body write re-arms the Nagle/delack deadlock. Or: s.sendmsg([hdr, mv])

def recv_tensor(s, rf, hidden=896):
    magic, ver, mtype, dt, n, req_id, seq, flags = HDR.unpack(rf.read(HDR.size))
    assert magic == 0xD1 and ver == 1, "bad frame"
    body = rf.read(n)
    if s.family == socket.AF_INET and hasattr(socket, "TCP_QUICKACK"):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)   # re-arm after every recv
    return np.frombuffer(body, dtype=DT[dt]).reshape(seq, hidden), req_id, mtype, flags  # zero-copy
```

Sockets are opened **once at startup**, killing v0's `async with httpx.AsyncClient()` inside
`forward_chain()` (3 fresh handshakes/token: 45.1 µs each loopback measured, ~0.3 ms each on 1 GbE modelled).
Linux demo box compose: `node0: {network_mode: host, ipc: shareable}`,
`node1/2: {network_mode: host, ipc: "container:decentralizedllm-node0"}`.

## 10. Recommendations

| # | change | tag | impact | effort |
|---|---|---|---|---|
| 1 | Replace HTTP+JSON+base64 with the 16-byte binary frame + `np.frombuffer` zero-copy | **v1** | 38.7x/hop @seq512, 4.7x @seq1 (measured); 114.2 → 2.95 ms per token (measured) | hours |
| 2 | One persistent socket per hop, opened at startup | **v1** | −135 µs/token loopback (measured), −0.9 ms/token on 1 GbE (modelled) | hours |
| 3 | `TCP_NODELAY` + single `sendall(hdr+body)` | **v1** | −31.5 µs/hop measured; removes up to 40 ms/hop Nagle stall (modelled) | minutes |
| 4 | `SO_SNDBUF/SO_RCVBUF = 4 MB` (mandatory on UDS) | **v1** | UDS 1.79 MB: 2 732 → 468 µs, 5.8x (measured) | minutes |
| 5 | UNIX domain socket for co-located nodes | **v1** | 25.8 → 6.6 µs on 3 584 B, 3.9x (measured) | hours |
| 6 | `dtype` byte in the header so T1-A3/T2 quantization needs no protocol change | **v1** | 0 now, unblocks 2–4x payload cut later | minutes |
| 7 | `network_mode: host` on a Linux demo box | **v1 (Linux)** | −10–25 µs/hop (modelled) | minutes |
| 8 | POSIX shm ring + eventfd doorbell for same-host nodes | v2 | 6.6 → 0.6 µs on 3 584 B (measured) | weeks |
| 9 | `TCP_QUICKACK` re-arm, `SO_BUSY_POLL=50`, IRQ affinity/RSS, `tuned-adm network-latency` | v2 | −15–40 µs + −10–30 µs p99 jitter (modelled) | days |
| 10 | macvlan/ipvlan per node; jumbo MTU 9000 across the switch | v2 | 6.2x fewer packets on prefill (arithmetic) | weeks |
| 11 | io_uring via `liburing` (Linux 6.1+) or a Rust sidecar; RoCEv2/UCX/NCCL once GPUs exist | v2 | ~5–10 µs/hop (modelled); RDMA RTT 2–5 µs vs 300 µs | months |

**Not worth doing:** `vmsplice`/`sendfile` zero-copy (saves 96.9 µs against a 38 ms problem — measured),
jumbo frames before KV caching lands (decode payloads are 3 584 B), `SO_BUSY_POLL` on the Mac (absent).

**Dependency note for the deck:** item 1 is the single biggest transport win and it is independent of the KV
cache. Do both and the 3-hop transport cost per token goes 114.2 ms → 0.25 ms.
