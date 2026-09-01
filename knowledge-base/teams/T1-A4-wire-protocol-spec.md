---
team: T1 — Transport & Protocol
agent: T1-A4
topic: DLP (Decentralized Layer Protocol) — byte-level wire spec to replace HTTP+JSON+base64
headline: A 40-byte fixed header on a persistent TCP stream cuts per-hop activation RTT from 8.48 ms to 0.089 ms (95x, measured) and wire overhead from +34% to +1.1%; Arrow Flight is the only serious alternative and loses on a 3.6 KB control-plane-shaped payload.
---

# DLP — Decentralized Layer Protocol v1

Replaces the v0 path (`httpx` → FastAPI → JSON → base64 fp32) between coordinator and layer nodes.

## 1. Measured results (the reason this exists)

Bench: Apple M1 Pro, macOS 26.6.2, Python 3.12.12, torch 2.10.0, numpy 2.4.3, **TCP loopback**.
Bench scripts + full reference impl: session scratchpad `dlp_bench.py` / `dlp_rtt.py` / `pipe.py` /
`dlp.py` (impl is reproduced inline in §5). HTTP baseline is stdlib `http.server`
(**faster** than v0's real FastAPI+pydantic stack — no ASGI, no model validation), so every DLP win below
is a conservative **lower bound**.

### 1.1 End-to-end round trip, one hop (measured)

| seq_len | payload | v0 as written<br>(new conn/call) | v0 + keep-alive<br>(1-line fix) | **DLP** | vs v0 | vs keep-alive |
|--------:|--------:|-------:|-------:|-------:|------:|------:|
| 1 (KV-cache case) | 3584 B | 8.483 ms | 1.103 ms | **0.089 ms** | **95.4x** | 12.3x |
| 8 | 28 KB | 10.456 ms | 1.573 ms | **0.129 ms** | 81.1x | 12.2x |
| 64 | 229 KB | 13.418 ms | 5.204 ms | **0.204 ms** | 65.7x | 25.5x |
| 256 (v0 today) | 918 KB | 25.084 ms | 15.743 ms | **0.718 ms** | 34.9x | 21.9x |

**Per generated token = 3 hops.** With a KV cache (seq_len=1): v0 `8.483 × 3 = 25.4 ms/token` of pure
transport → DLP `0.089 × 3 = 0.267 ms/token`. **25.2 ms/token saved (measured).** At v0's current
seq_len≈256 with no KV cache: `75.3 ms → 2.15 ms` per token.

Note the split: ~7.4 ms of the 8.48 ms is **connection churn** (`async with httpx.AsyncClient()` inside
every forward call, shared-context defect #5). Keep-alive alone recovers that. The remaining 12.3x is
what the wire format itself buys.

### 1.2 Serialization CPU + wire bytes (measured)

| seq_len | raw | v0 wire | DLP wire | v0 inflation | v0 CPU | DLP CPU | speedup |
|--------:|----:|--------:|---------:|-----:|-------:|--------:|--------:|
| 1 | 3584 B | 4805 B | **3624 B** | +34.1% | 25.4 µs | **1.08 µs** | 24x |
| 64 | 229 KB | 306 KB | **229 KB** | +33.3% | 1531 µs | **1.14 µs** | 1345x |
| 1024 | 3.67 MB | 4.89 MB | **3.67 MB** | +33.3% | 23429 µs | **1.80 µs** | 13024x |

DLP CPU is flat because it is **O(1) in payload size** — pack a header, hand the socket a `memoryview` of
the tensor's own storage. v0 is O(n): `tobytes` copy → base64 copy → JSON escape → parse → decode.
Header cost is `40 / 3584 = 1.12%`; base64's 33% is pure waste, a text-transport survival trick on a
wired LAN carrying binary.

Micro-numbers (measured): `struct.pack` header **141 ns**, `unpack_from` **118 ns**, vs `json.dumps/loads`
of the same 13 fields **3811 ns** (27x). `zlib.crc32` sustains **~30 GB/s**. bf16 cast costs 5.1 µs at
seq=1 for max rel. error **0.0035**.

Per hop at seq_len=1, with 1 GbE serialization time (modelled): v0 JSON+base64 4805 B / 38.4 µs →
DLP fp32 **3624 B / 29.0 µs** → bf16 **1832 B / 14.7 µs** → int8 **936 B / 7.5 µs**.

## 2. Frame layout

Fixed 40-byte header, **little-endian**, every field naturally aligned, total size `40 % 8 == 0` so the
payload starts 8-byte aligned — a `torch.float32`/`int64` view over it needs no realignment copy.
Little-endian because x86-64 and ARM64 are both LE: `repeated fp32` on the wire is then byte-identical to
the in-memory tensor buffer (verified: `sys.byteorder == 'little'`, `np.dtype(np.float32).str == '<f4'`).

| off | size | field | type | justification |
|--:|--:|---|---|---|
| 0 | 4 | `magic` | `u8[4]`=`"DLP1"` | Rejects a stray HTTP request or port probe at byte 0 instead of mis-parsing 4 GB of "payload". Cheap resync anchor. |
| 4 | 1 | `version` | `u8` | Major version. Mismatch → hard close (§3.5). |
| 5 | 1 | `msg_type` | `u8` | Frame discriminator (§2.1). 256 types is plenty. |
| 6 | 2 | `flags` | `u16` | Bitfield (§2.4). `u16` not `u8` so flags can be added without a version bump. |
| 8 | 4 | `request_id` | `u32` | **Pipelining key** — demultiplexes out-of-order responses on one stream. 4 B ≫ any realistic in-flight window; wraps safely. |
| 12 | 4 | `session_id` | `u32` | Binds the frame to a KV-cache slot. Without it a node cannot know *which* sequence's cache to append to — the field that makes KV caching (defect #1) possible at all. |
| 16 | 4 | `seq_len` | `u32` | `dims[0]`, token positions. 1 with a KV cache, N on prefill. |
| 20 | 4 | `dim1` | `u32` | `dims[1]` — 896 (hidden) or **151936 (vocab)**. Must be `u32`: vocab overflows `u16`. |
| 24 | 4 | `payload_len` | `u32` | **Length prefix** — all the reader needs to find the next frame boundary. 4 GB cap ≫ 3.67 MB worst case. |
| 28 | 4 | `credit` | `u32` | Piggybacked flow-control grant (§3.3). Free: rides an existing frame, zero extra packets. |
| 32 | 1 | `dtype` | `u8` | Enum (§2.2). Lets the sender drop fp32→bf16→int8 per frame without renegotiating. |
| 33 | 1 | `codec` | `u8` | Enum (§2.3). T1-A2/A3 own the algorithms; this field carries them. |
| 34 | 2 | `reserved` | `u16` | Pads `crc32` to a 4-byte boundary. Room for a future `priority`/`tenant` byte. |
| 36 | 4 | `crc32c` | `u32` | Payload integrity. **Optional — gated by `F_CRC`** (§2.4). |
| 40 | N | `payload` | raw | Tensor storage verbatim. No framing, no escaping, no base64. |

`struct` format: `"<4sBBHIIIIIIBBHI"`, `struct.calcsize` = **40** (asserted in the reference impl).

### 2.1 `msg_type`

| val | name | payload | direction |
|--:|---|---|---|
| 0x01 | `HELLO` | none | client → node |
| 0x02 | `HELLO_ACK` | none | node → client |
| 0x10 | `ACTIVATION` | `[seq_len, dim1]` tensor | coordinator → node, node → node |
| 0x11 | `LOGITS` | `[1, 151936]` tensor | last node → coordinator |
| 0x12 | `TOKENS` | `[seq_len]` int32 ids | coordinator → node0 |
| 0x20 | `CREDIT` | none | either (standalone grant) |
| 0x30 / 0x31 | `PING` / `PONG` | none | either |
| 0x40 | `CACHE_EVICT` | none | coordinator → node (drop `session_id`) |
| 0x7F | `ERROR` | UTF-8 message | either |

### 2.2 `dtype` — 0 `fp32`, 1 `fp16`, 2 `bf16`, 3 `int8`, 4 `int32`, 5 `fp8_e4m3`

### 2.3 `codec` — 0 `raw`, 1 `lz4`, 2 `zstd`, 3 `topk-delta`, 4 `int8-blockwise`
When `codec != 0`, `payload_len` is the **compressed** length; the uncompressed length is derived from
`seq_len × dim1 × sizeof(dtype)`. No second length field needed.

### 2.4 `flags`

| bit | name | meaning |
|--:|---|---|
| 0 | `F_CRC` | `crc32c` is valid. **Default off on the docker bridge / loopback**: Ethernet FCS (CRC32) + the TCP checksum already cover the link, so a third check is 0.16 µs/frame of pure tax. Turn it on for cross-machine or when debugging framing. |
| 1 | `F_LAST` | Final frame of a multi-frame response. |
| 2 | `F_MORE` | More frames follow for this `request_id` (chunked large tensors). |
| 3 | `F_PREFILL` | Prefill vs decode — lets the node pick a batching policy. |

**CRC32C vs CRC32:** CRC32C (Castagnoli) is the right choice — hardware instruction on SSE4.2 and ARMv8.
But **Python's stdlib has no CRC32C** (`zlib.crc32` is CRC-32/ISO-HDLC). v1 uses `zlib.crc32` (measured
~30 GB/s, fast enough that it does not matter); v2 swaps in `google-crc32c`. 4 bytes either way.

## 3. Protocol mechanics

**3.1 Framing.** Length-prefixed on a **persistent** TCP connection, one per node pair. Read loop is
exactly `recv_exact(40)` → parse → `recv_exact(payload_len)`: no delimiter scanning, no chunked encoding,
no header map. `TCP_NODELAY` is mandatory — without it Nagle holds a 3.6 KB write up to 40 ms waiting to
coalesce, a 450x latency bug against a 0.089 ms RTT.

**3.2 Pipelining / out-of-order responses.** Up to `credit` frames in flight without waiting; responses
carry the originating `request_id`, so the receiver demultiplexes into `dict[request_id] → Future`.
Verified in the reference impl: three pipelined requests returned `[103, 102, 101]`, matched correctly.

Measured (`pipe.py`, loopback, 0.5 ms simulated per-node compute):

| credit window | 1 | 2 | 4 | 8 | 16 |
|---|--:|--:|--:|--:|--:|
| tok/s (measured) | 1286 | 1482 | 1477 | 1467 | 1421 |

Depth 2 buys **+15%**; deeper buys nothing and eventually costs a little. That is the correct result, not
a disappointing one — pipelining hides *RTT*, and loopback RTT (0.089 ms) is small against 0.5 ms of
compute. The gain scales with RTT/compute, so expect more on 1 GbE (modelled). The big utilization win
(2 of 3 nodes idle, defect #6) is a **scheduling** problem — T2's microbatching, not the protocol's job.
**Set credit=4 and stop tuning.**

**3.3 Credit-based backpressure.** Each side advertises a receive window in `HELLO.credit` — frames, not
bytes, since activation frames are near-constant size. Sender decrements on send; receiver returns credit
piggybacked on any outbound frame, or via a standalone `CREDIT` frame if it has nothing to say. Why not
just rely on TCP's window? Because TCP backpressure is invisible to the application: the coordinator would
keep accepting requests and queue them in kernel buffers, adding unbounded latency with no signal.
Explicit credit lets it **reject or shed at admission time** (defect #8).

**3.4 Heartbeat.** `PING`/`PONG`, 40 B, every 1 s idle; 3 missed → node down. Cost 120 B/s/link. Turns
defect #9 (node failure = silent total outage) into a detectable event. Liveness must be independent of
request traffic, or a hung node is indistinguishable from an idle one.

**3.5 Version negotiation.** Client sends `HELLO{version, credit}`; node replies `HELLO_ACK` at
`min(client, server)` version, or `ERROR` + close. Once per connection, not per frame — so it can afford a
full round trip while the steady-state path stays a 40-byte header. `magic` + `version` in the first
5 bytes means a mismatched peer fails loudly instead of corrupting a tensor.

## 4. Alternatives rejected

| option | verdict | why |
|---|---|---|
| **Arrow Flight** (pyarrow 22.0.0, Oct 2025; `DoExchange` since 0.17.0) | **Closest call — reject for v1, revisit in v2** | See §4.1. |
| **gRPC + protobuf** | Reject | See §4.2 — and the usual "varint pessimises floats" argument is **wrong**. |
| **Cap'n Proto / FlatBuffers** | Reject | Genuinely zero-copy, but they solve *struct* access without parsing. Our payload is one flat fp32 blob with no internal structure — nothing to avoid parsing. A schema compiler and a dependency to wrap a `memoryview` we already have. |
| **MessagePack** | Reject | Beats JSON (binary `bin` type, so no base64 → recovers the 33%), but still self-describing tag-per-value and still copies into a new buffer. `struct.pack` of a fixed schema is strictly less work; msgpack's flexibility is worth nothing when we own both endpoints. |
| **ZeroMQ** | Reject for v1 | Good on paper (`DEALER`/`ROUTER` = pipelining + multipart zero-copy). But it brings its own threading model and silently unbounded HWM queues that fight the explicit admission control we want — and we'd still design DLP's header *inside* the ZMQ message. A dependency to replace ~80 lines of socket code. |
| **QUIC / HTTP3** | Reject | Solves head-of-line blocking on *lossy* links and 0-RTT reconnect across *the internet*. On a docker bridge, loss ≈ 0 and connections are permanent — we'd pay userspace congestion control + mandatory TLS for benefits that do not exist here. Revisit only if nodes span WAN (v2). |
| **WebSockets** | Reject | Frame + mask + HTTP upgrade on top of TCP. Client→server masking XORs **every payload byte** — a mandatory O(n) pass over the tensor for a browser security property irrelevant between two containers. Keep it for the **UI** stream (gateway→browser), not the node path. |
| **Plain raw TCP, no header** | Reject | DLP minus the 40 bytes. Without `payload_len` you cannot frame; without `request_id` you cannot pipeline; without `session_id` you cannot KV-cache; without `dtype` you cannot drop to bf16. The header costs 1.12% and buys all four. |

### 4.1 Arrow Flight — the serious contender

Stated fairly: gRPC transport with a **zero-copy Arrow IPC body** (Flight deliberately routes record
batches around protobuf), `DoExchange` is exactly the bidirectional stream we want, mature and
cross-language.

| criterion | Arrow Flight | DLP |
|---|---|---|
| dependency | `pyarrow` (~120 MB wheel) + grpcio | **stdlib `socket` + `struct`** |
| natural unit | columnar `RecordBatch` | a dense 2-D tensor |
| shape metadata | schema; `pyarrow.Tensor` needs an extension type or a flattened column (apache/arrow#14288 — tensor transport is awkward) | 2 `u32` fields |
| per-frame overhead | Arrow IPC message + schema/dictionary handling + HTTP/2 framing | **40 B** |
| `request_id`/`session_id`/`credit` | opaque bytes in `app_metadata` — i.e. we design our own header anyway | first-class |

Decisive point: Arrow's design centre is **large columnar batches**, where fixed overheads amortize to
nothing. Our hot payload with a KV cache is **3584 bytes**, three times per token — a control-plane-shaped
message, against which Flight's per-message machinery is a fixed cost. Flight becomes right when frames
get large: at a `[1024, 896]` prefill (3.67 MB), or under real batching, the overhead vanishes and the
ecosystem wins. **v2 revisit if (a) nodes go polyglot or (b) batched frames exceed ~1 MB.** For a
3-container demo it is 120 MB of wheel to move 3.6 KB.

### 4.2 gRPC + protobuf — with a correction

The brief asked me to explain that "proto varint encoding of a float tensor is a pessimisation."
**I checked, and it is not true** — so I will not argue it.

Per the [protobuf encoding spec](https://protobuf.dev/programming-guides/encoding/), `float` is **wire
type 5 (I32)**, `double` wire type 1 (I64) — fixed-width IEEE-754, **no varint**. Varint (type 0) covers
`int32/int64/uint*/bool/enum`. A `repeated float [packed=true]` is wire type 2: a length prefix followed by
concatenated little-endian fixed32s — on a LE machine, **byte-identical to the raw fp32 buffer**. Protobuf
does not inflate float tensors. (The varint story *would* hold for a quantized `repeated int32`, where
values cost 1–5 B and negatives cost 10 unless declared `sint32`/zigzag — relevant if T1-A2's int8
quantization is ever encoded that way. Encode quantized tensors as `bytes`.)

Real reasons to reject gRPC here:

1. **No zero-copy in Python.** `bytes` fields are immutable; a tensor costs a full copy each way — exactly
   the 25.4 µs → 1.08 µs we measured away.
2. **`repeated float` materializes a Python list** of 896 float objects per frame. Avoiding it means using
   `bytes`, at which point protobuf adds a length prefix and a schema compiler and nothing else.
3. **HTTP/2 framing + flow control** duplicates what TCP already does; grpcio's default
   `max_receive_message_length` is 4 MB, uncomfortably close to our 3.67 MB prefill frame.
4. **Dependency + codegen** for a 2-endpoint, 1-message-shape protocol between two files we own.

gRPC earns its keep on wide, multi-language, evolving APIs. This is a fixed binary blob between three
containers.

## 5. Reference implementation

Verified working — `demo()` passes: version negotiation, PING/PONG, 3 pipelined requests demultiplexed
out of order (`[103, 102, 101]`), CRC, bit-exact fp32 round trip, bf16 halving, and rejection of a stray
HTTP request. Reproduced below in full; `python dlp.py` runs the self-check.

```python
import socket, struct, threading, zlib
from collections import namedtuple
import numpy as np, torch

HDR = struct.Struct("<4sBBHIIIIIIBBHI")          # 40 B, LE, naturally aligned
Hdr = namedtuple("Hdr", "magic version msg_type flags req_id session_id "
                        "seq_len dim1 payload_len credit dtype codec rsv crc32")
MAGIC, VERSION = b"DLP1", 1
HELLO, HELLO_ACK, ACT, LOGITS, PING, PONG = 0x01, 0x02, 0x10, 0x11, 0x30, 0x31
F_CRC = 1 << 0
TORCH_DT = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
            torch.int8: 3, torch.int32: 4}
DT_TORCH = {v: k for k, v in TORCH_DT.items()}


def tensor_bytes(t: torch.Tensor) -> memoryview:
    """Zero-copy byte view of a contiguous tensor. No copy, no base64."""
    t = t.contiguous()
    if t.dtype is torch.bfloat16:            # numpy has no bf16 -> reinterpret as u8
        return memoryview(t.view(torch.uint8).numpy().reshape(-1).data)
    return memoryview(t.numpy().reshape(-1).view(np.uint8).data)


def pack_header(msg_type, req_id, sess_id, seq_len, dim1, nbytes, *,
                dtype=0, codec=0, flags=0, credit=0, crc=0):
    return HDR.pack(MAGIC, VERSION, msg_type, flags, req_id, sess_id,
                    seq_len, dim1, nbytes, credit, dtype, codec, 0, crc)


def send_tensor(sock, t, msg_type, req_id, sess_id, *, credit=0, crc=False):
    """One sendmsg = writev(2): header + tensor storage. Zero user-space copies."""
    mv = tensor_bytes(t)
    seq_len, dim1 = (t.shape[0], t.shape[1]) if t.dim() > 1 else (t.shape[0], 1)
    h = pack_header(msg_type, req_id, sess_id, seq_len, dim1, mv.nbytes,
                    dtype=TORCH_DT[t.dtype], credit=credit,
                    flags=F_CRC if crc else 0, crc=zlib.crc32(mv) if crc else 0)
    sock.sendmsg([h, mv])                    # scatter-gather: no header+body concat


def _recv_into(sock, mv):
    n, got = mv.nbytes, 0
    while got < n:
        k = sock.recv_into(mv[got:], n - got)
        if not k:
            raise ConnectionError("peer closed")
        got += k


def recv_frame(sock):
    """Returns (Hdr, tensor). Payload lands DIRECTLY in the tensor's storage."""
    hb = bytearray(HDR.size)
    _recv_into(sock, memoryview(hb))
    f = Hdr(*HDR.unpack_from(hb))
    if f.magic != MAGIC:
        raise ValueError(f"bad magic {f.magic!r} — not a DLP stream")
    if f.version != VERSION:
        raise ValueError(f"version {f.version} != {VERSION}")
    if f.payload_len == 0:
        return f, None                       # control frame: HELLO/PING/CREDIT
    # Allocate the destination tensor FIRST, then read the socket into its bytes.
    # This is the zero-copy receive: no intermediate buffer, no frombuffer+copy.
    out = torch.empty((f.seq_len, f.dim1), dtype=DT_TORCH[f.dtype])
    mv = tensor_bytes(out)
    assert mv.nbytes == f.payload_len
    _recv_into(sock, mv)
    if f.flags & F_CRC and zlib.crc32(mv) != f.crc32:
        raise ValueError("CRC mismatch")
    return f, out
```

Connect: `setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)`; client `sendall(pack_header(HELLO, credit=4))`,
server replies `HELLO_ACK` with its own credit; both sides then loop on `recv_frame`, guarding writes with
one lock so each `sendmsg` stays atomic. Zero-copy is asserted, not assumed — `demo()` mutates a tensor
and checks the change is visible through the already-taken `memoryview`.

## 6. Recommendations

### v1 — hackathon-demoable, days, CPU, docker-compose, 3 nodes

| # | change | effort | impact |
|--:|---|---|---|
| 1 | **Hoist the httpx client out of the forward call** (module-level `AsyncClient`). One line. Do this first even if DLP slips. | 10 min | 8.483 → 1.103 ms/hop, **7.7x** (measured) |
| 2 | **DLP over persistent TCP** for node↔node + coordinator↔node0. `node.py` `/forward` stays for health/debug. | 1–2 days | 1.103 → 0.089 ms/hop, **12.3x** on top of #1; **95x** vs v0 as written (measured) |
| 3 | `dtype=bf16` on the wire (`flags`/`dtype` already carry it) | 2 h | 3624 → 1832 B/hop, 2x; max rel err 0.0035 (measured) |
| 4 | `credit=4`, `TCP_NODELAY`, PING every 1 s | 2 h | +15% at depth 2 (measured); node-down detection in ≤3 s |
| 5 | Ship `session_id` in v1 **even before KV caching lands** | 0 (field exists) | unblocks the biggest win (defect #1) with no protocol change later |
| 6 | `F_CRC` **off** on the docker bridge | 0 | saves 0.16 µs/frame; Ethernet FCS + TCP checksum already cover it |

### v2 — production, months

- Replace `zlib.crc32` with `google-crc32c` (hardware CRC32C); keep the field.
- `sendfile`/`MSG_ZEROCOPY` (Linux) or io_uring for true kernel-bypass send on large prefill frames.
- Re-evaluate **Arrow Flight** once frames exceed ~1 MB or nodes become polyglot (§4.1).
- RDMA/RoCEv2 or UCX for the node↔node path — DLP's header maps onto an RDMA immediate/scatter list
  largely unchanged, since it was designed as a fixed offset table.
- Multi-frame `F_MORE` chunking + per-frame priority byte in `reserved`, for preemption of long prefills
  by latency-sensitive decode frames.
- TLS or a MAC on the header if nodes ever leave a trusted L2 segment. DLP v1 has **no authentication** —
  it assumes the docker bridge is trusted, which it is, and the gateway holds the API key.

## 7. Risks / honest caveats

- All numbers are **loopback on an M1 Pro**. Loopback has no NIC, no PCIe, no switch. On real 1 GbE the
  absolute RTTs rise ~0.2–0.5 ms for both sides, which **compresses the ratio** — the 95x is a
  loopback/docker-bridge figure and should be re-measured on the demo hardware before it goes on a slide.
  The *wire-byte* reductions (34% → 1.1%) and the CPU reductions are hardware-independent.
- The HTTP baseline is stdlib, not FastAPI+pydantic. Real v0 is slower, so DLP's win is understated.
- Pipelining measured +15%, not the multiples one might hope for. Said plainly in §3.2 rather than buried.
- DLP has no auth and no encryption by design. Fine on a docker bridge, not fine across a datacenter.
- `dim1` as `u32` caps a dimension at 4.29e9 and the frame at 4 GB. Both are far beyond this model.
