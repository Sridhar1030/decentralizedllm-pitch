---
id: ADR-002
title: DLP binary wire protocol replacing HTTP + JSON + base64
status: v1 accepted
date: 2026-09-01
sources: teams/T1-A4, T1-A1, T1-A2, T1-A3, T2-A5, T4-A2, T4-A5, 01-VERIFIED-FACTS F2/F4
---

# ADR-002 — DLP: a 40-byte binary frame on a persistent TCP stream

## Context

v0 moves a dense fp32 tensor as a base64 string inside a JSON body over HTTP/1.1, with
`async with httpx.AsyncClient()` constructed **inside every forward call**. Three independent harnesses
measured the same hop and agree on direction while disagreeing on magnitude — the spread is explained, not
averaged:

| harness | payload | v0 baseline | replacement | factor | why the spread |
|---|---|---|---|---|---|
| T1-A4 | 3,584 B (seq=1) | 8.483 ms (stdlib `http.server`) | 0.089 ms DLP | **95.4x** | baseline *faster* than v0's real FastAPI+pydantic stack ⇒ conservative lower bound |
| T1-A3 | 3,584 B (seq=1) | 10,406 µs (real httpx+FastAPI) | 30.1 µs framed TCP | **346x** | measures v0 as actually written |
| T1-A2 | 1.79 MB (seq=512) | 38,062 µs | 984 µs 16-B frame | **38.7x** | large payload — fixed per-call tax amortises |

All three isolate protocol cost with no model compute. Hardware-independent facts, safe to quote anywhere:
base64 is exactly **4/3 (+33.3%)** expansion (611,697 vs 458,760 B, measured at seq=128); serialization CPU is
**O(n) for v0 and O(1) for DLP** — 25.4 µs → 1.08 µs at seq=1, 23,429 µs → 1.80 µs at a 3.67 MB prefill frame
(**13,024x**) — because DLP hands the socket a `memoryview` of the tensor's own storage instead of copying it
three times. `struct.pack` of the header costs 141 ns vs 3,811 ns for `json.dumps` of the same 13 fields.

Two v0 facts dominate the byte budget and are fixed here, not by any codec. **FINDING 2:** node2 returns the
full fp32 logit vector — **607,744 B raw / 810,325 B base64 per token** — so the coordinator can `np.argmax`
and discard 151,935 of 151,936 floats: 97.7% of the post-KV byte budget, 4.285 ms/token of codec to do
0.0736 ms of work (**58x**). **v0 is a STAR, not a chain** (`coordinator.py:76-97`): 3 POSTs = **6 wire
crossings/token**, four activation-sized. FINDING 4's 1,821.7 MB counts only 2, so it is a *lower bound* on
v0's traffic. Keep 935x as published — understating our own baseline is the honest direction to err — and
chain routing deletes the extra crossings anyway.

## Options considered

| option | verdict | why |
|---|---|---|
| Keep HTTP/1.1, just fix keep-alive + raw body | partial — **do it first** | Recovers 7.4 of the 8.48 ms hop alone, one-line diff; ship it even if DLP slips. But ~2.2 ms of the remaining 2.577 ms hop is *httpx itself* (2,577 µs pooled-httpx vs 381 µs hand-rolled HTTP to the same uvicorn, measured). |
| **DLP: 40-byte fixed header + raw payload, persistent TCP, `TCP_NODELAY`** | **ACCEPTED v1** | `"<4sBBHIIIIIIBBHI"`, `calcsize == 40`, `40 % 8 == 0` so the payload is 8-byte aligned and needs no realignment copy. 1.12% overhead buys length-prefix framing, `request_id` pipelining, `session_id` (KV-cache binding, ADR-001), and per-frame `dtype` (ADR-003) — the four things v0 cannot do at all. |
| Arrow Flight (pyarrow 22.0.0, `DoExchange`) | **rejected v1, revisit v2** | Genuinely strong: gRPC transport with a zero-copy Arrow IPC body. Rejected on **payload shape, not quality** — our hot post-KV payload is 3,584 B, control-plane-shaped, against which Flight's per-message machinery is fixed cost; and pyarrow is a ~120 MB wheel to move 3.6 KB. Flips if batched frames exceed ~1 MB or nodes go polyglot. |
| gRPC + protobuf | rejected | **Correction to the brief:** the claim that "varint pessimises float tensors" is *false* — `float` is wire type 5 (I32) fixed-width and `repeated float [packed=true]` is byte-identical to the raw fp32 buffer on little-endian. Do not put that argument on a slide. Real reasons: no zero-copy in Python (`bytes` is immutable), `repeated float` materialises 896 Python floats per frame, HTTP/2 framing duplicates TCP, and grpcio's 4 MB default message limit sits uncomfortably close to our 3.67 MB prefill frame. |
| Cap'n Proto / FlatBuffers / MessagePack / ZeroMQ / QUIC / WebSockets / bare TCP | rejected | Cap'n Proto and FlatBuffers solve *struct* access without parsing — our payload is one flat blob with nothing to parse. MessagePack recovers the 33% but is still tag-per-value and still copies. ZeroMQ brings its own threading model and silently unbounded HWM queues that fight ADR-005's admission control, and we would still design DLP's header inside a ZMQ message. QUIC solves loss and 0-RTT reconnect across the internet; on a docker bridge loss ≈ 0. WebSocket masking XORs every payload byte — keep WS for the browser UI stream only. Bare TCP is DLP minus 40 B: no framing, no pipelining, no KV binding, no dtype switch. |

## Decision

1. **DLP v1** between coordinator↔node and node↔node: 40-byte LE header (`magic "DLP1" / version / msg_type /
   flags / request_id / session_id / seq_len / dim1 / payload_len / credit / dtype / codec / reserved /
   crc32c`) + raw payload, length-prefixed on one persistent TCP socket per peer pair. `TCP_NODELAY`
   mandatory — Nagle can otherwise hold a 3.6 KB write up to 40 ms, a 450x bug against a 0.089 ms RTT. Keep
   `node.py /forward` (HTTP) for health and debug.
2. **node2 returns the sampled token id, not logits.** 607,744 B → 4 B (**151,936x**). ~3-line diff. This is
   simultaneously the largest bandwidth win, a latency win, and the deletion of a prompt-inversion oracle
   (ADR-010). Optional top-k=50 `(id, logit)` pairs = 400 B for sampling clients.
3. **Chain routing.** Each node gets a `NEXT_URL`; node0→node1→node2 direct. 4 activation crossings → 2. The
   coordinator stays the **control** plane (holds `gen_ids`, mints admission and batch decisions) and leaves
   the data plane. Consequence for ADR-009: the boundary-activation journal moves to the *sender*.
4. Ship `session_id` before ADR-001 lands — it binds a frame to a cache slot, so KV caching then needs no
   protocol change.
5. `credit = 4` (measured 1286 → 1482 tok/s at depth 2, **+15%**, nothing beyond); `PING`/`PONG` every 1 s
   idle, 3 misses = node down. `F_CRC` **off** on the docker bridge — Ethernet FCS + TCP checksum already
   cover the link, so a third check is 0.16 µs/frame of tax. On for cross-machine.
6. Reserve `dtype` (0 fp32, 1 fp16, 2 bf16, 3 int8, 4 int32, 5 fp8_e4m3) and the revised `codec` enum
   (0 NONE, 1 ZSTD_1, 2 ZSTD_3, 3 BLOSC2_ZSTD_BITSHUF, 4 **retired-lz4**). LZ4 is retired before anyone
   implements it: measured 1.0036–1.0056 on activations, i.e. it *expands* them (ADR-003). Reserve `flags`
   bit 4 as `F_TRACE` (+32 B; 40+32 = 72, `72 % 8 == 0`, alignment survives) for ADR-012.

## Consequences

**Good.** 95–346x per hop at decode size (measured, loopback), 38.7x at prefill; wire overhead +34% → +1.1%;
serialization becomes O(1); the frame carries everything ADR-001/003/005/012 need with no version bump.

**Bad.**
- **All measured numbers are TCP loopback on an M1 Pro.** On real 1 GbE both sides gain ~0.2–0.5 ms of RTT,
  which **compresses** the ratio. Re-measure before any factor goes on a slide; the byte and CPU reductions
  are hardware-independent and safe.
- **No authentication and no encryption, by design.** Correct on a trusted docker bridge; unsafe the moment
  nodes span an untrusted network. ADR-010 sequences mTLS *after* connection pooling.
- **Python's stdlib has no CRC32C.** The field is named for the v2 `google-crc32c` swap; v1 fills it with
  `zlib.crc32` (CRC-32/ISO-HDLC, ~30 GB/s measured). Do not claim hardware CRC32C in v1.
- **Pipelining bought only +15%.** It hides RTT, and loopback RTT is small against compute. Do not claim
  protocol pipelining fixes the 1/3 utilisation ceiling — that is ADR-006's job.
- Two transports during migration (HTTP for health, DLP for tensors). After ADR-001 + this ADR, transport is
  ~0.4% of the per-token clock here; further protocol work earns nothing until GPUs or batch>1 (ADR-004).

## Status

**v1 accepted.** `google-crc32c`, MSG_ZEROCOPY/io_uring, Arrow Flight re-evaluation, `F_MORE` chunking with a
priority byte, and TLS/MAC over the header are **v2 proposed**.
