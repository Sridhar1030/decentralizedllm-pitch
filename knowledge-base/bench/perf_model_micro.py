#!/usr/bin/env python3
"""perf_model_micro.py — micro-benchmarks backing knowledge-base/30-PERF-MODEL.md

Every number this prints is (measured) on the host that runs it.

  python3 perf_model_micro.py            # full run, ~40 s, writes perf-model-micro-results.json
  python3 perf_model_micro.py --selftest # arithmetic identities only, ~0 s
  python3 perf_model_micro.py --server P # internal: uvicorn echo server on port P

Sections
  A  base64 encode/decode throughput
  B  json.dumps / json.loads on the base64 string (v0's actual serialisation path)
  C  struct.pack + raw bytes framing vs base64+JSON   (the T_ser/T_deser terms)
  D  lz4 / zstd / blosc2 ratio + throughput on activation-like float data (the T_codec term)
  E  loopback raw framed TCP RTT vs HTTP RTT via a real uvicorn+FastAPI  (the T_wire term)

ponytail: min-of-N estimator, no statistics package. Ratios transfer across hosts; absolute ms do not.
"""
import base64, json, os, socket, struct, subprocess, sys, time
import numpy as np

H = 896                      # hidden_size, Qwen2.5-0.5B-Instruct
V = 151936                   # vocab_size
SEQS = [1, 128, 293, 512, 2048]      # 293 rows x 896 x 4 B = 1,050,112 B ~ 1 MB
REPS = {1: 200, 128: 40, 293: 25, 512: 20, 2048: 6}


def best(fn, n):
    fn()                                    # warm
    return min((lambda t0=time.perf_counter(): (fn(), time.perf_counter() - t0)[1])() for _ in range(n))


def act(seq, seed=0):
    """Activation-like fp32: N(0, 1.75) with one 972x outlier channel (T2-A1 measured ch 62)."""
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((seq, H)) * 1.75).astype(np.float32)
    a[:, 62] *= 972.0
    return a


# ---------------------------------------------------------------- A, B, C
def sec_abc():
    rows = []
    for seq in SEQS:
        a = act(seq)
        raw = a.tobytes()
        b64 = base64.b64encode(raw)
        doc = {"hidden_states_b64": b64.decode(), "shape": [seq, H], "dtype": "float32"}
        js = json.dumps(doc).encode()
        n = REPS[seq]
        hdr = struct.Struct("<4sBBHIIIIIIBBHI")           # T1-A4 DLP 40-byte header
        t = dict(
            seq=seq, raw_B=len(raw), b64_B=len(b64), json_B=len(js),
            tobytes_ms=best(lambda: a.tobytes(), n) * 1e3,
            b64enc_ms=best(lambda: base64.b64encode(raw), n) * 1e3,
            b64dec_ms=best(lambda: base64.b64decode(b64), n) * 1e3,
            json_dumps_ms=best(lambda: json.dumps(doc), n) * 1e3,
            json_loads_ms=best(lambda: json.loads(js), n) * 1e3,
            frombuffer_ms=best(lambda: np.frombuffer(raw, np.float32).reshape(seq, H), n) * 1e3,
            structpack_ms=best(lambda: hdr.pack(b"DLP0", 1, 2, 0, 7, 7, seq, H, len(raw), 4, 0, 0, 0, 0), n) * 1e3,
        )
        t["v0_ser_ms"] = t["b64enc_ms"] + t["json_dumps_ms"]         # sender side
        t["v0_deser_ms"] = t["json_loads_ms"] + t["b64dec_ms"]       # receiver side
        t["v0_roundtrip_ms"] = t["v0_ser_ms"] + t["v0_deser_ms"]
        t["v1_roundtrip_ms"] = t["tobytes_ms"] + t["structpack_ms"] + t["frombuffer_ms"]
        t["v0_ms_per_MB"] = t["v0_roundtrip_ms"] / (len(raw) / 2**20)
        t["v1_ms_per_MB"] = t["v1_roundtrip_ms"] / (len(raw) / 2**20)
        t["codec_speedup"] = t["v0_roundtrip_ms"] / t["v1_roundtrip_ms"]
        t["b64enc_MBps"] = len(raw) / 2**20 / (t["b64enc_ms"] / 1e3)
        t["b64dec_MBps"] = len(raw) / 2**20 / (t["b64dec_ms"] / 1e3)
        t["dumps_MBps"] = len(raw) / 2**20 / (t["json_dumps_ms"] / 1e3)
        t["loads_MBps"] = len(raw) / 2**20 / (t["json_loads_ms"] / 1e3)
        rows.append(t)

    # the logits return path (VERIFIED FINDING 2)
    lg = np.random.default_rng(1).standard_normal(V).astype(np.float32)
    lraw = lg.tobytes(); lb64 = base64.b64encode(lraw)
    ldoc = {"logits_b64": lb64.decode()}; ljs = json.dumps(ldoc).encode()
    logits = dict(
        raw_B=len(lraw), b64_B=len(lb64), json_B=len(ljs),
        b64enc_ms=best(lambda: base64.b64encode(lraw), 60) * 1e3,
        b64dec_ms=best(lambda: base64.b64decode(lb64), 60) * 1e3,
        json_dumps_ms=best(lambda: json.dumps(ldoc), 60) * 1e3,
        json_loads_ms=best(lambda: json.loads(ljs), 60) * 1e3,
        argmax_ms=best(lambda: int(np.argmax(lg)), 200) * 1e3,
    )
    logits["codec_total_ms"] = sum(logits[k] for k in
                                   ("b64enc_ms", "b64dec_ms", "json_dumps_ms", "json_loads_ms"))
    logits["codec_over_argmax"] = logits["codec_total_ms"] / logits["argmax_ms"]
    return rows, logits


# ---------------------------------------------------------------- D
def sec_d():
    try:
        import lz4.frame, zstandard, blosc2
    except ImportError as e:
        return {"unavailable": str(e)}
    out = []
    zc1, zc3 = zstandard.ZstdCompressor(level=1), zstandard.ZstdCompressor(level=3)
    zd = zstandard.ZstdDecompressor()
    for seq, dt in [(1, "fp32"), (512, "fp32"), (2048, "fp32"), (1, "bf16"), (512, "bf16"),
                    (1, "int8"), (512, "int8")]:
        a = act(seq)
        if dt == "fp32":
            buf, ts = a.tobytes(), 4
        elif dt == "bf16":                        # bf16 == top 2 bytes of fp32, little-endian
            buf, ts = a.view(np.uint32).astype(np.uint32).__rshift__(16).astype(np.uint16).tobytes(), 2
        else:
            # T2-A1's shipped codec: outlier channels are carried in fp16 and EXCLUDED from the
            # per-token absmax, otherwise the 972x channel sets the scale and every other channel
            # quantises to ~0 — which compresses ~39x and is a pure artifact. (the corpus trap)
            keep = np.ones(H, bool); keep[62] = False
            s = np.abs(a[:, keep]).max(axis=1, keepdims=True) / 127.0
            buf, ts = np.clip(np.round(a[:, keep] / s), -127, 127).astype(np.int8).tobytes(), 1
        n = 200 if seq == 1 else (20 if seq == 512 else 6)
        MB = len(buf) / 2**20
        cands = {
            "lz4": (lambda b=buf: lz4.frame.compress(b, compression_level=0),
                    lambda c: lz4.frame.decompress(c)),
            "zstd-1": (lambda b=buf: zc1.compress(b), lambda c: zd.decompress(c)),
            "zstd-3": (lambda b=buf: zc3.compress(b), lambda c: zd.decompress(c)),
            "blosc2:lz4+bitshuf": (
                lambda b=buf, t=ts: blosc2.compress2(b, codec=blosc2.Codec.LZ4, clevel=5,
                                                     filters=[blosc2.Filter.BITSHUFFLE],
                                                     typesize=t, nthreads=1),
                lambda c: blosc2.decompress2(c)),
            "blosc2:zstd+bitshuf": (
                lambda b=buf, t=ts: blosc2.compress2(b, codec=blosc2.Codec.ZSTD, clevel=5,
                                                     filters=[blosc2.Filter.BITSHUFFLE],
                                                     typesize=t, nthreads=1),
                lambda c: blosc2.decompress2(c)),
        }
        for name, (cf, df) in cands.items():
            comp = cf()
            assert len(df(comp)) == len(buf), f"{name} roundtrip length mismatch"
            tc = best(cf, n); td = best(lambda c=comp: df(c), n)
            ratio = len(buf) / len(comp)                     # >1 shrinks, <1 expands
            # break-even link speed: bytes saved / CPU seconds spent
            saved = len(buf) - len(comp)
            be = saved / (tc + td) / 1e6 if (tc + td) > 0 else float("inf")   # MB/s
            out.append(dict(seq=seq, dtype=dt, codec=name, in_B=len(buf), out_B=len(comp),
                            ratio=ratio, comp_MBps=MB / tc, decomp_MBps=MB / td,
                            cpu_us=(tc + td) * 1e6, breakeven_MBps=be,
                            net_us_1GbE=saved / 125e6 * 1e6 - (tc + td) * 1e6))
    return out


# ---------------------------------------------------------------- E
SERVER_CODE = '''
import sys
from fastapi import FastAPI, Request, Response
app = FastAPI()

@app.post("/echo_json")
async def echo_json(request: Request):
    d = await request.json()
    return {"hidden_states_b64": d["hidden_states_b64"], "shape": d.get("shape")}

@app.post("/echo_raw")
async def echo_raw(request: Request):
    b = await request.body()
    return Response(b, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="error", access_log=False)
'''


def tcp_echo_server(port, stop):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen(4)
    c, _ = srv.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    f = c.makefile("rb")
    while True:
        h = f.read(8)
        if not h:
            break
        n = struct.unpack("<Q", h)[0]
        if n == 0:
            break
        body = f.read(n)
        c.sendall(h + body)
    c.close(); srv.close()


def sec_e():
    import threading, ssl, httpx
    res = {}
    # -- library construction costs (no I/O at all)
    res["ssl_create_default_context_ms"] = best(ssl.create_default_context, 30) * 1e3
    res["httpx_AsyncClient_ctor_ms"] = best(lambda: httpx.AsyncClient(), 30) * 1e3
    res["httpx_AsyncClient_noverify_ctor_ms"] = best(lambda: httpx.AsyncClient(verify=False), 30) * 1e3
    res["tcp_connect_close_ms"] = None

    # -- raw framed TCP ping-pong on loopback
    port = 47311
    th = threading.Thread(target=tcp_echo_server, args=(port, None), daemon=True); th.start()
    time.sleep(0.3)
    s = socket.create_connection(("127.0.0.1", port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sf = s.makefile("rb")

    def pingpong(payload):
        s.sendall(struct.pack("<Q", len(payload)) + payload)
        h = sf.read(8); n = struct.unpack("<Q", h)[0]
        return np.frombuffer(sf.read(n), np.float32)

    res["tcp_frame_rtt_ms"] = {}
    for seq in (1, 512):
        raw = act(seq).tobytes()
        res["tcp_frame_rtt_ms"][seq] = best(lambda r=raw: pingpong(r), 200 if seq == 1 else 20) * 1e3
    s.sendall(struct.pack("<Q", 0)); s.close()

    # -- fresh TCP connect+close cost
    lsrv = socket.socket(); lsrv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsrv.bind(("127.0.0.1", 47312)); lsrv.listen(64)

    def conn():
        c = socket.create_connection(("127.0.0.1", 47312)); a, _ = lsrv.accept(); c.close(); a.close()
    res["tcp_connect_close_ms"] = best(conn, 100) * 1e3
    lsrv.close()

    # -- real uvicorn + FastAPI
    hport = 47313
    srv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_perf_echo_server.py")
    open(srv_py, "w").write(SERVER_CODE)
    p = subprocess.Popen([sys.executable, srv_py, str(hport)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        try:
            socket.create_connection(("127.0.0.1", hport), 0.1).close(); break
        except OSError:
            time.sleep(0.05)
    else:
        p.kill(); return {**res, "http_unavailable": "uvicorn did not start"}

    base = f"http://127.0.0.1:{hport}"
    res["http_rtt_ms"] = {}
    try:
        with httpx.Client(base_url=base) as pooled:
            for seq in (1, 512):
                a = act(seq); raw = a.tobytes()
                doc = {"hidden_states_b64": base64.b64encode(raw).decode(), "shape": [seq, H]}
                n = 60 if seq == 1 else 15

                def v0_hop():           # fresh client, b64+JSON  == coordinator.py as written
                    with httpx.Client() as c:
                        r = c.post(base + "/echo_json", json=doc)
                    return np.frombuffer(base64.b64decode(r.json()["hidden_states_b64"]), np.float32)

                def pooled_json():
                    r = pooled.post("/echo_json", json=doc)
                    return np.frombuffer(base64.b64decode(r.json()["hidden_states_b64"]), np.float32)

                def pooled_raw():
                    r = pooled.post("/echo_raw", content=raw,
                                    headers={"content-type": "application/octet-stream"})
                    return np.frombuffer(r.content, np.float32)

                res["http_rtt_ms"][seq] = {
                    "v0_fresh_client_b64_json": best(v0_hop, n) * 1e3,
                    "pooled_b64_json": best(pooled_json, n) * 1e3,
                    "pooled_raw_binary": best(pooled_raw, n) * 1e3,
                }
    finally:
        p.terminate(); p.wait(timeout=5); os.remove(srv_py)
    return res


# ---------------------------------------------------------------- main
def selftest():
    a = act(4); raw = a.tobytes()
    assert len(raw) == 4 * H * 4 == 14336
    assert len(base64.b64encode(raw)) == 4 * -(-len(raw) // 3), "base64 is not exactly 4/3"
    assert struct.calcsize("<4sBBHIIIIIIBBHI") == 40, "DLP header is not 40 B"
    assert H * 4 == 3584 and V * 4 == 607744
    assert np.array_equal(np.frombuffer(raw, np.float32).reshape(4, H), a)
    print("selftest OK: 4/3 base64, 40 B DLP header, 3584 B fp32 hop, 607744 B logits, zero-copy identity")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    selftest()
    t0 = time.time()
    abc, logits = sec_abc()
    print(f"[A/B/C done {time.time()-t0:.1f}s]", file=sys.stderr)
    d = sec_d()
    print(f"[D done {time.time()-t0:.1f}s]", file=sys.stderr)
    e = sec_e()
    print(f"[E done {time.time()-t0:.1f}s]", file=sys.stderr)
    out = {
        "host": {"python": sys.version.split()[0], "platform": sys.platform,
                 "numpy": np.__version__, "uname": " ".join(os.uname())},
        "serialisation": abc, "logits_return_path": logits, "codecs": d, "transport": e,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf-model-micro-results.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {path}", file=sys.stderr)
