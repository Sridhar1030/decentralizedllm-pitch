#!/usr/bin/env python3
"""
T5-A4 micro-benchmarks: the cheap, independently-reproducible numbers behind the deck.

Everything here runs in seconds on a laptop with stdlib + numpy. No model weights, no torch.
Measures exactly the four things the pitch quotes as (measured):
  1. base64 encode/decode throughput            -> the "+33% and it costs CPU too" claim
  2. json.dumps/loads of a 1 MB payload         -> the "JSON is the most expensive step" claim
  3. codec throughput on activation-like data   -> the "compression does not pay" claim
  4. loopback TCP RTT vs HTTP round-trip        -> the "transport cost is software, not wire" claim
     + httpx.AsyncClient() construction cost    -> the 4 ms TLS-context-on-plain-http claim

Run:  python3 t5a4_micro.py            (writes t5a4-micro-results.json next to this file)
Self-check: python3 t5a4_micro.py --selftest
"""
import base64, json, math, os, socket, statistics, struct, sys, threading, time, zlib, lzma, bz2
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

H = 896                      # hidden_size, Qwen2.5-0.5B-Instruct
V = 151936                   # vocab_size
HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 30


def best(fn, reps=REPS):
    """min of reps, in seconds. min is the right estimator under OS noise."""
    fn()  # warm
    return min((lambda t0=time.perf_counter(): (fn(), time.perf_counter() - t0)[1])() for _ in range(reps))


def activation(seq, dtype=np.float32, outlier=True):
    """Activation-like tensor: N(0, 1.75) residual stream with one 972x outlier channel.
    Shape/scale/outlier taken from T2-A1's measurement of the real layer-7 output; we do not
    have the weights here, and codec throughput is dominated by byte statistics, not semantics."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.75, size=(seq, H)).astype(np.float32)
    if outlier:
        a[:, 62] = rng.normal(1701.9, 40.0, size=seq)     # T2-A1: channel 62, |1701.9|
    return a.astype(dtype) if dtype is np.float32 else bf16(a)


def bf16(a):
    """fp32 -> bf16 by truncating the low 16 bits. Returned as raw uint16 (the wire form)."""
    return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


# ---------------------------------------------------------------- 1 + 2: base64 / json
def bench_codec_tax():
    rows = []
    for seq in (1, 16, 128, 293, 512, 2048):     # 293 x 896 x 4 = 1,050,112 B ~ the "1 MB payload"
        raw = activation(seq).tobytes()
        b64 = base64.b64encode(raw)
        doc = {"hidden_states_b64": b64.decode()}
        blob = json.dumps(doc)
        n = len(raw)
        r = {
            "seq": seq, "raw_B": n, "b64_B": len(b64), "json_B": len(blob),
            "b64enc_ms": best(lambda: base64.b64encode(raw)) * 1e3,
            "b64dec_ms": best(lambda: base64.b64decode(b64)) * 1e3,
            "jsondumps_ms": best(lambda: json.dumps(doc)) * 1e3,
            "jsonloads_ms": best(lambda: json.loads(blob)) * 1e3,
            "frombuffer_ms": best(lambda: np.frombuffer(raw, dtype=np.float32)) * 1e3,
        }
        r["v0_path_ms"] = r["b64enc_ms"] + r["jsondumps_ms"] + r["jsonloads_ms"] + r["b64dec_ms"]
        r["raw_path_ms"] = r["frombuffer_ms"]
        r["b64enc_MBps"] = n / 1e6 / (r["b64enc_ms"] / 1e3)
        r["b64dec_MBps"] = n / 1e6 / (r["b64dec_ms"] / 1e3)
        r["jsondumps_MBps"] = n / 1e6 / (r["jsondumps_ms"] / 1e3)
        r["jsonloads_MBps"] = n / 1e6 / (r["jsonloads_ms"] / 1e3)
        r["tax_x"] = r["v0_path_ms"] / r["raw_path_ms"]
        rows.append(r)
    # the fp32 logit vector node2 ships back every token (VERIFIED FINDING 2)
    lg = np.random.default_rng(1).normal(0, 4, size=V).astype(np.float32).tobytes()
    lb = base64.b64encode(lg)
    ld = json.dumps({"logits_b64": lb.decode()})
    rows.append({
        "seq": "logits[151936]", "raw_B": len(lg), "b64_B": len(lb), "json_B": len(ld),
        "b64enc_ms": best(lambda: base64.b64encode(lg)) * 1e3,
        "b64dec_ms": best(lambda: base64.b64decode(lb)) * 1e3,
        "jsondumps_ms": best(lambda: json.dumps({"logits_b64": lb.decode()})) * 1e3,
        "jsonloads_ms": best(lambda: json.loads(ld)) * 1e3,
        "argmax_ms": best(lambda: int(np.argmax(np.frombuffer(lg, dtype=np.float32)))) * 1e3,
    })
    return rows


# ---------------------------------------------------------------- 3: byte codecs
CODECS = {
    "zlib-1": (lambda b: zlib.compress(b, 1), zlib.decompress),
    "zlib-6": (lambda b: zlib.compress(b, 6), zlib.decompress),
    "lzma-0": (lambda b: lzma.compress(b, preset=0), lzma.decompress),
    "bz2-1":  (lambda b: bz2.compress(b, 1), bz2.decompress),
    "base64": (base64.b64encode, base64.b64decode),   # for scale: v0's "codec" EXPANDS
}


def bench_codecs():
    rows = []
    for seq, dt, name in ((1, np.float32, "fp32"), (1, "bf16", "bf16"),
                          (512, np.float32, "fp32"), (512, "bf16", "bf16")):
        arr = activation(seq) if dt is np.float32 else bf16(activation(seq))
        raw = np.ascontiguousarray(arr).tobytes()
        for cname, (comp, decomp) in CODECS.items():
            reps = 5 if (cname in ("lzma-0", "bz2-1") and len(raw) > 100_000) else REPS
            cms = best(lambda: comp(raw), reps) * 1e3
            packed = comp(raw)
            dms = best(lambda: decomp(packed), reps) * 1e3
            n = len(raw)
            rows.append({
                "payload": f"seq{seq}_{name}", "raw_B": n, "codec": cname,
                "out_B": len(packed), "ratio": n / len(packed),
                "comp_MBps": n / 1e6 / (cms / 1e3), "decomp_MBps": n / 1e6 / (dms / 1e3),
                "cpu_us": (cms + dms) * 1e3,
                # bytes saved / link rate = wire us saved; negative net_us => codec LOSES
                "net_us_1GbE": (n - len(packed)) / 125e6 * 1e6 - (cms + dms) * 1e3,
                "net_us_100Mbit": (n - len(packed)) / 12.5e6 * 1e6 - (cms + dms) * 1e3,
            })
    return rows


# ---------------------------------------------------------------- 4: loopback TCP vs HTTP
class _Echo(threading.Thread):
    """Length-prefixed binary echo server. This is the DLP shape (T1-A4), minus the header."""
    daemon = True

    def __init__(self):
        super().__init__()
        self.s = socket.socket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind(("127.0.0.1", 0))
        self.s.listen(8)
        self.port = self.s.getsockname()[1]

    def run(self):
        while True:
            c, _ = self.s.accept()
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    @staticmethod
    def _serve(c):
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                hdr = _recvn(c, 4)
                if not hdr:
                    return
                body = _recvn(c, struct.unpack("!I", hdr)[0])
                c.sendall(hdr + body)     # echo, same size back
        except OSError:
            pass


def _recvn(c, n):
    buf = bytearray()
    while len(buf) < n:
        b = c.recv(n - len(buf))
        if not b:
            return None
        buf += b
    return bytes(buf)


class _H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def bench_transport():
    echo = _Echo(); echo.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    hport = httpd.server_address[1]

    sock = socket.create_connection(("127.0.0.1", echo.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def raw_rt(p):
        sock.sendall(struct.pack("!I", len(p)) + p)
        _recvn(sock, 4 + len(p))

    conn = http.client.HTTPConnection("127.0.0.1", hport)
    conn.connect()

    def http_pooled(p):
        conn.request("POST", "/forward", body=p,
                     headers={"Content-Length": str(len(p)), "Content-Type": "application/octet-stream"})
        conn.getresponse().read()

    def http_fresh(p):
        c = http.client.HTTPConnection("127.0.0.1", hport)
        c.request("POST", "/forward", body=p,
                  headers={"Content-Length": str(len(p)), "Content-Type": "application/octet-stream"})
        c.getresponse().read()
        c.close()

    def tcp_connect():
        s = socket.create_connection(("127.0.0.1", echo.port)); s.close()

    rows = []
    for seq in (1, 128, 512, 2048):
        raw = activation(seq).tobytes()
        b64json = json.dumps({"hidden_states_b64": base64.b64encode(raw).decode()}).encode()
        rows.append({
            "seq": seq, "raw_B": len(raw), "b64json_B": len(b64json),
            "tcp_frame_raw_ms": best(lambda: raw_rt(raw), 15) * 1e3,
            "http_pooled_raw_ms": best(lambda: http_pooled(raw), 15) * 1e3,
            "http_pooled_b64json_ms": best(lambda: http_pooled(b64json), 15) * 1e3,
            "http_fresh_b64json_ms": best(lambda: http_fresh(b64json), 15) * 1e3,
        })
    extra = {
        "tcp_connect_close_ms": best(tcp_connect, 20) * 1e3,
        "empty_frame_rtt_ms": best(lambda: raw_rt(b""), 40) * 1e3,
    }
    try:
        import ssl, httpx
        extra["ssl_create_default_context_ms"] = best(lambda: ssl.create_default_context(), 10) * 1e3
        extra["httpx_AsyncClient_ctor_ms"] = best(lambda: httpx.AsyncClient(), 10) * 1e3
        extra["httpx_AsyncClient_noverify_ctor_ms"] = best(lambda: httpx.AsyncClient(verify=False), 10) * 1e3
        extra["httpx_version"] = httpx.__version__
    except ImportError:
        extra["httpx_version"] = None
    sock.close(); conn.close(); httpd.shutdown()
    return rows, extra


def selftest():
    """One runnable check: the arithmetic identities the deck depends on."""
    a = activation(4)
    assert a.shape == (4, H) and a.dtype == np.float32
    assert bf16(a).nbytes * 2 == a.nbytes, "bf16 must be exactly half of fp32"
    raw = a.tobytes()
    assert len(base64.b64encode(raw)) == 4 * math.ceil(len(raw) / 3), "base64 is exactly 4/3"
    assert np.frombuffer(base64.b64decode(base64.b64encode(raw)), dtype=np.float32).reshape(4, H).tobytes() == raw
    assert 3584 == H * 4, "one position fp32 = 3584 B"
    assert 607744 == V * 4, "one logit vector fp32 = 607744 B"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    t = time.time()
    out = {
        "host": f"{os.uname().sysname} {os.uname().machine}", "python": sys.version.split()[0],
        "numpy": np.__version__, "estimator": f"min of {REPS} reps (fewer for slow codecs)",
        "note": "MB = 1e6 B. Activation-like data = N(0,1.75) + 972x outlier channel 62 (T2-A1 shape).",
        "b64_json_tax": bench_codec_tax(),
        "codecs": bench_codecs(),
    }
    out["transport"], out["transport_extra"] = bench_transport()
    out["wall_s"] = round(time.time() - t, 1)
    p = os.path.join(HERE, "t5a4-micro-results.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {p} in {out['wall_s']}s")
