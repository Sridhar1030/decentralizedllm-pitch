#!/usr/bin/env python3
"""
T2-A3: byte-level codecs on REAL Qwen2.5-0.5B-Instruct hidden states.

Captures the activation tensors that actually cross the wire in the
DecentralizedLLM PoC (output of layer 7 -> node1, output of layer 15 -> node2),
then benchmarks LZ4 / LZ4-HC / Snappy / Zstd / zlib / Brotli / Blosc2
(shuffle + bitshuffle) for ratio and comp/decomp throughput.

Run:
  PYTHONPATH=<codec-libs> /path/to/.venv/bin/python t2a3_byte_codecs.py

Throughput convention: BOTH compress and decompress MB/s are measured against
the UNCOMPRESSED byte count (the zstd/lz4 benchmark convention). MB = 1e6 B.
"""
import json
import os
import pathlib
import time
import zlib

import numpy as np

import blosc2
import brotli
import cramjam
import lz4.block
import zstandard

CACHE = pathlib.Path(os.getenv("ACT_CACHE", "/tmp/t2a3_acts"))
OUT = pathlib.Path(__file__).with_name("t2a3-byte-codecs-results.json")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
H = 896
SEQS = [1, 128, 512, 2048]


# ---------------------------------------------------------------- activations
def capture():
    """Real hidden states at the two PoC shard boundaries. Cached as .npy."""
    CACHE.mkdir(parents=True, exist_ok=True)
    want = {f"L{b}_s{s}": CACHE / f"L{b}_s{s}.npy" for b in (8, 16) for s in SEQS}
    if all(p.exists() for p in want.values()):
        return {k: np.load(p) for k, p in want.items()}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()

    # Real English prose, tiled to reach 2048 tokens. Not random noise.
    seed = (
        "The decentralized inference problem is fundamentally a communication problem. "
        "When a transformer is sharded across physical machines, every token forces the "
        "hidden state to cross a network boundary, and the cost of that crossing is set "
        "by bandwidth, latency, and the CPU you are willing to burn on the wire format. "
    )
    ids = tok(seed * 60, return_tensors="pt")["input_ids"][0]
    assert ids.numel() >= max(SEQS), f"only {ids.numel()} tokens"

    out = {}
    for s in SEQS:
        with torch.no_grad():
            hs = model(ids[:s].unsqueeze(0), output_hidden_states=True).hidden_states
        for b in (8, 16):  # hidden_states[8] = output of layer 7, [16] = layer 15
            a = hs[b][0].numpy().astype(np.float32)
            assert a.shape == (s, H), a.shape
            np.save(want[f"L{b}_s{s}"], a)
            out[f"L{b}_s{s}"] = a
        del hs
    return out


def to_bf16(a: np.float32) -> bytes:
    """bf16 = truncate-to-nearest-even the low 2 bytes of fp32. Raw 2-byte codes."""
    u = a.view(np.uint32)
    rounded = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    return ((u + rounded) >> 16).astype(np.uint16).tobytes()


# --------------------------------------------------------------------- codecs
ZC = {lvl: zstandard.ZstdCompressor(level=lvl) for lvl in (1, 3, 9, 19)}
ZD = zstandard.ZstdDecompressor()

# Element width of the CURRENT payload. Blosc's shuffle/bitshuffle filters need
# it to know the stride to transpose on; set per payload by run().
TYPESIZE = 4


CODECS = [
    # The PoC's current "codec". Ratio > 1 on purpose - it is an expansion.
    ("base64 (PoC v0)", lambda b: __import__("base64").b64encode(b),
     lambda b, n: __import__("base64").b64decode(b)),
    ("lz4-block", lambda b: lz4.block.compress(b, store_size=False),
     lambda b, n: lz4.block.decompress(b, uncompressed_size=n)),
    ("lz4hc-9", lambda b: lz4.block.compress(b, mode="high_compression",
                                             compression=9, store_size=False),
     lambda b, n: lz4.block.decompress(b, uncompressed_size=n)),
    ("snappy", lambda b: bytes(cramjam.snappy.compress_raw(b)),
     lambda b, n: bytes(cramjam.snappy.decompress_raw(b))),
    *[(f"zstd-{l}", (lambda l: lambda b: ZC[l].compress(b))(l),
       lambda b, n: ZD.decompress(b, max_output_size=n)) for l in (1, 3, 9, 19)],
    ("zlib-1", lambda b: zlib.compress(b, 1), lambda b, n: zlib.decompress(b)),
    ("zlib-6", lambda b: zlib.compress(b, 6), lambda b, n: zlib.decompress(b)),
    ("brotli-1", lambda b: brotli.compress(b, quality=1),
     lambda b, n: brotli.decompress(b)),
    ("brotli-5", lambda b: brotli.compress(b, quality=5),
     lambda b, n: brotli.decompress(b)),
]

def _blosc_comp(codec, filt, clevel):
    def c(b):
        # nthreads=1: a layer node has 2 CPUs and is busy doing matmuls.
        return blosc2.compress2(b, cparams=blosc2.CParams(
            codec=codec, clevel=clevel, typesize=TYPESIZE,
            filters=[filt], nthreads=1))
    return c


_bd = blosc2.Filter.BYTEDELTA
for _cd, _cn in ((blosc2.Codec.LZ4, "lz4"), (blosc2.Codec.LZ4HC, "lz4hc"),
                 (blosc2.Codec.ZSTD, "zstd")):
    for _ft, _fn in ((blosc2.Filter.NOFILTER, "none"),
                     (blosc2.Filter.SHUFFLE, "shuffle"),
                     (blosc2.Filter.BITSHUFFLE, "bitshuffle")):
        CODECS.append((f"blosc2:{_cn}+{_fn}", _blosc_comp(_cd, _ft, 5),
                       lambda b, n: blosc2.decompress2(b)))


def bench_one(fn, arg_n, data, budget=0.25, min_reps=5):
    """Median-of-reps wall time for fn(data[,n]). Returns seconds/call."""
    call = (lambda: fn(data, arg_n)) if arg_n is not None else (lambda: fn(data))
    call()  # warm
    ts, t_end = [], time.perf_counter() + budget
    while len(ts) < min_reps or time.perf_counter() < t_end:
        t0 = time.perf_counter()
        call()
        ts.append(time.perf_counter() - t0)
        if len(ts) > 20000:
            break
    ts.sort()
    return ts[len(ts) // 2]


def byte_plane_entropy(raw: bytes, typesize: int) -> dict:
    """Shannon entropy (bits) of each byte position within the float word.

    This is WHY shuffle/bitshuffle works. A float array interleaves a
    low-entropy exponent byte with near-random mantissa bytes every `typesize`
    bytes, so an LZ77 matcher never sees a repeat longer than 1-2 bytes.
    Transposing to plane-major order gives the matcher a long run of nearly
    identical exponent bytes to chew on.
    """
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, typesize)
    ents = []
    for k in range(typesize):
        cnt = np.bincount(a[:, k], minlength=256).astype(np.float64)
        p = cnt[cnt > 0] / cnt.sum()
        ents.append(float(-(p * np.log2(p)).sum()))
    # little-endian: plane 0 = LSB of mantissa, plane typesize-1 = sign+exponent
    return {"per_plane_bits": [round(e, 3) for e in ents],
            "order0_floor_ratio": round(sum(ents) / (8 * typesize), 4)}


def run(payloads):
    global TYPESIZE
    rows = []
    for pname, raw in payloads.items():
        n = len(raw)
        TYPESIZE = 2 if "bf16" in pname else 4
        for cname, comp, decomp in CODECS:
            try:
                packed = comp(raw)
                assert decomp(packed, n) == raw or bytes(decomp(packed, n)) == raw
            except Exception as e:  # codec refuses this payload
                rows.append(dict(payload=pname, bytes=n, codec=cname,
                                 error=f"{type(e).__name__}: {e}"))
                continue
            tc = bench_one(comp, None, raw)
            td = bench_one(decomp, n, packed)
            rows.append(dict(
                payload=pname, bytes=n, codec=cname, comp_bytes=len(packed),
                ratio=round(len(packed) / n, 4),
                comp_us=round(tc * 1e6, 3), decomp_us=round(td * 1e6, 3),
                comp_MBps=round(n / tc / 1e6, 1), decomp_MBps=round(n / td / 1e6, 1),
            ))
            print(f"{pname:24s} {cname:24s} r={rows[-1]['ratio']:.3f} "
                  f"c={rows[-1]['comp_MBps']:8.1f} d={rows[-1]['decomp_MBps']:8.1f} MB/s")
    return rows


def crossover(rows):
    """B_cross = (1-r) / (1/T_c + 1/T_d).  Compression pays iff link B < B_cross.

    Derivation (uncompressed-basis throughputs, S cancels):
      t_plain = S/B ; t_comp = S/T_c + S*r/B + S/T_d
      pays iff  S/B > S/T_c + S*r/B + S/T_d
            iff  (1-r)/B > 1/T_c + 1/T_d
            iff  B < (1-r) / (1/T_c + 1/T_d)
    """
    for r in rows:
        if "error" in r:
            continue
        t_eff = 1.0 / (1.0 / r["comp_MBps"] + 1.0 / r["decomp_MBps"])  # MB/s
        r["B_cross_MBps"] = round((1 - r["ratio"]) * t_eff, 1)
        r["cpu_us"] = round(r["comp_us"] + r["decomp_us"], 2)
        for label, B in (("1GbE", 125.0), ("10GbE", 1250.0), ("25GbE", 3125.0)):
            wire_saved = r["bytes"] * (1 - r["ratio"]) / (B * 1e6) * 1e6  # us
            r[f"net_us_{label}"] = round(wire_saved - r["cpu_us"], 2)
    return rows


def demo():
    """Self-check: the crossover algebra and the bf16 conversion."""
    a = np.array([1.0, -2.5, 3.14159, 0.0], dtype=np.float32)
    assert len(to_bf16(a)) == 8
    r = dict(ratio=0.5, comp_MBps=1000.0, decomp_MBps=1000.0, bytes=1000,
             comp_us=1.0, decomp_us=1.0)
    crossover([r])
    # T_eff = 500 MB/s, (1-r)=0.5 -> B_cross = 250 MB/s
    assert abs(r["B_cross_MBps"] - 250.0) < 1e-6, r["B_cross_MBps"]
    print("self-check ok")


if __name__ == "__main__":
    demo()
    acts = capture()
    payloads = {}
    for s in SEQS:
        a = acts[f"L8_s{s}"]
        payloads[f"L8_s{s}_fp32"] = a.tobytes()
        payloads[f"L8_s{s}_bf16"] = to_bf16(a)
    payloads["L16_s2048_fp32"] = acts["L16_s2048"].tobytes()
    # control: pure noise, same size as the 2048 prefill payload
    rng = np.random.default_rng(0)
    payloads["CTRL_gauss_s2048_fp32"] = rng.standard_normal(
        (2048, H)).astype(np.float32).tobytes()

    ent = {k: byte_plane_entropy(v, 2 if "bf16" in k else 4)
           for k, v in payloads.items()}
    for k, v in ent.items():
        print(f"ENTROPY {k:24s} planes(LSB..MSB)={v['per_plane_bits']} "
              f"order0_floor={v['order0_floor_ratio']}")

    rows = crossover(run(payloads))
    OUT.write_text(json.dumps(
        {"model": MODEL, "hidden_size": H,
         "note": "MB=1e6 B; comp/decomp MB/s both on uncompressed basis",
         "byte_plane_entropy": ent, "rows": rows}, indent=1))
    print(f"\nwrote {OUT}")
