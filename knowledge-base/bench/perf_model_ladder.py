#!/usr/bin/env python3
"""perf_model_ladder.py — evaluates the T_token model in knowledge-base/30-PERF-MODEL.md.

Every cell in the ladder table of that document is produced here. Nothing is typed by hand.

    T_token(R=1) = T_compute + SUM over the 3 POSTs of [ A + c*(B_req + B_resp) + B/BW + RTT ]
    X(R)         = min( R / T_token , 1 / D_max )      tokens/s

A and c are two-point fits to the four HTTP/TCP round trips measured by perf_model_micro.py.
T_compute and the per-stage split are (measured) by T1-A1 §5/§7 on the same host+interpreter.

    python3 perf_model_ladder.py            # prints the tables
    python3 perf_model_ladder.py --selftest # checks the fits against the raw measurements
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MB = 2 ** 20

# ---- transport parameters, two-point fit over the measured seq=1 / seq=512 round trips ----
# populated from perf-model-micro-results.json at import; see fit() below.
TRANSPORTS = {}          # name -> (A_ms_per_POST, c_ms_per_MB_per_crossing)

# ---- compute, (measured) T1-A1 §5 (v0 full-sequence) and §7 (KV-cached decode), seq=512 ----
COMPUTE_V0 = (205.81, 197.76, 308.97)          # node0, node1, node2   sum 712.54
COMPUTE_V1_SUM = 123.94                        # KV-cached decode, measured, flat in seq
# per-stage v1 apportioned by the measured v0 shares (T3-A4 §2) -- (modelled)
COMPUTE_V1 = tuple(COMPUTE_V1_SUM * s / sum(COMPUTE_V0) for s in COMPUTE_V0)

BW = {"loopback": None, "1GbE": 125e6, "10GbE": 1.25e9}   # bytes/s
RTT_MS = {"loopback": 0.0, "1GbE": 0.30, "10GbE": 0.08}   # per POST round trip (modelled)


def fit(p_small, p_big, b_small, b_big):
    """A + c*(B_req+B_resp); both round trips are symmetric so B_req=B_resp=b."""
    c = (p_big - p_small) / (2 * (b_big - b_small))        # ms per byte per crossing
    A = p_small - 2 * b_small * c
    return A, c * MB                                       # ms/POST, ms/MB/crossing


def load_measured():
    d = json.load(open(os.path.join(HERE, "perf-model-micro-results.json")))
    h = d["transport"]["http_rtt_ms"]
    t = d["transport"]["tcp_frame_rtt_ms"]
    b1, b512 = 896 * 4, 512 * 896 * 4
    TRANSPORTS["v0 fresh httpx + b64/JSON"] = fit(h["1"]["v0_fresh_client_b64_json"],
                                                  h["512"]["v0_fresh_client_b64_json"], b1, b512)
    TRANSPORTS["pooled httpx + b64/JSON"] = fit(h["1"]["pooled_b64_json"],
                                                h["512"]["pooled_b64_json"], b1, b512)
    TRANSPORTS["fresh httpx + raw binary"] = (TRANSPORTS["v0 fresh httpx + b64/JSON"][0],
                                              fit(h["1"]["pooled_raw_binary"],
                                                  h["512"]["pooled_raw_binary"], b1, b512)[1])
    TRANSPORTS["pooled httpx + raw binary"] = fit(h["1"]["pooled_raw_binary"],
                                                  h["512"]["pooled_raw_binary"], b1, b512)
    TRANSPORTS["framed TCP + zero-copy"] = fit(t["1"], t["512"], b1, b512)
    return d


# ---- payload sizes per POST, per generated token, at a 512-token context -------------------
def b64json(raw, wrapper=25):
    return 4 * -(-raw // 3) + wrapper


ACT_FP32 = 896 * 4                                    # 3584
LOGITS = 151936 * 4                                   # 607744
DLP = 40                                              # T1-A4 fixed header, struct.calcsize == 40

# (name, [(B_req, B_resp) per POST x3], transport key, T_compute)
LADDER = [
    ("v0 as written",
     [(3599, b64json(512 * ACT_FP32)),                             # ids -> H[512]
      (b64json(512 * ACT_FP32), b64json(512 * ACT_FP32)),
      (b64json(512 * ACT_FP32), b64json(LOGITS, 18))],
     "v0 fresh httpx + b64/JSON", sum(COMPUTE_V0), COMPUTE_V0),

    ("+ KV cache (last position only)",
     [(7, b64json(ACT_FP32)), (b64json(ACT_FP32), b64json(ACT_FP32)),
      (b64json(ACT_FP32), b64json(LOGITS, 18))],
     "v0 fresh httpx + b64/JSON", COMPUTE_V1_SUM, COMPUTE_V1),

    ("+ argmax on node2 (return the int)",
     [(7, b64json(ACT_FP32)), (b64json(ACT_FP32), b64json(ACT_FP32)), (b64json(ACT_FP32), 4)],
     "v0 fresh httpx + b64/JSON", COMPUTE_V1_SUM, COMPUTE_V1),

    ("+ binary frame (DLP 40 B, raw fp32)",
     [(DLP + 4, DLP + ACT_FP32), (DLP + ACT_FP32, DLP + ACT_FP32), (DLP + ACT_FP32, DLP + 4)],
     "fresh httpx + raw binary", COMPUTE_V1_SUM, COMPUTE_V1),

    ("+ bf16 on the wire",
     [(DLP + 4, DLP + ACT_FP32 // 2), (DLP + ACT_FP32 // 2, DLP + ACT_FP32 // 2),
      (DLP + ACT_FP32 // 2, DLP + 4)],
     "fresh httpx + raw binary", COMPUTE_V1_SUM, COMPUTE_V1),

    ("+ int8 + 8 fp16 outlier channels",
     [(DLP + 4, DLP + 906), (DLP + 906, DLP + 906), (DLP + 906, DLP + 4)],
     "fresh httpx + raw binary", COMPUTE_V1_SUM, COMPUTE_V1),

    ("+ connection reuse (pooled, then framed TCP)",
     [(DLP + 4, DLP + 906), (DLP + 906, DLP + 906), (DLP + 906, DLP + 4)],
     "framed TCP + zero-copy", COMPUTE_V1_SUM, COMPUTE_V1),
]


def t_token(posts, tkey, compute, link):
    A, c = TRANSPORTS[tkey]
    transport = 0.0
    for br, bs in posts:
        transport += A + c * (br + bs) / MB + RTT_MS[link]
        if BW[link]:
            transport += (br + bs) / BW[link] * 1e3
    return compute + transport, transport


def run(link):
    print(f"\n### LADDER @ seq=512 context, R=1, link = {link}"
          f"{'  (T_wire = 0)' if not BW[link] else f'  (BW={BW[link]/1e6:.0f} MB/s, RTT={RTT_MS[link]} ms/POST)'}")
    print(f"{'#':>2} {'step':<44} {'B/token':>10} {'B/act-hop':>10} {'T_cmp':>8} {'T_tr':>8} "
          f"{'T_tok ms':>9} {'tok/s':>7} {'cum x':>7}")
    base = None
    rows = []
    for i, (name, posts, tkey, compute, stages) in enumerate(LADDER):
        tt, tr = t_token(posts, tkey, compute, link)
        total_B = sum(br + bs for br, bs in posts)
        act_B = posts[1][0]
        base = base or tt
        rows.append((name, total_B, act_B, compute, tr, tt, 1000 / tt, base / tt, tkey, stages))
        print(f"{i:>2} {name:<44} {total_B:>10,} {act_B:>10,} {compute:>8.2f} {tr:>8.2f} "
              f"{tt:>9.1f} {1000/tt:>7.2f} {base/tt:>6.2f}x")
    return rows


def concurrency(rows, link):
    """X(R) = min(R/D, 1/D_max). D = T_token; D_max = slowest stage incl. its share of transport."""
    name, total_B, act_B, compute, tr, tt, tps, cum, tkey, stages = rows[-1]
    share = [s / sum(stages) for s in stages]
    st = [s + tr * f for s, f in zip(stages, share)]           # (modelled) apportion transport
    d_max = max(st)
    print(f"\n### CONCURRENCY, from row {len(rows)-1} (D = {tt:.2f} ms, link = {link})")
    print(f"  per-stage D_i (modelled apportionment of measured sums): "
          f"{st[0]:.2f} / {st[1]:.2f} / {st[2]:.2f} ms   D_max = {d_max:.2f} (node2)")
    bal = tt / 3
    print(f"  N* = D / D_max = {tt:.2f} / {d_max:.2f} = {tt/d_max:.3f} -> ceil = {-(-int(tt/d_max*1000)//1000) and 3}")
    for R in (1, 2, 3, 4, 8):
        x_now = min(R / tt, 1 / d_max) * 1000
        x_bal = min(R / tt, 1 / bal) * 1000
        print(f"  R={R:<2} X = min({R}/{tt:.2f}, 1/{d_max:.2f}) = {x_now:6.2f} tok/s"
              f"   | rebalanced 11/11/2 (D_max = D/3 = {bal:.2f}): {x_bal:6.2f} tok/s"
              f"   TPOT = {R/ (x_now/1000):7.1f} ms")
    return d_max, bal


def selftest():
    d = load_measured()
    h, t = d["transport"]["http_rtt_ms"], d["transport"]["tcp_frame_rtt_ms"]
    b1, b512 = 896 * 4, 512 * 896 * 4
    checks = [("v0 fresh httpx + b64/JSON", h["1"]["v0_fresh_client_b64_json"], h["512"]["v0_fresh_client_b64_json"]),
              ("pooled httpx + b64/JSON", h["1"]["pooled_b64_json"], h["512"]["pooled_b64_json"]),
              ("pooled httpx + raw binary", h["1"]["pooled_raw_binary"], h["512"]["pooled_raw_binary"]),
              ("framed TCP + zero-copy", t["1"], t["512"])]
    for k, m1, m512 in checks:
        A, c = TRANSPORTS[k]
        for b, m in ((b1, m1), (b512, m512)):
            pred = A + c * 2 * b / MB
            assert abs(pred - m) < 1e-6, f"{k} @ {b}: fit {pred} != measured {m}"
    assert b64json(ACT_FP32) == 4805 and b64json(LOGITS, 18) == 810346
    assert DLP == 40
    print("selftest OK: 4 two-point fits reproduce all 8 measured round trips exactly;"
          " b64+JSON byte counts match T1-A1 (4805, 810346)")


if __name__ == "__main__":
    load_measured()
    selftest()
    if "--selftest" in sys.argv:
        sys.exit(0)
    print("\n### TRANSPORT PARAMETERS  T_POST = A + c*(B_req+B_resp)   (measured, two-point fit)")
    print(f"{'transport':<32} {'A ms/POST':>10} {'c ms/MB/crossing':>18}")
    for k, (A, c) in TRANSPORTS.items():
        print(f"{k:<32} {A:>10.3f} {c:>18.3f}")
    for link in ("loopback", "1GbE", "10GbE"):
        rows = run(link)
        if link == "loopback":
            concurrency(rows, link)

    # ---- order sensitivity: same 6 changes, connection reuse moved from last to first ------
    P_KV = [(7, b64json(ACT_FP32)), (b64json(ACT_FP32), b64json(ACT_FP32)),
            (b64json(ACT_FP32), b64json(LOGITS, 18))]
    P_ARG = [(7, b64json(ACT_FP32)), (b64json(ACT_FP32), b64json(ACT_FP32)), (b64json(ACT_FP32), 4)]
    P_BIN = [(DLP + 4, DLP + ACT_FP32), (DLP + ACT_FP32, DLP + ACT_FP32), (DLP + ACT_FP32, DLP + 4)]
    P_BF = [(DLP + 4, DLP + ACT_FP32 // 2), (DLP + ACT_FP32 // 2, DLP + ACT_FP32 // 2),
            (DLP + ACT_FP32 // 2, DLP + 4)]
    P_I8 = [(DLP + 4, DLP + 906), (DLP + 906, DLP + 906), (DLP + 906, DLP + 4)]
    P_V0 = LADDER[0][1]
    ALT = [("v0 as written", P_V0, "v0 fresh httpx + b64/JSON", sum(COMPUTE_V0)),
           ("+ connection reuse FIRST (pooled httpx)", P_V0, "pooled httpx + b64/JSON", sum(COMPUTE_V0)),
           ("+ KV cache", P_KV, "pooled httpx + b64/JSON", COMPUTE_V1_SUM),
           ("+ argmax on node2", P_ARG, "pooled httpx + b64/JSON", COMPUTE_V1_SUM),
           ("+ binary frame (raw fp32, framed TCP)", P_BIN, "framed TCP + zero-copy", COMPUTE_V1_SUM),
           ("+ bf16", P_BF, "framed TCP + zero-copy", COMPUTE_V1_SUM),
           ("+ int8 + outliers", P_I8, "framed TCP + zero-copy", COMPUTE_V1_SUM)]
    for link in ("loopback", "1GbE"):
        print(f"\n### ORDER SENSITIVITY — same 6 changes, cheapest-first, link = {link}")
        print(f"{'#':>2} {'step':<44} {'B/token':>10} {'T_tr':>8} {'T_tok ms':>9} {'tok/s':>7} {'cum x':>7}")
        base = None
        for i, (name, posts, tkey, compute) in enumerate(ALT):
            tt, tr = t_token(posts, tkey, compute, link)
            base = base or tt
            print(f"{i:>2} {name:<44} {sum(a+b for a,b in posts):>10,} {tr:>8.2f} "
                  f"{tt:>9.1f} {1000/tt:>7.2f} {base/tt:>6.2f}x")

    # ---- when does a compression step pay? ------------------------------------------------
    print("\n### COMPRESSION CROSSOVER — link speed below which a dtype step is worth 10% of T_token")
    print("  criterion: bytes_saved / BW  >  0.10 * T_token   =>   BW < bytes_saved / (0.10*T_token)")
    for step, saved in (("fp32->bf16", 14584 - 7416), ("bf16->int8", 7416 - 3872),
                        ("fp32->int8", 14584 - 3872)):
        for label, tk in (("CPU decode, measured 123.94 ms", 123.94),
                          ("GPU-class runtime, modelled 1.5 ms", 1.5)):
            bw = saved / (0.10 * tk / 1e3)            # B/s
            print(f"  {step:<11} saves {saved:>6,} B/token | {label:<34} "
                  f"-> pays below {bw/1e6:8.2f} MB/s = {bw*8/1e6:9.1f} Mbit/s")
