#!/usr/bin/env python3
"""
T2-A4: quality guardrail harness for activation compression.

Takes a codec function and reports the metric table. A codec is:

    fn(a: np.ndarray[float32, (seq, 896)]) -> np.ndarray[float32, (seq, 896)]
    fn(a) -> (recon, wire_bytes)          # optional 2-tuple form

The codec is installed as a forward hook on the OUTPUT of layer 7 and layer 15
of Qwen/Qwen2.5-0.5B-Instruct -- i.e. exactly the two tensors that cross the
wire in the DecentralizedLLM PoC (node0->node1, node1->node2). Everything
downstream then runs on the reconstructed tensor, so error compounding is
real, not simulated.

Metrics: rel-L2 / cosine at the injection point, rel-L2 / cosine of the FINAL
hidden state (compounding gain), KL(P_fp32 || P_codec) mean+p99, top-1
agreement, wikitext-2 perplexity delta, and greedy-continuation divergence.

Run:
  PYTHONPATH=<pyarrow-dir> python t2a4_quality_harness.py            # all codecs
  python t2a4_quality_harness.py --codecs bf16,int8-tok --chunks 4
  python t2a4_quality_harness.py --trace                             # depth curve

Import:
  from t2a4_quality_harness import Harness
  Harness().report({"mycodec": my_fn})
"""
import argparse
import json
import os
import pathlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
H, NLAYER = 896, 24
CUTS = (7, 15)  # layer indices whose OUTPUT crosses the wire in the PoC
WIKI = pathlib.Path(os.getenv("WIKITEXT", "/tmp/wikitext2_test.txt"))
OUT = pathlib.Path(__file__).with_name("t2a4-quality-results.json")
PROMPTS = [
    "Explain why distributed inference is a communication problem.",
    "Write a short paragraph about the history of the steam engine.",
    "What is the difference between a list and a tuple in Python?",
    "Summarize the causes of the 1929 stock market crash.",
]


# ------------------------------------------------------------------- codecs
def _absmax(a, axis, bits, group=None):
    """Symmetric absmax quant/dequant. axis=None per-tensor, 1 per-token."""
    x = a.reshape(-1, group) if group else a
    ax = None if axis is None else (1 if group else axis)
    q = 2 ** (bits - 1) - 1
    s = np.abs(x).max(axis=ax, keepdims=True) / q
    s = np.maximum(s, 1e-12)
    return (np.rint(x / s).clip(-q - 1, q) * s).reshape(a.shape).astype(np.float32)


def _bits_per_elem(bits, group):
    """Payload bits/elem incl. one fp16 scale per group (or per 896-wide row)."""
    return bits + 16.0 / (group or H)


CODECS = {
    "fp32":       (lambda a: a,                              32.0),
    "bf16":       (lambda a: (a.view(np.uint32) & 0xFFFF0000).view(np.float32), 16.0),
    "fp16":       (lambda a: a.astype(np.float16).astype(np.float32), 16.0),
    "int8-tensor": (lambda a: _absmax(a, None, 8),           8.0),
    "int8-tok":   (lambda a: _absmax(a, 1, 8),               _bits_per_elem(8, None)),
    "int6-tok":   (lambda a: _absmax(a, 1, 6),               _bits_per_elem(6, None)),
    "int4-tok":   (lambda a: _absmax(a, 1, 4),               _bits_per_elem(4, None)),
    "int4-g128":  (lambda a: _absmax(a, 1, 4, 128),          _bits_per_elem(4, 128)),
    "int3-g128":  (lambda a: _absmax(a, 1, 3, 128),          _bits_per_elem(3, 128)),
}
# Control: Gaussian noise at the SAME rel-L2 as int4-g128 (measured 0.0148).
# If this hurts as much as int4 does, the damage is magnitude, not structure.
CODECS["gauss-1.5pct"] = (
    lambda a: a + np.random.default_rng(0).normal(
        0, 0.0148 * np.linalg.norm(a) / np.sqrt(a.size), a.shape).astype(np.float32),
    32.0,
)


def _norm(fn):
    """Accept fn->arr or fn->(arr, nbytes); always return (arr, nbytes|None)."""
    def g(a):
        r = fn(a)
        return r if isinstance(r, tuple) else (r, None)
    return g


# ------------------------------------------------------------------ harness
class Harness:
    def __init__(self, chunks=6, seqlen=512):
        self.chunks, self.seqlen = chunks, seqlen
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, low_cpu_mem_usage=True).eval()
        self.layers = self.model.model.layers
        assert len(self.layers) == NLAYER
        text = WIKI.read_text() if WIKI.exists() else " ".join(PROMPTS) * 400
        ids = self.tok(text[: 40 * chunks * seqlen], return_tensors="pt")["input_ids"][0]
        need = chunks * seqlen
        assert ids.numel() >= need, f"corpus too short: {ids.numel()} < {need}"
        self.ids = ids[:need].reshape(chunks, seqlen)

    # -- hook plumbing ----------------------------------------------------
    def _install(self, codec, log, cuts=CUTS):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            a = t[0].detach().numpy().astype(np.float32)
            ah, nb = codec(a)
            n = np.linalg.norm(a)
            log.append(dict(
                rel_l2=float(np.linalg.norm(ah - a) / n),
                cos=float(np.mean(np.sum(ah * a, 1) /
                                  (np.linalg.norm(ah, axis=1) * np.linalg.norm(a, axis=1) + 1e-12))),
                nbytes=nb))
            new = torch.from_numpy(ah).unsqueeze(0).to(t.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return [self.layers[c].register_forward_hook(hook) for c in cuts]

    def _fwd(self, ids, codec=None, hidden=False, cuts=CUTS):
        log = []
        handles = self._install(codec, log, cuts) if codec else []
        try:
            with torch.no_grad():
                o = self.model(ids.unsqueeze(0), output_hidden_states=hidden)
        finally:
            for h in handles:
                h.remove()
        return o, log

    # -- metrics ----------------------------------------------------------
    def report(self, codecs=None, bits=None):
        codecs = codecs or {k: v[0] for k, v in CODECS.items()}
        bits = bits or {k: v[1] for k, v in CODECS.items()}
        acc = {k: dict(kl=[], top1=0, n=0, nll=0.0, inj=[], cos=[], fin=[], fcos=[])
               for k in codecs}
        ref_nll = 0.0
        for ci in range(self.chunks):
            ids = self.ids[ci]
            ro, _ = self._fwd(ids, hidden=True)
            rlp = torch.log_softmax(ro.logits[0].float(), -1)
            rp = rlp.exp()
            rtop = rlp.argmax(-1)
            tgt = ids[1:]
            ref_nll += float(-rlp[:-1].gather(1, tgt[:, None]).mean())
            rfin = ro.hidden_states[-1][0].numpy()
            del ro
            for name, fn in codecs.items():
                f = _norm(fn)
                o, log = self._fwd(ids, codec=f, hidden=True)
                lp = torch.log_softmax(o.logits[0].float(), -1)
                kl = (rp * (rlp - lp)).sum(-1)
                a = acc[name]
                a["kl"] += kl.tolist()
                a["top1"] += int((lp.argmax(-1) == rtop).sum())
                a["n"] += ids.numel()
                a["nll"] += float(-lp[:-1].gather(1, tgt[:, None]).mean())
                a["inj"].append(np.mean([l["rel_l2"] for l in log]))
                a["cos"].append(np.mean([l["cos"] for l in log]))
                fin = o.hidden_states[-1][0].numpy()
                a["fin"].append(np.linalg.norm(fin - rfin) / np.linalg.norm(rfin))
                a["fcos"].append(float(np.mean(np.sum(fin * rfin, 1) / (
                    np.linalg.norm(fin, axis=1) * np.linalg.norm(rfin, axis=1) + 1e-12))))
                del o, lp
            del rlp, rp
        ref_ppl = float(np.exp(ref_nll / self.chunks))
        div = self.divergence(codecs)
        rows = []
        for name, a in acc.items():
            kl = np.array(a["kl"])
            inj, fin = float(np.mean(a["inj"])), float(np.mean(a["fin"]))
            b = bits.get(name, float("nan"))
            rows.append(dict(
                codec=name, bits_per_elem=round(b, 3),
                bytes_per_tok_hop=round(b * H / 8, 1),
                rel_l2_inject=round(inj, 6), cos_inject=round(float(np.mean(a["cos"])), 6),
                rel_l2_final=round(fin, 6), cos_final=round(float(np.mean(a["fcos"])), 6),
                compound_gain=round(fin / inj, 3) if inj > 0 else None,
                kl_mean=round(float(kl.mean()), 6), kl_p99=round(float(np.percentile(kl, 99)), 6),
                top1_agree=round(a["top1"] / a["n"], 5),
                ppl=round(float(np.exp(a["nll"] / self.chunks)), 4),
                ppl_delta_pct=round(100 * (np.exp(a["nll"] / self.chunks) / ref_ppl - 1), 3),
                greedy_exact=div[name][0], first_divergence=div[name][1]))
        return dict(model=MODEL, chunks=self.chunks, seqlen=self.seqlen,
                    tokens=self.chunks * self.seqlen, cuts=list(CUTS),
                    ref_ppl_fp32=round(ref_ppl, 4), rows=rows)

    def divergence(self, codecs, ntok=16):
        """End-task proxy: greedy decode, no KV cache (exactly like the PoC).
        Returns {codec: (exact_match_rate, mean_first_divergence_step)}."""
        refs = []
        for p in PROMPTS:
            ids = self.tok(self.tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True,
                tokenize=False), return_tensors="pt")["input_ids"][0]
            seq = ids
            for _ in range(ntok):
                o, _ = self._fwd(seq)
                seq = torch.cat([seq, o.logits[0, -1].argmax()[None]])
            refs.append((ids, seq[len(ids):]))
        out = {}
        for name, fn in codecs.items():
            f, ex, steps = _norm(fn), 0, []
            for ids, ref in refs:
                seq, d = ids, ntok
                for t in range(ntok):
                    o, _ = self._fwd(seq, codec=f)
                    nx = o.logits[0, -1].argmax()
                    if d == ntok and int(nx) != int(ref[t]):
                        d = t
                    seq = torch.cat([seq, nx[None]])
                steps.append(d)
                ex += d == ntok
            out[name] = (round(ex / len(refs), 3), round(float(np.mean(steps)), 2))
        return out

    def trace(self, codecs, cut=7):
        """Depth curve: inject at ONE cut, measure rel-L2 at every later layer.
        Answers 'do residual streams attenuate or amplify injected noise'."""
        ids = self.ids[0]
        with torch.no_grad():
            ref = self.model(ids.unsqueeze(0), output_hidden_states=True).hidden_states
        rn = [float(np.linalg.norm(h[0].numpy())) for h in ref]
        res = {"layer": list(range(NLAYER + 1)), "ref_norm": [round(x, 1) for x in rn]}
        for name, fn in codecs.items():
            o, _ = self._fwd(ids, codec=_norm(fn), hidden=True, cuts=(cut,))
            res[name] = [round(float(np.linalg.norm(o.hidden_states[i][0].numpy()
                                                    - ref[i][0].numpy()) / rn[i]), 6)
                         for i in range(NLAYER + 1)]
        return res

    def margin(self, codecs, chunk=0):
        """Does the fp32 top-2 logit margin predict which tokens a codec flips?
        If flips concentrate at low margin, the controller can gate on margin --
        a signal node2 already has, for free, with no reference forward."""
        ids = self.ids[chunk]
        with torch.no_grad():
            rl = self.model(ids.unsqueeze(0)).logits[0].float()
        t2 = rl.topk(2, -1).values
        m = (t2[:, 0] - t2[:, 1]).numpy()
        rtop = rl.argmax(-1)
        out = {"margin_deciles": [round(float(x), 3)
                                  for x in np.percentile(m, [10, 25, 50, 75, 90])]}
        for name, fn in codecs.items():
            o, _ = self._fwd(ids, codec=_norm(fn))
            flip = (o.logits[0].argmax(-1) != rtop).numpy()
            lo = m < np.percentile(m, 25)  # bottom quartile of margin
            out[name] = dict(
                flip_rate=round(float(flip.mean()), 5),
                flip_rate_lo_margin_q1=round(float(flip[lo].mean()), 5),
                flip_rate_hi_margin_q234=round(float(flip[~lo].mean()), 5),
                share_of_flips_in_q1=round(float(flip[lo].sum() / max(flip.sum(), 1)), 4),
                mean_margin_flipped=round(float(m[flip].mean()) if flip.any() else 0.0, 3),
                mean_margin_kept=round(float(m[~flip].mean()), 3))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", default="")
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--margin", action="store_true")
    A = ap.parse_args()
    hz = Harness(chunks=A.chunks)
    sel = A.codecs.split(",") if A.codecs else list(CODECS)
    cs = {k: CODECS[k][0] for k in sel}
    r = hz.report(cs, {k: CODECS[k][1] for k in sel})
    if A.trace:
        r["trace_cut7"] = hz.trace({k: cs[k] for k in sel if k != "fp32"})
    if A.margin:
        r["margin"] = hz.margin({k: cs[k] for k in sel if k != "fp32"})
    OUT.write_text(json.dumps(r, indent=1))
    hdr = ["codec", "B/tok/hop", "relL2_inj", "relL2_fin", "gain", "KL_mean",
           "KL_p99", "top1", "ppl", "dppl%", "exact", "div@"]
    print(f"fp32 reference ppl = {r['ref_ppl_fp32']}  ({r['tokens']} wikitext-2 tokens)")
    print(" | ".join(f"{h:>10}" for h in hdr))
    for x in r["rows"]:
        print(" | ".join(f"{str(v):>10}" for v in [
            x["codec"], x["bytes_per_tok_hop"], x["rel_l2_inject"], x["rel_l2_final"],
            x["compound_gain"], x["kl_mean"], x["kl_p99"], x["top1_agree"],
            x["ppl"], x["ppl_delta_pct"], x["greedy_exact"], x["first_divergence"]]))
    print(f"\nwrote {OUT}")

    # ponytail: one self-check, not a suite. Identity codec must be a no-op.
    z = [x for x in r["rows"] if x["codec"] == "fp32"]
    if z:
        assert z[0]["kl_mean"] == 0 and z[0]["top1_agree"] == 1.0, "harness is lying"
        print("self-check ok: fp32 identity -> KL 0, top-1 1.0")
