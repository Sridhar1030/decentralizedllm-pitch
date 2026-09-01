#!/usr/bin/env python3
"""parity_check.py — the correctness gate ADR-001 calls non-negotiable, made runnable.

ADR-001 / 10-ARCHITECTURE §7.4: slicing `model.model.layers` leaves every layer carrying its
ORIGINAL global `layer_idx`. A KV cache keyed by that index puts node1's layer 8 in cache slot 8
of an 8-slot cache, so slot 0 stays empty -> `get_seq_length()` returns 0 -> RoPE positions and the
causal mask are both wrong. There is no exception. The text is just quietly wrong.

Nothing in this knowledge base actually ran that check. This does. Three asserts:

  A  3 chained shards (stateless, v0 semantics) == 1 monolithic forward
  B  KV-cached token-at-a-time across shards  == stateless re-forward   (same greedy tokens)
  C  the same as B with the renumber REMOVED must FAIL                 (proves B has teeth)

Run (needs the PoC venv — torch + transformers + the cached Qwen checkpoint):

  /Users/srpillai/CODING/DecentralizedLLM/.venv/bin/python parity_check.py

ponytail: one process plays all three nodes by re-pointing `base.layers` at a slice. No deepcopy,
no containers, no network — this gate is about layer bookkeeping, and that is process-local.
"""
import sys

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CUTS = [(0, 8), (8, 16), (16, 24)]  # v0's split. Rebalanced 0-11/11-22/22-24 works the same way.
PROMPT = "The capital of France is"
NEW_TOKENS = 8
TOL = dict(rtol=1e-4, atol=1e-4)

torch.set_grad_enabled(False)
torch.set_num_threads(2)


class Pipeline:
    """Three shards over one set of weights. `renumber=False` reproduces the silent-wrongness bug."""

    _snap = None  # (all 24 layers, the real final norm) — taken once, before anything is sliced

    def __init__(self, model, renumber=True):
        if Pipeline._snap is None:
            Pipeline._snap = (list(model.model.layers), model.model.norm)
        self.layers, self.norm = Pipeline._snap
        self.full = model
        self.base = model.model
        self.base.norm = nn.Identity()  # shards must NOT norm mid-pipeline; final node does it
        self.renumber = renumber

    def _select(self, lo, hi):
        self.base.layers = nn.ModuleList(self.layers[lo:hi])
        for i, layer in enumerate(self.base.layers):
            # THE line. Without it, cache slot lookups are off by `lo`.
            layer.self_attn.layer_idx = i if self.renumber else lo + i

    def forward(self, input_ids, caches=None):
        """caches=None -> stateless (v0). caches=[c0,c1,c2] -> KV-cached (v1).

        Returns (logits, caches). The caches come back because each shard owns its own.
        """
        x, got = input_ids, []
        for n, (lo, hi) in enumerate(CUTS):
            self._select(lo, hi)
            kw = {"input_ids": x} if n == 0 else {"inputs_embeds": x}
            if caches is not None:
                kw["past_key_values"] = caches[n]
                kw["use_cache"] = True
            out = self.base(**kw)
            x, _ = out.last_hidden_state, got.append(out.past_key_values)
        return self.full.lm_head(self.norm(x)), got  # norm + lm_head: node2 only


def main():
    print(f"loading {MODEL} (fp32, cached) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print(f"prompt {tuple(ids.shape)} = {PROMPT!r}\n")

    # ---- reference: the monolithic model, exactly as a single box would run it -------------
    ref_logits = model(ids).logits
    ref_tokens = []
    cur = ids
    for _ in range(NEW_TOKENS):
        nxt = int(model(cur).logits[0, -1].argmax())
        ref_tokens.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]])], 1)
    print(f"A  reference (1 process) : {ref_tokens} {tok.decode(ref_tokens)!r}")

    pipe = Pipeline(model, renumber=True)

    # ---- A: 3 stateless shards == monolithic --------------------------------------------
    a_logits, _ = pipe.forward(ids)
    a_max = (a_logits - ref_logits).abs().max().item()
    assert torch.allclose(a_logits, ref_logits, **TOL), f"A FAILED: max |diff| = {a_max}"
    print(f"A  3 shards, stateless   : max |diff| = {a_max:.3e}  OK")

    # ---- B: KV-cached decode across shards == the same tokens ----------------------------
    def run_cached(renumber):
        p = Pipeline(model, renumber=renumber)
        caches = [None, None, None]  # transformers builds a DynamicCache when use_cache=True
        out = []
        for i in range(NEW_TOKENS):
            step_ids = ids if i == 0 else torch.tensor([[out[-1]]])
            logits, caches = p.forward(step_ids, caches=caches)
            out.append(int(logits[0, -1].argmax()))
        return out

    b_tokens = run_cached(renumber=True)
    print(f"B  3 shards, KV-cached   : {b_tokens} {tok.decode(b_tokens)!r}")
    assert b_tokens == ref_tokens, f"B FAILED: {b_tokens} != {ref_tokens}"
    print("B  cached == reference   : OK")

    # ---- C: without the renumber this must break -----------------------------------------
    try:
        c_tokens = run_cached(renumber=False)
        broke = c_tokens != ref_tokens
        detail = f"{c_tokens} {tok.decode(c_tokens)!r}"
    except Exception as e:  # an IndexError here is also "it broke", and louder
        broke, detail = True, f"raised {type(e).__name__}: {e}"
    print(f"C  no renumber           : {detail}")
    assert broke, "C FAILED: the bug did not reproduce, so test B proves nothing"
    print("C  bug reproduced        : OK (B has teeth)")

    print("\nPARITY OK — sharded+cached inference is token-identical to one-process inference.")


if __name__ == "__main__":
    sys.exit(main())
