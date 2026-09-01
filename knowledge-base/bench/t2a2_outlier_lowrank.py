"""
T2-A2 follow-up. t2a2_lowrank.py found channel 62 holds ~97% of activation energy at both cuts
(massive activations). Question: is the apparent redundancy (cross-prompt cos 0.98, rank-1 = 99%
energy) ENTIRELY that one channel, and does stripping the outliers rescue low-rank / delta coding?

Scheme tested: ship the C outlier channels exactly (their indices are static, so 0 index bytes)
+ a rank-k PCA of the remaining H-C dims. Compared against plain rank-k on all H dims.
Also: careful min-of-N timing of the projection matmuls.
"""
import os, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "Qwen/Qwen2.5-0.5B-Instruct"; CUTS = {7: 8, 15: 16}; KS = [16, 32, 64, 128, 224]
OUT = [62, 490, 570, 53]  # measured in t2a2_lowrank.py, identical at both cuts
torch.set_grad_enabled(False); torch.set_num_threads(2)
import ast  # reuse the exact same prompt set / split as t2a2_lowrank.py
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2a2_lowrank.py")).read()
PROMPTS = next(ast.literal_eval(ast.get_source_segment(_src, n.value)) for n in ast.parse(_src).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PROMPTS")
N_CAL = 32

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True).eval()
ids = lambda p: torch.tensor([tok(tok.apply_chat_template(
    [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False))["input_ids"]])

per = {c: [] for c in CUTS}; base = []
for p in PROMPTS:
    o = model(ids(p), output_hidden_states=True)
    for c, hi in CUTS.items(): per[c].append(o.hidden_states[hi][0].numpy().astype(np.float32))
    base.append(int(o.logits[0, -1].argmax()))

R = {"outlier_channels": OUT, "cuts": {}}
keep = np.setdiff1d(np.arange(896), OUT)

for cut in CUTS:
    cal = np.concatenate(per[cut][:N_CAL]); hold = np.concatenate(per[cut][N_CAL:])
    hn = np.linalg.norm(hold, axis=1)
    # --- how much is left once the 4 outlier channels are removed? ---
    hk = hold[:, keep]; ck = cal[:, keep]
    kn = np.linalg.norm(hk, axis=1)
    lnA = hold / hn[:, None]; lnB = hk / kn[:, None]
    lastA = np.stack([a[-1] for a in per[cut][N_CAL:]]); lastB = lastA[:, keep]
    cos = lambda X: (lambda n: (n @ n.T)[~np.eye(len(X), dtype=bool)])(X / np.linalg.norm(X, axis=1, keepdims=True))
    dts = []
    for a in per[cut][N_CAL:]:
        b = a[:, keep]
        dts += list(np.linalg.norm(b[1:] - b[:-1], axis=1) / np.linalg.norm(b[1:], axis=1))

    mu = ck.mean(0); _, S, Vt = np.linalg.svd(ck - mu, full_matrices=False)
    en = np.cumsum(S**2) / (S**2).sum()
    relerr = {}
    for k in KS:
        V = Vt[:k].T; rec = mu + (hk - mu) @ V @ V.T
        relerr[k] = round(float(np.median(np.linalg.norm(hk - rec, axis=1) / kn)), 5)

    # --- downstream top-1 with outliers exact + rank-k on the rest ---
    mu_t = torch.tensor(mu); kt = torch.tensor(keep)
    agree = {}
    for k in KS:
        V = torch.tensor(Vt[:k].T.copy())
        def hook(m, i, out, V=V):
            h = out[0] if isinstance(out, tuple) else out
            r = h.clone(); s = h[..., kt]
            r[..., kt] = mu_t + (s - mu_t) @ V @ V.T
            return (r,) + tuple(out[1:]) if isinstance(out, tuple) else r
        hd = model.model.layers[cut].register_forward_hook(hook)
        agree[k] = round(sum(int(model(ids(p)).logits[0, -1].argmax()) == base[N_CAL + i]
                             for i, p in enumerate(PROMPTS[N_CAL:])) / len(PROMPTS[N_CAL:]), 3)
        hd.remove()

    a = np.sort(np.abs(hk), axis=1)[:, ::-1] ** 2
    cum = np.cumsum(a, 1) / a.sum(1, keepdims=True)
    R["cuts"][cut] = {
        "outlier_energy_share": round(float(1 - (hk**2).sum() / (hold**2).sum()), 4),
        "cross_prompt_cos_median_all": round(float(np.median(cos(lastA))), 4),
        "cross_prompt_cos_median_minus_outliers": round(float(np.median(cos(lastB))), 4),
        "temporal_rel_delta_median_minus_outliers": round(float(np.median(dts)), 4),
        "energy_minus_outliers": {k: round(float(en[k - 1]), 5) for k in KS},
        "rank_for_90pct_energy": int(np.searchsorted(en, 0.90) + 1),
        "rank_for_99pct_energy": int(np.searchsorted(en, 0.99) + 1),
        "relerr_median_minus_outliers": relerr,
        "top1_agreement_outliers_exact_plus_rank_k": agree,
        "topk_mag_energy_minus_outliers": {k: round(float(cum[:, k - 1].mean()), 4) for k in KS},
    }
    print(cut, json.dumps(R["cuts"][cut], indent=1))

# --- careful projection timing: min of 5 trials x 1000 reps, 2 threads (matches PoC container) ---
def bench(h, k, trials=5, reps=1000):
    x = np.random.randn(1, h).astype(np.float32)
    E = np.random.randn(h, k).astype(np.float32); D = np.random.randn(k, h).astype(np.float32)
    for _ in range(300): (x @ E) @ D
    ts = []
    for _ in range(trials):
        t = time.perf_counter()
        for _ in range(reps): (x @ E) @ D
        ts.append((time.perf_counter() - t) / reps * 1e6)
    return round(min(ts), 2)

R["proj_us_min_of_5"] = {f"H{h}_k{h//d}": bench(h, h // d)
                         for h in (896, 4096, 8192) for d in (16, 8, 4)}
print(json.dumps(R["proj_us_min_of_5"], indent=1))
json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "t2a2_outlier_lowrank.json"), "w"), indent=1)
