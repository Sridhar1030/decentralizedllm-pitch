"""
T2-A2: structural compression of cut-layer activations for DecentralizedLLM.
Measures, on the real Qwen2.5-0.5B-Instruct, at the two PoC cuts (after layer 7, after layer 15):
  1. PCA/SVD spectrum + HOLDOUT reconstruction error vs rank k
  2. downstream top-1 token agreement when the cut activation is replaced by its rank-k reconstruction
  3. top-k magnitude sparsity energy
  4. temporal (token t vs t-1) and static-mean redundancy
  5. wall-clock cost of the projection matmuls, vs network time saved

Run:  /Users/srpillai/CODING/DecentralizedLLM/.venv/bin/python t2a2_lowrank.py
"""
import os, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "Qwen/Qwen2.5-0.5B-Instruct"
CUTS = {7: 8, 15: 16}  # layer index -> hidden_states tuple index (output of that layer)
KS = [16, 32, 64, 112, 128, 224, 256, 448]
torch.set_grad_enabled(False)
torch.set_num_threads(2)  # PoC nodes are 2-CPU containers

PROMPTS = [
    "Explain photosynthesis in two sentences.", "Write a haiku about winter rain.",
    "What is the capital of Australia?", "Summarise the causes of World War I.",
    "How do I reverse a linked list in Python?", "Why is the sky blue?",
    "Give me three ideas for a birthday gift.", "Translate 'good morning' to Japanese.",
    "What does a compiler do?", "Describe the taste of a mango.",
    "List the planets in order from the sun.", "How does a refrigerator work?",
    "What is the difference between TCP and UDP?", "Write a limerick about a cat.",
    "Explain recursion to a child.", "What causes inflation?",
    "How do vaccines work?", "Name four uses for baking soda.",
    "What is a black hole?", "Draft a polite decline to a meeting invite.",
    "Explain the CAP theorem.", "What is the boiling point of water at altitude?",
    "How do I make sourdough starter?", "Describe the plot of Hamlet briefly.",
    "What is gradient descent?", "Why do leaves change colour in autumn?",
    "Give a one-line definition of entropy.", "How does GPS determine position?",
    "What is the Fibonacci sequence?", "Explain HTTP caching headers.",
    "What are tectonic plates?", "How is cheese made?",
    # holdout starts here (last 8)
    "What is quantum entanglement?", "Write a short thank-you note to a teacher.",
    "How does a jet engine produce thrust?", "What is the Pythagorean theorem?",
    "Explain what a database index does.", "Name three types of clouds.",
    "Why do we dream?", "How do noise-cancelling headphones work?",
]
N_CAL = 32

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True).eval()
H = model.config.hidden_size
print(f"loaded, H={H}, layers={model.config.num_hidden_layers}, threads={torch.get_num_threads()}")


def ids(p):
    t = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
    return torch.tensor([tok(t)["input_ids"]])


# ---- 1. capture hidden states at both cuts -------------------------------------------------
per_prompt = {c: [] for c in CUTS}
base_argmax = []
t0 = time.time()
for p in PROMPTS:
    out = model(ids(p), output_hidden_states=True)
    for c, hi in CUTS.items():
        per_prompt[c].append(out.hidden_states[hi][0].numpy().astype(np.float32))
    base_argmax.append(int(out.logits[0, -1].argmax()))
print(f"captured {len(PROMPTS)} prompts in {time.time()-t0:.1f}s")

R = {"H": H, "n_prompts": len(PROMPTS), "n_cal": N_CAL, "cuts": {}}

for cut, hi in CUTS.items():
    cal = np.concatenate(per_prompt[cut][:N_CAL])          # [Ncal, H]
    hold = np.concatenate(per_prompt[cut][N_CAL:])          # [Nhold, H]
    mu = cal.mean(0)
    U, S, Vt = np.linalg.svd(cal - mu, full_matrices=False)
    energy = np.cumsum(S**2) / (S**2).sum()

    hn = np.linalg.norm(hold, axis=1)

    def relerr(k):  # holdout reconstruction, relative to raw vector norm
        V = Vt[:k].T
        rec = mu + (hold - mu) @ V @ V.T
        return float(np.linalg.norm(hold - rec) / np.linalg.norm(hold))

    def relerr_med(k):  # PER-TOKEN median; Frobenius is dominated by massive-activation tokens
        V = Vt[:k].T
        rec = mu + (hold - mu) @ V @ V.T
        return float(np.median(np.linalg.norm(hold - rec, axis=1) / hn))

    # top-k magnitude sparsity: fraction of squared energy in the k largest |coords|
    a = np.sort(np.abs(hold), axis=1)[:, ::-1] ** 2
    cume = np.cumsum(a, axis=1) / a.sum(1, keepdims=True)

    # temporal / static structure
    dts, coss = [], []
    for arr in per_prompt[cut][N_CAL:]:
        d = arr[1:] - arr[:-1]
        dts += list(np.linalg.norm(d, axis=1) / np.linalg.norm(arr[1:], axis=1))
        coss += list((arr[1:] * arr[:-1]).sum(1) /
                     (np.linalg.norm(arr[1:], axis=1) * np.linalg.norm(arr[:-1], axis=1)))
    mu_res = float(np.linalg.norm(hold - mu) / np.linalg.norm(hold))
    # cross-prompt (batch) redundancy: last-position vectors of different prompts
    last = np.stack([a_[-1] for a_ in per_prompt[cut][N_CAL:]])
    ln = last / np.linalg.norm(last, axis=1, keepdims=True)
    xc = ln @ ln.T
    off = xc[~np.eye(len(last), dtype=bool)]

    R["cuts"][cut] = {
        "n_cal_tok": int(len(cal)), "n_hold_tok": int(len(hold)),
        "energy": {k: round(float(energy[k - 1]), 5) for k in KS},
        "holdout_relerr": {k: round(relerr(k), 5) for k in KS},
        "holdout_relerr_median_per_token": {k: round(relerr_med(k), 5) for k in KS},
        # massive-activation diagnostics: is the spectrum dominated by a few raw CHANNELS?
        "chan_share_top1": round(float(np.sort((hold ** 2).mean(0))[::-1][0] / (hold ** 2).mean(0).sum()), 4),
        "chan_share_top4": round(float(np.sort((hold ** 2).mean(0))[::-1][:4].sum() / (hold ** 2).mean(0).sum()), 4),
        "top4_channel_idx": [int(i) for i in np.argsort((hold ** 2).mean(0))[::-1][:4]],
        "tok_norm_median": round(float(np.median(hn)), 1),
        "tok_norm_max": round(float(hn.max()), 1),
        "temporal_rel_delta_median": round(float(np.median(dts)), 4),
        "static_mean_resid_median": round(float(np.median(np.linalg.norm(hold - mu, axis=1) / hn)), 4),
        "cross_prompt_cos_median": round(float(np.median(off)), 4),
        "topk_mag_energy": {k: round(float(cume[:, k - 1].mean()), 5) for k in KS},
        "rank90": int(np.searchsorted(energy, 0.90) + 1),
        "rank99": int(np.searchsorted(energy, 0.99) + 1),
        "rank999": int(np.searchsorted(energy, 0.999) + 1),
        "temporal_rel_delta_norm": round(float(np.mean(dts)), 4),
        "temporal_cos": round(float(np.mean(coss)), 4),
        "static_mean_rel_resid": round(mu_res, 4),
        "cross_prompt_cos_mean": round(float(off.mean()), 4),
        "mean_norm_ratio": round(float(np.linalg.norm(mu) / np.linalg.norm(hold, axis=1).mean()), 4),
    }

# ---- 2. downstream quality: patch cut activation with rank-k reconstruction -----------------
for cut, hi in CUTS.items():
    cal = np.concatenate(per_prompt[cut][:N_CAL])
    mu = cal.mean(0); _, _, Vt = np.linalg.svd(cal - mu, full_matrices=False)
    mu_t = torch.tensor(mu)
    agree = {}
    for k in KS:
        V = torch.tensor(Vt[:k].T.copy())
        def hook(mod, inp, out, V=V):
            h = out[0] if isinstance(out, tuple) else out
            r = mu_t + (h - mu_t) @ V @ V.T
            return (r,) + tuple(out[1:]) if isinstance(out, tuple) else r
        hd = model.model.layers[cut].register_forward_hook(hook)
        ok = sum(int(model(ids(p)).logits[0, -1].argmax()) == base_argmax[N_CAL + i]
                 for i, p in enumerate(PROMPTS[N_CAL:]))
        hd.remove()
        agree[k] = round(ok / len(PROMPTS[N_CAL:]), 3)
    R["cuts"][cut]["top1_agreement_holdout"] = agree
    print(f"cut{cut} top1 agreement: {agree}")

# ---- 3. projection cost, measured, at several H --------------------------------------------
def bench(h, k, reps=2000):
    x = np.random.randn(1, h).astype(np.float32)
    E = np.random.randn(h, k).astype(np.float32); D = np.random.randn(k, h).astype(np.float32)
    for _ in range(200): (x @ E) @ D
    t0 = time.perf_counter()
    for _ in range(reps): (x @ E) @ D
    return (time.perf_counter() - t0) / reps * 1e6  # microseconds, encode+decode

R["proj_us_encode_plus_decode"] = {
    f"H{h}_k{k}": round(bench(h, k), 2)
    for h in (896, 4096, 8192) for k in (h // 8, h // 4)
}
print(json.dumps(R["proj_us_encode_plus_decode"], indent=1))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2a2_lowrank.json"), "w") as f:
    json.dump(R, f, indent=1)
print(json.dumps({c: {k: v for k, v in d.items() if k != "energy"} for c, d in R["cuts"].items()}, indent=1))
