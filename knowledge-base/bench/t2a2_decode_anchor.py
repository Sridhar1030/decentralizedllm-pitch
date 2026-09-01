"""T2-A2 anchor: measured cost of ONE cached decode step (24 layers, fp32, CPU, 2 threads),
so the microseconds saved by activation compression can be put in scale."""
import os, time, json
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False); torch.set_num_threads(2)
M="Qwen/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(M)
m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32,low_cpu_mem_usage=True).eval()
ids=torch.tensor([tok("Explain photosynthesis in two sentences, clearly and simply.")["input_ids"]])
o=m(ids,use_cache=True); pkv=o.past_key_values; nxt=o.logits[:,-1].argmax(-1,keepdim=True)
ts=[]
for _ in range(12):
    import copy
    t=time.perf_counter(); o2=m(nxt,past_key_values=pkv,use_cache=True); ts.append((time.perf_counter()-t)*1e3)
ts=sorted(ts)
r={"decode_step_ms_24layer_min":round(ts[0],2),"decode_step_ms_median":round(ts[len(ts)//2],2),
   "per_8layer_shard_ms_min":round(ts[0]/3,2),
   "note":"fp32 CPU, 2 threads, batch 1, KV cache on; lm_head included in the 24-layer number"}
print(json.dumps(r,indent=1)); json.dump(r,open("t2a2_decode_anchor.json","w"),indent=1)
