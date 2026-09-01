"""Ground-truth arithmetic for Qwen2.5-0.5B-Instruct split across 3 shards.
Config values fetched from HF config.json and asserted here."""
H, L, AH, KVH, V, INTER = 896, 24, 14, 2, 151936, 4864
HD = H // AH
assert HD == 64

# --- params ---
embed = H * V                                   # tied: embed_tokens IS lm_head
attn  = H*H + H*(KVH*HD) + H*(KVH*HD) + H*H     # q,k,v,o  (GQA => k,v are small)
mlp   = 3 * H * INTER                           # gate,up,down
layer = attn + mlp
total = embed + L*layer                          # tied embedding counted once
print(f"embed/lm_head params : {embed:,}   ({embed/total:.1%} of model)")
print(f"per-layer params     : {layer:,}")
print(f"total params         : {total:,}   (published: 494M)")

# --- compute cost per decoded token, in 'layer-equivalents' ---
lm_head_eq = embed / layer
print(f"\nlm_head compute      : {lm_head_eq:.2f} transformer layers' worth of MACs")
shards = {"node0 (embed+L0-7)": 8, "node1 (L8-15)": 8, "node2 (L16-23+lm_head)": 8 + lm_head_eq}
tot_eq = sum(shards.values())
for n, eq in shards.items():
    print(f"  {n:24s} {eq:6.2f} eq  {eq/tot_eq:6.1%}")
slow = max(shards.values()); ideal = tot_eq/3
print(f"  bottleneck stage {slow:.2f} eq vs balanced {ideal:.2f} eq -> {slow/ideal:.2f}x worse than balanced")
print(f"  balanced cut: node0=L0-10(11.0) node1=L11-21(11.0) node2=L22-23+lm_head({2+lm_head_eq:.2f})")

# --- KV cache (GQA makes this tiny) ---
kv_tok_layer = 2 * KVH * HD * 2                 # K+V, fp16
print(f"\nKV bytes/token/layer : {kv_tok_layer}   all {L} layers: {kv_tok_layer*L:,} B ({kv_tok_layer*L/1024:.0f} KB)")
print(f"KV for 2048 ctx      : {kv_tok_layer*L*2048/1e6:.1f} MB whole model, {kv_tok_layer*8*2048/1e6:.1f} MB per 8-layer shard")

# --- v0 O(n^2) resend waste ---
P, G = 32, 512
v0_pos = G*P + G*(G-1)//2                       # sum of sequence lengths resent
v1_pos = P + (G-1)
print(f"\nposition-forwards per node, P={P} G={G}:  v0 {v0_pos:,}   v1(KV cache) {v1_pos:,}   -> {v0_pos/v1_pos:.0f}x recompute")

# --- wire bytes, whole generation ---
B64 = 4/3
hid_v0   = v0_pos * H*4 * B64 * 2               # 2 hidden hops, fp32, base64
logit_v0 = G * (V*4) * B64                      # node2 -> coordinator, full fp32 logit vector
v0 = hid_v0 + logit_v0
hid_v1   = (P + (G-1)) * H*2 * 2                # bf16, binary, KV-cached (1 vector/step)
tok_v1   = G * 4                                # node2 argmaxes locally, returns a token id
v1 = hid_v1 + tok_v1
print(f"\nwire bytes for one {G}-token generation:")
print(f"  v0 hidden  {hid_v0/1e6:9.1f} MB   v0 logits {logit_v0/1e6:9.1f} MB   v0 total {v0/1e6:9.1f} MB")
print(f"  v1 hidden  {hid_v1/1e6:9.3f} MB   v1 tokens {tok_v1/1e6:9.6f} MB   v1 total {v1/1e6:9.3f} MB")
print(f"  reduction  {v0/v1:,.0f}x")

# --- the logits-return-path crossover ---
print(f"\nlogits payload {V*4:,} B vs hidden payload seq*{H*4} B")
print(f"  -> full logit vector is the LARGEST payload on the wire until seq_len > {V*4/(H*4):.0f} tokens")
print(f"  -> with a KV cache (hidden = {H*4} B) logits are {V*4/(H*4):.0f}x larger than the activation")
