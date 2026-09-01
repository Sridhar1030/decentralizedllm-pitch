---
status: completeness audit — read after 00-SHARED-CONTEXT and 01-VERIFIED-FACTS
role: does the knowledge base actually answer the seven things that were asked for?
date: 2026-09-01
---

# 91 — GAPS: what is covered, what is thin, what nobody owned

Audit of all 25 `teams/` files, 13 ADRs, the four synthesis docs and the two assets, against the seven
items in the brief. **Three gaps are filled at the bottom of this file (§5); the rest are listed, ranked,
and left.** Nothing here invents a number.

## 0. Verdict, one row per ask

| # | Ask | Covered? | Primary home | Thin part |
|---|---|---|---|---|
| 1 | Wired fast-path protocol | **Yes, over-covered** | ADR-002/004, `10-ARCHITECTURE` §3, T1-A1…A5 | Never run over a real NIC. Every figure is loopback. |
| 2 | Compression of inter-layer embeddings | **Yes** | ADR-003, T2-A1…A5 | Answer is "cast to bf16, ship no codec" — a *negative* result that needs framing, not more work. |
| 3 | Queueing / maximise compute | **Yes** | ADR-005/006, T3-A1…A5 | Whole thing is modelled. `demo.sh` still sends one curl, so it demonstrates nothing. |
| 4 | Animated prototype | **Was stale — now patched** | `assets/split-model-bench.html` | Scene 3 still *draws* a chain; only the caption says star. No `demo.mp4`. |
| 5 | Infra / runtime recommendations | **Yes, strongest section** | `20-INFRA-AND-STACK`, ADR-008, T4-A1…A5 | vLLM PP=3 and llama.cpp `--rpc` baselines are recommended and **never run**. |
| 6 | 4–5 slide deck | **Yes, built** | `40-PITCH` §3, `assets/DecentralizedLLM-deck.pptx` (5 slides, verified) | Two `<FILL>` blocks and the ask are still placeholders. |
| 7 | Shared knowledge base | **Yes** | this tree | No index mapping the seven asks to files — that is §0 above, and it did not exist until now. |

Nothing in the brief is unanswered. **The gaps are all of one kind: the knowledge base is far ahead of the
running code.** Every headline is composed from components measured separately; not one end-to-end v1 run
exists. That is the honest summary, and ADR-013 already says so — this file is about the corners it misses.

## 1. Per-item, what is actually missing

### 1 · Wired fast path — thin: reality
Five agents, four transports, a full 40-byte frame spec with a self-checked reference implementation
(T1-A4 §5), and a resolved three-way contradiction on per-hop speedup (ADR-013 C2). **Missing:**

- **Zero measurements over a physical NIC.** Loopback on an M1 Pro *compresses* the ratio (both sides gain
  0.2–0.5 ms of RTT on 1 GbE), so 95x is a loopback number. `20-INFRA` §5 has the runbook; nobody ran it.
- **Docker was not running on the bench host**, so every container-networking figure (veth, bridge, host
  mode) is modelled. The demo runs in exactly that environment.
- **The 40 ms Nagle/delayed-ACK stall — T1-A2's largest single claimed win — cannot fire on loopback.** It
  needs a real switch and a deliberate split write. Do not promise it in a live demo.
- **No auth on DLP by design.** Correct on a trusted bridge, wrong the moment the "decentralised, untrusted
  nodes" story is taken literally. ADR-010 owns the trust claim; the *protocol* has no answer.

### 2 · Compression — thin: nothing, it is complete and the answer is "don't"
T2 measured five families (dtype, byte codecs, low-rank/sparsity, quality guardrails, and the decision) on
the real model. Verdict: **bf16, no codec**, int8+outliers gated to WAN. The one process risk: **`T2-A2`
(low-rank and sparsity) exists on disk and is cited by ADR-003, but was absent from the summary index
handed to synthesis.** It was not lost — check that its negative results (low-rank rejected) survive into
any slide, because a judge asking "did you try SVD?" needs that answer.

### 3 · Queueing — thin: it has never been switched on
`N* = D/D_max = 3`, `K = λ·W_SLO = 6`, `Semaphore(3)`, `Queue(6)`, `429 + Retry-After: 8`. Derived, not
guessed — the best-reasoned section in the KB. **Missing:**

- **The `anyio` 40-thread default is the actual bug** and it is a one-line fix nobody has applied.
- **`demo.sh` sends one curl.** Every queueing number is invisible at R=1, and slide 4 will visibly
  disagree with the screen (8.62 vs 24.21 tok/s). Flagged in four files; fixed in none.
- **Continuous batching's 32.9x is from a different measurement run (ctx=128)** and must never be composed
  with the 19x. Correctly fenced in ADR-013 — the risk is a slide author ignoring the fence.

### 4 · Animated prototype — was the worst-maintained deliverable
T5-A3 specified 8 defects (D1–D8) and `40-PITCH` §6 lists them as checklist item #4, owner "code",
**never applied**. Patched in §5.1 below. **Still missing after the patch:**

- **Scene 3 draws the logical chain.** The caption now says star / 6 crossings; the SVG geometry does not.
  Redrawing it is real work and was left rather than risked blind.
- **No `assets/demo.mp4`.** PowerPoint strips SVG animation and inserts a static frame — the deck's
  centrepiece is a still image until someone screen-records the six scenes.
- **The file is edited concurrently.** Its mtime moved twice during this audit. Re-read before editing.

### 5 · Infra / runtimes — thin: no baseline was ever run
Four runtimes compared with versions and dates, a BOM with `$/(GB/s)`, a rejection table with a number per
row, and the correct buy/build line (buy the per-node engine, build the fabric). **Missing:**

- **The vLLM PP=3 baseline and the llama.cpp `--rpc` baseline are both recommended and unrun.** ADR-008
  and ADR-013 both mark that row of the ledger *empty*. This is the single most valuable hour of work left:
  it converts "we claim we are in the right ballpark" into a number.
- **`ray-vllm/` scaffolding exists in the PoC repo and nobody brought it up.**

### 6 · Deck — thin: only the personal fields
Five slides, built, with tags and caveats in body type (verified by reading the `.pptx` XML). Slide 3 quotes
`8.483 → 0.089 ms` slightly barer than ADR-013 C2 prefers, but the panel carries the reason. **Missing:**
the two load-bearing `<FILL>` blocks (PRIOR BUILDS, STAND OUT) and the ask on slide 5. Those are the user's
to write; inventing them would be fabrication.

### 7 · Knowledge base — thin: navigation
Well-structured, cross-referenced, contradiction-resolved. It had no map from the *brief* to the *files* —
§0 of this document is that map.

## 2. Topics no team owned

| Topic | Why it matters | Severity |
|---|---|---|
| **Model licence and shard redistribution** | Qwen2.5-0.5B is Apache-2.0, but "we ship pre-sliced weight shards" is a *redistribution*, and the 70B story implies Llama, whose community licence has naming and AUP terms. Mentioned nowhere as a decision. | Low for the demo, real for the product |
| **Correctness of the pipeline itself** | Every team optimised; nobody tested that the optimised thing produces the same tokens. ADR-001 called for a check and no check existed. **Filled — §5.2.** | **Was high** |
| **KV cache at 70B** | The memory-wall slide quotes weights only. **Filled — §5.3.** | **Was high** |
| **Coordinator SPOF** | Named in four files as a residual, owned by no ADR as a v1 decision. The failover demo kills node1 — a judge will ask about the coordinator. Answer exists (`40-PITCH` Q4 concedes it); the *fix* is v2. | Medium |
| **Byzantine / malicious-node verification** | T5-A2 designed a v1 scheme (norm bounds + BLAKE3 commit-reveal + 5% re-execution, 80.6% catch over 32 tokens, +1.7% compute) and `40-PITCH` Q7 answers it — but **no ADR records it**, so it is in the pitch and not in the plan. | Medium |
| **Cold-start / weight-load time as a product property** | Treated only as a demo landmine (60–90 s). For a swarm where nodes join and leave, time-to-serve *is* the availability metric. Nobody modelled it. | Medium |
| **Tokenizer placement and the plaintext hop-0 leak** | node0 receives raw token ids. Petals' own fix (client-side embedding) is listed as v2 in two files; the *current* leak is stated but never priced. | Low |
| **Energy / thermal under sustained load** | Only appears as an electricity line in the cost model. Three CPU containers pinned at 100% on a laptop will thermally throttle mid-demo. | Low, but it is a demo risk |
| **What happens at N > 3 nodes** | Placement DP generalises; the registry, the `N*=3` semaphore, the `K=6` queue and the failover ladder are all sized for exactly three. | Low for v1 |

## 3. Judge questions nobody answered

Ranked by how likely they are and how badly the silence lands.

1. **"Your shard is 11.8 GB of weights — where does the KV cache live?"** The natural next sentence after
   the memory-wall slide, and it had no answer anywhere. **Filled — §5.3.**
2. **"How do you know the split version produces the same output as the unsplit one?"** Asked of any
   distributed-inference demo. The KB had a *plan* for this check, not a result. **Filled — §5.2.**
3. **"You recommend vLLM. How do you compare to it?"** No baseline was run. There is no answer. Say
   "we have not run it, here is the exact command we would run" — that is `20-INFRA` §8.
4. **"Kill the coordinator."** Honest answer: total outage, v2 fixes it. Rehearse saying it calmly; the
   failover demo invites the question.
5. **"You are on 0.5B. What have you actually run at 70B?"** Nothing. Every 70B figure is arithmetic. The
   memory-wall slide is *derived*, and must be introduced as such.
6. **"Is this legal to redistribute?"** See §2. Unowned.
7. **"Your demo is an animation. Does the real thing run?"** It does (v0), but `40-PITCH` §2 explicitly
   advises against running it live. Have `docker compose ps` and a pre-recorded terminal ready.

## 4. Ranked backlog of what is left (nothing below was done here)

| # | Item | Effort | Why it is worth it |
|---|---|---|---|
| 1 | Run the vLLM PP=3 + llama.cpp `--rpc` baselines | hours | Fills the only empty row of the claims ledger |
| 2 | Drive `demo.sh` at concurrency ≥3 | minutes | Converts 2.8x of the 19x from modelled to measured, and stops the screen contradicting slide 4 |
| 3 | Record `assets/demo.mp4` | hours | PowerPoint strips SVG animation |
| 4 | Redraw scene 3 as a star | hours | The last surviving factual defect in the prototype |
| 5 | Fill the two `<FILL>` blocks and the ask | minutes | User-only; blocking for submission |
| 6 | An ADR for Byzantine verification | hours | It is in the pitch and not in the plan |
| 7 | One real-NIC transport measurement | hours | Every transport number is loopback |

## 5. What this audit filled

### 5.1 The prototype now matches the decided architecture — `assets/split-model-bench.html`

Applied T5-A3's defect list, which no one had. Twenty edits, JS syntax-checked (`node --check`):

- **The scene-4 clock, which was entirely missing.** A sixth instrument cell, `TOKENS / SEC (modelled)`,
  stepping **1.27 → 6.94 → 8.05 → 8.05 → 24.21** across the four levers. **Lever 3 deliberately does not
  move it**, and its caption now says so — a lever that visibly fails is why the other three are believed.
- **D1 — the DLP header is 40 bytes, not 32**, with the real field list and `struct '<4sBBHIIIIIIBBHI'`.
- **bf16, not fp16.** The prototype claimed `fp16 … KL = 0, 100% top-1`, contradicting ADR-003 and the
  demo script in `40-PITCH` §2. Now bf16 with the measured figures (KL 5.7e-5, 99.74% top-1, 3.5 µs), plus
  the reason bf16 and not fp16: bf16 keeps fp32's exponent range, and the max activation here is 1701.9.
- **D2 — the star.** Scene 3's caption and an on-canvas note now state `3 POSTs = 6 wire crossings/token`.
  *(The SVG still draws the chain — see §1.4.)*
- **The honesty clause in scene 1**: `from_pretrained` loads the whole checkpoint today, and
  `tie_word_embeddings` makes the real split 51.7 / 24.1 / 51.7%.
- **D3 — the three Google Fonts tags are gone.** The file now makes no network request: it renders from
  disk, offline, under a strict CSP.
- **Accessibility and stage control**: `aria-live="polite"` on the caption, a keymap
  (`space` play/pause, `←/→` scrub, `1`–`6` jump to scene, `r` restart), and `overflow-x:auto` on the stage
  so wide content scrolls inside itself rather than the page.

### 5.2 The correctness gate is now run, not asserted — `bench/parity_check.py`

ADR-001 called a `torch.allclose` parity check "non-negotiable" and nobody wrote one. Now written and run
against **real Qwen2.5-0.5B-Instruct weights** (torch 2.10.0, transformers 5.3.0), output in
`bench/parity_check.out`. All three results are **measured**:

| # | what | result |
|---|---|---|
| **A** | 3 chained shards, stateless, vs one monolithic forward | **max abs diff 0.000e+00** — the split is bit-exact |
| **B** | 3 shards, KV-cached, token-at-a-time, `layer_idx` renumbered | `" Paris. It is the largest city in"` — token-identical to reference |
| **C** | same, renumber removed (the §7.4 trap) | `" Paris. Paris is the capital of France"` |

**Row C is the finding.** The bug does not crash, does not warn, and does not produce nonsense — it
produces a *different, perfectly reasonable sentence*. No eyeball review catches that. This upgrades the
sharded-inference correctness claim from "we reasoned about it" to "we ran it", and it is a better
demo-day answer than any speedup. Written into `10-ARCHITECTURE` §7.4 and `ADR-001`.

The script also pins two things the prose left implicit: non-final shards must run with `norm = Identity`
(the final node owns `norm` + `lm_head`), and **each shard owns its own cache object** — three independent
caches handed back per hop, never shared.

### 5.3 The 70B KV-cache arithmetic — `40-PITCH` §Q10 and the slide-5 preamble

Every 70B memory figure in the KB was weights-only. Derived from Llama-3.3-70B's published config (80
layers, 8 KV heads, head_dim 128, fp16): **327,680 B = 320 KiB per token whole-model, 106.7 KiB per 3-way
shard.** Three consequences now on the record:

1. **The laptop claim survives with a stated bound** — 16 GB − 11.8 GB int4 shard = 4.2 GB headroom =
   **5 concurrent 8k sessions, or one 41,287-token session.** Quote the bound, not just the fit.
2. **Concurrency breaks it before context does** — 10 concurrent 8k sessions is 8.33 GiB of KV per shard.
   Admission control (ADR-005) is a *memory* controller here, not only a latency one.
3. **Past 115,998 tokens of context, KV per shard (13.3 GiB at 128k) exceeds the int4 weight shard
   (11.8 GiB).** At long context the thing worth sharding is the cache, not the model — which is the
   strongest v2 statement of our own thesis and the best answer to "why will this matter at 70B".
