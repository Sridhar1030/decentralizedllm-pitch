---
team: T5 — Product, Narrative & Deliverables
agent: T5-A5
topic: Literal submission-form copy, elevator pitch, timed demo script, and 5 slide titles + speaker notes
headline: >
  Ship the title "No Node Knows" only if the pre-sliced-weights fix lands before submission — today
  node.py:36 loads the full checkpoint on every node, so the name would be a claim the code disproves;
  fallback title is "Shardmind". Every personal credential field is left as <FILL: ...> on purpose:
  fabricating a hackathon win or an employer on a real submission is not a risk worth any amount of polish.
---

# T5-A5 — The literal words

> ## READ THIS FIRST — nothing below invents your credentials
> Every place a real personal fact is required is marked **`<FILL: ...>`**. I did not write your hackathon
> wins, your employers, your repos, or your job title, because I do not know them and inventing them on a
> real submission form is fraud, not copywriting. The **only** biographical fact used below is the one
> that is verifiably true from `/Users/srpillai/CODING/DecentralizedLLM/`: a working PoC exists, splits
> `Qwen/Qwen2.5-0.5B-Instruct` across three Docker containers, and runs.
> **Two `<FILL>` blocks are load-bearing** (`PRIOR BUILDS`, `STAND OUT`). Submitting them unfilled is worse
> than submitting them short. Delete a placeholder you cannot honestly fill — an empty line beats a vague one.

---

## 1. IDEA TITLE — three options, one pick

| # | Title | Chars | Why it works | Why it might not |
|---|---|---|---|---|
| **A** | **No Node Knows** | 13 | Names the *property*, not the mechanism. Three stressed syllables, sticks after one hearing. A judge repeats it to another judge without a slide. Doubles as the demo's punchline (kill a node → outage). | Doesn't contain "LLM". And it is a **claim the current code disproves** — `node.py:36` calls `AutoModelForCausalLM.from_pretrained()` on every node, so every node holds the full checkpoint at boot (T1-A5, T5-A2). |
| B | Shardmind | 9 | Brandable, single word, no claim to disprove, survives a logo. | Says nothing. Judge has to read the description to learn anything. |
| C | Split the Model, Not the Data | 29 | Carries the whole value prop; the sovereignty buyer (T5-A1) recognises themselves instantly. | Not a *name*, it's a tagline. Form says "short, punchy name". Hard to say twice. |

**PICK: A — "No Node Knows"** — conditional on shipping the pre-sliced-safetensors fix (~1 day, T5-A2 §Q-load)
before you submit. Name the property you have, not the one you plan. **If that fix does not land, submit B and use
A as the deck's subtitle**, where an aspirational phrase is allowed and a product name is not.

---

## 2. DESCRIPTION — "What are you building, for whom, and how?"

### 2a. 60-word version (use if the field is cramped or a card view truncates)

> No Node Knows splits one LLM across several machines so that no single machine ever holds the whole model.
> Our working prototype runs Qwen2.5-0.5B across three Docker containers: node 0 owns layers 0–7, node 1
> layers 8–15, node 2 layers 16–23 plus the output head. Kill any one and the model stops. For teams whose
> data cannot leave the building.

*(60 words / 355 chars. Last sentence is the buyer; keep it even under truncation pressure.)*

### 2b. 146-word version (default — use this)

> We are building **No Node Knows**: one language model sharded by layer across N machines, where no single
> machine can hold or reconstruct the whole model. A prompt enters the coordinator; node 0 embeds it and runs
> its layers, hands a hidden state to node 1, then to node 2, which owns the output head and returns the token.
>
> **For whom:** hospital consortia, EU banks under DORA, and any team whose data may not leave the building
> but whose model no longer fits on one device.
>
> **How:** a working PoC already runs Qwen2.5-0.5B-Instruct across three CPU containers today. From that
> measured baseline we are landing four fixes — a per-shard KV cache (271x less redundant compute), argmax
> moved onto the tail node (607,744 bytes to 4 per token), a binary bf16 frame replacing JSON and base64, and
> a rebalanced split that deletes a 1.55x pipeline stall.

*(146 words. Every number is (derived) from `config.json` — see `01-VERIFIED-FACTS.md`.)*

**Optional device-forward swap** — the form asks for Android proficiency, so if this hackathon is on-device
Android, replace the *For whom* paragraph with:
> **For whom:** anyone holding three ordinary devices and a model that fits on none of them. An 8B int4 model
> is 4.0 GB — too much for one phone's app heap; 1.3 GB per device across three. Same trick, consumer scale.

*(1.3 GB/shard is (derived): 8e9 params x 0.5 B/param int4 = 4.0 GB, / 3 = 1.33 GB. Do **not** claim a phone build you have not run.)*

---

## 3. THE OTHER FORM FIELDS

| Field | What to put | Note |
|---|---|---|
| VIDEO WALKTHROUGH URL | `<FILL: unlisted YouTube/Loom URL>` | Optional on the form, **not optional in practice.** Record §6 verbatim over the animated prototype. 2:00 hard cap. |
| PROTOTYPE URL | `<FILL: URL of the published animated prototype>` | `knowledge-base/assets/split-model-bench.html` — self-contained, publishable as-is. |
| DECK / DOCUMENT | 5 slides, §7 | Required. |
| ANDROID PROFICIENCY | **Leave "Basic"** unless it is untrue | Do not inflate. §5 turns Basic into an asset by scoping Android to v2 explicitly. A judge forgives "Basic"; nobody forgives a demo that contradicts a claimed "Advanced". |
| LLM PROFICIENCY | **"Deployed local LLMs on-device"** — keep **only if true** | If you have only run local models on a laptop, say so. `<FILL: confirm this is accurate — which model, which device, which runtime>` |

---

## 4. PRIOR BUILDS & HACKATHONS — draft (first person)

> I built the prototype in this submission end to end: **DecentralizedLLM**, a layer-sharded inference stack
> that splits `Qwen/Qwen2.5-0.5B-Instruct` across three Docker containers. Each container is a FastAPI layer
> node holding a contiguous slice of the 24 transformer layers; a coordinator drives the forward pass hop by
> hop; an API gateway in front adds key auth and a circuit breaker; Prometheus and Grafana are wired in. It
> works today — `docker compose up`, one curl, a completion comes back, with an SSE streaming path for live
> token flow. Stopping node 1 takes the model down, which is exactly the property I am pitching.
>
> `<FILL: hackathons — event name, year, what you built, placing. One line each. "Won X with Y" only if you won X. Delete this paragraph entirely if you have none — no hackathon history is a neutral fact, a vague one is a red flag.>`
>
> `<FILL: shipped at work — one line each: what shipped, at what scale, what was YOURS. Do not list team output as personal output.>`
>
> `<FILL: OSS — github.com/<user>/<repo>, one line on what it does. Include stars/downloads only if the number helps you.>`
>
> `<FILL: on-device / Android work — any app you shipped, any local model you have actually run on a phone (llama.cpp, MediaPipe LLM Inference API, MLC-LLM, ONNX Runtime Mobile), naming the device and the model. Delete if none — §5 already handles the gap.>`

---

## 5. WHAT MAKES YOU AND YOUR TEAM STAND OUT — draft (first person)

> **I brought a working system, not a slide.** The three-node split runs before the pitch starts, so the demo
> is the product rather than a mock of it.
>
> **I profiled my own build before defending it, and the deck leads with what I found wrong.** The 8/8/8 layer
> split looks balanced and is not: `lm_head` is 136M parameters — 9.13 transformer layers of compute per token
> — so node 2 really carries 17.13 layer-equivalents against 8.00, a 1.55x throughput loss. The coordinator
> runs `argmax`, so node 2 ships a 607,744-byte fp32 logit vector back per token when a 4-byte token id would
> do. There is no KV cache, so a 32-token prompt generating 512 tokens performs 147,200 position-forwards per
> node where 543 suffice — 271x redundant. I would rather a judge hear those three numbers from me than find
> them in my repo.
>
> **I know exactly what this is not.** It is not cheaper than a hosted API and I will not claim it is. Below
> ~13B parameters, splitting a model across machines is a stunt; the architecture earns its place at the point
> where the weights stop fitting on the device you are allowed to use. That boundary is the product.
>
> `<FILL: your domain edge — the industry, regulation, or infrastructure you have actually worked in that makes THIS problem yours. Distributed systems? Healthcare or fintech data handling? On-device ML? One specific sentence beats three general ones.>`
>
> `<FILL: team — who else, what they own, and whether you have shipped together before. Delete this block if you are solo, and say "solo" plainly; solo is not a weakness on a working prototype.>`
>
> `<FILL: Android specifically — if your Android proficiency is Basic, say so here in one line and state the plan: the node is a plain HTTP service today, and the v2 path to a phone node is a known, bounded piece of work, not a mystery. Owning the gap is stronger than hiding it.>`

---

## 6. THE 30-SECOND ELEVATOR PITCH (74 words = 30 s — say it at ~150 wpm)

> A hospital consortium wants one model trained on all of their data, and no member — and no vendor — may
> ever hold the whole thing. So we cut the model up. Twenty-four layers, three machines, eight layers each.
> Every machine computes its slice and passes a hidden state along; the last one emits the token. Take any
> machine away and the model is gone. It runs today, on three CPU containers, no GPUs.

*Delivery notes: pause after "so we cut the model up" — that is the idea. The last sentence is the credibility
line; land it flat, no emphasis. Do not say "revolutionary", "decentralized AI", or "democratize".*

---

## 7. THE 2-MINUTE DEMO SCRIPT — keyed to `assets/split-model-bench.html`

Prototype has 6 scenes (8+10+14+20+9+10 = 71 s of animation). Speak at **~145 wpm**; scrub/pause the prototype
to the cues below rather than letting it autoplay — the timings assume you control the scrubber.

| Time | Prototype scene | Spoken words (verbatim) | w |
|---|---|---|---|
| **0:00–0:14** | **01 — One model, three machines** | "This is one language model — twenty-four transformer layers. We cut the stack into three shards, one per machine, each in its own container. No single node holds the full model. Everything else follows from that." | 35 |
| **0:14–0:32** | **02 — One token, three hops** | "Here's one token. The prompt hits the coordinator. Node zero embeds it, runs layers zero through seven, hands a hidden state to node one, which hands it to node two. Node two owns the output head — so node two is where a token is born." | 43 |
| **0:32–0:58** | **03 — What v0 actually costs** | "Now the honest part. We measured our own baseline. There's no KV cache, so the whole sequence is re-sent and re-computed every single token — watch the packet grow. It travels as float32, inside base64, inside JSON. And argmax runs on the coordinator, so node two ships six hundred thousand bytes of logits back per token. Meanwhile two of three nodes sit dark." | 62 |
| **0:58–1:30** | **04 — Four levers** *(step through all four)* | "Four levers. One: cache K/V per shard, send only the newest position, and move argmax onto node two — the return path becomes a four-byte token id instead of six hundred kilobytes. Two and three: a binary frame on a persistent socket, bf16 instead of float32. Four: the output head is nine layers' worth of compute, so our 'equal' eight-eight-eight split really runs eight-eight-seventeen. Re-cut it, and run three requests at once — one request can never fill a pipeline." | 78 |
| **1:30–1:46** | **05 — When a node disappears** | "Kill node one. A third of the model is gone — today we return a five hundred. With a standby holding the same layers, the coordinator reroutes; the honest cost is that the dead node's cache dies with it." | 38 |
| **1:46–1:58** | **06 — What the levers bought** | "Nine hundred thirty-five times fewer bytes, two hundred seventy-one times less redundant compute. Caveat, same slide: on a fast LAN we're compute-bound — the wall clock comes from the recompute, not the bytes." | 32 |
| **1:58–2:00** | hold on scene 06 | "One model. Three machines. None of them knows it." | 9 |

*(297 words = 123 s at 145 wpm, ±3 s. If you run long, cut "Two and three" from 0:58–1:30 — levers 1 and 4
carry the numbers judges remember. If you run short, restore "Bytes win on WAN, on one-gig ethernet, and at
long context" to the 1:46 row.)*

**If you demo live instead of on video:** `docker compose up -d`, then `./demo.sh` — steps 2 and 3 show token
flow and a real completion, step 5 stops node1 and shows the outage. Do **not** demo the live stack and the
animation both; pick one. The animation is safer and 30x faster.

---

## 8. THE DECK — 5 slide titles + speaker notes

| # | Slide title (on the slide, verbatim) | On-slide content | Speaker notes |
|---|---|---|---|
| 1 | **No node knows** | One line: *"One model. Three machines. None of them holds it."* + the 3-shard diagram (`assets/pipeline.svg`) | Open with the hospital sentence from §6 — a named buyer in the first ten seconds. Do not explain the architecture yet. 20 s. |
| 2 | **One token, three hops** | The forward path: coordinator → node0 (embed + L0-7) → node1 (L8-15) → node2 (L16-23 + lm_head) → token | Say "this is pipeline parallelism, GPipe defined it in 2019 and Petals shipped it decentralized in 2022 — we did not invent the mechanism." Claiming novelty here is how you lose a technical judge. What's new is that the shard boundary is a **trust** boundary. 25 s. |
| 3 | **We profiled our own baseline first** | Three rows only: no KV cache → **271x** redundant position-forwards; argmax on coordinator → **607,744 B → 4 B**; 8/8/8 split is really 8/8/17 → **1.55x** stall | This slide buys every other slide's credibility. Deliver it as confession, not as achievement. "We measured this on our own code this week." 30 s. |
| 4 | **Four levers, 935x fewer bytes — and the caveat** | KV cache + local argmax · binary DLP frame · bf16 wire · rebalance + R=3 concurrency. Then, in the same size type: *"wire bytes, not wall clock. On a fast LAN we are compute-bound."* | The caveat is not a weakness, it is the reason the judge believes the 935x. Put it on the slide so you are the one who says it. 30 s. |
| 5 | **What this is, and what it is not** | IS: the option when the weights don't fit the device you're allowed to use. IS NOT: cheaper than an API; not novel pipeline parallelism; not useful under ~13B. Then the ask. | Close on the memory wall: at 70B fp16 a 3-way shard is 47 GB and fits nothing; at int4 it's 11.8 GB and fits a laptop. End with the ask, then stop talking. 25 s. |

*(5 slides = ~2:10 of speaking, leaving the rest of a 5-minute slot for the demo and questions. If the deck is
capped at 4, merge 1 and 2 — the diagram carries both.)*

---

## 9. RISKS IN THIS COPY

| Risk | Mitigation |
|---|---|
| Title claims a property the code lacks (`node.py:36` loads the full checkpoint) | Ship pre-sliced safetensors before submitting, or fall back to title B. Non-negotiable — this is the first thing a judge who opens the repo will find. |
| `<FILL>` blocks submitted unfilled | Two of them are load-bearing (§4, §5). Set a hard checkpoint before you hit submit. |
| "935x" quoted without its caveat | It is baked into the §7 script and slide 4 on purpose. Never say it naked. |
| Android proficiency "Basic" vs an Android-judged hackathon | §5's last `<FILL>` owns the gap explicitly. Do not upgrade the self-rating to match the audience. |
| The 1.3 GB/phone figure in §2b | (derived) from int4 arithmetic only — no phone build exists. Never state it as measured. |
