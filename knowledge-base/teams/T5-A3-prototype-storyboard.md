---
team: T5 — Product, Narrative & Deliverables
agent: T5-A3
topic: Build spec for the animated demo prototype — 6-scene storyboard, exact copy, visual language, and the "animated MD" variant
headline: >
  The prototype already exists at assets/split-model-bench.html and its scene skeleton is correct — this
  spec freezes it and lists 8 defects to fix, of which three are factual: the DLP header is 40 B not 32 B,
  v0 is a STAR (3 POSTs = 6 wire crossings) not the chain scene 3 draws, and "no node holds the full model"
  is false at boot today (node.py loads the entire checkpoint, then `del full`). Scene 4 also has no clock,
  and the brief demands one: 1.27 → 6.94 → 8.05 → 8.05 → 24.21 tok/s, 19.0x, with lever 3 deliberately
  not moving it. "Animated markdown" means exactly one thing that works: a CSS-keyframed SVG embedded as
  an image. PowerPoint strips SVG animation — the deck needs a screen recording.
---

# T5-A3 — Prototype build spec

**Status: not greenfield.** `assets/split-model-bench.html` (712 lines) already implements all six scenes,
the scrubber, per-scene buttons, play/pause and speed. This document is the **contract**: §1 freezes what
exists, §2 lists the deltas, §4 is the copy the builder may not deviate from.

## 1. Deliverables

| # | file | what | status |
|---|---|---|---|
| P1 | `assets/split-model-bench.html` | the demo prototype, 6 scenes, 71 s loop | exists — apply §2 |
| P2 | `assets/pipeline.svg` | the README/"animated MD" variant | exists (SMIL) — convert per §7 |
| P3 | `bench/wire_anim.py` | terminal ASCII variant, stdlib only, ~40 lines | **new**, §7.3 |
| P4 | `assets/demo.mp4` | screen recording of P1 for the deck | **new** — PowerPoint cannot animate SVG (§7.2) |

Hard constraints, all currently met by P1 except D1: single file · zero network requests · inline CSS/JS/SVG ·
light **and** dark · responsive · opens from `file://` · publishes unchanged as an Artifact.

## 2. Defects to fix (this is the work)

| id | defect | fix | source |
|---|---|---|---|
| **D1** | `<link>` to `fonts.googleapis.com` — breaks "runs offline from disk" and is the only network request in the file | delete the three font tags; the CSS already falls back to `ui-monospace`/system sans | brief |
| **D2** | lever-2 copy says "fixed **32**-byte little-endian header" | **40-byte** header | T1-A4 headline; T2-A5 `3584+40=3624` |
| **D3** | every scene draws a chain `coord→n0→n1→n2`. v0 is a **star**: 3 POSTs = **6 wire crossings**, 4 of them activation-sized | in scene 03 only, add the return leg n_i→coord after each hop and the note in §4 | T1-A1 §1 |
| **D4** | scene 01 asserts "no single node ever holds the full model" — false at boot | append the honesty clause in §4; it is the strongest credibility move in the deck | `node.py:40-45` `from_pretrained` then `del full`; FINDING 1 `tie_word_embeddings=true` |
| **D5** | scene 04 has **no clock**, and the brief requires one that visibly speeds up | add the tok/s readout + metronome, §5 | brief |
| **D6** | scene 03 sizes hop-0 ids at 6 B/token | 7 B/token | T1-A1 §2 (127 B at seq=16, JSON-wrapped) |
| **D7** | no keyboard control, caption changes are silent to a screen reader | §6 keymap; `aria-live="polite"` on `#sceneCap`, `aria-atomic="true"` | a11y baseline |
| **D8** | at 375 px the 1180-unit viewBox scales 10.5 px type down to 3.3 px | wrap `.stage` in `overflow-x:auto`, `svg{min-width:900px}` below 760 px; meters/captions stack | responsive rule |

Do **not** change: the scene count, order, durations, the packet-width law, or the token palette.

## 3. Visual language (frozen — already in P1)

**Colour.** Tokens on bare `:root`, redefined under `@media (prefers-color-scheme: dark) :root:not([data-theme="light"])`
**and** `:root[data-theme="dark"]`. No colour may be defined only inside a media block.

| token | light | dark | means |
|---|---|---|---|
| `--ground` / `--surface` / `--surface-2` | `#eaeef4` / `#ffffff` / `#dfe6ef` | `#0a0f15` / `#111a23` / `#17212c` | page, card, inset |
| `--line` | `#c3cfdd` | `#26323e` | wire at rest, borders |
| `--ink` / `--ink-2s` / `--ink-3` | `#0e141c` / `#445265` / `#78879a` | `#e9eff6` / `#9fb0c2` / `#6a7c8f` | body / caption / annotation |
| `--n0` / `--n1` / `--n2` | `#0d7d74` / `#4c4bd0` / `#c2306b` | `#2dd4bf` / `#8f8ffb` / `#f472a6` | **node identity — a packet is always the colour of the node that sent it** |
| `--ok` / `--bad` / `--accent` | `#15803d` / `#be123c` / `#a86200` | `#4ade80` / `#fb7185` / `#f0a83c` | v1 / v0-defect / narration |
| `--stack` | `#5a6b7e` | `#7d8ea1` | unassigned layer bar |

**Type.** Headings `Archivo` 600/700 → system sans. Body `IBM Plex Sans` → `-apple-system`. Every number,
label and annotation in `IBM Plex Mono` → `ui-monospace, SFMono-Regular, Menlo, monospace`. With D1 applied
the webfonts are gone; the fallbacks are the design.

**Glyphs.**

| glyph | spec |
|---|---|
| **layer** | `rect` 46×9 r2, `--stack` when unassigned, node hue at 78% opacity once sharded. 24 of them, 13 px pitch. |
| **node** | `rect` 176×182 r5, `--surface`, 1.5 px `--line`. Active: stroke → node hue, 2.5 px. Idle: `opacity .45`. Dead: stroke `--bad`, `opacity .42`, badge `DOWN`. Name top-left in node hue, layer range bottom-left in `--ink-3`, badge top-right. |
| **wire** | 2 px line at `--line`; carrying → sender's hue. Return path is the same weight, `stroke-dasharray 5 5`. |
| **packet** | `rect` height 22 r3, filled sender hue, plus a 20%-opacity blurred copy behind it. Byte label centred 9 px above in the same hue. **Width = `clamp(9 + 15·log10(bytes), 9, 132)`** — the caption `packet width ∝ log(bytes) — a 400,000x range will not fit a linear scale` stays on screen permanently. This is the single most important honesty device in the piece. |
| **util meter** | 3 vertical bars, one per node, filled to `min(1,u)` in that node's hue; empty = transparent, not grey. |
| **readouts** | 5 fixed slots: forward payload · return path · utilisation · cumulative wire · encoding. They never disappear between scenes, so the viewer's eye learns one location per fact. |

## 4. Storyboard and EXACT copy

Durations are the contract. Total **71 s** — one loop inside a 90 s demo slot leaves 19 s of talking room.
Caption HTML below is verbatim; `<b>` = emphasis, `<code>` = identifier. Nothing here is a paraphrase target.

| # | dur | title | what moves | what the viewer learns |
|---|---|---|---|---|
| 01 | 8 s | One model, three machines | 24 stacked bars fly apart into 3 shards; node cards + wires fade in behind them | the split is physical, not logical |
| 02 | 10 s | One token, three hops | one packet walks seg0→seg1→seg2, cards light in sequence, fat logits packet returns along the dashed path, token `"The"` prints | the hop is the unit of work; node2 owns `lm_head` |
| 03 | 14 s | What v0 actually costs | simulated token counter runs 1→300, seq grows, packet grows with it; 2 of 3 cards dark at all times; util pinned 33%; **(D3)** each hop returns to the coordinator before the next | the three defects are one picture |
| 04 | 20 s | Four levers | 4 sub-scenes of 5 s; packet shrinks twice, split re-cuts to 11/11/2, 3 packets ride at once, **(D5)** clock accelerates | each fix, isolated, with its price |
| 05 | 9 s | When a node disappears | node1 goes red then dark; pipeline stalls; standby takes the range; traffic resumes | failure is designed for, not hidden |
| 06 | 10 s | What the levers bought | 6 scoreboard rows wipe in at 0.5 s intervals | the numbers, with the caveat attached |

**Scene captions — verbatim.**

- **01** — `Qwen2.5-0.5B is <b>24 transformer layers</b>. We cut the stack into three contiguous shards and give one to each machine. Every shard runs in its own container, and at inference <b>no single node ever holds the full model</b>. Honest footnote, on the record: today each container still calls <code>from_pretrained</code> and loads the whole checkpoint at boot before discarding the layers it does not own — and because <code>tie_word_embeddings</code> is true, node&nbsp;0's embedding and node&nbsp;2's <code>lm_head</code> are the same 136M-parameter matrix. Per-shard weight files are v1.`  **(D4)**
- **02** — `A prompt enters the coordinator. Node&nbsp;0 embeds it and runs its layers, then hands the <b>hidden state</b> to node&nbsp;1, which hands it to node&nbsp;2. Node&nbsp;2 owns <code>lm_head</code>, so it produces the logits — and one token comes back.`
- **03** — `Three defects, all visible at once. There is <b>no KV cache</b>, so the whole sequence is re-sent and re-computed every single token — the packet grows without bound. It travels as <b>fp32 inside base64 inside JSON</b>. And <code>argmax</code> runs on the coordinator, so node&nbsp;2 ships back the entire <b>607,744-byte logit vector</b> per token. Meanwhile two of three nodes sit dark.`
- **03 · on-canvas note (D3)** — `every hop returns to the coordinator first — 3 POSTs, 6 wire crossings per token`
- **05** — `Kill node&nbsp;1 and one third of the model is gone — v0 returns a 500. With a hot standby holding the same layer range, the coordinator re-routes. The cost is honest: the dead node's <b>KV cache dies with it</b>, so the session replays its prompt on the standby before decoding resumes.`
- **06** — `<b>935× fewer bytes</b> on the wire for a 512-token generation, and <b>271×</b> less redundant compute. The caveat that belongs on the same slide: on a fast LAN v0 is <b>compute-bound</b>, so the wall-clock win comes from the recompute and the rebalance, not the bytes. Bytes dominate on WAN, on 1 GbE, and at long context.`

**Lever captions — verbatim, one per 5 s sub-scene.**

- **L1 · KV cache + local argmax** — `<b>Lever 1 — stop sending what the other end already has.</b> Cache K/V per shard and send only the newest position: <code>[1, 896]</code> instead of <code>[seq, 896]</code>. GQA makes this nearly free — 2 KV heads × 64 dims = <b>512 B per token per layer</b>, so a 2048-token context is 25 MB for the entire model. Move <code>argmax</code> onto node 2 and the return path becomes a <b>4-byte token id</b>.`
- **L2 · DLP binary frame** — `<b>Lever 2 — stop encoding tensors as text.</b> A fixed <b>40-byte</b> little-endian header (magic, version, request_id, session_id, dtype, codec, shape, CRC32C) then the raw buffer, length-prefixed on a <b>persistent, pipelined TCP connection</b> with <code>TCP_NODELAY</code> set. That removes base64's 33% inflation, the JSON parse of a multi-megabyte string, and — worst of all — the fresh TCP handshake v0 pays on <em>every hop of every token</em>.`  **(D2)**
- **L3 · bf16 on the wire** — `<b>Lever 3 — compress the activation itself.</b> fp32 → bf16 halves it for free: 3,584 B → 1,792 B, measured at 99.41% top-1 agreement for 3.5 µs. int8 with a <b>per-token fp16 scale plus 8 fp16 outlier channels</b> reaches 906 B — and the outlier handling is the part that matters, because channel 62 of this model runs 972× the median and collapses naive int8 to 1.4% agreement. Watch the clock: <b>it does not move.</b> On a LAN this lever buys bytes, not time. It pays on WAN and at long context.`
- **L4 · Rebalance + concurrency** — `<b>Lever 4 — fill the pipeline.</b> <code>lm_head</code> is 136M params — <b>9.13 layers' worth of compute</b> — so the "equal" 8/8/8 split really runs 8/8/17 and is <b>1.55× off balance</b>. Re-cut to 11/11/2+head. Then note that a single request <em>cannot</em> fill a pipeline: token t+1 depends on token t. Only <b>concurrent requests</b> can. With R requests over S stages, utilisation is min(1, R/S) — so R=3 lights all three nodes.`

**Footer — verbatim.** `Every figure is derived from the real config.json of Qwen2.5-0.5B-Instruct (H=896, 24 layers, 2 KV heads, V=151,936) and from the v0 source, not estimated. Byte counts and recompute factors are exact arithmetic. Wire totals are payload-only on both sides; the 40 B DLP header adds 43 KB over a 512-token generation (+2.2%).`

## 5. The clock (D5) — scene 04's spine

A `tok/s` digit in the readout strip plus a metronome dot whose period is `1/(tok/s)` seconds, scaled ×4 so
the acceleration is visible. Values step at each lever boundary; the digit tweens over 400 ms.

| after lever | per-token budget | tok/s | tag | arithmetic |
|---|---|---|---|---|
| — (v0) | 785.3 ms | **1.27** | measured | T3-A4: measured v0 wall clock incl. transport |
| L1 KV cache | 123.94 + 20.2 = 144.1 ms | **6.94** | modelled | decode compute 123.94 ms (measured, T3-A4) + 3 POSTs × 6.717 ms httpx build (measured, T1-A1) |
| L2 DLP frame | 123.94 + 3×0.089 = 124.2 ms | **8.05** | modelled | DLP hop RTT 0.089 ms measured, T1-A4 |
| L3 bf16 | 124.2 ms | **8.05** | modelled | **unchanged on purpose** — compute-bound |
| L4 rebalance + R=3 | bottleneck 41.31 ms | **24.21** | modelled | T3-A4: balanced D_max, N\*=3 |

**19.0× end to end (modelled)** = 24.21 / 1.2734. Cumulative wire per lever, 512-token generation:
`5.19 MB → 3.89 MB → 1.95 MB → 1.95 MB` (543 positions × 2 crossings × {4779, 3584, 1792, 1792} B).

## 6. Interaction

| control | behaviour |
|---|---|
| autoplay | on, loops at T=TOTAL. `prefers-reduced-motion: reduce` → start paused at scene 01 end-state, all six scenes still reachable by button. |
| render model | `render(T)` is a **pure function of absolute time** — no accumulated state. Scrubbing is exact, not approximate. Keep it that way; every new effect must be `f(T)`. |
| scrubber | `<input type=range min=0 max=1000>`, maps to `T/TOTAL`; `input` renders immediately without unpausing |
| scene buttons | 6 buttons `NN  Title`, jump to `Σt+0.01`, active one gets `.on` |
| play/pause | toggles, label Pause/Play, `aria-pressed` |
| speed | cycles `1× → 2× → 0.5×` |
| restart | `T=0` |
| **keys (D7)** | `Space` play/pause · `←/→` ±1 s · `Shift+←/→` prev/next scene · `1`–`6` jump · `+/-` speed · `R` restart. Ignore when focus is in the range input. |

## 7. The "animated MD" variant

**Markdown cannot animate. The only thing that animates in a rendered README is an image.** So "animated MD"
= an animated **SVG embedded as an image**: `![DecentralizedLLM](assets/pipeline.svg)`. No `<script>` — the
`<img>` context disables JS. Nothing else in the markdown moves.

### 7.1 Surface matrix

| surface | animates? | note |
|---|---|---|
| GitHub README (github.com) | **yes, with CSS `@keyframes` inside an inline `<style>`** | proven by `DenverCoder1/readme-typing-svg`; SMIL through the camo proxy is reported inconsistently — **convert `pipeline.svg` off SMIL** |
| VS Code / IntelliJ markdown preview | yes | webview, both CSS and SMIL |
| GitLab, Gitea | yes (CSS) | same img semantics |
| PyPI / npm README | no | images sanitised |
| **PowerPoint / Keynote** | **no — animation is stripped, a static frame is inserted** | ship P4 `demo.mp4`, a screen recording |
| Slack, Notion | no SVG animation | export a GIF |
| terminal | no images | §7.3 |

### 7.2 `pipeline.svg` conversion rules

1. Replace every `<animate attributeName="stroke" …>` with a CSS class + `@keyframes` in a single inline
   `<style>` inside the SVG. Replace `<animateMotion path="…">` with `offset-path:path('…')` +
   `offset-distance` keyframes (Chrome/Firefox/Safari 16+).
2. No external refs of any kind — no webfonts (`monospace` only), no `xlink:href`, no `<image>`.
3. **Theme-neutral by construction**: keep the opaque `#0d1420` rounded-rect ground. `prefers-color-scheme`
   inside an image-context SVG resolves against the *browser*, not the README, so do not rely on it. One file
   that looks identical in GitHub light and dark beats two files behind a `<picture>`.
4. `role="img"` + `<title>` + `<desc>`; add `@media (prefers-reduced-motion: reduce){ *{animation:none !important} }`
   so the frozen frame is still a legible diagram.
5. Two lanes only — `v0` (6 s cycle, packet width growing 22→108) and `v1` (2 s cycle, constant width). The
   3× period ratio *is* the message; keep the durations coupled.
6. **Acceptance: push to a scratch repo and look at the rendered README.** Not verifiable locally.

### 7.3 `bench/wire_anim.py` — terminal variant

Stdlib only, ~40 lines. `\x1b[H\x1b[J` home+clear per frame, `--fps 12 --frames 240`, `--mode v0|v1`,
`--once` dumps a single frame for pasting into a fenced block, colour suppressed when `not sys.stdout.isatty()`.
78 columns. Frame template, exact:

```
DecentralizedLLM · v0                                token 041/512   seq 073
 coord ══[####################]═══▶ node0 ░░░░░ node1 ░░░░░ node2
 payload 261,632 B  fp32·b64·json   util ███░░░░░░ 33%      1.27 tok/s
```
```
DecentralizedLLM · v1                                token 041/512   seq 073
 coord ══[##]══▶ node0 ██████ node1 ██████ node2 ══▶ id 4 B
 payload   1,792 B  bf16·DLP       util █████████ ~100%    24.21 tok/s
```

`[####]` bar length = `clamp(1 + round(4·log10(bytes)), 1, 20)` — the same log law as P1, so the two variants
never disagree on scale. Active node `██████`, idle `░░░░░`, dead `╳╳╳╳╳`.

## 8. Number provenance — nothing on screen may lack a row here

| on screen | value | tag | source |
|---|---|---|---|
| hidden state fp32 | 3,584 B | derived | H=896 × 4 |
| hidden state bf16 | 1,792 B | derived | H=896 × 2 |
| logit vector, base64 | 810,325 B | derived | V=151,936 × 4 × 4/3, FINDING 2 |
| return path v1 | 4 B | design | argmax on node2 |
| v0 wire, 512-token gen | 1,821.7 MB | derived | FINDING 4 |
| v1 wire, 512-token gen | 1.95 MB | derived | FINDING 4 (1.948) |
| ratio | 935× | derived | FINDING 4 |
| position-forwards | 147,200 → 543 | derived | FINDING 3 |
| recompute factor | 271× | derived | FINDING 3 |
| lm_head share | 9.13 layer-equivalents, 27.6% of params | derived | FINDING 1 |
| imbalance | 1.55× | derived | FINDING 1 |
| KV per token, whole model | 12 KB (512 B/layer) | derived | FINDING 3, GQA 2 KV heads |
| utilisation | 33% → ~100% | modelled | T3-A2, `min(1, R/S)` |
| tok/s ladder | 1.27 / 6.94 / 8.05 / 8.05 / 24.21 | 1 measured, 4 modelled | §5 |
| 512-token wall clock | 2,908 s → 101.6 s, 28.6× | modelled | T3-A1 |
| bf16 fidelity | 99.41% top-1, 3.5 µs | measured | T2-A5 |
| int8 outlier channel | ch 62, 972× median, 1.4% naive top-1 | measured | T2-A1 |
| DLP hop RTT | 8.48 ms → 0.089 ms, 95× | measured | T1-A4 |

**Scene 06 scoreboard — exactly these six rows, in this order:**

| row | v0 | v1 | factor |
|---|---|---|---|
| bytes on the wire | 1,821.7 MB | 1.95 MB | 935× |
| redundant position-forwards | 147,200 | 543 | 271× |
| return path / token | 810,325 B | 4 B | 202,581× |
| throughput (R=3, balanced) | 1.27 tok/s | 24.21 tok/s | 19.0× |
| pipeline utilisation | 33% | ~100% | 3× |
| 512-token generation | 2,908 s | 101.6 s | 28.6× |

## 9. Acceptance checklist

`file://` open with wifi off → no console errors, no failed requests · toggle OS theme mid-play → every colour
flips, none stick · 375 px wide → stage scrolls, nothing overflows the body · scrub to any T twice → identical
frame · `prefers-reduced-motion` → paused, legible, all 6 scenes reachable · publish as Artifact → identical ·
grep the file for `http` → only the `<title>`-free footer text · every number appears in §8 · every caption
byte-identical to §4.
