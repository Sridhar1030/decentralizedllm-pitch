# DecentralizedLLM

**Status: design and planning.** This repository holds the architecture we intend to build, the research
behind it, and the analysis that shaped it. None of the system described here has been built yet. Where a
number appears, it comes from the model's own configuration or from measuring the model directly, and
anything about how the finished system will perform is a projection rather than a result.

![The architecture we are planning](knowledge-base/assets/pipeline.svg)

---

## The idea

Open source models are really good now, but running one yourself is still expensive. You either buy a
serious GPU, or you rent a cluster and send your prompts off to someone else's servers.

We want to take one model and spread it across whatever devices a small team already has. A phone, a
laptop, an old desktop with a decent GPU, a spare box in the corner. Each device would hold a few layers of
the model instead of the whole thing. A prompt goes to the first device, the partial result gets passed to
the next over wifi or a VPN, and the last one sends back the answer. No single device would ever hold the
full model.

The devices would not have to match. A shard is just a range of layers, so the machine with the GPU can
take more of them and the phone can take fewer. You are pooling whatever the group happens to own.

| Device | Rough shard | Worked example |
|---|---|---|
| Phone | under 1 GB | an 8B model in int4 is 4.0 GB, so 1.0 GB each across four phones |
| Laptop | a few GB | a 70B model in int4 is 35.3 GB, so 7.1 GB each across five laptops |
| Desktop with a GPU | the biggest slice | takes more layers, so it is not left idle waiting on the others |

## What we are planning to build

Six pieces, each with a decision record explaining the alternatives we considered and rejected.

| Piece | What it does | Decision record |
|---|---|---|
| Layer shards | each device loads a contiguous range of layers and nothing else | [ADR-011](knowledge-base/decisions/ADR-011-weight-distribution-and-loader.md) |
| KV cache per shard | stop resending the whole sequence on every token | [ADR-001](knowledge-base/decisions/ADR-001-kv-cache-stateful-shards.md) |
| DLP, a binary wire protocol | a fixed 40 byte header on a persistent socket, replacing HTTP and JSON | [ADR-002](knowledge-base/decisions/ADR-002-dlp-binary-wire-protocol.md) |
| Activation compression | bf16 on the wire, with a quality gate that can back off | [ADR-003](knowledge-base/decisions/ADR-003-activation-compression.md) |
| A queue and admission control | one bounded queue, because a single request cannot fill a pipeline | [ADR-005](knowledge-base/decisions/ADR-005-queueing-admission-backpressure.md) |
| Cost aware layer placement | pick the cut points from each device's speed, not by counting layers | [ADR-007](knowledge-base/decisions/ADR-007-layer-placement-dp.md) |

All thirteen records are in [`decisions/`](knowledge-base/decisions/), including a
[claims ledger](knowledge-base/decisions/ADR-013-published-claims-ledger.md) that tracks which figures we
consider defensible and which we have retired.

## Why the design looks like this

Four decisions came out of measuring the model before we designed anything. Each one is the answer to a
specific cost, and each is reproducible with `python3 knowledge-base/bench/verify_constants.py`. Full
detail in [`01-VERIFIED-FACTS.md`](knowledge-base/01-VERIFIED-FACTS.md).

1. **Placement is cost aware, not layer counting.** The output head is 136M parameters, which is 9.13
   transformer layers' worth of compute on every token. Cutting 24 layers evenly across three devices would
   really run 8/8/17, and a pipeline moves at the speed of its slowest stage. Cutting by cost gives
   11/11/11 layer-equivalents instead, and it is the same three lines of configuration either way.
2. **Sampling happens on the last device, not the coordinator.** The logit vector is 607,744 bytes. Choosing
   the token where the logits are produced means 4 bytes travel back per token rather than 607,744.
3. **Every device caches its own keys and values.** Without that, each new token redoes the entire sequence:
   147,200 position forwards per device across a 512 token reply, against 543. Grouped query attention keeps
   the cache at 12 KB per token for the whole model, so there is no memory argument for skipping it.
4. **Devices hand off directly to each other.** Relaying every hop through a coordinator would double the
   traffic, four activation crossings per token instead of two.

## Ideas we tested and dropped

Both of these looked good on paper. We ran them against the real model, they did not hold up, and we would
rather record that than quietly leave them in the design.

- **Aggressive quantisation.** Per tensor int8 destroys the model: cosine similarity 0.039, perplexity
  411,041 against 18.6 for the unquantised reference. The cause is measurable, one channel out of 896 carries 972x the
  median magnitude. The plan stops at bf16.
- **Compressing the activation.** Rank 224 of 896 reproduces every next token choice exactly, but the
  projection costs 27.31 µs to save 11 µs of network time. It only starts paying below 394 Mbit/s, so it
  stays out of the design and behind a link speed check.

There is also a trap worth recording: rank 16 captures 99.95% of activation energy and still yields only
12.5% top-1 agreement. Any quality gate has to measure end to end agreement, never a distance between
hidden states.

## What we are not claiming

- **None of this is built.** The projected figures are modelled from measured stage times and exact
  arithmetic on the model config. The integrated system has not been run.
- **Not cheaper than an API.** A fleet of idle machines burns more in electricity than the tokens are worth
  at market prices. The argument is fit, not cost. The devices are already yours and already on, and the
  model does not fit on any one of them.
- **Not encryption.** Splitting a model across machines raises the cost of recovering a prompt from
  intermediate state. It does not prevent it.
- **Not a new idea.** Layer sharded inference goes back to SplitNN in 2018, through Petals in 2022 and exo
  in 2024, and vLLM ships cross node pipeline parallelism behind one flag. What we think is defensible is
  the trust and heterogeneity model, not the mechanism.
- **Phones are the goal, not the starting point.** Nothing has been run on a handset.

## What is in this repository

| | |
|---|---|
| [`01-VERIFIED-FACTS.md`](knowledge-base/01-VERIFIED-FACTS.md) | the measured facts, with the script that regenerates them |
| [`10-ARCHITECTURE.md`](knowledge-base/10-ARCHITECTURE.md) | the target design, component by component |
| [`20-INFRA-AND-STACK.md`](knowledge-base/20-INFRA-AND-STACK.md) | serving runtimes surveyed, and what we would build on |
| [`30-PERF-MODEL.md`](knowledge-base/30-PERF-MODEL.md) | the latency model and where the projections come from |
| [`40-PITCH.md`](knowledge-base/40-PITCH.md) | the pitch, the demo script and the slides |
| [`90-AUDIT.md`](knowledge-base/90-AUDIT.md) | an adversarial audit of our own numbers, 23 findings |
| [`decisions/`](knowledge-base/decisions/) | 13 architecture decision records |
| [`teams/`](knowledge-base/teams/) | 25 research reports behind the decisions |
| [`bench/`](knowledge-base/bench/) | benchmark scripts and raw results |
| [`assets/`](knowledge-base/assets/) | the animated walkthrough, the diagram and the deck |

Start with [`90-AUDIT.md`](knowledge-base/90-AUDIT.md) before quoting any figure from here. It lists what we
had to correct in our own work, including the numbers we retired outright.

## Reproduce the analysis

```bash
python3 knowledge-base/bench/verify_constants.py     # the numbers above, from the model config
python3 knowledge-base/assets/build_deck.py          # rebuild the slides
```

An animated walkthrough of the design is at
[decentralizedllm-pitch.vercel.app](https://decentralizedllm-pitch.vercel.app).
