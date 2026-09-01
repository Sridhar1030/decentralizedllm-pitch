---
team: T4 — Infrastructure & Serving Runtimes
agent: T4-A4
topic: Deployment plan (docker-compose v1, Kubernetes/Ray v2), wired 3-machine demo, hardware BOM and $/1M-token economics
headline: "The 2026 GPU market is the pitch: a used RTX 3090 buys memory bandwidth at $1.07/(GB/s) and an RTX 5090 at $3.22/(GB/s), so the only cheap bandwidth on sale comes in 24 GB units — splitting the model is the only way to spend it. But be honest: three GPUs in ONE chassis is $1.39/(GB/s) vs $1.93 across three boxes, so decentralization is justified by constraint (capacity you can't buy in one box, hardware you already own, trust boundaries), never by 'three small beats one big'."
---

# T4-A4 — Deployment & BOM

Prices checked 2026-09-01 via WebSearch; tagged `(list)`, `(street)`, `(est.)`. Throughput tagged
`(measured)` / `(modelled)` / `(reported)` = third-party, not benchmarked here. Source read:
`docker-compose.yml`, `layer-nodes/node.py`, `layer-nodes/coordinator.py`, `gateway/app.py`, `ray-vllm/Dockerfile`.

## 1. v1 — docker-compose changes (hackathon, days, CPU, 3 nodes)

### 1a. The one-laptop demo. Ranked by value/effort.

| # | Change | Where | Why / number | Effort |
|---|---|---|---|---|
| 1 | `NODE_LAYERS: "0-11" / "11-22" / "22-24"` | compose env, 3 lines | FINDING 1: 8/8/8 puts `lm_head` (9.13 layer-eq) on node2 → 17.13 vs 11.04 eq. **1.55x throughput, free.** `node.py`'s `start_layer==0` / `end_layer==24` branches still hold. | 3 edits |
| 2 | `OMP_NUM_THREADS: "2"`, `MKL_NUM_THREADS: "2"` per node | compose env | `deploy.resources.limits.cpus: "2"` is a **cgroup quota, not a core mask**. `torch` reads the host's 10 cores and spawns 10 OpenMP threads that then fight over 2 CPUs of quota → context-switch thrash. Torch cannot see the quota. This is the single most-missed docker+torch bug. | 6 lines |
| 3 | Shared HF cache volume `hf-cache:/root/.cache/huggingface` + `HF_HOME` on all 3 nodes | compose volumes | Today each node downloads the **full 1 GB** safetensors independently and `from_pretrained` loads the whole model before discarding 2/3 of it. Cold start ~255 s → ~65 s, **~4x** (modelled). | 5 lines |
| 4 | `mem_limit: 6G` (or `deploy.resources.limits.memory: 6G`) — uncomment and **raise**, don't use 4G | compose | T3-A5 FINDING 6: the loader allocates the model twice → **3.95 GB peak/container**, not the 2 GB the comment claims. 4G OOM-kills on startup. | 3 lines |
| 5 | Docker Desktop VM ≥ 16 GB / 8 CPU | Desktop settings | 3 x 3.95 GB peak = 11.9 GB. Default 8 GB VM = OOM at startup. **Demo-day landmine.** | 1 min |
| 6 | Keep the `sleep 45` + `depends_on: service_healthy` chain **until** #3 and #5 land | compose | The hack is load-bearing: it serialises the peak allocation. Remove it only together with the cache + VM bump, then start all 3 in parallel. | — |
| 7 | `CIRCUIT_FAILURE_THRESHOLD: "10"`, `CIRCUIT_COOLDOWN_SEC: "5"` | gateway env | `gateway/app.py` opens the breaker after 3 failures for 30 s. On a cold CPU demo the first request is slow; the breaker trips and the demo is dead for 30 s. | 2 lines |
| 8 | `TOKENIZERS_PARALLELISM: "false"` on coordinator | compose env | Kills the HF fork warning + a thread pool that does nothing here. | 1 line |

### 1b. The three questions the brief asked, answered with a verdict

| Knob | Verdict | Reason |
|---|---|---|
| `network_mode: host` | **Yes on Linux, no-op on macOS.** | Linux: skips the veth + NAT hop, ~0.05–0.1 ms/hop (modelled). macOS: Docker Desktop runs a Linux VM, so "host" is the VM's host, not yours — you still cross the VM boundary. Also incompatible with the `ports:` mappings, so it is a *deploy-target* switch, not a default. |
| `shm_size` / `ipc: host` | **No. Skip it entirely.** | `/dev/shm` (64 MB default) only matters for `torch.utils.data.DataLoader` worker IPC and NCCL shared-memory transports. We have neither: one process, HTTP between containers. Raising it changes nothing measurable. Stated explicitly so nobody spends a day on it. |
| Resource limits | **Yes, but they are not the lever.** | CPU limits are a cgroup quota torch can't see (#2 above is the real fix). Memory limits must be **6G, not 4G** (#4). |

## 2. v1 — the 3-machine wired demo (the user's ask)

Two topologies. **If the machines are Macs, use Thunderbolt and buy no switch at all.**

### 2a. Topology A — Thunderbolt mesh (Macs, 3 nodes, zero switch)

Each Mac mini M4 has **3 Thunderbolt 4 ports**. Three machines + three cables = a full triangle; every
pair is a direct point-to-point link. N ≤ 4 needs no switch. TB4 bridge sustains **~2.5–3 GB/s (20–24 Gbit/s)**
(reported); TB5 on M4 Pro sustains **~60 Gbit/s TCP** (reported) — 2–6x a 10GbE switch, for $90 of cable.

```
System Settings → Network → Thunderbolt Bridge → Manually:
  mac0 10.42.0.1/24   mac1 10.42.0.2/24   mac2 10.42.0.3/24     (one /24 per link if you prefer /30s)
```

### 2b. Topology B — ethernet switch (mixed hardware)

| Item | Choice | $ |
|---|---|---|
| Switch | TP-Link TL-SX105, 5-port 10GBASE-T, unmanaged, fanless | ~200 (street) |
| Cables | 3x Cat6a 1 m | ~24 |
| Mac NIC | Mac mini 10GbE BTO, +$100 each | 300 |
| PC NIC | Intel X550-T2 ~$130 new / used Mellanox ConnectX-3 EN + SFP+ DAC ~$40 | — |

Not the MikroTik CRS305-1G-4S+IN ($135 street): SFP+ only, so RJ45 Macs need 3x 10GBASE-T modules
@ ~$40 → $255 total, and those modules run hot.

### 2c. Setup, in order. Do not skip step 4.

1. **Static IPs, no DHCP, no mDNS.** `10.42.0.{1,2,3}/24` on the *dedicated* NIC; leave Wi-Fi/DHCP on the
   other interface for internet. Put the literal IPs in `NODE0_URL`/`NODE1_URL`/`NODE2_URL`. Rationale:
   `.local` mDNS costs 1–5 ms on first resolve, macOS `mDNSResponder` re-resolves on cache expiry, and
   Docker containers do not see host mDNS by default. **Zero DNS in the data path** is a one-line saving.
2. **Publish the ports.** The compose file already does `ports: 8001:8001` etc., so the 3-machine version is
   the same file split into three — one node service per machine, coordinator + gateway + Prometheus on
   machine 0. No code change.
3. **MTU 9000 on all three NICs *and* the switch *and* the Docker bridge.**
   `sudo ifconfig en1 mtu 9000` (macOS) / `ip link set en1 mtu 9000` (Linux), plus
   `networks: {decentralized-net: {driver_opts: {com.docker.network.driver.mtu: "9000"}}}` — miss the Docker
   line and containers still emit 1500 B segments. Payload efficiency 1448/1538 = 94.1% → 8948/9038 = 99.0%.
   At seq=512 the v0 hidden state is **1,236 frames at MTU 1500 vs 205 at 9000**; with a KV cache, 3,584 B
   goes from **3 segments to 1**. Jumbo must be set *everywhere* or PMTUD silently black-holes.
4. **Verify with iperf3 before blaming the model.**
   ```
   node1$ iperf3 -s
   node0$ iperf3 -c 10.42.0.2 -t 10 -P 4        # bulk
   node0$ ping -c 100 -i 0.01 10.42.0.2         # RTT
   ```

| Link | iperf3 expected | RTT expected | Tag |
|---|---|---|---|
| 1 GbE, MTU 1500 | 940–990 Mbit/s (theoretical ceiling 1000 x 1448/1538 = **941**) | 0.20–0.35 ms | derived / house numbers |
| 10 GbE, MTU 1500 / 9000 | 9.4 / 9.8 Gbit/s | 0.06–0.12 ms | modelled |
| TB4 bridge | 15–26 Gbit/s | 0.10–0.20 ms | reported |
| TB5 bridge (M4 Pro) | ~60 Gbit/s | <0.10 ms | reported |

   **Gate:** < 900 Mbit/s on a 1 GbE link = duplex mismatch or bad cable. Fix it first.
5. **Then run the test that actually matters.** `iperf3` measures bulk bandwidth; our v1 payload is 3,584 B.
   At 10 GbE that transfer is `3584 / 1.25e9` = **2.9 µs** against a 60–120 µs RTT — the wire is
   **100% latency, 0% bandwidth**. Use a 3,584 B ping-pong (T1-A2 §2 frame server) against T1-A2's measured
   0.084 ms/hop. **Do not buy 10GbE for a 0.5B model.** 1 GbE saturates only at (big model x big batch):
   70B, H=8192, bf16 hidden+residual = 32,768 B/hop; at B=64 that is **2.097 MB = 16.8 ms on 1 GbE** against
   a ~19 ms GPU stage — 88% of the stage. On 10 GbE, 1.68 ms.

## 3. v2 — production stack (months)

### 3a. StatefulSet vs Deployment — StatefulSet, and one STS *per stage*

| Requirement | Deployment | StatefulSet | Verdict |
|---|---|---|---|
| Stage *i* must address stage *i+1* by identity | random pod names, one LB Service | `stage-1-0.stage-1.ns.svc` | **STS.** A Service would round-robin decode step *n+1* of request R to a pod that does **not** hold R's KV cache → full re-prefill. This is the argument; everything else is secondary. |
| Per-pod NVMe weight cache | shared PVC or re-pull | `volumeClaimTemplates` | STS. A 40 GB shard must not be re-pulled per restart. |
| Canary one stage | all-or-nothing | `updateStrategy.rollingUpdate.partition` | STS. |

**Honest counter:** the KV cache is *ephemeral* — it dies with the pod anyway, so "stateful" here means
**identity + affinity**, not durability. Deployment + headless Service + a consistent-hash-on-request-id
router also works (vLLM `--kv-transfer-config` / LMCache / NVIDIA Dynamo do exactly that) — it is just more
code you own. StatefulSet gives you the identity for free.

**One STS per stage, not one STS of 3 pods** — the stages have different layer counts, GPU memory and
scaling curves, and one STS = one pod template = one resource request. `sts/stage-{0,1,2}`, each
`replicas = R` (R = parallel pipelines), each with a headless Service; the coordinator addresses
`stage-{s}-{i}` and pins request → pipeline *i* by request id. `topologySpreadConstraints` keep pipeline
*i*'s three stages **in one AZ** and spread *pipelines* across AZs: cross-AZ RTT ~0.5–1 ms x 2 hops =
**1–2 ms/token** — 5–10% of a 20 ms/token budget, **20–40%** of a 5 ms GPU budget (modelled).

### 3b. Service mesh on the tensor path: **no**, and here is the number

Istio's published benchmark (1 KB HTTP/1.1, mTLS, 2 proxy workers): **p90 ≈ 0.63 ms sidecar vs ≈ 0.16 ms
ambient L4** (documented).

| | per hop | per token (2 inter-stage hops) | 512-token response |
|---|---|---|---|
| Sidecar (Envoy, 2 traversals/hop) | +0.63 ms | **+1.26 ms** | **+645 ms** |
| Ambient / ztunnel (L4) | +0.16 ms | +0.32 ms | +164 ms |
| No mesh, direct pod IP | 0 | 0 | 0 |

6.3% of a 20 ms/token budget, **25%** of a 5 ms one — paid on *every token, forever*. Worse, Envoy is a
userspace proxy: kernel→envoy→kernel→app is **two extra memcpys of every 3.5 KB tensor per hop**, exactly
the copy T1-A2's zero-copy `frombuffer` path deleted.

**Rule: mesh the control plane (gateway, coordinator API, metrics, admin). Never the tensor path.** Exclude
it explicitly — `traffic.sidecar.istio.io/excludeOutboundPorts: "9100"` on the binary-frame port. If the
tensor path needs mTLS, do it **in-process** in the node's frame server (one copy) or with WireGuard
(kernel-space, ~0.05–0.1 ms, one copy). If policy forces a mesh, use Istio **ambient** and never attach an
L7 waypoint to the tensor ports.

### 3c. Ray as an alternative control plane

Not hypothetical — the repo already ships `ray-vllm/` (`vllm/vllm-openai-cpu` + `ray[default]>=2.9.0`).

| Ray gives you | Ray costs you |
|---|---|
| Placement groups (`STRICT_SPREAD` across nodes for the 3 stages, `PACK` within), actor lifetime + auto-restart, a real scheduler, `ray.util.collective` (NCCL/Gloo) | The object-store data path is Plasma→gRPC→Plasma; `ray.get` on a *remote* object carries ~0.5–1 ms overhead (documented) vs T1-A2's **0.084 ms** measured raw TCP frame. Plasma is zero-copy **same-node only**. |

**Verdict: Ray is a control plane, not a data plane.** `@ray.remote` actors for placement, health and
lifecycle; the actors talk over the T1-A4 binary socket. This is vLLM's own split —
`--distributed-executor-backend ray` for orchestration, NCCL/gloo for tensors. Not either/or with K8s:
KubeRay's `RayCluster` CRD runs Ray *on* K8s.

### 3d. Autoscaling, GPU operator, multi-region

| Topic | Recommendation | Number / reason |
|---|---|---|
| Scaling unit | **A whole pipeline, never a stage.** PDB `minAvailable: R-1` per stage; Karpenter NodePool with `consolidationPolicy: WhenEmpty` so a half-provisioned pipeline is never billed. | Stage 1 alone is useless — the model is incomplete. |
| HPA metric | Queue depth / TTFT from the T3-A4 admission controller via KEDA or prometheus-adapter. **Not CPU.** | CPU is ~100% on every stage by construction; it carries no signal. |
| Scale-out latency | Keep **one warm spare pipeline**. Gate routing on a `ReadinessGate` that flips only when all 3 stages report weights loaded. | 40 GB image pull + weight load = 3–10 min. HPA cannot react on that timescale. |
| GPU drivers | `nvidia/gpu-operator` Helm chart (driver, container-toolkit, device-plugin, DCGM exporter, NFD, MIG manager). **Lazier and usually correct:** if the managed node group already runs a GPU AMI, ship only the `nvidia-device-plugin` DaemonSet. | Fewer moving parts. DCGM exporter → Prometheus replaces the PoC's hand-rolled `/metrics` counters. |
| Packing small stages | **MPS** over time-slicing (no context-switch quantum → lower latency); MIG only on A100/H100 when you need hard isolation. | Our 0.5B stages are tiny; several fit per GPU. |
| Multi-region | **Never stripe a pipeline across regions. Replicate whole pipelines per region**, route at the edge (Route53 latency records / Global Accelerator), and replicate only *weights* across regions (S3 CRR / registry pull-through). KV cache never leaves its region. | us-east-1↔us-west-2 ≈ 60–70 ms. 2 hops x 65 ms = **+130 ms per token**; a 512-token response = **66.6 s of pure RTT**. Fatal for chat. |
| The one exception | Offline/batch generation, where TTFT is irrelevant: pipeline depth hides RTT once you have ⌈65/19⌉+1 = **5 microbatches in flight** per stage. | Works for batch. Never for chat. |

## 4. Hardware BOM, three tiers

All 3-node clusters sized to hold **Llama-3.3-70B Q4_K_M ≈ 40 GB** — the smallest model that does *not*
fit one cheap node, i.e. the smallest model for which this architecture has a reason to exist.

| Tier | Line items | $ |
|---|---|---|
| **A — 3x Mac mini M4** (16 GB/256 GB) **over Thunderbolt** | 3 x $599 (launch list; Apple US list moved to $799 in 2026 → $2,397) + 3 x TB4 cable @$30 (Apple TB4 Pro 1 m is $69) | **$1,887** |
| A' — same, over 10GbE switch | 3 x $599 + 3 x $100 10GbE BTO + TL-SX105 $200 + Cat6a $24 | **$2,321** |
| **B — 3x consumer GPU box** | 3 x [ used RTX 3090 24 GB ~$1,000 (est.) + mobo/CPU/64 GB/PSU/case ~$700 ] + TL-SX105 $200 + NICs/cables ~$114 | **$5,414** |
| B' — 3 GPUs in **one** chassis | 1 x host $900 + 3 x $1,000 GPU. Same 72 GB, same 2,808 GB/s, no network at all. | **$3,900** |
| **C — cloud** | 3 x `g5.xlarge` (A10G 24 GB) @ $1.006/hr us-east-1 = **$3.018/hr**; single `g6e.xlarge` (L40S 48 GB) = **$1.861/hr** | hourly |

### 4a. The table that is the whole argument — $ per unit of memory bandwidth

LLM decode is memory-bandwidth-bound, so `$/(GB/s)` is the real price of throughput and `$/GB` the real
price of capacity.

| Device | GB | GB/s | $ | $/GB | **$/(GB/s)** |
|---|---:|---:|---:|---:|---:|
| Mac mini M4 16 GB | 16 | 120 | 599 | 37.4 | **4.99** |
| **3x Mac mini M4 (Tier A)** | 48 | **360** | 1,887 | 39.3 | **5.24** |
| Mac mini M4 Pro 64 GB | 64 | 273 | 1,999 | 31.2 | 7.32 |
| Mac Studio M3 Ultra 512 GB | 512 | 819 | 9,499 | 18.6 | 11.60 |
| **RTX 3090 24 GB (used)** | 24 | 936 | 1,000 | 41.7 | **1.07** |
| **3x 3090, three boxes (Tier B)** | 72 | 2,808 | 5,414 | 75.2 | **1.93** |
| 3x 3090, **one** box (B') | 72 | 2,808 | 3,900 | 54.2 | **1.39** |
| RTX 5090 32 GB | 32 | 1,792 | 5,769 (street, 2026) | 180.3 | 3.22 |
| RTX PRO 6000 96 GB | 96 | 1,792 | 8,500 (est.) | 88.5 | 4.74 |

Three readings, all load-bearing. **(1) The 2026 GPU market is the pitch:** a used 3090 is **3.0x** cheaper
per GB/s than a 5090 and **4.4x** cheaper than an RTX PRO 6000, and cheap bandwidth is only sold in **24 GB
units** — splitting the model is the only way to spend it. **(2) On Apple silicon, clustering loses:** 3x M4
mini = $5.24/(GB/s) beats one M4 Pro at $7.32 on bandwidth, but Apple's marginal memory is $12.50/GB (the
+$200-per-16 GB BTO) against $37.4/GB for a whole extra Mac — **buy RAM, not Macs**, until you exceed the
largest single box. **(3) Even on GPUs, one chassis beats three:** $1.39 vs $1.93 per GB/s, **1.39x**.

### 4b. $/1M tokens — 70B Q4, decode, pipeline full

Assumptions stated: 3-year straight-line amortization (26,280 h), $0.17/kWh, 100% utilization,
`tok/s ≈ 0.65–0.75 x BW / 40 GB` per stage with the pipe full, **labour excluded**.

| Tier | tok/s (modelled) | $/hr | h per 1M | **$/1M tok** | Breakeven utilization vs Haiku 4.5 |
|---|---:|---:|---:|---:|---:|
| A — 3x Mac mini M4 | 5.86 | 0.0922 (capex 0.0718 + power 0.0204) | 47.4 | **$4.37** | **87%** |
| B — 3x used 3090 | 52.7 | 0.4355 (capex 0.2060 + power 0.2295) | 5.27 | **$2.30** | **46%** |
| C — 3x g5.xlarge on-demand | 33.8 | 3.018 | 8.22 | **$24.80** | never |
| C' — 1x g6e.xlarge on-demand | 16.2 | 1.861 | 17.15 | **$31.91** | never |
| **Claude Haiku 4.5** (API, output) | — | — | — | **$5.00** | — |
| Claude Sonnet 5 (API, output) | — | — | — | $10.00 | — |
| Claude Opus 5 (API, output) | — | — | — | $25.00 | — |

Cross-AZ egress for Tier C is **noise**: 70B bf16 hidden+residual = 32,768 B/hop x 2 hops = 65.5 KB/token
→ 65.5 GB per 1M tokens x $0.02/GB = **$1.31**. This independently confirms T4-A1: PP is latency-bound,
not bandwidth-bound. Compression saves cents.

**Four caveats that must travel with this table.** (1) **$/token across model qualities is meaningless** —
Haiku 4.5 is not Llama-3.3-70B; the only defensible claim is *if a 70B-class open model is good enough*,
Tier B is ~2.2x cheaper than the cheapest frontier API at 100% utilization. (2) **Utilization is the whole
game** — at 10% utilization Tier B is $23/1M and loses to every API listed. (3) **Labour is excluded and it
dominates**: one engineer-week ≈ $4–6k ≈ **2,000 M tokens** of Haiku 4.5. (4) **In the cloud, owning loses
outright** — every cloud row is 5–14x the on-prem rows, because you rent the capex at ~100%/yr.

## 5. When does a decentralized cluster of cheap heterogeneous nodes beat one big GPU?

### 5a. The formal condition

Let `P(m)` = market price of the cheapest single device with `m` bytes of memory — a **convex** curve
(memory is priced superlinearly at the top). Let `U ∈ (0,1]` = pipeline utilization.

> **The cluster wins iff  `P(M) / (N · P(C)) > 1/U`,** where `M` = model bytes, `C` = per-node memory.

`1/U` is the pipeline penalty. Without microbatching a 3-stage chain caps at `U = 1/3` (shared-context
defect #6: 2 of 3 nodes idle at any instant), so the big box must be **3x** more expensive to break even.
With T3-A2 microbatching `U → ~0.9`, `1/U → 1.11`, and an **11% price edge is enough**.

> **This is why the queueing/microbatching work — not the compression work — makes the economics real:**
> it moves the required price advantage from 3.0x to 1.11x, a **2.7x** swing in the break-even condition.

### 5b. The crossover, worked

| Scenario | `M` | `C` | Cheapest 1-box `P(M)` | 3-node `3·P(C)` | Ratio | With `1/U`=1.11 | Verdict |
|---|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B fp32 (the PoC)** | 2.0 GB | 16 GB | $599 (1 Mac mini) | $1,887 | **0.32x** | 0.32 < 1.11 | **Cluster loses, 3.5x.** `M < C` — one node holds the whole model, so the split buys nothing and costs 2 hops. State this on the slide. |
| Llama-3.3-70B Q4 (40 GB), Apple | 40 GB | 16 GB | $1,999 (M4 Pro 64 GB) | $1,887 | 1.06x | 1.06 < 1.11 | Dead heat. Buy the M4 Pro — simpler. |
| **Llama-3.3-70B Q4, NVIDIA** | 40 GB | 24 GB | $8,500 (RTX PRO 6000 96 GB) | $3,900 (one box) / $5,414 (three) | **2.18x / 1.57x** | > 1.11 | **Cluster wins.** No consumer GPU ships >32 GB; above it the price curve jumps. |
| DeepSeek-V3 671B fp8 (671 GB) | 671 GB | 512 GB | >$9,499; no single consumer device exists | 8x Mac mini M4 | ∞ | ∞ | **Cluster is the only option.** EXO's 8x Mac mini reached 5.37 tok/s on DeepSeek-V3 in Apr 2026 after kernel-level TB5 RDMA, up from <1 tok/s in Mar 2026 (reported). |

### 5c. The honest statement for the deck

> **Decentralization is justified by constraint, never by "three small beat one big".** Inside one chassis,
> three GPUs are 1.39x cheaper per GB/s than three chassis. The four real justifications are:
> **(a) capacity you cannot buy in one box** — above 32 GB consumer VRAM or 512 GB unified memory the
> price curve goes vertical; **(b) hardware you already own**, sitting in different rooms, already paid for
> — marginal capex $0, so any `P(M) > 0` wins; **(c) trust and sovereignty boundaries** — no single party
> holds the weights, which is the one thing *no* amount of money buys you in one box; **(d) the 2026 GPU
> market**, where the only cheap bandwidth on sale comes in 24 GB units.
>
> For a **0.5B model on CPU it does not work and we should say so** — one Mac mini is **3.5x** cheaper per
> token than three. The PoC is a correctness demo of the mechanism, not an economic one. Demo it on a model
> that does not fit one node, or demo it on hardware you already own.
