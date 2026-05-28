# Roadmap

This roadmap is the path from the current code to a real trainer sustaining
goodput on a 220k-GB300-class cluster.

Two axes run in parallel:
- **Scale** (M0→M6): correctness from one GPU up to the full cluster.
- **Performance** (P1→P3): the speedup tiers layered on top.

Legend: ✅ done · 🚧 in progress · ⬜ planned · 🔒 needs real GPUs (can't run here)

## Status (this build)

Everything implementable without a GPU is done and tested:
- **Real training works** on CPU: `refmodel` is a dense transformer with fp64
  forward/backward/AdamW, its backward **verified against finite differences**
  (L2 rel err ~3e-5); `reftrain` learns a real task to 100% held-out accuracy.
- **Whole-cluster validation**: `plan_verify` checks all 220,032 ranks (communicator
  tiling, hash agreement, intra-stage op identity, pipeline p2p conservation) in ~17 ms.
- **Higher-fidelity simulator**: alpha-beta collective cost; EP-aware MoE memory;
  per-op trace + per-stream occupancy telemetry.
- **Interleaved 1F1B** (`vpp`): planning-level virtual pipeline (bubble 14.3%→7.7% at vpp=2).
- **Spare-rank recovery**: `plan_recover` hot-swaps a failed rank onto a spare GPU in
  another rack, emitting the identical replacement plan.

What remains (M2–M6 device paths, P1–P3) is **gated on real hardware** — a CUDA
toolchain and ultimately a GB300 cluster — and is marked 🔒 below.

---

## Scale milestones

### M0 — Scaffold ✅ (current)
The planning + simulation layer that lives in this repo.
- offline plan compiler: decomposition, comm groups, 1F1B schedule, memory layout
- per-rank `plan.bin` / `topology.bin` with `plan_hash` / `topology_hash` guards
- analytic CPU backend + event-driven runtime (step time / bubble / MFU / tokens/s)
- `topology_linter`, unit tests, 1-rack / 220k / MoE configs

**Runnable today:** compile + validate all 220,032 ranks and replay the schedule to
estimate performance — with no GPU.

### M1 — Reference & spec freeze ✅
*Done:* `src/refmodel.{c,h}` is the golden reference (fp64, real fwd/bwd/AdamW),
backward verified against central finite differences; `tools/reftrain.c` trains it on
CPU. This is the correctness oracle the device path must match.

### M2 — Single-GPU C trainer (real compute)  CPU ✅ · device 🔒
*CPU realization (done):* `reftrain` runs a real forward/backward/AdamW loop on CPU
and learns (loss 4.16→0.003, 100% held-out acc) — the training algorithm is proven.
*Device (needs GPU):* fill in `src/backend_cuda.c` (`cai_backend_cuda_create`):
device bring-up, cubins via the Driver API, CUDA-Graph steady step matching the
reference loss with zero runtime allocation. → unlocks **P1**.

### M3 — Intra-rack (tray → 72 GPU) 🔒
Build NCCL communicators from the plan's comm groups; TP/DP groups; NVLink-local
collectives; rack-local checkpoint shards.
*Exit:* identical loss curve at 4 / 8 / 72 GPU; TP beats the baseline intra-rack;
collective shape mismatches are caught **before** launch; long single-rack run.

### M4 — Multi-rack pipeline 🔒
Real `PIPE_SEND/RECV`, 1F1B executed across racks, comm/compute overlap, NIC
utilization telemetry, pod-level checkpoint restore, failure injection.
*Exit:* measured pipeline bubble within the model's prediction; stage imbalance is
detectable; restore from checkpoint at pod granularity.

### M5 — Pod scale  recovery-planning ✅ · run 🔒
*Done (no GPU):* spare-rank recovery planning via `plan_recover` — picks a spare GPU
(preferring another rack) and regenerates the failed rank's identical plan.
*Needs GPU:* hierarchical launcher + control plane, DP reduce-scatter/all-gather at
scale, MoE all-to-all, straggler detection, auto-recovery on injected failure. → P2.

### M6 — Full 220k-class  validation ✅ · run 🔒
*Done (no GPU):* full-topology plan generation; **pre-launch validation** via
`plan_verify` rules out collective deadlock — communicator tiling, `plan_hash` /
`topology_hash` agreement, intra-stage op identity, and pipeline p2p conservation,
checked across all 220,032 ranks.
*Needs GPU:* hierarchical job launch, global checkpoint manifest, rack/pod failure
policy, smoke step → sustained-goodput run, JAX comparison report.

---

## Performance tiers

All performance tiers are 🔒 (need real GPUs to measure); the simulator already
*predicts* where the wins are (e.g. comm_dp dominates at 1719-way FSDP over NICs).

| tier | unlocked by | levers | target |
|---|---|---|---|
| **P1** | M2–M3 | static memory + CUDA Graphs + topology-aware mapping + comm/compute overlap | 1.3–2.5x goodput vs an optimized JAX baseline |
| **P2** | M4–M5 | custom fused kernels + pipeline optimization (interleaved 1F1B *modeled*; kernels need GPU) | 3–5x |
| **P3** | post-M6 | megakernels + GPU-initiated scheduler (NVSHMEM) + model/topology co-design | >5x under the right conditions |

10x is **not** "because it's C" — it is the compound effect of throwing away
generality and fully specializing to the physical layout of 220k GB300s.

---

## Cross-cutting tracks (run throughout)

- **Telemetry** ✅ (no-GPU parts) — per-op trace (CSV) + per-stream occupancy from
  `trainer`; `cai_trace_*` API. Pod aggregation / NIC / NVLink counters come with M4–M5.
- **Simulator fidelity** ✅ (this pass) — alpha-beta collective cost (latency ~log2(size)
  + ring bandwidth) and EP-aware MoE memory. Still to do: rails/congestion, and
  **validating against measured M2/M3 numbers** once a GPU is available.
- **Kernels** 🔒 — handwritten RMSNorm / RoPE / SwiGLU / fused attention fwd+bwd /
  AdamW / MoE dispatch+combine; start on cuBLASLt + cuDNN, replace hot paths as the
  profile dictates. (The fp64 `refmodel` is the per-kernel correctness oracle.)
- **Communication** 🔒 — NCCL for bulk collectives → NVSHMEM for fine-grained
  pipeline / expert routing → cluster-specific custom collectives.
- **Fault tolerance** 🚧 — checkpoint save/restore + rolling "latest" pointer +
  topology-hash guard + spare-rank recovery planning (`plan_recover`) exist; async /
  two-phase commit and live substitution need the device runtime.

---

## Next three steps

The next milestones all require a CUDA toolchain / GPU (not available in this
environment), so they are the first things to do once hardware is in hand:

1. **M2 device**: implement `backend_cuda.c` bring-up; match the fp64 `refmodel`
   loss on one GPU with zero runtime allocation and a CUDA-Graph steady step.
2. **M3**: wire NCCL communicators from the (already-verified) comm groups; confirm
   an identical loss curve at 4 / 8 / 72 GPU.
3. **Close the loop**: feed measured op times into the simulator's cost model and
   check the predictions in this build against reality.
