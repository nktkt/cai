# Roadmap

cai today is a **GPU-free scaffold**: the offline compiler, the analytic runtime,
the plan/topology formats, and the tooling are real and tested, but no training
math runs on a device yet. This roadmap is the path from that scaffold to a real
trainer sustaining goodput on a 220k-GB300-class cluster.

Two axes run in parallel:
- **Scale** (M0→M6): correctness from one GPU up to the full cluster.
- **Performance** (P1→P3): the speedup tiers layered on top.

Legend: ✅ done · 🚧 in progress · ⬜ planned

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

### M1 — Reference & spec freeze ⬜
*Deliverables:* a golden JAX/PyTorch reference for a small model; frozen model /
optimizer / precision / dataset format; correctness tolerances; a baseline profile
to compare against later.
*Exit:* small model reproduces loss/grad/optimizer-update; the hot path cai must
replace is enumerated; the speedup hypothesis is decomposed per op.

### M2 — Single-GPU C trainer (real compute) ⬜
Fill in `src/backend_cuda.c` (`cai_backend_cuda_create`): device bring-up, modules,
precompiled cubins launched via the Driver API, real forward/backward/optimizer.
*Exit:* 1 GPU matches the M1 reference loss; **zero runtime allocation**; the steady
step runs as a CUDA Graph; per-op timeline captured. → unlocks **P1**.

### M3 — Intra-rack (tray → 72 GPU) ⬜
Build NCCL communicators from the plan's comm groups; TP/DP groups; NVLink-local
collectives; rack-local checkpoint shards.
*Exit:* identical loss curve at 4 / 8 / 72 GPU; TP beats the baseline intra-rack;
collective shape mismatches are caught **before** launch; long single-rack run.

### M4 — Multi-rack pipeline ⬜
Real `PIPE_SEND/RECV`, 1F1B executed across racks, comm/compute overlap, NIC
utilization telemetry, pod-level checkpoint restore, failure injection.
*Exit:* measured pipeline bubble within the model's prediction; stage imbalance is
detectable; restore from checkpoint at pod granularity.

### M5 — Pod scale ⬜
Hierarchical launcher + control plane, DP reduce-scatter/all-gather at scale, MoE
all-to-all, spare-rank recovery, straggler detection.
*Exit:* per-pod goodput beats the baseline; auto-recovery after injected failure;
checkpoint pause does not dominate the step; stragglers isolated. → unlocks **P2**.

### M6 — Full 220k-class ⬜
Global plan compilation at full scale (already works in simulation), hierarchical
job launch, global checkpoint manifest + telemetry, rack/pod failure policy.
*Exit:* full-topology plan generation completes; pre-launch validation rules out
collective deadlock; all ranks agree on `plan_hash` / `topology_hash`; smoke step →
sustained-goodput run; JAX comparison report.

---

## Performance tiers

| tier | unlocked by | levers | target |
|---|---|---|---|
| **P1** | M2–M3 | static memory + CUDA Graphs + topology-aware mapping + comm/compute overlap | 1.3–2.5x goodput vs an optimized JAX baseline |
| **P2** | M4–M5 | custom fused kernels + pipeline optimization (interleaved / zero-bubble 1F1B) | 3–5x |
| **P3** | post-M6 | megakernels + GPU-initiated scheduler (NVSHMEM) + model/topology co-design | >5x under the right conditions |

10x is **not** "because it's C" — it is the compound effect of throwing away
generality and fully specializing to the physical layout of 220k GB300s.

---

## Cross-cutting tracks (run throughout)

- **Kernels** ⬜ — handwritten RMSNorm / RoPE / SwiGLU / fused attention fwd+bwd /
  AdamW / MoE dispatch+combine; start on cuBLASLt + cuDNN, replace hot paths as the
  profile dictates.
- **Communication** ⬜ — NCCL for bulk collectives → NVSHMEM for fine-grained
  pipeline / expert routing → cluster-specific custom collectives.
- **Fault tolerance** ⬜ — async checkpoint, two-phase commit, rank-local NVMe →
  remote, spare-rank substitution; elastic training later (not V1).
- **Telemetry** ⬜ — per-rank GPU-event ring buffer, pod aggregation, end-to-end
  goodput / MFU / NIC / NVLink dashboards (compile-time excluded, checkpoint &
  data-loader stalls included).
- **Simulator fidelity** 🚧 — tighten the analytic cost model (rails, congestion,
  EP storage sharding) and **validate it against measured M2/M3 numbers** so the
  planner stays predictive as scale grows.

---

## Next three steps

1. **M1**: stand up the small-model golden reference and freeze the spec.
2. **M2**: implement `backend_cuda.c` device bring-up and match reference loss on 1 GPU.
3. **Close the loop**: feed M2's measured op times back into the simulator's cost
   model and check the M0 estimates against reality.
