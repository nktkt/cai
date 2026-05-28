# cai — a 220k GB300 specialized C training stack (Design / V1.0)

## 0. What this repository is

This takes the idea of "a 220,000-GPU-specialized AI training stack written in C"
and turns it into something you can **actually build, run, and verify**.

That said, real training (running GEMM/collectives/kernels on GPUs) **cannot run
here**, because there are no 220k GPUs available. So V1.0 implements the parts the
design considers "most important" ——

- offline plan compiler (pre-generating the parallelism decomposition, memory
  layout, communication groups, and pipeline schedule)
- topology mapping (rank ↔ rack/tray/gpu)
- static schedule (1F1B issue order)
- static memory arena (a layout premised on zero runtime malloc)
- op-table runtime (event-driven replay of the schedule, computing step time /
  bubble / MFU / tokens/s)

—— as **real, GPU-independent C code**. The parts that need a GPU (GEMM / cuDNN /
NCCL / NVSHMEM / CUDA Graphs / cubin) are isolated behind `#ifdef CAI_WITH_CUDA`,
and the whole thing runs on a CPU analytic backend.

In other words: **"training can't run, but generating and validating plans for all
220,032 ranks, and streaming the schedule through the runtime to estimate
performance characteristics, works for real."**

## 1. Goals / non-goals

What we build (V1.0):
- a **static, distributed training executor** specialized for a fixed model, fixed
  shapes, fixed precision, fixed topology, and fixed parallelism config
- runtime / scheduler / memory planner / comm coordinator / checkpoint / telemetry

What we do **not** build (V1.0):
- autodiff, dynamic shapes, a Python runtime, a generic operator dispatcher
- runtime malloc/free, runtime graph compilation
- becoming "C-flavored JAX" (throwing away generality *is* the winning move)

## 2. Architecture

All decisions happen offline; the runtime merely replays them.

```
                 model + parallelism + topology (config)
                                 |
                    +------------v------------+
                    |   offline plan compiler  |   tools/plan_compiler
                    |  decompose -> validate   |
                    |  -> emit per-rank plans  |
                    +------------+------------+
                                 |
              topology.bin + plan.rankNNNNNN.bin  (binary)
                                 |
                    +------------v------------+
                    |       C runtime          |   src/runtime.c
                    |  load + replay op-table  |
                    |  event-driven streams    |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |        backend           |   src/backend_*.c
                    |  CPU analytic (here)     |
                    |  CUDA launch (#ifdef)    |
                    +-------------------------+
```

The runtime "doesn't think": it advances each stream's ops in issue order and
synchronizes streams via record/wait events. That's all.

## 3. Binary formats

Everything is fixed-layout POD with sizes pinned by `_Static_assert` (assuming a
homogeneous cluster = same endianness).

- `topology.bin` : magic `CTOP` + `cai_topology_t` (64B)
- `plan.rankNNNNNN.bin` : `cai_plan_header_t` (128B) + tensor table + comm groups + op table
  - the header embeds `plan_hash` / `topology_hash`, so the runtime **refuses a plan
    that was compiled for a different topology** (accident prevention at 220k scale).

Key structs:
| struct | size | role |
|---|---|---|
| `cai_phys_id_t` | 12B | physical location of a rank (rack/tray/gpu/nic/rail) |
| `cai_topology_t` | 64B | cluster description (incl. bandwidth/FLOPS/memory) |
| `cai_op_t` | 32B | one instruction (flops or bytes, stream, event/group) |
| `cai_comm_group_t` | 16B | communicator (kind/color/size; ranks are derived) |
| `cai_tensor_desc_t` | 24B | a tensor in the arena (class/offset/bytes) |
| `cai_plan_header_t` | 128B | plan metadata + hashes + derived quantities |

## 4. Rank mapping and parallelism

GB300 NVL72 = one rack of 72 GPUs = one NVLink domain. The rank coordinate puts
**TP innermost**:

```
tp_id = rank % tp
cp_id = (rank / tp) % cp
pp_id = (rank / (tp*cp)) % pp      <- pipeline stage
dp_id =  rank / (tp*cp*pp)
```

This keeps a TP group on consecutive ranks = the same rack, closed inside NVLink
(the linter checks `TP <= gpus_per_rack` and divisibility).
- Inside a rack: tensor / sequence / small-scale expert parallel
- Across racks: pipeline activations / DP reduce-scatter / MoE all-to-all

If `tp*pp*cp` does not divide the world size, the compiler errors out and points you
at `spare_gpus` to adjust.

## 5. The op-table and the simulation model

The compiler expands one step per stage (all microbatches' fwd+bwd + optimizer)
into ops in 1F1B issue order (in `src/model.c`, not `tools`).

There are 6 streams: `compute_hi / compute_lo / comm_tp / comm_pp / comm_dp / io`.
Dependencies are expressed with **EVENT_RECORD / EVENT_WAIT**:
- the TP all-reduce is serialized with compute (unhideable comm, realistic)
- the DP weight all-gather (FSDP) is issued so it **overlaps** the next layer's compute
- the DP grad reduce-scatter **overlaps** backward compute, with a barrier just
  before the optimizer

The runtime simulation (`cai_train_step`):
1. Walk the ops once in issue order. Advance each stream's clock; RECORD stamps the
   event's time, WAIT stalls the stream via `max()` (the compiler guarantees a DAG,
   so one pass is correct).
2. `busy = max(stream clocks)` — whether TP/DP comm was hidden under compute falls
   out naturally.
3. Pipeline fill/drain isn't visible from a single rank, so it's corrected
   analytically:
   `step_time = busy / (1 - bubble)`, `bubble = (pp-1)/(m+pp-1)`.
4. Compute `MFU = (useful 6N flops/GPU) / (peak FLOPS * step_time)`, tokens/s, etc.

An op's duration comes from the backend (CPU: `flops/peak` and `bytes/bandwidth`,
collectives use a ring factor + intra/inter-rack bandwidth). The CUDA backend is
designed to swap the same function for "real launch + measured time."

## 6. The memory arena

No runtime malloc. Per-class segments (param/grad/optstate/activation/comm/workspace)
are laid out contiguously, and tensors are rebased from a class-local offset to a
whole-arena offset (`src/arena.c`).
The per-GPU memory estimate assumes FSDP/ZeRO-3 sharding and is checked against the
`budget` (= HBM capacity or an override), reporting OK / OVER.

## 7. File map

| design concept | implementation |
|---|---|
| offline plan compiler | `tools/plan_compiler.c` + `src/model.c` |
| topology linter | `tools/topology_linter.c` |
| runtime driver (1 proc = 1 GPU) | `tools/trainer.c` + `src/runtime.c` |
| rank/topology mapping | `src/topology.c` |
| static memory arena | `src/arena.c` |
| 1F1B schedule + bubble | `src/pipeline.c` |
| op-table / sizing / decomposition | `src/model.c` |
| plan (de)serialize | `src/plan_io.c` |
| backend abstraction | `src/backend.h` |
| CPU analytic backend | `src/backend_cpu.c` |
| CUDA backend (stub) | `src/backend_cuda.c` |
| io / hash / log / fmt | `src/common.c` |
| fp64 correctness oracle (real fwd/bwd/AdamW) | `src/refmodel.c` |
| real CPU training demo | `tools/reftrain.c` |
| whole-cluster pre-launch validator | `tools/plan_verify.c` |
| spare-rank recovery planner | `tools/plan_recover.c` (`cai_spare_rank`) |
| per-op trace + per-stream occupancy | `src/runtime.c` (`cai_trace_*`) |
| interleaved 1F1B (virtual pipeline) | `vpp` in `src/pipeline.c` / `src/model.c` |

## 8. V1.0 approximations and TODO

Intentional simplifications (the numbers land in the right order of magnitude, but
aren't exact):
- collectives use a ring factor + single-link bandwidth approximation (rails and
  congestion are not modeled)
- MoE expert storage sharding isn't split out by EP precisely; it's folded into the
  DP shard (all-to-all and active-param counts are still reflected)
- activation/comm buffer factors are representative values (switched by whether
  recompute is on)

Order in which to fill in the CUDA implementation (`backend_cuda.c`):
1. Launch precompiled cubins via the Driver API (`cuLaunchKernel`); realize streams/events
2. Build NCCL communicators from the comm groups; launch real collectives
3. Use NVSHMEM for fine-grained pipeline / MoE communication, and CUDA Graphs to bake
   the steady step
4. Swap `op_seconds` for measured time (CUDA events) → the same runtime moves to
   production as-is

Realistic speed targets (breaking down the "10x vs JAX" claim):
- V1.0: static memory + CUDA Graphs + topology-aware mapping + comm/compute overlap →
  **goodput 1.3–2.5x**
- V1.5: custom kernels + pipeline optimization → **3–5x**
- V2 : megakernels + a GPU-initiated scheduler + model/topology co-design →
  **>5x under the right conditions**
- 10x isn't "because it's C"; aim for it as the compound effect of **throwing away
  generality and fully specializing to the physical layout of 220k GB300s**.

## 9. The biggest failure modes (notes to self)

1. Turning it into a general framework (→ a degraded copy of JAX/PyTorch)
2. Writing your own GEMM/NCCL from day one (→ V1 collapses). Call the NVIDIA stack
   first, replace only the true bottlenecks
3. Underrating pipeline stage imbalance (the slowest stage dominates everything)
4. Deferring checkpointing (a no-failure assumption doesn't hold at 220k GPUs → it's a
   V1 feature)
5. Sloppy JAX comparisons (measure under identical model/tokens/precision/convergence/
   failure conditions)
