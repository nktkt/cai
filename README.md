# cai — a C AI trainer for a 220k GB300 cluster

A static, schedule-driven distributed training runtime specialized for **one fixed
model on one fixed cluster**. It is deliberately *not* a framework: no autodiff, no
dynamic shapes, no Python, no runtime allocation, no graph compilation on the hot
path. The cluster is treated as a single machine that gets compiled against.

**It builds and runs with no GPU.** Everything that needs a GPU (GEMM, cuDNN,
NCCL, NVSHMEM, cubins) lives behind `#ifdef CAI_WITH_CUDA`. An analytic CPU
backend stands in for it, so the whole pipeline — compile per-rank plans, validate
them, and replay the schedule to estimate performance — runs on a laptop.

See [DESIGN.md](DESIGN.md) for the full design (in Japanese).

> Status: V1.0 scaffold. The offline compiler, topology mapping, static schedule,
> memory arena, and op-table runtime are real, tested C. The on-device CUDA
> backend is stubbed (see `src/backend_cuda.c`).

## What it does

The system splits cleanly into an **offline compiler** and a **dumb runtime**:

1. `plan_compiler` reads a model + parallelism + topology config, computes the
   cluster decomposition (TP/PP/DP/EP/CP, global batch, memory, pipeline bubble),
   and emits `topology.bin` plus one `plan.rankNNNNNN.bin` per rank.
2. `trainer` (the runtime) loads a plan and replays its op stream over 6 logical
   streams, coupling them with record/wait events. With the analytic backend this
   is a faithful **schedule simulation**: it reports per-step critical path,
   pipeline bubble, MFU, and tokens/s. No training math runs.
3. `topology_linter` catches the failures that are catastrophic at scale before
   you compile 220k plans: indivisible world size, tensor-parallel groups that
   spill across NVLink domains, oversized per-GPU memory, bad rank↔location math.

Plans embed a `topology_hash`/`plan_hash`; the runtime refuses a plan that was
compiled for a different cluster.

## Build & run

```sh
make            # libcai.a + tools + tests
make test       # unit tests (topology / arena / pipeline / plan / decompose)
make demo       # lint -> compile 220k plans -> replay, end to end
make CUDA=1     # also build the (stubbed) CUDA backend path
```

Requires only a C11 compiler and `make`.

## Tools

```sh
# 1) validate a config (divisibility, TP-in-rack, rank mapping, bubble, memory)
bin/topology_linter configs/full220k.cfg

# 2) compile plans (writes topology.bin + plan.rankNNNNNN.bin)
bin/plan_compiler configs/full220k.cfg out/full            # representative ranks (--sample)
bin/plan_compiler configs/small.cfg   out/small --all      # all 72 ranks
bin/plan_compiler configs/full220k.cfg out/full --rank 128 # a specific rank

# 3) replay a plan through the runtime (CPU analytic backend)
bin/trainer out/full/plan.rank000000.bin out/full/topology.bin --steps 5
bin/trainer out/full/plan.rank000000.bin out/full/topology.bin --steps 8 --ckpt /tmp/ck
```

## Bundled configs

| config | cluster | model |
|---|---|---|
| `configs/small.cfg` | 1 rack / 72 GPU | toy dense (24 layers) |
| `configs/full220k.cfg` | 3056 racks / 220,032 GPU | ~408B dense |
| `configs/full220k_moe.cfg` | 3056 racks / 220,032 GPU | ~3.7T MoE (180B active) |

Example: replaying stage 0 of the 408B-dense / 220,032-GPU plan reports ~45% MFU,
10.5% pipeline bubble, and ~9.6 GiB resident per GPU (FSDP-sharded) under the
default GB300 NVL72 bandwidth/FLOP assumptions.

## Config format

`key = value` (`#` starts a comment; numbers may use exponents like `288e9`).

- **model**: `hidden_size ffn_hidden num_layers num_heads num_kv_heads head_dim vocab_size seq_len is_moe num_experts moe_top_k`
- **precision**: `param_dtype optim_dtype optim_states master_weights recompute`
- **parallelism**: `tp pp ep cp microbatch grad_accum`
- **topology**: `gpus_per_rack trays_per_rack gpus_per_tray num_racks spare_gpus intra_rack_bw inter_rack_bw hbm_bw gpu_flops_bf16 gpu_mem_bytes`
- **budget**: `mem_budget` (0 → use `gpu_mem_bytes`)

dtype names: `fp32 bf16 fp16 fp8e4m3 fp8e5m2 int32 uint8`

## Layout

```
include/   public headers (cai.h, cai_plan.h, cai_topology.h, cai_tensor.h)
src/       library: topology, arena, model/op-table, pipeline, plan_io,
           runtime, backend_cpu, backend_cuda (stub), common
tools/     plan_compiler, topology_linter, trainer
tests/     cai_test
configs/   small / full220k / full220k_moe
```

## License

MIT — see [LICENSE](LICENSE).
