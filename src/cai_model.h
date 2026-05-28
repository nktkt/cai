/* cai_model.h - the fixed model spec, parallelism config, and the offline
 * "compiler" front-end (config -> decomposition -> per-rank plan).
 *
 * Internal to the build; consumed by tools/plan_compiler and tools/topology_linter. */
#ifndef CAI_MODEL_H
#define CAI_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "cai_plan.h"
#include "cai_topology.h"

typedef struct {
    /* model graph (fixed) */
    uint32_t hidden_size;
    uint32_t ffn_hidden;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads; /* GQA; == num_heads for MHA */
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t seq_len;

    /* MoE (0 -> dense) */
    int is_moe;
    uint32_t num_experts;
    uint32_t moe_top_k;

    /* precision */
    uint16_t param_dtype; /* cai_dtype_t for weights/activations */
    uint16_t optim_dtype; /* cai_dtype_t for optimizer state */
    uint32_t optim_states; /* fp states per param (AdamW = 2) */
    int master_weights;    /* keep an fp32 master copy of params */
    int recompute;         /* recompute activations in backward */

    /* parallelism (fixed) */
    uint32_t tp;
    uint32_t pp;
    uint32_t ep; /* expert parallel (MoE); 1 if dense */
    uint32_t cp; /* context parallel; 1 if unused */
    uint32_t microbatch;
    uint32_t grad_accum;

    uint64_t mem_budget; /* per-GPU byte budget; 0 -> use topology gpu_mem */
} cai_model_spec_t;

typedef struct {
    cai_model_spec_t model;
    cai_topology_t topo;
} cai_config_t;

void cai_config_defaults(cai_config_t *cfg);
int cai_config_load(const char *path, cai_config_t *cfg, char *err, size_t errlen);

/* Derived, cluster-wide quantities. Pure function of (model, finalized topo). */
typedef struct {
    uint32_t tp, pp, ep, cp, dp;
    uint32_t model_replica_gpus; /* tp*pp*cp */
    uint64_t global_batch;       /* sequences per optimizer step */
    uint64_t global_tokens;      /* tokens per optimizer step */
    uint32_t num_microbatches;   /* in-flight microbatches per stage */
    uint32_t layers_per_stage;
    double bubble_ratio;

    /* per-GPU memory estimate, by class */
    uint64_t mem_param, mem_grad, mem_opt, mem_act, mem_comm, mem_ws, mem_total;

    /* compute */
    double flops_per_token_train; /* fwd+bwd(+recompute) over active params */
    double step_flops;            /* whole cluster, one optimizer step */
    uint64_t total_params;        /* full model parameter count */
    uint64_t active_params;       /* params touched per token (MoE-aware) */
} cai_decomp_t;

int cai_decompose(const cai_config_t *cfg, cai_decomp_t *out, char *err,
                  size_t errlen);

/* Per-GPU memory budget actually applied (model override or topology HBM). */
uint64_t cai_mem_budget(const cai_config_t *cfg);

/* Human-readable decomposition report. Returns 1 if memory fits the budget,
 * 0 if it is over budget. */
#include <stdio.h>
int cai_report_decomp(FILE *f, const cai_config_t *cfg, const cai_decomp_t *dec);

/* Build the plan for one specific global rank. topo must be finalized. */
int cai_build_plan(const cai_config_t *cfg, const cai_decomp_t *dec,
                   uint32_t rank, uint64_t plan_hash, uint64_t topo_hash,
                   cai_plan_t *out, char *err, size_t errlen);

/* Communicator groups a rank belongs to (cheap; no op-table). topo finalized. */
void cai_rank_groups(const cai_decomp_t *dec, const cai_topology_t *topo, int is_moe,
                     uint32_t rank, cai_comm_group_t groups[CAI_GROUP_COUNT],
                     uint32_t *ng_out);

/* Pipeline stage owning a rank. */
uint32_t cai_rank_stage(const cai_decomp_t *dec, uint32_t rank);

/* Stable hash of the logical plan (independent of which rank). */
uint64_t cai_plan_hash(const cai_config_t *cfg, const cai_decomp_t *dec);
uint64_t cai_topology_hash(const cai_topology_t *topo);

/* ---- pipeline schedule ----------------------------------------------- */

typedef enum { CAI_TICK_FWD = 0, CAI_TICK_BWD = 1 } cai_tick_kind_t;
typedef struct {
    uint16_t kind;       /* cai_tick_kind_t */
    uint16_t microbatch; /* 0 .. m-1 */
} cai_tick_t;

/* Steady-state, non-interleaved 1F1B issue order for one pipeline stage.
 * Writes 2*m ticks into buf (caller-owned, length >= 2*m). Returns 2*m. */
uint32_t cai_pipeline_schedule_1f1b(uint32_t stage, uint32_t pp, uint32_t m,
                                    cai_tick_t *buf);

/* Fraction of the step a stage spends idle from pipeline fill/drain. */
double cai_pipeline_bubble(uint32_t pp, uint32_t m);

#endif /* CAI_MODEL_H */
