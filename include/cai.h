/* cai.h - public training runtime API.
 *
 * cai = a C AI trainer specialized for one fixed model on one fixed cluster.
 * It is deliberately NOT a framework: no autodiff, no dynamic shapes, no Python,
 * no runtime allocation, no graph compilation on the hot path. The runtime
 * loads a precompiled plan + topology and replays it. */
#ifndef CAI_H
#define CAI_H

#include <stddef.h>
#include <stdint.h>

#include "cai_plan.h"
#include "cai_tensor.h"
#include "cai_topology.h"

typedef enum {
    CAI_OK = 0,
    CAI_ERR_IO = -1,
    CAI_ERR_INVALID = -2,
    CAI_ERR_OOM = -3,
    CAI_ERR_TOPOLOGY = -4,
    CAI_ERR_PLAN = -5,
    CAI_ERR_BACKEND = -6,
    CAI_ERR_DEPENDENCY = -7,
    CAI_ERR_BUDGET = -8
} cai_status_t;

const char *cai_status_str(int status);

typedef struct cai_context cai_context_t;

typedef struct {
    uint32_t global_rank;
    uint32_t world_size;
    uint32_t local_rank;
    const char *plan_path;
    const char *topology_path;
    const char *checkpoint_path; /* NULL to start from scratch */
    uint32_t checkpoint_every;   /* steps; 0 disables */
    uint32_t max_steps;          /* 0 = run until next_batch reports EOF */
} cai_init_desc_t;

typedef struct {
    uint32_t step;
    uint64_t tokens; /* tokens in this global batch */
    uint64_t seed;
} cai_batch_t;

/* Per-step telemetry produced by the (simulated, in V1) backend. */
typedef struct {
    double step_time_s;     /* wall-clock of the step (critical path) */
    double busy_compute_s;  /* GPU compute occupancy on the critical path */
    double bubble_ratio;    /* pipeline idle fraction */
    double mfu;             /* model FLOP utilization 0..1 */
    double tokens_per_s;    /* global tokens / step_time */
    double tokens_per_s_per_gpu;
    uint64_t arena_bytes;
} cai_step_stats_t;

int cai_init(cai_context_t **ctx, const cai_init_desc_t *desc);
int cai_load_plan(cai_context_t *ctx, const char *path);
int cai_load_checkpoint(cai_context_t *ctx, const char *path);

int cai_next_batch(cai_context_t *ctx, cai_batch_t *batch);
int cai_train_step(cai_context_t *ctx, const cai_batch_t *batch);

int cai_should_checkpoint(cai_context_t *ctx);
int cai_save_checkpoint(cai_context_t *ctx, const char *tag);

int cai_last_step_stats(cai_context_t *ctx, cai_step_stats_t *out);
int cai_finalize(cai_context_t *ctx);

#endif /* CAI_H */
