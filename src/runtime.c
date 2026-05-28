#include <stdlib.h>
#include <string.h>

#include "backend.h"
#include "cai.h"
#include "cai_common.h"
#include "cai_model.h"

struct cai_context {
    cai_init_desc_t desc;
    cai_topology_t topo;
    cai_backend_t *be;
    cai_plan_t plan;
    int plan_loaded;
    uint32_t step;
    cai_step_stats_t last;
    double stream_time[CAI_STREAM_COUNT];
    double *event_time; /* [plan.hdr.num_events] */
};

static const cai_comm_group_t *find_group(const cai_plan_t *p, uint32_t id) {
    if (id < p->hdr.num_groups && p->groups[id].group_id == id)
        return &p->groups[id];
    for (uint32_t i = 0; i < p->hdr.num_groups; i++)
        if (p->groups[i].group_id == id) return &p->groups[i];
    return NULL;
}

int cai_init(cai_context_t **out, const cai_init_desc_t *desc) {
    cai_context_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return CAI_ERR_OOM;
    ctx->desc = *desc;

    char err[128];
    if (desc->topology_path) {
        int rc = cai_topology_read(desc->topology_path, &ctx->topo);
        if (rc != CAI_OK) {
            CAI_ERROR("cannot read topology '%s'", desc->topology_path);
            free(ctx);
            return rc;
        }
        if (cai_topology_finalize(&ctx->topo, err, sizeof(err)) != CAI_OK) {
            CAI_ERROR("bad topology: %s", err);
            free(ctx);
            return CAI_ERR_TOPOLOGY;
        }
    } else {
        cai_topology_default(&ctx->topo);
        cai_topology_finalize(&ctx->topo, err, sizeof(err));
    }

    ctx->be = cai_backend_cuda_create(&ctx->topo);
    if (!ctx->be) ctx->be = cai_backend_cpu_create(&ctx->topo);
    if (!ctx->be) {
        free(ctx);
        return CAI_ERR_BACKEND;
    }
    CAI_INFO("rank %u/%u backend=%s", desc->global_rank, ctx->topo.world_size,
             ctx->be->name);

    if (desc->plan_path) {
        int rc = cai_load_plan(ctx, desc->plan_path);
        if (rc != CAI_OK) {
            cai_finalize(ctx);
            return rc;
        }
    }
    *out = ctx;
    return CAI_OK;
}

int cai_load_plan(cai_context_t *ctx, const char *path) {
    int rc = cai_plan_read(path, &ctx->plan);
    if (rc != CAI_OK) {
        CAI_ERROR("cannot read plan '%s': %s", path, cai_status_str(rc));
        return rc;
    }
    if (ctx->plan.hdr.topology_hash != cai_topology_hash(&ctx->topo)) {
        CAI_ERROR("plan was compiled for a different topology (hash mismatch)");
        cai_plan_free(&ctx->plan);
        return CAI_ERR_TOPOLOGY;
    }
    if (ctx->plan.hdr.world_size != ctx->topo.world_size) {
        CAI_ERROR("plan world_size %u != topology world_size %u",
                  ctx->plan.hdr.world_size, ctx->topo.world_size);
        cai_plan_free(&ctx->plan);
        return CAI_ERR_PLAN;
    }
    free(ctx->event_time);
    ctx->event_time = calloc(ctx->plan.hdr.num_events ? ctx->plan.hdr.num_events : 1,
                             sizeof(double));
    if (!ctx->event_time) {
        cai_plan_free(&ctx->plan);
        return CAI_ERR_OOM;
    }
    ctx->plan_loaded = 1;
    CAI_INFO("plan loaded: stage %u/%u ops=%u groups=%u arena=%.2f GiB",
             ctx->plan.hdr.pipeline_stage, ctx->plan.hdr.num_stages,
             ctx->plan.hdr.num_ops, ctx->plan.hdr.num_groups,
             ctx->plan.hdr.arena_bytes / (1024.0 * 1024.0 * 1024.0));
    return CAI_OK;
}

int cai_load_checkpoint(cai_context_t *ctx, const char *path) {
    char latest[600];
    snprintf(latest, sizeof(latest), "%s.latest.rank%06u", path,
             ctx->desc.global_rank);
    FILE *f = fopen(latest, "rb"); /* prefer the rolling pointer */
    if (!f) f = fopen(path, "rb"); /* else an explicit checkpoint file */
    if (!f) return CAI_OK;         /* neither: fresh start */
    uint32_t step = 0;
    if (cai_read_all(f, &step, sizeof(step)) == CAI_OK) ctx->step = step;
    fclose(f);
    CAI_INFO("resumed from checkpoint at step %u", ctx->step);
    return CAI_OK;
}

int cai_next_batch(cai_context_t *ctx, cai_batch_t *batch) {
    if (ctx->desc.max_steps && ctx->step >= ctx->desc.max_steps) return 1; /* EOF */
    batch->step = ctx->step;
    batch->tokens = ctx->plan_loaded ? ctx->plan.hdr.global_tokens_per_step : 0;
    batch->seed = 0x9E3779B97F4A7C15ull ^ ctx->step;
    return CAI_OK;
}

int cai_train_step(cai_context_t *ctx, const cai_batch_t *batch) {
    (void)batch;
    if (!ctx->plan_loaded) return CAI_ERR_PLAN;
    const cai_plan_t *p = &ctx->plan;

    for (int s = 0; s < CAI_STREAM_COUNT; s++) ctx->stream_time[s] = 0.0;
    for (uint32_t e = 0; e < p->hdr.num_events; e++) ctx->event_time[e] = 0.0;

    /* Single forward pass over the program: each stream advances in issue order;
     * record/wait events couple streams. The compiler guarantees a DAG (every
     * WAIT follows its RECORD in program order), so one pass yields correct
     * stream end-times and naturally exposes comm/compute overlap via max(). */
    for (uint32_t i = 0; i < p->hdr.num_ops; i++) {
        const cai_op_t *op = &p->ops[i];
        uint16_t st = op->stream_id;
        if (st >= CAI_STREAM_COUNT) return CAI_ERR_PLAN;

        if (op->kind == CAI_OP_EVENT_RECORD) {
            if (op->aux0 >= p->hdr.num_events) return CAI_ERR_PLAN;
            ctx->event_time[op->aux0] = ctx->stream_time[st];
            continue;
        }
        if (op->kind == CAI_OP_EVENT_WAIT) {
            if (op->aux0 >= p->hdr.num_events) return CAI_ERR_DEPENDENCY;
            double et = ctx->event_time[op->aux0];
            if (et > ctx->stream_time[st]) ctx->stream_time[st] = et;
            continue;
        }

        const cai_comm_group_t *g = NULL;
        if (cai_op_is_comm(op->kind)) {
            g = find_group(p, op->aux0);
            if (!g) return CAI_ERR_PLAN;
        }
        ctx->stream_time[st] += ctx->be->op_seconds(ctx->be, op, g);
    }

    double busy = 0.0, busy_compute = 0.0;
    for (int s = 0; s < CAI_STREAM_COUNT; s++)
        if (ctx->stream_time[s] > busy) busy = ctx->stream_time[s];
    if (ctx->stream_time[CAI_STREAM_COMPUTE_HI] > busy_compute)
        busy_compute = ctx->stream_time[CAI_STREAM_COMPUTE_HI];
    if (ctx->stream_time[CAI_STREAM_COMPUTE_LO] > busy_compute)
        busy_compute = ctx->stream_time[CAI_STREAM_COMPUTE_LO];

    double bubble = cai_pipeline_bubble(p->hdr.num_stages, p->hdr.num_microbatches);
    double step_time = (bubble < 1.0) ? busy / (1.0 - bubble) : busy;

    double useful_per_gpu = (double)p->hdr.global_tokens_per_step *
                            (double)p->hdr.flops_per_token_useful /
                            (double)p->hdr.world_size;
    double cap = ctx->topo.gpu_flops_bf16 * step_time;

    cai_step_stats_t *L = &ctx->last;
    L->step_time_s = step_time;
    L->busy_compute_s = busy_compute;
    L->bubble_ratio = bubble;
    L->mfu = cap > 0 ? useful_per_gpu / cap : 0.0;
    L->tokens_per_s = step_time > 0 ? (double)p->hdr.global_tokens_per_step / step_time : 0.0;
    L->tokens_per_s_per_gpu = L->tokens_per_s / (double)p->hdr.world_size;
    L->arena_bytes = p->hdr.arena_bytes;

    ctx->step++;
    return CAI_OK;
}

int cai_should_checkpoint(cai_context_t *ctx) {
    return ctx->desc.checkpoint_every && (ctx->step % ctx->desc.checkpoint_every) == 0;
}

static int write_ckpt_file(const char *path, uint32_t step, uint64_t plan_hash) {
    FILE *f = fopen(path, "wb");
    if (!f) return CAI_ERR_IO;
    int rc = cai_write_all(f, &step, sizeof(step));
    if (rc == CAI_OK) rc = cai_write_all(f, &plan_hash, sizeof(plan_hash));
    fclose(f);
    return rc;
}

int cai_save_checkpoint(cai_context_t *ctx, const char *tag) {
    if (!ctx->desc.checkpoint_path) return CAI_OK;
    char tagged[600], latest[600];
    uint32_t r = ctx->desc.global_rank;
    snprintf(tagged, sizeof(tagged), "%s.%s.rank%06u", ctx->desc.checkpoint_path, tag, r);
    snprintf(latest, sizeof(latest), "%s.latest.rank%06u", ctx->desc.checkpoint_path, r);
    int rc = write_ckpt_file(tagged, ctx->step, ctx->plan.hdr.plan_hash);
    /* roll the "latest" pointer so resume finds it without knowing the tag */
    if (rc == CAI_OK) rc = write_ckpt_file(latest, ctx->step, ctx->plan.hdr.plan_hash);
    return rc;
}

int cai_last_step_stats(cai_context_t *ctx, cai_step_stats_t *out) {
    if (!ctx->plan_loaded) return CAI_ERR_PLAN;
    *out = ctx->last;
    return CAI_OK;
}

int cai_finalize(cai_context_t *ctx) {
    if (!ctx) return CAI_OK;
    if (ctx->be && ctx->be->destroy) ctx->be->destroy(ctx->be);
    cai_plan_free(&ctx->plan);
    free(ctx->event_time);
    free(ctx);
    return CAI_OK;
}
