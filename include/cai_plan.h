/* cai_plan.h - the compiled, per-rank execution plan.
 *
 * A plan is a flat instruction stream over a small fixed set of streams. The
 * runtime contains no scheduling policy: it replays ops in order per stream and
 * honors cross-stream ordering via record/wait events. All the "thinking"
 * (parallelism layout, memory packing, collective grouping, pipeline schedule)
 * happens once, offline, in tools/plan_compiler. */
#ifndef CAI_PLAN_H
#define CAI_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "cai_tensor.h"

#define CAI_PLAN_MAGIC 0x50494143u /* 'CAIP' */
#define CAI_PLAN_VERSION 1u

/* Logical streams; map 1:1 onto CUDA streams in the real backend. */
typedef enum {
    CAI_STREAM_COMPUTE_HI = 0, /* GEMM / attention */
    CAI_STREAM_COMPUTE_LO = 1, /* norm / pointwise / optimizer */
    CAI_STREAM_COMM_TP = 2,    /* tensor-parallel collectives */
    CAI_STREAM_COMM_PP = 3,    /* pipeline point-to-point */
    CAI_STREAM_COMM_DP = 4,    /* data-parallel reduce-scatter / all-gather */
    CAI_STREAM_IO = 5,         /* checkpoint / prefetch */
    CAI_STREAM_COUNT
} cai_stream_t;

const char *cai_stream_name(uint16_t s);

typedef enum {
    CAI_OP_GEMM = 0,
    CAI_OP_ATTENTION_FWD,
    CAI_OP_ATTENTION_BWD,
    CAI_OP_RMSNORM,
    CAI_OP_SWIGLU,
    CAI_OP_ELEMENTWISE,
    CAI_OP_REDUCE_SCATTER,
    CAI_OP_ALL_GATHER,
    CAI_OP_ALL_REDUCE,
    CAI_OP_ALL_TO_ALL,
    CAI_OP_PIPE_SEND,
    CAI_OP_PIPE_RECV,
    CAI_OP_OPTIMIZER,
    CAI_OP_EVENT_RECORD,
    CAI_OP_EVENT_WAIT,
    CAI_OP_COUNT
} cai_op_kind_t;

const char *cai_op_name(uint16_t kind);
int cai_op_is_comm(uint16_t kind);
int cai_op_is_compute(uint16_t kind);

/* One instruction. 32 bytes, no padding.
 *  - compute ops use `flops`
 *  - comm/pipe ops use `bytes` and aux0 = comm_group id
 *  - EVENT_RECORD / EVENT_WAIT use aux0 = event id
 *  - aux1 generally carries the microbatch id (for tracing) */
typedef struct {
    uint64_t flops;
    uint64_t bytes;
    uint32_t aux0;
    uint32_t aux1;
    uint16_t kind;
    uint16_t stream_id;
    uint32_t pad;
} cai_op_t;

_Static_assert(sizeof(cai_op_t) == 32, "cai_op_t layout");

typedef enum {
    CAI_GROUP_TP = 0,
    CAI_GROUP_DP = 1,
    CAI_GROUP_PP = 2,
    CAI_GROUP_EP = 3,
    CAI_GROUP_COUNT
} cai_group_kind_t;

/* A collective communicator the runtime must create. Ranks are not listed (at
 * 220k scale that is wasteful); they are derived from (kind, color, size). 16
 * bytes. */
typedef struct {
    uint32_t group_id;
    uint16_t kind; /* cai_group_kind_t */
    uint16_t intra_rack; /* 1 if every member shares one NVLink domain */
    uint32_t size;       /* ranks in the group */
    uint32_t color;      /* communicator color (members share it) */
} cai_comm_group_t;

_Static_assert(sizeof(cai_comm_group_t) == 16, "cai_comm_group_t layout");

/* Fixed-size file header. 128 bytes. Followed by, in order:
 *   tensor table   (num_tensors  * cai_tensor_desc_t)
 *   comm groups    (num_groups   * cai_comm_group_t)
 *   op table       (num_ops      * cai_op_t)            */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t plan_hash;     /* identical across all ranks of one job */
    uint64_t topology_hash; /* must match the loaded topology.bin */

    uint32_t global_rank;
    uint32_t world_size;
    uint32_t pipeline_stage;
    uint32_t num_stages; /* == PP */

    uint32_t tp;
    uint32_t dp;
    uint32_t ep;
    uint32_t cp;

    uint32_t microbatch;
    uint32_t grad_accum;
    uint32_t num_microbatches;
    uint32_t seq_len;

    uint32_t hidden_size;
    uint32_t layers_in_stage;
    uint32_t num_tensors;
    uint32_t num_groups;

    uint32_t num_ops;
    uint32_t num_events;
    uint64_t arena_bytes; /* total static arena this rank reserves */

    uint64_t global_tokens_per_step;
    uint64_t flops_per_token_useful; /* 6N-style, excludes recompute; for MFU */
    uint32_t vpp;                    /* interleave / virtual-pipeline factor (>=1) */
    uint32_t reserved;
} cai_plan_header_t;

_Static_assert(sizeof(cai_plan_header_t) == 128, "cai_plan_header_t layout");

/* In-memory plan owned by the runtime after load. */
typedef struct {
    cai_plan_header_t hdr;
    cai_tensor_desc_t *tensors;
    cai_comm_group_t *groups;
    cai_op_t *ops;
} cai_plan_t;

int cai_plan_write(const char *path, const cai_plan_t *plan);
int cai_plan_read(const char *path, cai_plan_t *plan);
void cai_plan_free(cai_plan_t *plan);

/* Arena bytes broken down by tensor class (for reporting / budgeting). */
void cai_plan_arena_breakdown(const cai_plan_t *plan,
                              uint64_t out_bytes[CAI_TCLASS_COUNT]);

#endif /* CAI_PLAN_H */
