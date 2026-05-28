/* backend_cpu.c - analytic backend. It does not move tensors; it returns the
 * time each op *would* occupy its stream on the resolved topology. This is what
 * lets the whole trainer build and run a faithful schedule simulation with no
 * GPU. The CUDA backend (see backend_cuda.c) replaces op_seconds with launches
 * and measured time. */
#include <stdlib.h>

#include "backend.h"
#include "cai_plan.h"

#define CAI_COLL_LATENCY 5e-6 /* fixed per-collective launch+sync cost */

typedef struct {
    cai_topology_t topo;
} cpu_impl_t;

/* Ring-algorithm bytes-on-wire factor relative to the message size. */
static double coll_factor(uint16_t kind, uint32_t size) {
    if (size <= 1) return 0.0;
    double n = (double)size;
    switch (kind) {
        case CAI_OP_ALL_REDUCE: return 2.0 * (n - 1.0) / n;
        case CAI_OP_REDUCE_SCATTER:
        case CAI_OP_ALL_GATHER:
        case CAI_OP_ALL_TO_ALL: return (n - 1.0) / n;
        case CAI_OP_PIPE_SEND:
        case CAI_OP_PIPE_RECV: return 1.0;
        default: return 1.0;
    }
}

static double cpu_op_seconds(cai_backend_t *be, const cai_op_t *op,
                             const cai_comm_group_t *g) {
    cpu_impl_t *s = (cpu_impl_t *)be->impl;
    if (cai_op_is_compute(op->kind))
        return (double)op->flops / s->topo.gpu_flops_bf16;
    if (cai_op_is_comm(op->kind)) {
        int intra = (g && g->intra_rack);
        double bw = intra ? (double)s->topo.intra_rack_bw
                          : (double)s->topo.inter_rack_bw;
        uint32_t size = g ? g->size : 1;
        double f = coll_factor(op->kind, size);
        if (op->kind == CAI_OP_PIPE_SEND || op->kind == CAI_OP_PIPE_RECV)
            f = 1.0; /* point-to-point */
        return CAI_COLL_LATENCY + f * (double)op->bytes / bw;
    }
    return 0.0; /* events have no duration */
}

static int cpu_sync(cai_backend_t *be) {
    (void)be;
    return 0;
}

static void cpu_destroy(cai_backend_t *be) {
    if (!be) return;
    free(be->impl);
    free(be);
}

cai_backend_t *cai_backend_cpu_create(const cai_topology_t *topo) {
    cai_backend_t *be = calloc(1, sizeof(*be));
    cpu_impl_t *impl = calloc(1, sizeof(*impl));
    if (!be || !impl) {
        free(be);
        free(impl);
        return NULL;
    }
    impl->topo = *topo;
    be->name = "cpu-analytic";
    be->impl = impl;
    be->op_seconds = cpu_op_seconds;
    be->sync = cpu_sync;
    be->destroy = cpu_destroy;
    return be;
}
