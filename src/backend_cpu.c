/* backend_cpu.c - analytic backend. It does not move tensors; it returns the
 * time each op *would* occupy its stream on the resolved topology. This is what
 * lets the whole trainer build and run a faithful schedule simulation with no
 * GPU. The CUDA backend (see backend_cuda.c) replaces op_seconds with launches
 * and measured time. */
#include <math.h>
#include <stdlib.h>

#include "backend.h"
#include "cai_plan.h"

/* alpha-beta cost: time = alpha * hops + beta * bytes. alpha is one network hop;
 * bandwidth-optimal collectives move ~ring_factor * message bytes, while latency
 * scales with the number of synchronization steps (tree-like, ~log2(size)). */
#define CAI_HOP_LATENCY 1.5e-6

typedef struct {
    cai_topology_t topo;
} cpu_impl_t;

static double ilog2(double n) {
    double s = 0;
    while (n > 1.0) { n *= 0.5; s += 1.0; }
    return s; /* ceil(log2(size)) */
}

/* bytes-on-wire factor (ring/bandwidth-optimal) for a collective. */
static double coll_factor(uint16_t kind, double n) {
    if (n <= 1.0) return 0.0;
    switch (kind) {
        case CAI_OP_ALL_REDUCE: return 2.0 * (n - 1.0) / n;
        case CAI_OP_REDUCE_SCATTER:
        case CAI_OP_ALL_GATHER:
        case CAI_OP_ALL_TO_ALL: return (n - 1.0) / n;
        default: return 1.0;
    }
}

/* number of latency-bound synchronization steps for a collective. */
static double coll_hops(uint16_t kind, double n) {
    if (n <= 1.0) return 0.0;
    switch (kind) {
        case CAI_OP_ALL_REDUCE: return 2.0 * ilog2(n);
        case CAI_OP_REDUCE_SCATTER:
        case CAI_OP_ALL_GATHER: return ilog2(n);
        case CAI_OP_ALL_TO_ALL: return 1.0; /* one all-to-all exchange */
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
        double n = g ? (double)g->size : 1.0;
        if (op->kind == CAI_OP_PIPE_SEND || op->kind == CAI_OP_PIPE_RECV)
            return CAI_HOP_LATENCY + (double)op->bytes / bw; /* point-to-point */
        double f = coll_factor(op->kind, n);
        double hops = coll_hops(op->kind, n);
        return CAI_HOP_LATENCY * hops + f * (double)op->bytes / bw;
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
