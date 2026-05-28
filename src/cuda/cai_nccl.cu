/* cai_nccl.cu - NCCL communicators from the compiled parallel layout (roadmap M3).
 *
 * UNVERIFIED: needs nvcc + CUDA + NCCL + GPUs; not compiled/run in the authoring
 * env. Builds two communicators per rank — a data-parallel group (grad all-reduce)
 * and a tensor-parallel group (activation all-reduce) — using the same rank
 * layout the offline compiler validated (TP innermost). The ncclUniqueId is
 * exchanged through a shared filesystem path keyed by group color; production
 * would bootstrap via MPI / torch.distributed instead. */
#include "cai_cuda.h"

#ifdef CAI_WITH_NCCL

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NK(call)                                                          \
    do {                                                                  \
        ncclResult_t r_ = (call);                                         \
        if (r_ != ncclSuccess) {                                          \
            fprintf(stderr, "NCCL %s:%d %s\n", __FILE__, __LINE__,        \
                    ncclGetErrorString(r_));                              \
            abort();                                                      \
        }                                                                 \
    } while (0)

typedef struct {
    ncclComm_t dp, tp;
    int dp_size, tp_size;
} cai_nccl_t;

/* group_rank==0 publishes the unique id to a shared file; the rest poll for it. */
static void exchange_id(const char *tag, int color, int group_rank, ncclUniqueId *id) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/cai_nccl_%s_%d.id", tag, color);
    if (group_rank == 0) {
        NK(ncclGetUniqueId(id));
        FILE *f = fopen(path, "wb");
        if (!f) { perror("nccl id write"); abort(); }
        fwrite(id, sizeof(*id), 1, f);
        fclose(f);
    } else {
        FILE *f = NULL;
        for (int tries = 0; tries < 100000 && !(f = fopen(path, "rb")); tries++)
            usleep(1000);
        if (!f) { fprintf(stderr, "nccl id timeout %s\n", path); abort(); }
        if (fread(id, sizeof(*id), 1, f) != 1) { abort(); }
        fclose(f);
    }
}

extern "C" void *cai_nccl_init(const cai_gpu_parallel_t *par) {
    cai_nccl_t *h = (cai_nccl_t *)calloc(1, sizeof(*h));
    int replica = par->tp * par->pp * par->cp;
    int tp_color = par->global_rank / par->tp;   /* TP innermost */
    int dp_color = par->global_rank % replica;    /* DP outermost */

    if (par->dp > 1) {
        ncclUniqueId id;
        exchange_id("dp", dp_color, par->dp_rank, &id);
        NK(ncclCommInitRank(&h->dp, par->dp, id, par->dp_rank));
        h->dp_size = par->dp;
    }
    if (par->tp > 1) {
        ncclUniqueId id;
        exchange_id("tp", tp_color, par->tp_rank, &id);
        NK(ncclCommInitRank(&h->tp, par->tp, id, par->tp_rank));
        h->tp_size = par->tp;
    }
    return h;
}

extern "C" void cai_nccl_allreduce_dp(void *vh, float *buf, size_t n, void *stream) {
    cai_nccl_t *h = (cai_nccl_t *)vh;
    if (!h || h->dp_size <= 1) return;
    /* average gradients across data-parallel replicas */
    NK(ncclAllReduce(buf, buf, n, ncclFloat, ncclAvg, h->dp, (cudaStream_t)stream));
}

extern "C" void cai_nccl_allreduce_tp(void *vh, float *buf, size_t n, void *stream) {
    cai_nccl_t *h = (cai_nccl_t *)vh;
    if (!h || h->tp_size <= 1) return;
    /* sum partial activations across the tensor-parallel group */
    NK(ncclAllReduce(buf, buf, n, ncclFloat, ncclSum, h->tp, (cudaStream_t)stream));
}

extern "C" void cai_nccl_free(void *vh) {
    cai_nccl_t *h = (cai_nccl_t *)vh;
    if (!h) return;
    if (h->dp_size > 1) ncclCommDestroy(h->dp);
    if (h->tp_size > 1) ncclCommDestroy(h->tp);
    free(h);
}

#else /* built without NCCL: single-GPU only (collectives are no-ops) */

#include <stdio.h>
#include <stdlib.h>

extern "C" void *cai_nccl_init(const cai_gpu_parallel_t *par) {
    if (par->world_size > 1) {
        fprintf(stderr, "cai_nccl: built without NCCL; world_size>1 unsupported\n");
        abort();
    }
    return 0;
}
extern "C" void cai_nccl_allreduce_dp(void *h, float *b, size_t n, void *s) {
    (void)h; (void)b; (void)n; (void)s;
}
extern "C" void cai_nccl_allreduce_tp(void *h, float *b, size_t n, void *s) {
    (void)h; (void)b; (void)n; (void)s;
}
extern "C" void cai_nccl_free(void *h) { (void)h; }

#endif /* CAI_WITH_NCCL */
