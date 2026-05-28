/* backend_cuda.c - real GPU backend.
 *
 * V1 intentionally leaves the device path stubbed: the production plan is to
 * launch precompiled cubins via the CUDA Driver API (cuLaunchKernel) and drive
 * collectives through NCCL / NVSHMEM, recording CUDA events so op_seconds can
 * return measured time. None of that can run here (no GPU), so unless built with
 * -DCAI_WITH_CUDA and a device present, creation fails and the runtime falls
 * back to the analytic CPU backend. */
#include <stddef.h>

#include "backend.h"

#ifdef CAI_WITH_CUDA

/* Real implementation would: cuInit, cuDeviceGet, cuCtxCreate, load module
 * cubins, create streams/events, wire NCCL communicators from the plan's comm
 * groups, then time each launched op. Left unimplemented in this scaffold. */
cai_backend_t *cai_backend_cuda_create(const cai_topology_t *topo) {
    (void)topo;
    return NULL; /* TODO: device bring-up */
}

#else

cai_backend_t *cai_backend_cuda_create(const cai_topology_t *topo) {
    (void)topo;
    return NULL; /* built without CUDA */
}

#endif
