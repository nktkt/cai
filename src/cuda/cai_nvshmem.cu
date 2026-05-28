/* cai_nvshmem.cu - GPU-initiated fine-grained communication (roadmap P3).
 *
 * UNVERIFIED: needs nvcc + CUDA + NVSHMEM + GPUs; not compiled/run here. Only
 * built with CAI_WITH_NVSHMEM. NVSHMEM lets a GPU kernel put/get remote GPU
 * memory without a host launch per message — the right tool for pipeline-stage
 * activation handoff and MoE token routing at strong-scaling limits, where the
 * per-op launch/sync overhead of host-driven NCCL send/recv starts to dominate. */
#include "cai_cuda.h"

#ifdef CAI_WITH_NVSHMEM

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

extern "C" void cai_nvshmem_init(void) { nvshmem_init(); }
extern "C" int cai_nvshmem_my_pe(void) { return nvshmem_my_pe(); }
extern "C" int cai_nvshmem_n_pes(void) { return nvshmem_n_pes(); }

/* allocate from the symmetric heap (same address on every PE) */
extern "C" void *cai_nvshmem_malloc(size_t bytes) { return nvshmem_malloc(bytes); }
extern "C" void cai_nvshmem_free(void *p) { nvshmem_free(p); }

/* host-initiated bulk put on a stream (e.g. pipeline activation to next stage) */
extern "C" void cai_nvshmem_put_stream(float *dst, const float *src, size_t n,
                                       int peer, void *stream) {
    nvshmemx_float_put_on_stream(dst, src, n, peer, (cudaStream_t)stream);
}

/* device-side fine-grained handoff: a kernel can push a stage's activation chunk
 * straight into the next stage's symmetric buffer, then signal completion. */
__global__ void pipe_push_k(float *dst, const float *src, size_t n, int peer,
                            uint64_t *flag, uint64_t token) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) nvshmem_float_p(&dst[i], src[i], peer);
    __syncthreads();
    if (i == 0) {
        nvshmem_fence();
        nvshmemx_signal_op(flag, token, NVSHMEM_SIGNAL_SET, peer);
    }
}

extern "C" void cai_nvshmem_pipe_push(float *dst, const float *src, size_t n,
                                      int peer, uint64_t *flag, uint64_t token,
                                      void *stream) {
    int tpb = 256, blocks = (int)((n + tpb - 1) / tpb);
    pipe_push_k<<<blocks, tpb, 0, (cudaStream_t)stream>>>(dst, src, n, peer, flag, token);
}

extern "C" void cai_nvshmem_finalize(void) { nvshmem_finalize(); }

#else

/* built without NVSHMEM: the executor falls back to NCCL send/recv for pipeline. */
extern "C" void cai_nvshmem_init(void) {}
extern "C" int cai_nvshmem_my_pe(void) { return -1; }
extern "C" int cai_nvshmem_n_pes(void) { return 0; }

#endif
