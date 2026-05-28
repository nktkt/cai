/* cai_cuda.h - host-callable interface to the GPU executor (roadmap M2-device/M3).
 *
 *  ┌─────────────────────────────────────────────────────────────────────┐
 *  │ UNVERIFIED. This file and the .cu files in this directory require a   │
 *  │ CUDA toolchain (nvcc + CUDA headers), NCCL, and NVIDIA GPUs to        │
 *  │ compile and run. None were available in the authoring environment, so │
 *  │ this code is *code-complete but not compiled or executed*. It mirrors  │
 *  │ the math of the fp64 reference (src/refmodel.c), which IS verified, so │
 *  │ the intended bring-up test is: match refmodel's loss on one GPU.      │
 *  └─────────────────────────────────────────────────────────────────────┘
 *
 * Built only when CUDA=1 (see Makefile). The default GPU-free build never
 * compiles this and uses the analytic CPU backend instead. */
#ifndef CAI_CUDA_H
#define CAI_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* The on-device model mirrors src/refmodel.h's spec (MHA, RMSNorm, SwiGLU, tied
 * embedding) but runs in fp32/bf16 on the GPU. */
typedef struct {
    int vocab, seq, hidden, layers, n_heads, head_dim, ffn;
    float eps;
} cai_gpu_spec_t;

/* Parallel placement for this rank (from the compiled plan / decomposition). */
typedef struct {
    int global_rank, world_size, local_device;
    int tp, pp, dp, ep, cp;
    int tp_rank, pp_rank, dp_rank; /* coordinate within each group */
} cai_gpu_parallel_t;

typedef struct cai_gpu_ctx cai_gpu_ctx_t;

/* Bring up CUDA, allocate the static arena, build NCCL communicators from the
 * parallel layout, and (optionally) capture the steady step as a CUDA graph. */
cai_gpu_ctx_t *cai_gpu_init(const cai_gpu_spec_t *spec, const cai_gpu_parallel_t *par);
void cai_gpu_free(cai_gpu_ctx_t *c);

/* One real training step on device tokens/targets ([B*seq]); returns mean loss.
 * Runs forward, backward, gradient all-reduce/reduce-scatter, and AdamW. */
float cai_gpu_train_step(cai_gpu_ctx_t *c, int B, const int *d_tokens,
                         const int *d_targets, float lr);

/* Same, but tokens/targets are host arrays (uploaded internally). Lets a plain-C
 * driver (tools/gpu_trainer.c) run the device step without touching CUDA. */
float cai_gpu_train_step_host(cai_gpu_ctx_t *c, int B, const int *h_tokens,
                              const int *h_targets, float lr);

/* ---- kernel launchers (cai_kernels.cu), all on a cudaStream_t (void* here to
 * keep this header CUDA-header-free for the C callers) -------------------- */
void cai_k_embed_fwd(const float *E, const float *P, const int *tok, float *out,
                     int BT, int T, int H, void *stream);
void cai_k_embed_bwd(float *dE, float *dP, const int *tok, const float *dout,
                     int BT, int T, int H, void *stream);
void cai_k_rmsnorm_fwd(const float *x, const float *g, float *y, float *recip,
                       int rows, int H, float eps, void *stream);
void cai_k_rmsnorm_bwd(const float *x, const float *g, const float *recip,
                       const float *dy, float *dx, float *dg, int rows, int H,
                       void *stream);
void cai_k_rope(float *q, float *k, int B, int T, int nh, int hd, void *stream);
void cai_k_attn_fwd(const float *Q, const float *K, const float *V, float *probs,
                    float *ctx, int B, int T, int nh, int hd, void *stream);
void cai_k_attn_bwd(const float *Q, const float *K, const float *V,
                    const float *probs, const float *dctx, float *dQ, float *dK,
                    float *dV, int B, int T, int nh, int hd, void *stream);
void cai_k_swiglu_fwd(const float *gate, const float *up, float *act, int n,
                      void *stream);
void cai_k_swiglu_bwd(const float *gate, const float *up, const float *dact,
                      float *dgate, float *dup, int n, void *stream);
void cai_k_add(const float *a, const float *b, float *out, int n, void *stream);
void cai_k_cross_entropy(const float *logits, const int *tgt, float *sm,
                         float *loss_partial, int BT, int V, void *stream);
void cai_k_ce_grad(const float *sm, const int *tgt, float *dlogits, int BT, int V,
                   void *stream);
void cai_k_adamw(float *p, const float *g, float *m, float *v, int n, float lr,
                 float b1, float b2, float eps, float wd, long t, void *stream);

#ifdef __cplusplus
}
#endif

#endif /* CAI_CUDA_H */
