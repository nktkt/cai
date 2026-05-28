/* cai_gpu_train.cu - the real on-device executor (roadmap M2-device, + M3 DDP).
 *
 * UNVERIFIED: needs nvcc + CUDA + GPU; not compiled/run in the authoring env.
 *
 * This is a GPU port of the fp64 reference (src/refmodel.c): same graph (embed +
 * learned pos, RMSNorm, causal MHA, SwiGLU, tied logits, softmax CE), in fp32.
 * For tp=dp=1 it must reproduce refmodel's loss on one GPU (the M2 bring-up test).
 * Data-parallel (dp>1) is wired via an NCCL all-reduce of gradients (M3 DDP).
 * Tensor/pipeline/expert parallel and FSDP sharding are the next increments and
 * are marked TODO at the exact call sites. GEMMs use a naive correct kernel here;
 * swap in cuBLASLt/tensor cores for performance (P1). */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cai_cuda.h"

/* NCCL helpers (cai_nccl.cu) - opaque so this file needn't include nccl.h. */
extern "C" void *cai_nccl_init(const cai_gpu_parallel_t *par); /* returns handle or NULL */
extern "C" void cai_nccl_allreduce_dp(void *h, float *buf, size_t n, void *stream);
extern "C" void cai_nccl_allreduce_tp(void *h, float *buf, size_t n, void *stream);
extern "C" void cai_nccl_free(void *h);

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(e_));                                  \
            abort();                                                          \
        }                                                                     \
    } while (0)

#define TPB 256
static inline int grid(long n) { return (int)((n + TPB - 1) / TPB); }

/* ---- naive row-major GEMMs (correctness V1; replace with cuBLASLt).
 * acc=0 overwrites C, acc=1 adds into C. ------------------------------- */
__global__ void gemm_nn_k(const float *A, const float *B, float *C, int M, int K, int N, int acc) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)M * N) return;
    int i = idx / N, j = idx % N;
    float s = acc ? C[idx] : 0.f;
    for (int k = 0; k < K; k++) s += A[(long)i * K + k] * B[(long)k * N + j];
    C[idx] = s;
}
__global__ void gemm_nt_k(const float *A, const float *B, float *C, int M, int K, int N, int acc) {
    /* C[M,N] = A[M,K] * B[N,K]^T */
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)M * N) return;
    int i = idx / N, j = idx % N;
    float s = acc ? C[idx] : 0.f;
    for (int k = 0; k < K; k++) s += A[(long)i * K + k] * B[(long)j * K + k];
    C[idx] = s;
}
__global__ void gemm_tn_k(const float *A, const float *B, float *C, int M, int K, int N, int acc) {
    /* C[K,N] = A[M,K]^T * B[M,N] */
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)K * N) return;
    int k = idx / N, j = idx % N;
    float s = acc ? C[idx] : 0.f;
    for (int i = 0; i < M; i++) s += A[(long)i * K + k] * B[(long)i * N + j];
    C[idx] = s;
}
static void g_nn(const float *A, const float *B, float *C, int M, int K, int N, cudaStream_t s) {
    gemm_nn_k<<<grid((long)M * N), TPB, 0, s>>>(A, B, C, M, K, N, 0);
}
static void g_nt(const float *A, const float *B, float *C, int M, int K, int N, int acc, cudaStream_t s) {
    gemm_nt_k<<<grid((long)M * N), TPB, 0, s>>>(A, B, C, M, K, N, acc);
}
static void g_tn(const float *A, const float *B, float *C, int M, int K, int N, cudaStream_t s) {
    gemm_tn_k<<<grid((long)K * N), TPB, 0, s>>>(A, B, C, M, K, N, 0);
}
__global__ void zero_k(float *p, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.f;
}

/* ---- parameter layout (mirrors refmodel) ------------------------------ */
struct GP {
    float *E, *P, *gf;
    float **g1, **Wq, **Wk, **Wv, **Wo, **g2, **Wg, **Wu, **Wd;
};
static size_t gp_layout(const cai_gpu_spec_t *s, float *base, GP *p) {
    int H = s->hidden, Hq = s->n_heads * s->head_dim, F = s->ffn, L = s->layers;
    size_t o = 0;
#define T1(f, n) do { if (base) p->f = base + o; o += (size_t)(n); } while (0)
#define TL(f, l, n) do { if (base) p->f[l] = base + o; o += (size_t)(n); } while (0)
    T1(E, (size_t)s->vocab * H); T1(P, (size_t)s->seq * H); T1(gf, H);
    for (int l = 0; l < L; l++) {
        TL(g1, l, H); TL(Wq, l, (size_t)H * Hq); TL(Wk, l, (size_t)H * Hq);
        TL(Wv, l, (size_t)H * Hq); TL(Wo, l, (size_t)Hq * H); TL(g2, l, H);
        TL(Wg, l, (size_t)H * F); TL(Wu, l, (size_t)H * F); TL(Wd, l, (size_t)F * H);
    }
#undef T1
#undef TL
    return o;
}
static void gp_alloc_views(int L, GP *p) {
    p->g1 = (float **)malloc(sizeof(float *) * L); p->Wq = (float **)malloc(sizeof(float *) * L);
    p->Wk = (float **)malloc(sizeof(float *) * L); p->Wv = (float **)malloc(sizeof(float *) * L);
    p->Wo = (float **)malloc(sizeof(float *) * L); p->g2 = (float **)malloc(sizeof(float *) * L);
    p->Wg = (float **)malloc(sizeof(float *) * L); p->Wu = (float **)malloc(sizeof(float *) * L);
    p->Wd = (float **)malloc(sizeof(float *) * L);
}

struct cai_gpu_ctx {
    cai_gpu_spec_t s;
    cai_gpu_parallel_t par;
    int Hq;
    size_t N;
    float *params, *grads, *m, *v;
    GP P, G;
    long step;
    void *nccl;
    cudaStream_t stream;
    /* activations (B fixed at first step) */
    int B, BT;
    int *d_tok, *d_tgt; /* cached upload buffers for the host wrapper */
    float *h0, *hf, *rf, *logits, *sm, *lossp;
    float *x1, *Q, *K, *V, *probs, *ctx, *ao, *hmid, *x2, *gate, *up, *act, *hout, *r1, *r2;
    float *dstream, *dx, *dq, *dk, *dv, *dctx, *dgate, *dup, *dact, *dlogits, *dhf, *dmid;
};

static float *dmalloc(size_t n) { float *p; CK(cudaMalloc(&p, n * sizeof(float))); return p; }

extern "C" cai_gpu_ctx_t *cai_gpu_init(const cai_gpu_spec_t *spec,
                                       const cai_gpu_parallel_t *par) {
    cai_gpu_ctx_t *c = (cai_gpu_ctx_t *)calloc(1, sizeof(*c));
    c->s = *spec; c->par = *par; c->Hq = spec->n_heads * spec->head_dim;
    CK(cudaSetDevice(par->local_device));
    CK(cudaStreamCreate(&c->stream));
    c->N = gp_layout(spec, NULL, NULL);
    c->params = dmalloc(c->N); c->grads = dmalloc(c->N);
    c->m = dmalloc(c->N); c->v = dmalloc(c->N);
    CK(cudaMemset(c->m, 0, c->N * sizeof(float)));
    CK(cudaMemset(c->v, 0, c->N * sizeof(float)));
    gp_alloc_views(spec->layers, &c->P); gp_alloc_views(spec->layers, &c->G);
    gp_layout(spec, c->params, &c->P); gp_layout(spec, c->grads, &c->G);

    /* host init (xavier-ish), deterministic per global_rank-independent seed so
     * all DP replicas start identical, then copy to device. */
    float *h = (float *)malloc(c->N * sizeof(float));
    unsigned long st = 0x2545F4914F6CDD1Dul;
    for (size_t i = 0; i < c->N; i++) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        double u = ((st >> 11) + 1.0) / 9007199254740994.0;
        h[i] = (float)(0.02 * (u * 2.0 - 1.0));
    }
    CK(cudaMemcpy(c->params, h, c->N * sizeof(float), cudaMemcpyHostToDevice));
    free(h);

    c->nccl = (par->world_size > 1) ? cai_nccl_init(par) : NULL;
    return c;
}

static void alloc_acts(cai_gpu_ctx_t *c, int B) {
    const cai_gpu_spec_t *s = &c->s;
    int T = s->seq, H = s->hidden, Hq = c->Hq, F = s->ffn, L = s->layers, nh = s->n_heads;
    long BT = (long)B * T; c->B = B; c->BT = BT;
    c->h0 = dmalloc(BT * H); c->hf = dmalloc(BT * H); c->rf = dmalloc(BT);
    c->logits = dmalloc(BT * s->vocab); c->sm = dmalloc(BT * s->vocab); c->lossp = dmalloc(BT);
    c->x1 = dmalloc((long)L * BT * H); c->Q = dmalloc((long)L * BT * Hq);
    c->K = dmalloc((long)L * BT * Hq); c->V = dmalloc((long)L * BT * Hq);
    c->probs = dmalloc((long)L * B * nh * T * T); c->ctx = dmalloc((long)L * BT * Hq);
    c->ao = dmalloc((long)L * BT * H); c->hmid = dmalloc((long)L * BT * H);
    c->x2 = dmalloc((long)L * BT * H); c->gate = dmalloc((long)L * BT * F);
    c->up = dmalloc((long)L * BT * F); c->act = dmalloc((long)L * BT * F);
    c->hout = dmalloc((long)L * BT * H); c->r1 = dmalloc((long)L * BT);
    c->r2 = dmalloc((long)L * BT);
    c->dstream = dmalloc(BT * H); c->dx = dmalloc(BT * H); c->dq = dmalloc(BT * Hq);
    c->dk = dmalloc(BT * Hq); c->dv = dmalloc(BT * Hq); c->dctx = dmalloc(BT * Hq);
    c->dgate = dmalloc(BT * F); c->dup = dmalloc(BT * F); c->dact = dmalloc(BT * F);
    c->dlogits = dmalloc(BT * s->vocab); c->dhf = dmalloc(BT * H); c->dmid = dmalloc(BT * H);
}

extern "C" float cai_gpu_train_step(cai_gpu_ctx_t *c, int B, const int *d_tok,
                                    const int *d_tgt, float lr) {
    if (!c->h0) alloc_acts(c, B);
    const cai_gpu_spec_t *s = &c->s;
    int T = s->seq, H = s->hidden, Hq = c->Hq, F = s->ffn, Vv = s->vocab, L = s->layers,
        nh = s->n_heads, hd = s->head_dim;
    long BT = c->BT;
    cudaStream_t st = c->stream;
    void *vs = (void *)st;

    cai_k_embed_fwd(c->P.E, c->P.P, d_tok, c->h0, BT, T, H, vs);
    float *hs = c->h0;
    for (int l = 0; l < L; l++) {
        float *x1 = c->x1 + (long)l * BT * H, *r1 = c->r1 + (long)l * BT;
        cai_k_rmsnorm_fwd(hs, c->P.g1[l], x1, r1, BT, H, s->eps, vs);
        float *Q = c->Q + (long)l * BT * Hq, *K = c->K + (long)l * BT * Hq, *V = c->V + (long)l * BT * Hq;
        g_nn(x1, c->P.Wq[l], Q, BT, H, Hq, st);
        g_nn(x1, c->P.Wk[l], K, BT, H, Hq, st);
        g_nn(x1, c->P.Wv[l], V, BT, H, Hq, st);
        float *pr = c->probs + (long)l * B * nh * T * T, *cx = c->ctx + (long)l * BT * Hq;
        cai_k_attn_fwd(Q, K, V, pr, cx, B, T, nh, hd, vs);
        float *ao = c->ao + (long)l * BT * H;
        g_nn(cx, c->P.Wo[l], ao, BT, Hq, H, st);
        if (c->par.tp > 1) cai_nccl_allreduce_tp(c->nccl, ao, BT * H, vs); /* row-parallel out-proj */
        float *hm = c->hmid + (long)l * BT * H;
        cai_k_add(hs, ao, hm, BT * H, vs);
        float *x2 = c->x2 + (long)l * BT * H, *r2 = c->r2 + (long)l * BT;
        cai_k_rmsnorm_fwd(hm, c->P.g2[l], x2, r2, BT, H, s->eps, vs);
        float *ga = c->gate + (long)l * BT * F, *up = c->up + (long)l * BT * F, *ac = c->act + (long)l * BT * F;
        g_nn(x2, c->P.Wg[l], ga, BT, H, F, st);
        g_nn(x2, c->P.Wu[l], up, BT, H, F, st);
        cai_k_swiglu_fwd(ga, up, ac, BT * F, vs);
        float *ho = c->hout + (long)l * BT * H;
        g_nn(ac, c->P.Wd[l], ho, BT, F, H, st);
        if (c->par.tp > 1) cai_nccl_allreduce_tp(c->nccl, ho, BT * H, vs); /* row-parallel down-proj */
        cai_k_add(hm, ho, ho, BT * H, vs);
        hs = ho;
    }
    cai_k_rmsnorm_fwd(hs, c->P.gf, c->hf, c->rf, BT, H, s->eps, vs);
    g_nt(c->hf, c->P.E, c->logits, BT, H, Vv, 0, st); /* logits = hf E^T */
    cai_k_cross_entropy(c->logits, d_tgt, c->sm, c->lossp, BT, Vv, vs);

    /* ---- backward ---- */
    zero_k<<<grid((long)c->N), TPB, 0, st>>>(c->grads, c->N);
    cai_k_ce_grad(c->sm, d_tgt, c->dlogits, BT, Vv, vs);
    g_nn(c->dlogits, c->P.E, c->dhf, BT, Vv, H, st);   /* dhf = dlogits E */
    g_tn(c->dlogits, c->hf, c->G.E, BT, Vv, H, st);    /* dE += (acc handled: separate buffer) */
    const float *hs_last = c->hout + (long)(L - 1) * BT * H;
    cai_k_rmsnorm_bwd(hs_last, c->P.gf, c->rf, c->dhf, c->dstream, c->G.gf, BT, H, vs);

    for (int l = L - 1; l >= 0; l--) {
        const float *hs_in = (l == 0) ? c->h0 : c->hout + (long)(l - 1) * BT * H;
        const float *x1 = c->x1 + (long)l * BT * H, *r1 = c->r1 + (long)l * BT;
        const float *Q = c->Q + (long)l * BT * Hq, *K = c->K + (long)l * BT * Hq, *V = c->V + (long)l * BT * Hq;
        const float *pr = c->probs + (long)l * B * nh * T * T, *cx = c->ctx + (long)l * BT * Hq;
        const float *hm = c->hmid + (long)l * BT * H, *x2 = c->x2 + (long)l * BT * H;
        const float *ga = c->gate + (long)l * BT * F, *up = c->up + (long)l * BT * F, *ac = c->act + (long)l * BT * F;

        /* h_out = h_mid + mlp_out: residual carries dstream into d h_mid */
        CK(cudaMemcpyAsync(c->dmid, c->dstream, BT * H * sizeof(float), cudaMemcpyDeviceToDevice, st));
        /* mlp_out = act Wd */
        g_nt(c->dstream, c->P.Wd[l], c->dact, BT, H, F, 0, st); /* dact = dmlp Wd^T */
        g_tn(ac, c->dstream, c->G.Wd[l], BT, F, H, st);          /* dWd = act^T dmlp */
        cai_k_swiglu_bwd(ga, up, c->dact, c->dgate, c->dup, BT * F, vs);
        /* dx2 = dgate Wg^T + dup Wu^T */
        g_nt(c->dgate, c->P.Wg[l], c->dx, BT, H, F, 0, st); /* overwrite */
        g_nt(c->dup, c->P.Wu[l], c->dx, BT, H, F, 1, st);   /* accumulate */
        g_tn(x2, c->dgate, c->G.Wg[l], BT, H, F, st);
        g_tn(x2, c->dup, c->G.Wu[l], BT, H, F, st);
        cai_k_rmsnorm_bwd(hm, c->P.g2[l], c->r2 + (long)l * BT, c->dx, c->dmid, c->G.g2[l], BT, H, vs);

        /* h_mid = hs_in + attn_out: residual carries dmid into d hs_in */
        CK(cudaMemcpyAsync(c->dstream, c->dmid, BT * H * sizeof(float), cudaMemcpyDeviceToDevice, st));
        g_nt(c->dmid, c->P.Wo[l], c->dctx, BT, Hq, H, 0, st); /* dctx = dmid Wo^T */
        g_tn(cx, c->dmid, c->G.Wo[l], BT, Hq, H, st);         /* dWo */
        cai_k_attn_bwd(Q, K, V, pr, c->dctx, c->dq, c->dk, c->dv, B, T, nh, hd, vs);
        /* dx1 = dq Wq^T + dk Wk^T + dv Wv^T */
        g_nt(c->dq, c->P.Wq[l], c->dx, BT, H, Hq, 0, st);
        g_nt(c->dk, c->P.Wk[l], c->dx, BT, H, Hq, 1, st);
        g_nt(c->dv, c->P.Wv[l], c->dx, BT, H, Hq, 1, st);
        g_tn(x1, c->dq, c->G.Wq[l], BT, H, Hq, st);
        g_tn(x1, c->dk, c->G.Wk[l], BT, H, Hq, st);
        g_tn(x1, c->dv, c->G.Wv[l], BT, H, Hq, st);
        if (c->par.tp > 1) cai_nccl_allreduce_tp(c->nccl, c->dx, BT * H, vs); /* gather TP input grads */
        cai_k_rmsnorm_bwd(hs_in, c->P.g1[l], r1, c->dx, c->dstream, c->G.g1[l], BT, H, vs);
    }
    cai_k_embed_bwd(c->G.E, c->G.P, d_tok, c->dstream, BT, T, H, vs);

    /* data-parallel gradient all-reduce (M3 DDP) */
    if (c->par.dp > 1) cai_nccl_allreduce_dp(c->nccl, c->grads, c->N, vs);

    cai_k_adamw(c->params, c->grads, c->m, c->v, c->N, lr, 0.9f, 0.999f, 1e-8f, 0.01f,
                ++c->step, vs);

    /* mean loss to host */
    float *hl = (float *)malloc(BT * sizeof(float));
    CK(cudaMemcpyAsync(hl, c->lossp, BT * sizeof(float), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    double sum = 0;
    for (long i = 0; i < BT; i++) sum += hl[i];
    free(hl);
    return (float)(sum / BT);
}

extern "C" float cai_gpu_train_step_host(cai_gpu_ctx_t *c, int B, const int *h_tok,
                                         const int *h_tgt, float lr) {
    long BT = (long)B * c->s.seq;
    if (!c->d_tok) {
        CK(cudaMalloc(&c->d_tok, BT * sizeof(int)));
        CK(cudaMalloc(&c->d_tgt, BT * sizeof(int)));
    }
    CK(cudaMemcpy(c->d_tok, h_tok, BT * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
    return cai_gpu_train_step(c, B, c->d_tok, c->d_tgt, lr);
}

extern "C" void cai_gpu_free(cai_gpu_ctx_t *c) {
    if (!c) return;
    if (c->nccl) cai_nccl_free(c->nccl);
    cudaFree(c->params); cudaFree(c->grads); cudaFree(c->m); cudaFree(c->v);
    /* activations + views omitted for brevity (process exit frees device) */
    cudaStreamDestroy(c->stream);
    free(c);
}
