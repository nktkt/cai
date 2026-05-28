/* cai_kernels.cu - handwritten GPU kernels (roadmap "kernels" track).
 *
 * UNVERIFIED: requires nvcc + CUDA to compile and an NVIDIA GPU to run; not
 * compiled or executed in the authoring environment. Math mirrors the verified
 * fp64 reference (src/refmodel.c) so a single-GPU run can be checked against it.
 *
 * V1 favors correctness/clarity over peak performance (one thread per row/element,
 * naive reductions). Fusion + flash-attention + tensor-core tiling come later. */
#include <cuda_runtime.h>
#include <math.h>

#include "cai_cuda.h"

#define TPB 256
static inline int grid(int n, int b) { return (n + b - 1) / b; }
#define ST(s) ((cudaStream_t)(s))

/* ---- embedding (tied) + learned absolute position ---------------------- */
__global__ void embed_fwd_k(const float *E, const float *P, const int *tok,
                            float *out, int BT, int T, int H) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)BT * H) return;
    int bt = idx / H, j = idx % H, t = bt % T;
    out[idx] = E[(long)tok[bt] * H + j] + P[(long)t * H + j];
}
__global__ void embed_bwd_k(float *dE, float *dP, const int *tok, const float *dout,
                            int BT, int T, int H) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)BT * H) return;
    int bt = idx / H, j = idx % H, t = bt % T;
    float g = dout[idx];
    atomicAdd(&dE[(long)tok[bt] * H + j], g);
    atomicAdd(&dP[(long)t * H + j], g);
}

/* ---- rmsnorm (one block per row) -------------------------------------- */
__global__ void rmsnorm_fwd_k(const float *x, const float *g, float *y,
                              float *recip, int H, float eps) {
    int r = blockIdx.x;
    const float *xr = x + (long)r * H;
    __shared__ float red[TPB];
    float s = 0;
    for (int j = threadIdx.x; j < H; j += blockDim.x) s += xr[j] * xr[j];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    float rr = rsqrtf(red[0] / H + eps);
    if (threadIdx.x == 0) recip[r] = rr;
    float *yr = y + (long)r * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) yr[j] = xr[j] * rr * g[j];
}
__global__ void rmsnorm_bwd_k(const float *x, const float *g, const float *recip,
                              const float *dy, float *dx, float *dg, int H) {
    int r = blockIdx.x;
    const float *xr = x + (long)r * H;
    const float *dyr = dy + (long)r * H;
    float rr = recip[r];
    __shared__ float red[TPB];
    float s = 0;
    for (int j = threadIdx.x; j < H; j += blockDim.x) s += dyr[j] * g[j] * xr[j];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    float S = red[0];
    float *dxr = dx + (long)r * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        float dxhat = dyr[j] * g[j];
        dxr[j] = rr * dxhat - rr * rr * rr * xr[j] * S / H;
        atomicAdd(&dg[j], dyr[j] * xr[j] * rr);
    }
}

/* ---- causal MHA (one thread per (b,head,query)) ----------------------- */
__global__ void attn_fwd_k(const float *Q, const float *K, const float *V,
                           float *probs, float *ctx, int B, int T, int nh, int hd) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)B * nh * T) return;
    int i = idx % T, h = (idx / T) % nh, b = idx / ((long)T * nh);
    int Hq = nh * hd;
    float scale = rsqrtf((float)hd);
    const float *qi = Q + ((long)(b * T + i) * Hq) + h * hd;
    float *pr = probs + (((long)(b * nh + h) * T) + i) * T;
    float mx = -1e30f;
    for (int j = 0; j <= i; j++) {
        const float *kj = K + ((long)(b * T + j) * Hq) + h * hd;
        float sc = 0;
        for (int d = 0; d < hd; d++) sc += qi[d] * kj[d];
        sc *= scale;
        pr[j] = sc;
        if (sc > mx) mx = sc;
    }
    float sum = 0;
    for (int j = 0; j <= i; j++) { pr[j] = __expf(pr[j] - mx); sum += pr[j]; }
    for (int j = 0; j <= i; j++) pr[j] /= sum;
    for (int j = i + 1; j < T; j++) pr[j] = 0;
    float *ci = ctx + ((long)(b * T + i) * Hq) + h * hd;
    for (int d = 0; d < hd; d++) ci[d] = 0;
    for (int j = 0; j <= i; j++) {
        const float *vj = V + ((long)(b * T + j) * Hq) + h * hd;
        float p = pr[j];
        for (int d = 0; d < hd; d++) ci[d] += p * vj[d];
    }
}
__global__ void attn_bwd_k(const float *Q, const float *K, const float *V,
                           const float *probs, const float *dctx, float *dQ,
                           float *dK, float *dV, int B, int T, int nh, int hd) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)B * nh * T) return;
    int i = idx % T, h = (idx / T) % nh, b = idx / ((long)T * nh);
    int Hq = nh * hd;
    float scale = rsqrtf((float)hd);
    const float *pr = probs + (((long)(b * nh + h) * T) + i) * T;
    const float *dci = dctx + ((long)(b * T + i) * Hq) + h * hd;
    const float *qi = Q + ((long)(b * T + i) * Hq) + h * hd;
    float *dqi = dQ + ((long)(b * T + i) * Hq) + h * hd;
    float sps = 0;
    for (int j = 0; j <= i; j++) {
        const float *vj = V + ((long)(b * T + j) * Hq) + h * hd;
        float dp = 0;
        for (int d = 0; d < hd; d++) dp += dci[d] * vj[d];
        sps += pr[j] * dp;
        float *dvj = dV + ((long)(b * T + j) * Hq) + h * hd;
        for (int d = 0; d < hd; d++) atomicAdd(&dvj[d], pr[j] * dci[d]);
    }
    for (int j = 0; j <= i; j++) {
        const float *vj = V + ((long)(b * T + j) * Hq) + h * hd;
        float dp = 0;
        for (int d = 0; d < hd; d++) dp += dci[d] * vj[d];
        float dscore = pr[j] * (dp - sps) * scale;
        const float *kj = K + ((long)(b * T + j) * Hq) + h * hd;
        float *dkj = dK + ((long)(b * T + j) * Hq) + h * hd;
        for (int d = 0; d < hd; d++) {
            dqi[d] += dscore * kj[d];
            atomicAdd(&dkj[d], dscore * qi[d]);
        }
    }
}

/* ---- SwiGLU / residual / loss / optimizer (elementwise) --------------- */
__global__ void swiglu_fwd_k(const float *gate, const float *up, float *act, int n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    act[i] = (g / (1.0f + __expf(-g))) * up[i];
}
__global__ void swiglu_bwd_k(const float *gate, const float *up, const float *dact,
                             float *dgate, float *dup, int n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z = gate[i], sg = 1.0f / (1.0f + __expf(-z)), silu = z * sg;
    float dsilu = sg * (1.0f + z * (1.0f - sg));
    dgate[i] = dact[i] * up[i] * dsilu;
    dup[i] = dact[i] * silu;
}
__global__ void add_k(const float *a, const float *b, float *o, int n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}
__global__ void ce_k(const float *logits, const int *tgt, float *sm, float *lossp,
                     int BT, int V) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= BT) return;
    const float *lo = logits + (long)r * V;
    float *s = sm + (long)r * V, mx = -1e30f, sum = 0;
    for (int v = 0; v < V; v++) if (lo[v] > mx) mx = lo[v];
    for (int v = 0; v < V; v++) { s[v] = __expf(lo[v] - mx); sum += s[v]; }
    for (int v = 0; v < V; v++) s[v] /= sum;
    lossp[r] = -__logf(s[tgt[r]] + 1e-30f);
}
__global__ void ce_grad_k(const float *sm, const int *tgt, float *dlogits, int BT,
                          int V) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)BT * V) return;
    int r = idx / V, v = idx % V;
    float g = sm[idx];
    if (v == tgt[r]) g -= 1.0f;
    dlogits[idx] = g / BT;
}
__global__ void adamw_k(float *p, const float *g, float *m, float *v, int n,
                        float lr, float b1, float b2, float eps, float wd,
                        float bc1, float bc2) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gg = g[i];
    float mm = b1 * m[i] + (1 - b1) * gg, vv = b2 * v[i] + (1 - b2) * gg * gg;
    m[i] = mm; v[i] = vv;
    p[i] -= lr * ((mm / bc1) / (sqrtf(vv / bc2) + eps) + wd * p[i]);
}

/* ---- extern "C" launchers -------------------------------------------- */
extern "C" {

void cai_k_embed_fwd(const float *E, const float *P, const int *tok, float *out,
                     int BT, int T, int H, void *s) {
    embed_fwd_k<<<grid((long)BT * H, TPB), TPB, 0, ST(s)>>>(E, P, tok, out, BT, T, H);
}
void cai_k_embed_bwd(float *dE, float *dP, const int *tok, const float *dout, int BT,
                     int T, int H, void *s) {
    embed_bwd_k<<<grid((long)BT * H, TPB), TPB, 0, ST(s)>>>(dE, dP, tok, dout, BT, T, H);
}
void cai_k_rmsnorm_fwd(const float *x, const float *g, float *y, float *recip,
                       int rows, int H, float eps, void *s) {
    rmsnorm_fwd_k<<<rows, TPB, 0, ST(s)>>>(x, g, y, recip, H, eps);
}
void cai_k_rmsnorm_bwd(const float *x, const float *g, const float *recip,
                       const float *dy, float *dx, float *dg, int rows, int H,
                       void *s) {
    rmsnorm_bwd_k<<<rows, TPB, 0, ST(s)>>>(x, g, recip, dy, dx, dg, H);
}
void cai_k_rope(float *q, float *k, int B, int T, int nh, int hd, void *s) {
    (void)q; (void)k; (void)B; (void)T; (void)nh; (void)hd; (void)s;
    /* the reference uses learned absolute position embeddings, so RoPE is a
     * no-op here; provided for models that opt into it. */
}
void cai_k_attn_fwd(const float *Q, const float *K, const float *V, float *probs,
                    float *ctx, int B, int T, int nh, int hd, void *s) {
    attn_fwd_k<<<grid((long)B * nh * T, TPB), TPB, 0, ST(s)>>>(Q, K, V, probs, ctx, B,
                                                              T, nh, hd);
}
void cai_k_attn_bwd(const float *Q, const float *K, const float *V,
                    const float *probs, const float *dctx, float *dQ, float *dK,
                    float *dV, int B, int T, int nh, int hd, void *s) {
    attn_bwd_k<<<grid((long)B * nh * T, TPB), TPB, 0, ST(s)>>>(Q, K, V, probs, dctx,
                                                              dQ, dK, dV, B, T, nh, hd);
}
void cai_k_swiglu_fwd(const float *gate, const float *up, float *act, int n, void *s) {
    swiglu_fwd_k<<<grid(n, TPB), TPB, 0, ST(s)>>>(gate, up, act, n);
}
void cai_k_swiglu_bwd(const float *gate, const float *up, const float *dact,
                      float *dgate, float *dup, int n, void *s) {
    swiglu_bwd_k<<<grid(n, TPB), TPB, 0, ST(s)>>>(gate, up, dact, dgate, dup, n);
}
void cai_k_add(const float *a, const float *b, float *o, int n, void *s) {
    add_k<<<grid(n, TPB), TPB, 0, ST(s)>>>(a, b, o, n);
}
void cai_k_cross_entropy(const float *logits, const int *tgt, float *sm,
                         float *lossp, int BT, int V, void *s) {
    ce_k<<<grid(BT, TPB), TPB, 0, ST(s)>>>(logits, tgt, sm, lossp, BT, V);
}
void cai_k_ce_grad(const float *sm, const int *tgt, float *dlogits, int BT, int V,
                   void *s) {
    ce_grad_k<<<grid((long)BT * V, TPB), TPB, 0, ST(s)>>>(sm, tgt, dlogits, BT, V);
}
void cai_k_adamw(float *p, const float *g, float *m, float *v, int n, float lr,
                 float b1, float b2, float eps, float wd, long t, void *s) {
    float bc1 = 1.0f - powf(b1, (float)t), bc2 = 1.0f - powf(b2, (float)t);
    adamw_k<<<grid(n, TPB), TPB, 0, ST(s)>>>(p, g, m, v, n, lr, b1, b2, eps, wd, bc1,
                                             bc2);
}

} /* extern "C" */
