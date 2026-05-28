#include "refmodel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- matmul kernels (fp64, naive) ------------------------------------ *
 *  mm    : C[M,N] (+)= A[M,K] B[K,N]
 *  mm_nt : C[M,N] (+)= A[M,K] B[N,K]^T
 *  mm_tn : C[K,N] (+)= A[M,K]^T B[M,N]                                    */
static void mm(const double *A, const double *B, double *C, int M, int K, int N, int acc) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double s = acc ? C[i * N + j] : 0.0;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}
static void mm_nt(const double *A, const double *B, double *C, int M, int K, int N, int acc) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double s = acc ? C[i * N + j] : 0.0;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[j * K + k];
            C[i * N + j] = s;
        }
}
static void mm_tn(const double *A, const double *B, double *C, int M, int K, int N, int acc) {
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) {
            double s = acc ? C[k * N + j] : 0.0;
            for (int i = 0; i < M; i++) s += A[i * K + k] * B[i * N + j];
            C[k * N + j] = s;
        }
}

/* linear Y[M,No]=X[M,Ni]W[Ni,No]; backward into dX (acc) and dW (acc). */
static void lin_fwd(const double *X, const double *W, double *Y, int M, int Ni, int No) {
    mm(X, W, Y, M, Ni, No, 0);
}
static void lin_bwd(const double *X, const double *W, const double *dY, double *dX,
                    double *dW, int M, int Ni, int No, int dx_acc) {
    mm_nt(dY, W, dX, M, No, Ni, dx_acc); /* dX += dY W^T */
    mm_tn(X, dY, dW, M, Ni, No, 1);      /* dW += X^T dY */
}

/* ---- rmsnorm --------------------------------------------------------- */
static void rms_fwd(const double *x, const double *g, double *y, double *recip, int rows,
                    int H, double eps) {
    for (int r = 0; r < rows; r++) {
        const double *xr = x + (size_t)r * H;
        double ms = 0;
        for (int j = 0; j < H; j++) ms += xr[j] * xr[j];
        ms /= H;
        double rr = 1.0 / sqrt(ms + eps);
        recip[r] = rr;
        double *yr = y + (size_t)r * H;
        for (int j = 0; j < H; j++) yr[j] = xr[j] * rr * g[j];
    }
}
/* dx (+)= rmsnorm_bwd; dg += sum dy*xhat */
static void rms_bwd(const double *x, const double *g, const double *recip, const double *dy,
                    double *dx, double *dg, int rows, int H, int dx_acc) {
    for (int r = 0; r < rows; r++) {
        const double *xr = x + (size_t)r * H;
        const double *dyr = dy + (size_t)r * H;
        double rr = recip[r];
        double S = 0;
        for (int j = 0; j < H; j++) S += dyr[j] * g[j] * xr[j];
        double *dxr = dx + (size_t)r * H;
        for (int j = 0; j < H; j++) {
            double dxhat = dyr[j] * g[j];
            double v = rr * dxhat - rr * rr * rr * xr[j] * S / H;
            dxr[j] = dx_acc ? dxr[j] + v : v;
            dg[j] += dyr[j] * xr[j] * rr;
        }
    }
}

static double sigmoidd(double z) { return 1.0 / (1.0 + exp(-z)); }

/* ====================================================================== */

struct ref_model {
    ref_spec_t s;
    int Hq;
    size_t N;
    double *p, *g, *mom, *vel;
    ref_params_t P, G;
    long t;

    int Bcur, BT;
    /* activation tape (sized for Bcur) */
    double *h0, *hf, *rf, *logits, *sm;
    double *x1, *Qb, *Kb, *Vb, *probs, *ctx, *attn_out, *h_mid, *x2, *gate, *up, *act,
        *h_out, *r1, *r2;
    /* backward scratch */
    double *dstream, *dx, *dq, *dk, *dv, *dctx, *dgate, *dup, *dact, *dlogits, *dhf,
        *dmid;
};

/* assign pointers into base (or just count if base==NULL); returns count. */
static size_t layout(const ref_spec_t *s, double *base, ref_params_t *P) {
    int H = s->hidden, Hq = s->n_heads * s->head_dim, F = s->ffn, L = s->layers;
    size_t o = 0;
#define TAKE(field, n)                              \
    do {                                            \
        if (base) P->field = base + o;              \
        o += (size_t)(n);                           \
    } while (0)
#define TAKEL(field, l, n)                          \
    do {                                            \
        if (base) P->field[l] = base + o;           \
        o += (size_t)(n);                           \
    } while (0)
    TAKE(E, (size_t)s->vocab * H);
    TAKE(P, (size_t)s->seq * H);
    TAKE(gf, H);
    for (int l = 0; l < L; l++) {
        TAKEL(g1, l, H);
        TAKEL(Wq, l, (size_t)H * Hq);
        TAKEL(Wk, l, (size_t)H * Hq);
        TAKEL(Wv, l, (size_t)H * Hq);
        TAKEL(Wo, l, (size_t)Hq * H);
        TAKEL(g2, l, H);
        TAKEL(Wg, l, (size_t)H * F);
        TAKEL(Wu, l, (size_t)H * F);
        TAKEL(Wd, l, (size_t)F * H);
    }
#undef TAKE
#undef TAKEL
    return o;
}

static void alloc_views(const ref_spec_t *s, ref_params_t *P) {
    int L = s->layers;
    P->g1 = malloc(sizeof(double *) * L);
    P->Wq = malloc(sizeof(double *) * L);
    P->Wk = malloc(sizeof(double *) * L);
    P->Wv = malloc(sizeof(double *) * L);
    P->Wo = malloc(sizeof(double *) * L);
    P->g2 = malloc(sizeof(double *) * L);
    P->Wg = malloc(sizeof(double *) * L);
    P->Wu = malloc(sizeof(double *) * L);
    P->Wd = malloc(sizeof(double *) * L);
}
static void free_views(ref_params_t *P) {
    free(P->g1); free(P->Wq); free(P->Wk); free(P->Wv); free(P->Wo);
    free(P->g2); free(P->Wg); free(P->Wu); free(P->Wd);
}

static unsigned long xs_state;
static double urand(void) { /* xorshift64 -> (0,1) */
    xs_state ^= xs_state << 13;
    xs_state ^= xs_state >> 7;
    xs_state ^= xs_state << 17;
    return ((xs_state >> 11) + 1.0) / 9007199254740994.0;
}
static double nrand(void) { /* Box-Muller */
    double u1 = urand(), u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

ref_model_t *ref_model_create(const ref_spec_t *spec, unsigned long seed) {
    ref_model_t *m = calloc(1, sizeof(*m));
    m->s = *spec;
    m->Hq = spec->n_heads * spec->head_dim;
    m->N = layout(spec, NULL, NULL);
    m->p = calloc(m->N, sizeof(double));
    m->g = calloc(m->N, sizeof(double));
    m->mom = calloc(m->N, sizeof(double));
    m->vel = calloc(m->N, sizeof(double));
    alloc_views(spec, &m->P);
    alloc_views(spec, &m->G);
    layout(spec, m->p, &m->P);
    layout(spec, m->g, &m->G);

    xs_state = seed ? seed : 0x2545F4914F6CDD1Dul;
    int H = spec->hidden, Hq = m->Hq, F = spec->ffn;
    for (size_t i = 0; i < (size_t)spec->vocab * H; i++) m->P.E[i] = 0.02 * nrand();
    for (size_t i = 0; i < (size_t)spec->seq * H; i++) m->P.P[i] = 0.02 * nrand();
    for (int j = 0; j < H; j++) m->P.gf[j] = 1.0;
    for (int l = 0; l < spec->layers; l++) {
        for (int j = 0; j < H; j++) m->P.g1[l][j] = 1.0;
        for (int j = 0; j < H; j++) m->P.g2[l][j] = 1.0;
        double sq = 1.0 / sqrt((double)H), so = 1.0 / sqrt((double)Hq);
        double sf = 1.0 / sqrt((double)H), sd = 1.0 / sqrt((double)F);
        for (size_t i = 0; i < (size_t)H * Hq; i++) {
            m->P.Wq[l][i] = sq * nrand();
            m->P.Wk[l][i] = sq * nrand();
            m->P.Wv[l][i] = sq * nrand();
        }
        for (size_t i = 0; i < (size_t)Hq * H; i++) m->P.Wo[l][i] = so * nrand();
        for (size_t i = 0; i < (size_t)H * F; i++) {
            m->P.Wg[l][i] = sf * nrand();
            m->P.Wu[l][i] = sf * nrand();
        }
        for (size_t i = 0; i < (size_t)F * H; i++) m->P.Wd[l][i] = sd * nrand();
    }
    return m;
}

static void free_acts(ref_model_t *m) {
    free(m->h0); free(m->hf); free(m->rf); free(m->logits); free(m->sm);
    free(m->x1); free(m->Qb); free(m->Kb); free(m->Vb); free(m->probs);
    free(m->ctx); free(m->attn_out); free(m->h_mid); free(m->x2);
    free(m->gate); free(m->up); free(m->act); free(m->h_out);
    free(m->r1); free(m->r2);
    free(m->dstream); free(m->dx); free(m->dq); free(m->dk); free(m->dv);
    free(m->dctx); free(m->dgate); free(m->dup); free(m->dact);
    free(m->dlogits); free(m->dhf); free(m->dmid);
    memset(&m->h0, 0, 0); /* no-op, clarity */
}

static void ensure_acts(ref_model_t *m, int B) {
    if (m->Bcur == B && m->h0) return;
    if (m->h0) free_acts(m);
    const ref_spec_t *s = &m->s;
    int T = s->seq, H = s->hidden, Hq = m->Hq, F = s->ffn, L = s->layers, nh = s->n_heads;
    int BT = B * T;
    m->Bcur = B;
    m->BT = BT;
    m->h0 = calloc((size_t)BT * H, sizeof(double));
    m->hf = calloc((size_t)BT * H, sizeof(double));
    m->rf = calloc((size_t)BT, sizeof(double));
    m->logits = calloc((size_t)BT * s->vocab, sizeof(double));
    m->sm = calloc((size_t)BT * s->vocab, sizeof(double));
    m->x1 = calloc((size_t)L * BT * H, sizeof(double));
    m->Qb = calloc((size_t)L * BT * Hq, sizeof(double));
    m->Kb = calloc((size_t)L * BT * Hq, sizeof(double));
    m->Vb = calloc((size_t)L * BT * Hq, sizeof(double));
    m->probs = calloc((size_t)L * B * nh * T * T, sizeof(double));
    m->ctx = calloc((size_t)L * BT * Hq, sizeof(double));
    m->attn_out = calloc((size_t)L * BT * H, sizeof(double));
    m->h_mid = calloc((size_t)L * BT * H, sizeof(double));
    m->x2 = calloc((size_t)L * BT * H, sizeof(double));
    m->gate = calloc((size_t)L * BT * F, sizeof(double));
    m->up = calloc((size_t)L * BT * F, sizeof(double));
    m->act = calloc((size_t)L * BT * F, sizeof(double));
    m->h_out = calloc((size_t)L * BT * H, sizeof(double));
    m->r1 = calloc((size_t)L * BT, sizeof(double));
    m->r2 = calloc((size_t)L * BT, sizeof(double));
    m->dstream = calloc((size_t)BT * H, sizeof(double));
    m->dx = calloc((size_t)BT * H, sizeof(double));
    m->dq = calloc((size_t)BT * Hq, sizeof(double));
    m->dk = calloc((size_t)BT * Hq, sizeof(double));
    m->dv = calloc((size_t)BT * Hq, sizeof(double));
    m->dctx = calloc((size_t)BT * Hq, sizeof(double));
    m->dgate = calloc((size_t)BT * F, sizeof(double));
    m->dup = calloc((size_t)BT * F, sizeof(double));
    m->dact = calloc((size_t)BT * F, sizeof(double));
    m->dlogits = calloc((size_t)BT * s->vocab, sizeof(double));
    m->dhf = calloc((size_t)BT * H, sizeof(double));
    m->dmid = calloc((size_t)BT * H, sizeof(double));
}

/* causal multi-head attention forward for one layer */
static void attn_fwd(ref_model_t *m, int B, const double *Q, const double *K,
                     const double *V, double *probs, double *ctx) {
    const ref_spec_t *s = &m->s;
    int T = s->seq, nh = s->n_heads, hd = s->head_dim, Hq = m->Hq;
    double scale = 1.0 / sqrt((double)hd);
    memset(ctx, 0, sizeof(double) * (size_t)B * T * Hq);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < nh; h++) {
            for (int i = 0; i < T; i++) {
                const double *qi = Q + ((size_t)(b * T + i) * Hq) + h * hd;
                double *pr = probs + (((size_t)(b * nh + h) * T) + i) * T;
                double mx = -1e30;
                for (int j = 0; j <= i; j++) {
                    const double *kj = K + ((size_t)(b * T + j) * Hq) + h * hd;
                    double sc = 0;
                    for (int d = 0; d < hd; d++) sc += qi[d] * kj[d];
                    sc *= scale;
                    pr[j] = sc;
                    if (sc > mx) mx = sc;
                }
                double sum = 0;
                for (int j = 0; j <= i; j++) { pr[j] = exp(pr[j] - mx); sum += pr[j]; }
                for (int j = 0; j <= i; j++) pr[j] /= sum;
                for (int j = i + 1; j < T; j++) pr[j] = 0;
                double *ci = ctx + ((size_t)(b * T + i) * Hq) + h * hd;
                for (int j = 0; j <= i; j++) {
                    const double *vj = V + ((size_t)(b * T + j) * Hq) + h * hd;
                    double pij = pr[j];
                    for (int d = 0; d < hd; d++) ci[d] += pij * vj[d];
                }
            }
        }
}

/* attention backward: dctx -> dQ,dK,dV (all zeroed inside) */
static void attn_bwd(ref_model_t *m, int B, const double *Q, const double *K,
                     const double *V, const double *probs, const double *dctx,
                     double *dQ, double *dK, double *dV) {
    const ref_spec_t *s = &m->s;
    int T = s->seq, nh = s->n_heads, hd = s->head_dim, Hq = m->Hq;
    double scale = 1.0 / sqrt((double)hd);
    memset(dQ, 0, sizeof(double) * (size_t)B * T * Hq);
    memset(dK, 0, sizeof(double) * (size_t)B * T * Hq);
    memset(dV, 0, sizeof(double) * (size_t)B * T * Hq);
    double *dp = malloc(sizeof(double) * T);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < nh; h++)
            for (int i = 0; i < T; i++) {
                const double *pr = probs + (((size_t)(b * nh + h) * T) + i) * T;
                const double *dci = dctx + ((size_t)(b * T + i) * Hq) + h * hd;
                double sps = 0;
                for (int j = 0; j <= i; j++) {
                    const double *vj = V + ((size_t)(b * T + j) * Hq) + h * hd;
                    double d = 0;
                    for (int dd = 0; dd < hd; dd++) d += dci[dd] * vj[dd];
                    dp[j] = d;
                    sps += pr[j] * d;
                    double *dvj = dV + ((size_t)(b * T + j) * Hq) + h * hd;
                    for (int dd = 0; dd < hd; dd++) dvj[dd] += pr[j] * dci[dd];
                }
                const double *qi = Q + ((size_t)(b * T + i) * Hq) + h * hd;
                double *dqi = dQ + ((size_t)(b * T + i) * Hq) + h * hd;
                for (int j = 0; j <= i; j++) {
                    double dscore = pr[j] * (dp[j] - sps) * scale;
                    const double *kj = K + ((size_t)(b * T + j) * Hq) + h * hd;
                    double *dkj = dK + ((size_t)(b * T + j) * Hq) + h * hd;
                    for (int dd = 0; dd < hd; dd++) {
                        dqi[dd] += dscore * kj[dd];
                        dkj[dd] += dscore * qi[dd];
                    }
                }
            }
    free(dp);
}

double ref_forward(ref_model_t *m, int B, const int *tokens, const int *targets) {
    ensure_acts(m, B);
    const ref_spec_t *s = &m->s;
    int T = s->seq, H = s->hidden, Hq = m->Hq, F = s->ffn, V = s->vocab, L = s->layers;
    int BT = B * T;

    /* embedding + position */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            int idx = b * T + t;
            const double *e = m->P.E + (size_t)tokens[idx] * H;
            const double *pp = m->P.P + (size_t)t * H;
            double *o = m->h0 + (size_t)idx * H;
            for (int j = 0; j < H; j++) o[j] = e[j] + pp[j];
        }

    const double *hs = m->h0;
    for (int l = 0; l < L; l++) {
        double *x1 = m->x1 + (size_t)l * BT * H;
        double *r1 = m->r1 + (size_t)l * BT;
        rms_fwd(hs, m->P.g1[l], x1, r1, BT, H, s->eps);
        double *Q = m->Qb + (size_t)l * BT * Hq;
        double *K = m->Kb + (size_t)l * BT * Hq;
        double *Vv = m->Vb + (size_t)l * BT * Hq;
        lin_fwd(x1, m->P.Wq[l], Q, BT, H, Hq);
        lin_fwd(x1, m->P.Wk[l], K, BT, H, Hq);
        lin_fwd(x1, m->P.Wv[l], Vv, BT, H, Hq);
        double *pr = m->probs + (size_t)l * B * s->n_heads * T * T;
        double *cx = m->ctx + (size_t)l * BT * Hq;
        attn_fwd(m, B, Q, K, Vv, pr, cx);
        double *ao = m->attn_out + (size_t)l * BT * H;
        lin_fwd(cx, m->P.Wo[l], ao, BT, Hq, H);
        double *hm = m->h_mid + (size_t)l * BT * H;
        for (int i = 0; i < BT * H; i++) hm[i] = hs[i] + ao[i];

        double *x2 = m->x2 + (size_t)l * BT * H;
        double *r2 = m->r2 + (size_t)l * BT;
        rms_fwd(hm, m->P.g2[l], x2, r2, BT, H, s->eps);
        double *ga = m->gate + (size_t)l * BT * F;
        double *up = m->up + (size_t)l * BT * F;
        double *ac = m->act + (size_t)l * BT * F;
        lin_fwd(x2, m->P.Wg[l], ga, BT, H, F);
        lin_fwd(x2, m->P.Wu[l], up, BT, H, F);
        for (int i = 0; i < BT * F; i++) ac[i] = (ga[i] * sigmoidd(ga[i])) * up[i];
        double *ho = m->h_out + (size_t)l * BT * H;
        lin_fwd(ac, m->P.Wd[l], ho, BT, F, H); /* ho = act@Wd */
        for (int i = 0; i < BT * H; i++) ho[i] += hm[i];
        hs = ho;
    }

    rms_fwd(hs, m->P.gf, m->hf, m->rf, BT, H, s->eps);
    mm_nt(m->hf, m->P.E, m->logits, BT, H, V, 0); /* logits = hf E^T */

    double loss = 0;
    for (int i = 0; i < BT; i++) {
        const double *lo = m->logits + (size_t)i * V;
        double *sm = m->sm + (size_t)i * V;
        double mx = -1e30;
        for (int v = 0; v < V; v++) if (lo[v] > mx) mx = lo[v];
        double sum = 0;
        for (int v = 0; v < V; v++) { sm[v] = exp(lo[v] - mx); sum += sm[v]; }
        for (int v = 0; v < V; v++) sm[v] /= sum;
        loss += -log(sm[targets[i]] + 1e-30);
    }
    return loss / BT;
}

void ref_backward(ref_model_t *m, int B, const int *tokens, const int *targets) {
    const ref_spec_t *s = &m->s;
    int T = s->seq, H = s->hidden, Hq = m->Hq, F = s->ffn, V = s->vocab, L = s->layers;
    int BT = B * T;
    memset(m->g, 0, sizeof(double) * m->N);

    /* softmax cross-entropy */
    for (int i = 0; i < BT; i++) {
        const double *sm = m->sm + (size_t)i * V;
        double *dl = m->dlogits + (size_t)i * V;
        for (int v = 0; v < V; v++) dl[v] = sm[v] / BT;
        dl[targets[i]] -= 1.0 / BT;
    }
    /* logits = hf E^T */
    mm(m->dlogits, m->P.E, m->dhf, BT, V, H, 0);       /* dhf = dlogits E */
    mm_tn(m->dlogits, m->hf, m->G.E, BT, V, H, 1);     /* dE += dlogits^T hf */

    /* final rmsnorm; input was h_out[L-1] */
    const double *hs_last = m->h_out + (size_t)(L - 1) * BT * H;
    rms_bwd(hs_last, m->P.gf, m->rf, m->dhf, m->dstream, m->G.gf, BT, H, 0);

    for (int l = L - 1; l >= 0; l--) {
        const double *hs_in = (l == 0) ? m->h0 : m->h_out + (size_t)(l - 1) * BT * H;
        const double *x1 = m->x1 + (size_t)l * BT * H;
        const double *r1 = m->r1 + (size_t)l * BT;
        const double *Q = m->Qb + (size_t)l * BT * Hq;
        const double *K = m->Kb + (size_t)l * BT * Hq;
        const double *Vv = m->Vb + (size_t)l * BT * Hq;
        const double *pr = m->probs + (size_t)l * B * s->n_heads * T * T;
        const double *cx = m->ctx + (size_t)l * BT * Hq;
        const double *hm = m->h_mid + (size_t)l * BT * H;
        const double *x2 = m->x2 + (size_t)l * BT * H;
        const double *ga = m->gate + (size_t)l * BT * F;
        const double *up = m->up + (size_t)l * BT * F;
        const double *ac = m->act + (size_t)l * BT * F;

        /* h_out = h_mid + mlp_out ; dstream is dL/d h_out */
        for (int i = 0; i < BT * H; i++) m->dmid[i] = m->dstream[i]; /* via residual */
        /* mlp: mlp_out = act@Wd ; dact = dmlp Wd^T ; dWd += act^T dmlp */
        memset(m->dact, 0, sizeof(double) * (size_t)BT * F);
        lin_bwd(ac, m->P.Wd[l], m->dstream, m->dact, m->G.Wd[l], BT, F, H, 0);
        /* act = silu(gate)*up */
        for (int i = 0; i < BT * F; i++) {
            double sg = sigmoidd(ga[i]);
            double silu = ga[i] * sg;
            double dsilu = sg * (1.0 + ga[i] * (1.0 - sg));
            m->dgate[i] = m->dact[i] * up[i] * dsilu;
            m->dup[i] = m->dact[i] * silu;
        }
        /* gate=x2 Wg ; up=x2 Wu ; accumulate dx2 then rms2 */
        memset(m->dx, 0, sizeof(double) * (size_t)BT * H);
        lin_bwd(x2, m->P.Wg[l], m->dgate, m->dx, m->G.Wg[l], BT, H, F, 1);
        lin_bwd(x2, m->P.Wu[l], m->dup, m->dx, m->G.Wu[l], BT, H, F, 1);
        /* rms2: input h_mid ; dmid += contribution */
        rms_bwd(hm, m->P.g2[l], m->r2 + (size_t)l * BT, m->dx, m->dmid, m->G.g2[l], BT, H, 1);

        /* h_mid = hs_in + attn_out ; dstream(next) starts from dmid (residual) */
        for (int i = 0; i < BT * H; i++) m->dstream[i] = m->dmid[i];
        /* attn_out = ctx Wo ; dctx = dmid Wo^T ; dWo += ctx^T dmid */
        memset(m->dctx, 0, sizeof(double) * (size_t)BT * Hq);
        lin_bwd(cx, m->P.Wo[l], m->dmid, m->dctx, m->G.Wo[l], BT, Hq, H, 0);
        /* attention */
        attn_bwd(m, B, Q, K, Vv, pr, m->dctx, m->dq, m->dk, m->dv);
        /* Q,K,V = x1 W{q,k,v} ; accumulate dx1 then rms1 */
        memset(m->dx, 0, sizeof(double) * (size_t)BT * H);
        lin_bwd(x1, m->P.Wq[l], m->dq, m->dx, m->G.Wq[l], BT, H, Hq, 1);
        lin_bwd(x1, m->P.Wk[l], m->dk, m->dx, m->G.Wk[l], BT, H, Hq, 1);
        lin_bwd(x1, m->P.Wv[l], m->dv, m->dx, m->G.Wv[l], BT, H, Hq, 1);
        rms_bwd(hs_in, m->P.g1[l], r1, m->dx, m->dstream, m->G.g1[l], BT, H, 1);
    }

    /* embedding + position grads */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            int idx = b * T + t;
            const double *ds = m->dstream + (size_t)idx * H;
            double *de = m->G.E + (size_t)tokens[idx] * H;
            double *dp = m->G.P + (size_t)t * H;
            for (int j = 0; j < H; j++) { de[j] += ds[j]; dp[j] += ds[j]; }
        }
}

void ref_adamw(ref_model_t *m, double lr, double b1, double b2, double eps, double wd) {
    m->t++;
    double bc1 = 1.0 - pow(b1, (double)m->t);
    double bc2 = 1.0 - pow(b2, (double)m->t);
    for (size_t i = 0; i < m->N; i++) {
        double g = m->g[i];
        m->mom[i] = b1 * m->mom[i] + (1 - b1) * g;
        m->vel[i] = b2 * m->vel[i] + (1 - b2) * g * g;
        double mh = m->mom[i] / bc1, vh = m->vel[i] / bc2;
        m->p[i] -= lr * (mh / (sqrt(vh) + eps) + wd * m->p[i]);
    }
}

double ref_grad_check(ref_model_t *m, int B, const int *tokens, const int *targets,
                      int n_probe, double h) {
    ref_forward(m, B, tokens, targets);
    ref_backward(m, B, tokens, targets);
    double *gsave = malloc(sizeof(double) * m->N);
    memcpy(gsave, m->g, sizeof(double) * m->N);

    /* L2 relative error over the probed params: robust to near-zero-gradient
     * params whose pointwise relative error is dominated by round-off. */
    double sd2 = 0, sa2 = 0;
    int total = (n_probe <= 0 || n_probe > (int)m->N) ? (int)m->N : n_probe;
    for (int t = 0; t < total; t++) {
        size_t i = (n_probe <= 0) ? (size_t)t
                                  : (size_t)(urand() * (double)m->N) % m->N;
        double save = m->p[i];
        m->p[i] = save + h;
        double lp = ref_forward(m, B, tokens, targets);
        m->p[i] = save - h;
        double lm = ref_forward(m, B, tokens, targets);
        m->p[i] = save;
        double num = (lp - lm) / (2 * h);
        double ana = gsave[i];
        sd2 += (num - ana) * (num - ana);
        sa2 += ana * ana;
    }
    free(gsave);
    return sqrt(sd2 / (sa2 + 1e-30));
}

double ref_accuracy(ref_model_t *m, int B, const int *tokens, const int *targets) {
    ref_forward(m, B, tokens, targets);
    int V = m->s.vocab, BT = B * m->s.seq, correct = 0;
    for (int i = 0; i < BT; i++) {
        const double *lo = m->logits + (size_t)i * V;
        int arg = 0;
        for (int v = 1; v < V; v++) if (lo[v] > lo[arg]) arg = v;
        if (arg == targets[i]) correct++;
    }
    return (double)correct / BT;
}

void ref_model_free(ref_model_t *m) {
    if (!m) return;
    if (m->h0) free_acts(m);
    free(m->p); free(m->g); free(m->mom); free(m->vel);
    free_views(&m->P);
    free_views(&m->G);
    free(m);
}

size_t ref_param_count(const ref_model_t *m) { return m->N; }
