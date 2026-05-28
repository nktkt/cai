/* refmodel.h - the correctness oracle (roadmap M1).
 *
 * A small, dense, fp32 transformer with REAL forward / backward / AdamW, computed
 * on the CPU with no parallelism. This is not the at-scale executor; it is the
 * golden reference the at-scale path (CUDA backend) must match, and it is what
 * lets cai demonstrate actual learning with no GPU. Backward is verified against
 * finite differences (see ref_grad_check). */
#ifndef CAI_REFMODEL_H
#define CAI_REFMODEL_H

#include <stddef.h>

typedef struct {
    int vocab;
    int seq;      /* max positions (learned absolute pos embedding) */
    int hidden;
    int layers;
    int n_heads;  /* MHA: kv heads == heads */
    int head_dim;
    int ffn;
    float eps;    /* rmsnorm epsilon */
} ref_spec_t;

/* Named views into the flat parameter / gradient vectors (same layout for both).
 * Computed in fp64: this is the precision oracle, deliberately more accurate than
 * the bf16/fp32 on-device path it will validate. */
typedef struct {
    double *E;  /* [vocab, hidden]  (tied: also the output projection) */
    double *P;  /* [seq, hidden] */
    double *gf; /* [hidden]  final rmsnorm gain */
    /* per-layer arrays of pointers */
    double **g1; /* [hidden] */
    double **Wq; /* [hidden, Hq] */
    double **Wk; /* [hidden, Hq] */
    double **Wv; /* [hidden, Hq] */
    double **Wo; /* [Hq, hidden] */
    double **g2; /* [hidden] */
    double **Wg; /* [hidden, ffn] */
    double **Wu; /* [hidden, ffn] */
    double **Wd; /* [ffn, hidden] */
} ref_params_t;

typedef struct ref_model ref_model_t;

ref_model_t *ref_model_create(const ref_spec_t *spec, unsigned long seed);
void ref_model_free(ref_model_t *m);
size_t ref_param_count(const ref_model_t *m);

/* Forward over tokens[B*seq] (0-based ids), targets[B*seq]; returns mean loss and
 * caches activations for the matching backward. */
double ref_forward(ref_model_t *m, int B, const int *tokens, const int *targets);

/* Backward of the last forward; accumulates into the gradient vector (zeroed). */
void ref_backward(ref_model_t *m, int B, const int *tokens, const int *targets);

/* One AdamW step using the current gradients. */
void ref_adamw(ref_model_t *m, double lr, double b1, double b2, double eps, double wd);

/* Top-1 next-token accuracy of the last/just-run forward over B*seq positions. */
double ref_accuracy(ref_model_t *m, int B, const int *tokens, const int *targets);

/* Finite-difference check of backward; returns max relative error over a sample
 * of `n_probe` parameters. Pass n_probe<=0 to check all. */
double ref_grad_check(ref_model_t *m, int B, const int *tokens, const int *targets,
                      int n_probe, double h);

#endif /* CAI_REFMODEL_H */
