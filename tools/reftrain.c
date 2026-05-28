/* reftrain - roadmap M2 on CPU: actually train the reference model.
 *
 *   reftrain [--steps N] [--lr X] [--batch B] [--seed S]
 *
 * Task: predict the *previous* token (a shift), which forces the model to attend
 * one position back — so this exercises attention, the MLP, and the optimizer, not
 * just the embedding. Fresh random batches each step, evaluated on a held-out
 * batch, so falling loss == real generalization, not memorization. No GPU. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "refmodel.h"

static unsigned long rng;
static unsigned long nextr(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

/* shift task: target[t] = tokens[t-1] (target[0] = 0) */
static void gen_batch(int B, int T, int V, int *tok, int *tgt) {
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            int idx = b * T + t;
            tok[idx] = (int)(nextr() % (unsigned long)V);
            tgt[idx] = (t == 0) ? 0 : tok[idx - 1];
        }
}

int main(int argc, char **argv) {
    int steps = 400, B = 16;
    double lr = 0.01;
    unsigned long seed = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoul(argv[++i], NULL, 10);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    ref_spec_t s = {.vocab = 64, .seq = 32, .hidden = 64, .layers = 2,
                    .n_heads = 4, .head_dim = 16, .ffn = 128, .eps = 1e-6};
    ref_model_t *m = ref_model_create(&s, seed);
    rng = seed ? seed * 2654435761ul + 1 : 99;

    int BT = B * s.seq;
    int *tok = malloc(sizeof(int) * BT), *tgt = malloc(sizeof(int) * BT);
    int *etok = malloc(sizeof(int) * BT), *etgt = malloc(sizeof(int) * BT);
    gen_batch(B, s.seq, s.vocab, etok, etgt); /* held-out eval batch */

    printf("reference model: %zu params | task=prev-token | vocab=%d seq=%d\n",
           ref_param_count(m), s.vocab, s.seq);
    printf("chance loss ~ ln(vocab) = %.3f\n\n", 4.158883);
    printf(" step |  train loss | eval loss | eval acc\n");
    printf("------+-------------+-----------+---------\n");

    double eval0 = ref_forward(m, B, etok, etgt);
    int every = steps / 20 ? steps / 20 : 1;
    double tl = eval0;
    for (int step = 0; step < steps; step++) {
        gen_batch(B, s.seq, s.vocab, tok, tgt);
        tl = ref_forward(m, B, tok, tgt);
        ref_backward(m, B, tok, tgt);
        ref_adamw(m, lr, 0.9, 0.999, 1e-8, 0.01);
        if (step % every == 0 || step == steps - 1) {
            double el = ref_forward(m, B, etok, etgt);
            double ea = ref_accuracy(m, B, etok, etgt);
            printf("%5d | %11.4f | %9.4f | %6.1f%%\n", step, tl, el, ea * 100.0);
        }
    }
    double eval1 = ref_forward(m, B, etok, etgt);
    double acc1 = ref_accuracy(m, B, etok, etgt);
    printf("\nresult: eval loss %.4f -> %.4f  (%.1fx lower), eval acc %.1f%%\n",
           eval0, eval1, eval0 / (eval1 + 1e-9), acc1 * 100.0);

    free(tok); free(tgt); free(etok); free(etgt);
    ref_model_free(m);
    return (eval1 < eval0 * 0.5) ? 0 : 1; /* nonzero if it failed to learn */
}
