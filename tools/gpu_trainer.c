/* gpu_trainer - drive the on-device executor (roadmap M2-device bring-up test).
 *
 * UNVERIFIED: built only with CUDA=1 and runnable only on a GPU. It trains the
 * same shift task as tools/reftrain.c (CPU fp64 oracle); on one GPU the loss
 * curve should track reftrain's, which is the bring-up acceptance test. Reads
 * RANK/WORLD_SIZE/LOCAL_RANK from the environment (set by tools/launch.sh). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cai_cuda.h"

static unsigned long rng;
static unsigned long nextr(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}
static void gen_batch(int B, int T, int V, int *tok, int *tgt) {
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            int i = b * T + t;
            tok[i] = (int)(nextr() % (unsigned long)V);
            tgt[i] = (t == 0) ? 0 : tok[i - 1];
        }
}

static int envi(const char *k, int dflt) {
    const char *v = getenv(k);
    return v ? atoi(v) : dflt;
}

int main(int argc, char **argv) {
    int steps = 400, B = 16;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) B = atoi(argv[++i]);
    }
    cai_gpu_spec_t s = {.vocab = 64, .seq = 32, .hidden = 64, .layers = 2,
                        .n_heads = 4, .head_dim = 16, .ffn = 128, .eps = 1e-6f};
    cai_gpu_parallel_t par;
    memset(&par, 0, sizeof(par));
    par.global_rank = envi("RANK", 0);
    par.world_size = envi("WORLD_SIZE", 1);
    par.local_device = envi("LOCAL_RANK", 0);
    par.tp = 1; par.pp = 1; par.dp = par.world_size; par.ep = 1; par.cp = 1;
    par.tp_rank = 0; par.pp_rank = 0; par.dp_rank = par.global_rank;

    cai_gpu_ctx_t *c = cai_gpu_init(&s, &par);
    if (!c) { fprintf(stderr, "gpu init failed\n"); return 1; }
    rng = 99u + (unsigned)par.global_rank;

    int BT = B * s.seq;
    int *tok = malloc(sizeof(int) * BT), *tgt = malloc(sizeof(int) * BT);
    if (par.global_rank == 0)
        printf("gpu_trainer: world=%d, task=prev-token, vocab=%d seq=%d\n",
               par.world_size, s.vocab, s.seq);
    for (int step = 0; step < steps; step++) {
        gen_batch(B, s.seq, s.vocab, tok, tgt);
        float loss = cai_gpu_train_step_host(c, B, tok, tgt, 0.01f);
        if (par.global_rank == 0 && (step % (steps / 20 ? steps / 20 : 1) == 0))
            printf("step %4d  loss %.4f\n", step, loss);
    }
    free(tok); free(tgt);
    cai_gpu_free(c);
    return 0;
}
