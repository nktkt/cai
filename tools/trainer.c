/* trainer - the runtime driver (one process == one GPU rank).
 *
 *   trainer <plan.bin> <topology.bin> [--steps N] [--ckpt PATH]
 *
 * Loads a compiled plan and replays it. With the analytic CPU backend this is a
 * faithful schedule *simulation*: it reports the per-step critical path, pipeline
 * bubble, MFU, and tokens/s the real cluster would see. No training math runs. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cai.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <plan.bin> <topology.bin> [--steps N] [--ckpt PATH]\n",
                argv[0]);
        return 2;
    }
    cai_init_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.plan_path = argv[1];
    desc.topology_path = argv[2];
    desc.max_steps = 10;
    desc.checkpoint_every = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc)
            desc.max_steps = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ckpt") && i + 1 < argc) {
            desc.checkpoint_path = argv[++i];
            desc.checkpoint_every = 5;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    cai_context_t *ctx = NULL;
    int rc = cai_init(&ctx, &desc);
    if (rc != CAI_OK) {
        fprintf(stderr, "init failed: %s\n", cai_status_str(rc));
        return 1;
    }
    if (desc.checkpoint_path) cai_load_checkpoint(ctx, desc.checkpoint_path);

    printf("step | step_time | bubble |   MFU  | tokens/s    | tok/s/gpu\n");
    printf("-----+-----------+--------+--------+-------------+----------\n");

    double sum_mfu = 0, sum_tps = 0;
    uint32_t n = 0;
    for (;;) {
        cai_batch_t batch;
        rc = cai_next_batch(ctx, &batch);
        if (rc != CAI_OK) break; /* EOF */
        rc = cai_train_step(ctx, &batch);
        if (rc != CAI_OK) {
            fprintf(stderr, "step %u failed: %s\n", batch.step, cai_status_str(rc));
            cai_save_checkpoint(ctx, "emergency");
            cai_finalize(ctx);
            return 1;
        }
        cai_step_stats_t st;
        cai_last_step_stats(ctx, &st);
        printf("%4u | %7.2f ms| %5.1f%% | %5.1f%% | %9.2e | %8.1f\n", batch.step,
               st.step_time_s * 1e3, st.bubble_ratio * 100.0, st.mfu * 100.0,
               st.tokens_per_s, st.tokens_per_s_per_gpu);
        sum_mfu += st.mfu;
        sum_tps += st.tokens_per_s;
        n++;
        if (cai_should_checkpoint(ctx)) cai_save_checkpoint(ctx, "periodic");
    }

    if (n) {
        cai_step_stats_t st;
        cai_last_step_stats(ctx, &st);
        printf("\nsummary: %u steps | avg MFU %.1f%% | avg %.3e tokens/s | "
               "arena %.2f GiB/gpu\n",
               n, sum_mfu / n * 100.0, sum_tps / n,
               st.arena_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    cai_finalize(ctx);
    return 0;
}
