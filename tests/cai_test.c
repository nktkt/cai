/* cai_test - unit tests for the GPU-free core: topology mapping, arena layout,
 * pipeline schedule, plan serialization, decomposition, and hash guards. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend.h"
#include "cai.h"
#include "cai_common.h"
#include "cai_model.h"
#include "refmodel.h"

static int g_fail;
#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            printf("  FAIL: ");                                \
            printf(__VA_ARGS__);                               \
            printf("  (%s:%d)\n", __FILE__, __LINE__);         \
            g_fail++;                                          \
        }                                                      \
    } while (0)

static void test_topology(void) {
    printf("[topology]\n");
    cai_topology_t t;
    cai_topology_default(&t);
    t.num_racks = 3056;
    char err[128];
    CHECK(cai_topology_finalize(&t, err, sizeof(err)) == CAI_OK, "finalize: %s", err);
    CHECK(t.world_size == 220032, "world=%u", t.world_size);

    uint32_t probes[] = {0, 1, 71, 72, 73, 1000, 220031};
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        uint32_t r = probes[i];
        cai_phys_id_t p = cai_phys_of_rank(&t, r);
        uint32_t back = cai_rank_of(&t, p.rack_id, p.tray_id, p.gpu_in_tray);
        CHECK(back == r, "round-trip rank %u -> %u", r, back);
        CHECK(p.gpu_in_rack < 72, "gpu_in_rack %u", p.gpu_in_rack);
    }
    CHECK(cai_same_rack(&t, 0, 71), "0,71 same rack");
    CHECK(!cai_same_rack(&t, 71, 72), "71,72 different rack");
}

static void test_arena(void) {
    printf("[arena]\n");
    uint64_t want[CAI_TCLASS_COUNT] = {100, 200, 0, 4096, 50, 7};
    cai_arena_t a;
    cai_arena_layout(&a, want);
    CHECK(a.seg_off[0] == 0, "first seg at 0");
    uint64_t sum = 0;
    for (int c = 0; c < CAI_TCLASS_COUNT; c++) {
        CHECK(a.seg_off[c] % CAI_ARENA_ALIGN == 0, "seg %d aligned", c);
        CHECK(a.seg_len[c] == want[c], "seg %d len", c);
        sum += want[c];
    }
    CHECK(a.total >= sum, "total covers all segments");
    CHECK(a.total % CAI_ARENA_ALIGN == 0, "total aligned");

    cai_tensor_desc_t td[CAI_TCLASS_COUNT];
    for (int c = 0; c < CAI_TCLASS_COUNT; c++) {
        memset(&td[c], 0, sizeof(td[c]));
        td[c].tclass = (uint16_t)c;
        td[c].offset = 0;
    }
    cai_arena_relocate(&a, td, CAI_TCLASS_COUNT);
    for (int c = 0; c < CAI_TCLASS_COUNT; c++)
        CHECK(td[c].offset == a.seg_off[c], "tensor %d rebased", c);
}

static void test_pipeline(void) {
    printf("[pipeline]\n");
    CHECK(cai_pipeline_bubble(1, 100) == 0.0, "pp=1 no bubble");
    double b = cai_pipeline_bubble(16, 128);
    CHECK(b > 0.10 && b < 0.11, "pp=16 m=128 bubble=%.4f", b);

    uint32_t pp = 4, m = 8;
    cai_tick_t buf[64];
    for (uint32_t stage = 0; stage < pp; stage++) {
        uint32_t n = cai_pipeline_schedule_1f1b(stage, pp, m, buf);
        CHECK(n == 2 * m, "stage %u ticks=%u", stage, n);
        int fseen[8] = {0}, bseen[8] = {0};
        int fcount = 0, bcount = 0;
        int last_fwd_of[8];
        for (int i = 0; i < 8; i++) last_fwd_of[i] = -1;
        for (uint32_t i = 0; i < n; i++) {
            uint16_t mb = buf[i].microbatch;
            if (buf[i].kind == CAI_TICK_FWD) {
                fseen[mb]++;
                last_fwd_of[mb] = (int)i;
                fcount++;
            } else {
                bseen[mb]++;
                CHECK(last_fwd_of[mb] >= 0, "stage %u bwd %u before its fwd", stage, mb);
                bcount++;
            }
        }
        CHECK(fcount == (int)m && bcount == (int)m, "stage %u balance", stage);
        for (uint32_t mb = 0; mb < m; mb++)
            CHECK(fseen[mb] == 1 && bseen[mb] == 1, "stage %u mb %u once", stage, mb);
    }
}

static void make_small(cai_config_t *cfg) {
    cai_config_defaults(cfg);
    cfg->model.hidden_size = 4096;
    cfg->model.ffn_hidden = 14336;
    cfg->model.num_layers = 24;
    cfg->model.num_heads = 32;
    cfg->model.num_kv_heads = 8;
    cfg->model.head_dim = 128;
    cfg->model.seq_len = 4096;
    cfg->model.tp = 4;
    cfg->model.pp = 3;
    cfg->model.cp = 1;
    cfg->model.grad_accum = 12;
    cfg->topo.num_racks = 1;
}

static void test_decompose(void) {
    printf("[decompose]\n");
    cai_config_t cfg;
    make_small(&cfg);
    char err[128];
    CHECK(cai_topology_finalize(&cfg.topo, err, sizeof(err)) == CAI_OK, "finalize");
    cai_decomp_t d;
    CHECK(cai_decompose(&cfg, &d, err, sizeof(err)) == CAI_OK, "decompose: %s", err);
    CHECK(d.model_replica_gpus == 12, "replica=%u", d.model_replica_gpus);
    CHECK(d.dp == 6, "dp=%u", d.dp);
    CHECK(d.global_batch == (uint64_t)1 * 12 * 6, "global_batch=%llu",
          (unsigned long long)d.global_batch);
    CHECK(d.mem_total > 0, "mem_total>0");

    /* indivisible world should be rejected */
    cfg.model.tp = 5; /* 5*3 = 15 does not divide 72 */
    cai_decomp_t d2;
    CHECK(cai_decompose(&cfg, &d2, err, sizeof(err)) == CAI_ERR_INVALID,
          "expected indivisible rejection");
}

static void test_plan_roundtrip(void) {
    printf("[plan-roundtrip]\n");
    cai_config_t cfg;
    make_small(&cfg);
    char err[128];
    cai_topology_finalize(&cfg.topo, err, sizeof(err));
    cai_decomp_t d;
    cai_decompose(&cfg, &d, err, sizeof(err));

    cai_topology_t topo = cfg.topo;
    cai_topology_finalize(&topo, err, sizeof(err));
    uint64_t th = cai_topology_hash(&topo);
    uint64_t ph = cai_plan_hash(&cfg, &d);

    cai_plan_t a;
    CHECK(cai_build_plan(&cfg, &d, 0, ph, th, &a, err, sizeof(err)) == CAI_OK,
          "build: %s", err);
    CHECK(a.hdr.num_ops > 0, "ops>0");
    CHECK(a.hdr.num_groups == 3, "groups=%u", a.hdr.num_groups);
    CHECK(a.hdr.pipeline_stage == 0, "stage0");

    const char *path = "/tmp/cai_test_plan.bin";
    CHECK(cai_plan_write(path, &a) == CAI_OK, "write");
    cai_plan_t b;
    CHECK(cai_plan_read(path, &b) == CAI_OK, "read");

    CHECK(memcmp(&a.hdr, &b.hdr, sizeof(a.hdr)) == 0, "header identical");
    CHECK(a.hdr.num_ops == b.hdr.num_ops &&
              memcmp(a.ops, b.ops, a.hdr.num_ops * sizeof(cai_op_t)) == 0,
          "ops identical");
    CHECK(memcmp(a.groups, b.groups, a.hdr.num_groups * sizeof(cai_comm_group_t)) == 0,
          "groups identical");

    uint64_t bd[CAI_TCLASS_COUNT];
    cai_plan_arena_breakdown(&b, bd);
    uint64_t sum = 0;
    for (int c = 0; c < CAI_TCLASS_COUNT; c++) sum += bd[c];
    CHECK(sum <= b.hdr.arena_bytes, "breakdown within arena (%llu<=%llu)",
          (unsigned long long)sum, (unsigned long long)b.hdr.arena_bytes);

    /* a plan must not load against a different topology */
    CHECK(b.hdr.topology_hash == th, "topo hash stored");
    cai_topology_t t2 = topo;
    t2.num_racks = 2;
    cai_topology_finalize(&t2, err, sizeof(err));
    CHECK(cai_topology_hash(&t2) != th, "different topo -> different hash");

    cai_plan_free(&a);
    cai_plan_free(&b);
    remove(path);
}

static void test_refmodel(void) {
    printf("[refmodel]\n");
    ref_spec_t s = {.vocab = 7, .seq = 4, .hidden = 8, .layers = 2,
                    .n_heads = 2, .head_dim = 4, .ffn = 16, .eps = 1e-6};
    ref_model_t *m = ref_model_create(&s, 12345);
    int B = 2, BT = B * s.seq;
    int tokens[8], targets[8];
    for (int i = 0; i < BT; i++) {
        tokens[i] = (i * 3 + 1) % s.vocab;
        targets[i] = (i * 5 + 2) % s.vocab;
    }
    /* backward must match central finite differences (fp64 oracle) */
    double rel = ref_grad_check(m, B, tokens, targets, 0, 1e-4);
    CHECK(rel < 1e-4, "grad check max rel err = %.2e", rel);

    /* training must actually reduce loss on a learnable (periodic) target */
    for (int i = 0; i < BT; i++) targets[i] = tokens[i]; /* identity is learnable */
    double l0 = ref_forward(m, B, tokens, targets);
    for (int step = 0; step < 300; step++) {
        ref_forward(m, B, tokens, targets);
        ref_backward(m, B, tokens, targets);
        ref_adamw(m, 0.02, 0.9, 0.999, 1e-8, 0.0);
    }
    double l1 = ref_forward(m, B, tokens, targets);
    CHECK(l1 < l0 * 0.5, "loss should drop: %.4f -> %.4f", l0, l1);
    ref_model_free(m);
}

static void test_costmodel(void) {
    printf("[costmodel]\n");
    cai_topology_t t;
    cai_topology_default(&t);
    t.num_racks = 64;
    char e[128];
    cai_topology_finalize(&t, e, sizeof(e));
    cai_backend_t *be = cai_backend_cpu_create(&t);
    CHECK(be != NULL, "backend create");
    if (!be) return;

    cai_op_t ar;
    memset(&ar, 0, sizeof(ar));
    ar.kind = CAI_OP_ALL_REDUCE;
    ar.bytes = 1u << 20; /* 1 MiB */
    cai_comm_group_t intra = {0}, inter = {0}, big = {0};
    intra.size = 8; intra.intra_rack = 1;
    inter.size = 8; inter.intra_rack = 0;
    big.size = 1024; big.intra_rack = 0;
    double ti = be->op_seconds(be, &ar, &intra);
    double te = be->op_seconds(be, &ar, &inter);
    double tb = be->op_seconds(be, &ar, &big);
    CHECK(ti < te, "intra-rack faster than inter (%.3e < %.3e)", ti, te);
    CHECK(tb > te, "bigger group has more latency (%.3e > %.3e)", tb, te);

    cai_op_t gm;
    memset(&gm, 0, sizeof(gm));
    gm.kind = CAI_OP_GEMM;
    gm.flops = (uint64_t)t.gpu_flops_bf16; /* ~1 second of compute */
    double tg = be->op_seconds(be, &gm, NULL);
    CHECK(tg > 0.99 && tg < 1.01, "gemm time = flops/peak ~1s (%.4f)", tg);
    be->destroy(be);
}

static void test_ep_memory(void) {
    printf("[ep-memory]\n");
    cai_config_t cfg;
    cai_config_defaults(&cfg);
    cfg.model.is_moe = 1;
    cfg.model.num_experts = 36;
    cfg.model.moe_top_k = 2;
    cfg.model.hidden_size = 1024;
    cfg.model.ffn_hidden = 2048;
    cfg.model.num_layers = 4;
    cfg.model.tp = 2;
    cfg.model.pp = 1;
    cfg.model.cp = 1;
    cfg.model.grad_accum = 8;
    cfg.topo.num_racks = 1; /* world=72, replica=2, dp=36 */
    char e[128];
    CHECK(cai_topology_finalize(&cfg.topo, e, sizeof(e)) == CAI_OK, "finalize");
    cai_decomp_t d2, d36;
    cfg.model.ep = 2;
    CHECK(cai_decompose(&cfg, &d2, e, sizeof(e)) == CAI_OK, "ep=2: %s", e);
    cfg.model.ep = 36;
    CHECK(cai_decompose(&cfg, &d36, e, sizeof(e)) == CAI_OK, "ep=36: %s", e);
    /* sharding experts over more EP ranks lowers per-GPU expert memory */
    CHECK(d2.mem_param > d36.mem_param,
          "ep=2 mem (%llu) > ep=36 mem (%llu)",
          (unsigned long long)d2.mem_param, (unsigned long long)d36.mem_param);
}

static void check_tiling(const cai_decomp_t *d, const cai_topology_t *t, int moe,
                         uint16_t kind, const char *name) {
    uint32_t W = t->world_size, *cnt = calloc(W, sizeof(uint32_t)), size = 0;
    for (uint32_t r = 0; r < W; r++) {
        cai_comm_group_t gs[CAI_GROUP_COUNT];
        uint32_t ng = 0;
        cai_rank_groups(d, t, moe, r, gs, &ng);
        for (uint32_t i = 0; i < ng; i++)
            if (gs[i].kind == kind) {
                size = gs[i].size;
                CHECK(gs[i].color < W, "%s color in range", name);
                cnt[gs[i].color]++;
            }
    }
    uint64_t total = 0;
    int buckets = 0, bad = 0;
    for (uint32_t c = 0; c < W; c++)
        if (cnt[c]) { buckets++; total += cnt[c]; if (cnt[c] != size) bad++; }
    CHECK(bad == 0 && total == W && (uint64_t)buckets * size == W,
          "%s tiles world (buckets=%d size=%u)", name, buckets, size);
    free(cnt);
}

static void test_grouptiling(void) {
    printf("[group-tiling]\n");
    cai_config_t cfg;
    make_small(&cfg);
    char e[128];
    cai_topology_finalize(&cfg.topo, e, sizeof(e));
    cai_decomp_t d;
    cai_decompose(&cfg, &d, e, sizeof(e));
    check_tiling(&d, &cfg.topo, 0, CAI_GROUP_TP, "TP");
    check_tiling(&d, &cfg.topo, 0, CAI_GROUP_DP, "DP");
    check_tiling(&d, &cfg.topo, 0, CAI_GROUP_PP, "PP");
}

int main(void) {
    cai_log_set_level(CAI_LOG_ERROR); /* quiet */
    test_costmodel();
    test_ep_memory();
    test_grouptiling();
    test_topology();
    test_arena();
    test_pipeline();
    test_decompose();
    test_plan_roundtrip();
    test_refmodel();
    if (g_fail == 0)
        printf("\nALL TESTS PASSED\n");
    else
        printf("\n%d CHECK(S) FAILED\n", g_fail);
    return g_fail ? 1 : 0;
}
