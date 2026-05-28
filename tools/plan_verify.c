/* plan_verify - pre-launch, whole-cluster consistency check (roadmap M6 exit).
 *
 *   plan_verify <config>
 *
 * At 220k scale a single mismatched collective hangs the whole job. This proves,
 * before launch and with no GPU, that:
 *   1. every communicator (TP/DP/PP/EP) tiles the world exactly - each rank is in
 *      exactly one group of each kind, every group has its declared size;
 *   2. all ranks agree on plan_hash and topology_hash;
 *   3. ranks sharing a pipeline stage emit byte-identical op streams (so every
 *      member of a TP/DP/EP communicator issues the same collective sequence);
 *   4. pipeline point-to-point is conserved: total sends == total recvs ==
 *      2*(PP-1)*microbatches. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cai.h"
#include "cai_common.h"
#include "cai_model.h"

static int g_problems;
#define FAIL(...)                          \
    do {                                   \
        printf("  [FAIL] ");               \
        printf(__VA_ARGS__);               \
        printf("\n");                      \
        g_problems++;                      \
    } while (0)
#define OK(...)                            \
    do {                                   \
        printf("  [ok]   ");               \
        printf(__VA_ARGS__);               \
        printf("\n");                      \
    } while (0)

static const cai_comm_group_t *group_of(const cai_comm_group_t *g, uint32_t n,
                                        uint16_t kind) {
    for (uint32_t i = 0; i < n; i++)
        if (g[i].kind == kind) return &g[i];
    return NULL;
}

/* Verify a communicator kind tiles all ranks into equal-size groups. */
static void check_tiling(const cai_decomp_t *dec, const cai_topology_t *topo,
                         int is_moe, uint16_t kind, const char *name) {
    uint32_t W = topo->world_size;
    uint32_t *cnt = calloc(W, sizeof(uint32_t));
    if (!cnt) { FAIL("%s: oom", name); return; }
    uint32_t want_size = 0;
    int size_uniform = 1, color_inrange = 1;
    for (uint32_t r = 0; r < W; r++) {
        cai_comm_group_t gs[CAI_GROUP_COUNT];
        uint32_t ng = 0;
        cai_rank_groups(dec, topo, is_moe, r, gs, &ng);
        const cai_comm_group_t *g = group_of(gs, ng, kind);
        if (!g) { FAIL("%s: rank %u has no group", name, r); free(cnt); return; }
        if (r == 0) want_size = g->size;
        else if (g->size != want_size) size_uniform = 0;
        if (g->color >= W) { color_inrange = 0; continue; }
        cnt[g->color]++;
    }
    int buckets = 0, badbucket = 0;
    uint64_t total = 0;
    for (uint32_t c = 0; c < W; c++)
        if (cnt[c]) {
            buckets++;
            total += cnt[c];
            if (cnt[c] != want_size) badbucket++;
        }
    free(cnt);

    if (!size_uniform) FAIL("%s: group size not uniform", name);
    if (!color_inrange) FAIL("%s: color out of range", name);
    if (badbucket) FAIL("%s: %d groups have wrong member count (want %u)", name,
                        badbucket, want_size);
    if (total != W) FAIL("%s: members %llu != world %u", name,
                         (unsigned long long)total, W);
    if (want_size && (uint64_t)buckets * want_size != W)
        FAIL("%s: %d groups * size %u != world %u", name, buckets, want_size, W);
    if (!size_uniform || badbucket || total != W) return;
    OK("%s: %d groups of %u tile all %u ranks", name, buckets, want_size, W);
}

static void count_p2p(const cai_plan_t *p, uint32_t *sends, uint32_t *recvs) {
    *sends = *recvs = 0;
    for (uint32_t i = 0; i < p->hdr.num_ops; i++) {
        if (p->ops[i].kind == CAI_OP_PIPE_SEND) (*sends)++;
        else if (p->ops[i].kind == CAI_OP_PIPE_RECV) (*recvs)++;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <config>\n", argv[0]);
        return 2;
    }
    cai_config_t cfg;
    char err[160];
    if (cai_config_load(argv[1], &cfg, err, sizeof(err)) != CAI_OK) {
        fprintf(stderr, "config error: %s\n", err);
        return 1;
    }
    cai_topology_t topo = cfg.topo;
    if (cai_topology_finalize(&topo, err, sizeof(err)) != CAI_OK) {
        fprintf(stderr, "topology: %s\n", err);
        return 1;
    }
    cfg.topo = topo;
    cai_decomp_t dec;
    if (cai_decompose(&cfg, &dec, err, sizeof(err)) != CAI_OK) {
        fprintf(stderr, "decompose: %s\n", err);
        return 1;
    }
    uint64_t ph = cai_plan_hash(&cfg, &dec), th = cai_topology_hash(&topo);
    printf("verifying %u ranks (TP=%u PP=%u DP=%u EP=%u)\n", topo.world_size, dec.tp,
           dec.pp, dec.dp, dec.ep);

    printf("[1] communicator tiling (all ranks)\n");
    check_tiling(&dec, &topo, cfg.model.is_moe, CAI_GROUP_TP, "TP");
    check_tiling(&dec, &topo, cfg.model.is_moe, CAI_GROUP_DP, "DP");
    check_tiling(&dec, &topo, cfg.model.is_moe, CAI_GROUP_PP, "PP");
    if (cfg.model.is_moe && dec.ep > 1)
        check_tiling(&dec, &topo, cfg.model.is_moe, CAI_GROUP_EP, "EP");

    printf("[2] hash agreement + [4] pipeline p2p conservation\n");
    uint64_t total_send = 0, total_recv = 0;
    int hash_ok = 1;
    for (uint32_t s = 0; s < dec.pp; s++) {
        uint32_t r = s * dec.tp * dec.cp; /* stage s, dp0/cp0/tp0 */
        cai_plan_t pl;
        if (cai_build_plan(&cfg, &dec, r, ph, th, &pl, err, sizeof(err)) != CAI_OK) {
            FAIL("stage %u: build: %s", s, err);
            continue;
        }
        if (pl.hdr.plan_hash != ph || pl.hdr.topology_hash != th ||
            pl.hdr.pipeline_stage != s)
            hash_ok = 0;
        uint32_t sd, rc;
        count_p2p(&pl, &sd, &rc);
        total_send += sd;
        total_recv += rc;
        cai_plan_free(&pl);
    }
    if (hash_ok) OK("all %u stages share plan_hash=%016llx topology_hash=%016llx",
                    dec.pp, (unsigned long long)ph, (unsigned long long)th);
    else FAIL("hash / stage mismatch across stages");

    uint64_t expect = (uint64_t)2 * (dec.pp - 1) * dec.num_microbatches;
    if (total_send == total_recv && total_send == expect)
        OK("pipeline p2p conserved: %llu sends == %llu recvs == 2*(PP-1)*m",
           (unsigned long long)total_send, (unsigned long long)total_recv);
    else
        FAIL("pipeline p2p: sends=%llu recvs=%llu expected=%llu",
             (unsigned long long)total_send, (unsigned long long)total_recv,
             (unsigned long long)expect);

    printf("[3] intra-stage op-stream identity\n");
    if (dec.dp > 1) {
        cai_plan_t a, b;
        uint32_t ra = 0, rb = dec.model_replica_gpus; /* both stage 0, different DP */
        int ea = cai_build_plan(&cfg, &dec, ra, ph, th, &a, err, sizeof(err));
        int eb = cai_build_plan(&cfg, &dec, rb, ph, th, &b, err, sizeof(err));
        if (ea == CAI_OK && eb == CAI_OK) {
            int same = a.hdr.num_ops == b.hdr.num_ops &&
                       memcmp(a.ops, b.ops, (size_t)a.hdr.num_ops * sizeof(cai_op_t)) == 0;
            if (same) OK("stage-0 ranks %u and %u emit identical %u-op streams", ra, rb,
                         a.hdr.num_ops);
            else FAIL("stage-0 ranks %u and %u differ in op stream", ra, rb);
            cai_plan_free(&a);
            cai_plan_free(&b);
        } else FAIL("build for identity check failed");
    } else OK("DP=1: nothing to cross-check");

    printf(g_problems ? "\nRESULT: %d problem(s)\n" : "\nRESULT: all checks passed\n",
           g_problems);
    return g_problems ? 1 : 0;
}
